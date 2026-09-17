"""Application entry point: startup, text ingestion, and database migrations."""

import argparse
import json
import logging
from collections.abc import Iterator, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from pathlib import Path
from uuid import UUID

import httpx
from alembic.util.exc import CommandError
from pydantic import ValidationError
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from sqlalchemy.exc import SQLAlchemyError

from agentic_rag.core.config import Settings
from agentic_rag.core.logging import configure_logging
from agentic_rag.embeddings.e5 import E5EmbeddingProvider
from agentic_rag.ingestion.models import ChunkingConfig
from agentic_rag.ingestion.parsing import DocumentInputError
from agentic_rag.ingestion.service import prepare_document
from agentic_rag.llm.base import LLM, LLMError
from agentic_rag.llm.compatible import CompatibleLLM
from agentic_rag.llm.fake import FakeLLM
from agentic_rag.rag.service import answer
from agentic_rag.reranking.cross_encoder import CrossEncoderProvider
from agentic_rag.reranking.service import rerank
from agentic_rag.retrieval.hybrid import HybridRetriever
from agentic_rag.retrieval.models import SearchFilter, SearchHit
from agentic_rag.retrieval.qdrant import QdrantVectorIndex
from agentic_rag.retrieval.service import DenseRetriever
from agentic_rag.retrieval.sparse import TOKENIZER_VERSION, BM25Config, SparseRetriever
from agentic_rag.storage.database import create_database_engine, upgrade_database
from agentic_rag.storage.repository import PostgresDocumentRepository
from agentic_rag.storage.retrieval import PostgresIndexCatalog

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ML/LLM document ingestion")
    commands = parser.add_subparsers(dest="command")
    for name in ("preview", "ingest"):
        command = commands.add_parser(name, help=f"{name.capitalize()} a UTF-8 Markdown/TXT file")
        command.add_argument("path", type=Path)
        command.add_argument(
            "--source-uri", help="Stable file/HTTP/HTTPS identity; no URL is fetched"
        )
        command.add_argument("--max-chars", type=int, default=1200)
        command.add_argument("--overlap", type=int, default=200)
    commands.add_parser("db-upgrade", help="Apply packaged Alembic migrations")
    commands.add_parser("model-check", help="Load pinned CPU model and verify embedding output")
    for name in ("index", "search", "ask", "evaluate"):
        command = commands.add_parser(name, help=f"{name.capitalize()} dense retrieval")
        command.add_argument("--collection-prefix")
        if name in {"search", "ask", "evaluate"}:
            command.add_argument(
                "--rerank", action="store_true", help="Score retrieved pairs with CPU cross-encoder"
            )
            command.add_argument(
                "--rerank-k",
                type=int,
                default=20,
                help="Candidate budget for reranking (default: 20)",
            )
        if name == "evaluate":
            command.add_argument("dataset", type=Path)
            command.add_argument("--output", type=Path, required=True)
            command.add_argument(
                "--compare", action="store_true", help="Compare dense, BM25 and RRF"
            )
        else:
            command.add_argument("--document-id", type=UUID, action="append", default=[])
            command.add_argument("--source-uri")
            command.add_argument("--media-type", choices=["text/plain", "text/markdown"])
        if name in {"search", "ask"}:
            command.add_argument("query")
            command.add_argument("--mode", choices=["dense", "bm25", "hybrid"], default="dense")
            command.add_argument("--candidate-k", type=int, default=20)
            command.add_argument("--k", type=int, default=5)
            command.add_argument("--approximate", action="store_true")
        if name == "ask":
            command.add_argument("--llm", choices=["fake", "compatible"], default="fake")
    evaluation = commands.add_parser(
        "evaluate-rag", help="Evaluate retrieval/context and export answers"
    )
    evaluation.add_argument("dataset", type=Path)
    evaluation.add_argument("--output", type=Path, required=True)
    evaluation.add_argument("--mode", choices=["dense", "bm25", "hybrid"], default="bm25")
    evaluation.add_argument("--llm", choices=["fake", "compatible"], default="fake")
    evaluation.add_argument("--k", type=int, default=5)
    evaluation.add_argument("--candidate-k", type=int, default=20)
    evaluation.add_argument("--rerank", action="store_true")
    evaluation.add_argument("--rerank-k", type=int, default=20)
    evaluation.add_argument("--collection-prefix")
    review = commands.add_parser("review-rag", help="Export a review template or score annotations")
    review.add_argument("report", type=Path)
    review.add_argument("--annotations", type=Path)
    review.add_argument("--output", type=Path, required=True)
    show = commands.add_parser("show", help="Read a stored document and its chunks as JSON")
    show.add_argument("document_id", type=UUID)
    show.add_argument(
        "--revision", type=UUID, help="Read a retained revision instead of the current one"
    )
    return parser


def _execute(args: argparse.Namespace, settings: Settings) -> int:
    if args.command == "review-rag":
        from agentic_rag.evaluation.rag_review import Review, review_template, score_review

        report = json.loads(args.report.read_text(encoding="utf-8"))
        review_result = (
            score_review(
                report, Review.model_validate_json(args.annotations.read_text(encoding="utf-8"))
            )
            if args.annotations
            else review_template(report)
        )
        _write_new_json(args.output, review_result)
        print(json.dumps({"output": str(args.output)}))
        return 0
    if args.command == "evaluate-rag":
        return _evaluate_rag(args, settings)
    if args.command == "model-check":
        model = _load_embeddings(settings)
        vector = model.embed_query("embedding model check")
        print(
            json.dumps(
                {
                    "spec": asdict(model.spec),
                    "fingerprint": model.spec.fingerprint,
                    "norm_squared": sum(v * v for v in vector),
                }
            )
        )
        return 0
    if args.command in {"search", "ask", "evaluate"}:
        if not 1 <= args.rerank_k <= 100:
            raise ValueError("rerank-k must be 1..100")
        if args.rerank and args.rerank_k < (args.k if args.command in {"search", "ask"} else 5):
            raise ValueError("rerank-k must cover final k")
    if args.command in {"search", "ask"}:
        if not args.query.strip() or not 1 <= args.k <= 100:
            raise ValueError("Query must be nonempty; k must be 1..100")
        if args.rerank and args.mode == "hybrid" and args.rerank_k > args.candidate_k:
            raise ValueError("Hybrid candidate-k must cover rerank-k")
    if args.command == "evaluate" and args.rerank and args.compare:
        raise ValueError("Use --compare or --rerank, not both")
    if args.command == "ask" and args.llm == "compatible" and not settings.llm_model:
        raise ValueError("Set RAG_LLM_MODEL for compatible generation")
    prepared = None
    if args.command in {"preview", "ingest"}:
        prepared = prepare_document(
            args.path,
            ChunkingConfig(args.max_chars, args.overlap),
            source_uri=args.source_uri,
        )
        if args.command == "preview":
            print(json.dumps(asdict(prepared), ensure_ascii=False, default=str))
            return 0
    if settings.database_url is None:
        logger.error("database_url_required")
        return 2
    engine = create_database_engine(settings.database_url.get_secret_value())
    try:
        if args.command in {"search", "ask"} and args.mode == "bm25":
            if args.approximate or args.collection_prefix:
                raise ValueError("BM25 does not use an approximate index or vector collection")
            hits = SparseRetriever(PostgresIndexCatalog(engine)).search(
                args.query,
                k=args.rerank_k if args.rerank else args.k,
                filters=SearchFilter(tuple(args.document_id), args.source_uri, args.media_type),
            )
            if args.rerank and hits:
                hits = list(
                    rerank(
                        args.query,
                        hits,
                        _load_reranker(settings),
                        PostgresIndexCatalog(engine),
                        None,
                        k=args.k,
                    )
                )
            _print_search(args, settings, hits, None)
            return 0
        if args.command in {"index", "search", "ask", "evaluate"}:
            model = _load_embeddings(settings)
            client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key.get_secret_value()
                if settings.qdrant_api_key
                else None,
                timeout=30,
            )
            try:
                index = QdrantVectorIndex(
                    client, model.spec, args.collection_prefix or settings.collection_prefix
                )
                retriever = DenseRetriever(model, index, PostgresIndexCatalog(engine))
                if args.command == "evaluate":
                    from agentic_rag.evaluation.runner import compare, compare_reranking, evaluate

                    if args.rerank:
                        report = compare_reranking(
                            args.dataset,
                            retriever,
                            PostgresDocumentRepository(engine),
                            _load_reranker(settings),
                            candidate_k=args.rerank_k,
                        )
                    else:
                        run = compare if args.compare else evaluate
                        report = run(args.dataset, retriever, PostgresDocumentRepository(engine))
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(
                        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                    )
                    print(json.dumps({"output": str(args.output), "metrics": report["metrics"]}))
                else:
                    filters = SearchFilter(
                        tuple(args.document_id), args.source_uri, args.media_type
                    )
                    if args.command == "index":
                        print(json.dumps(asdict(retriever.sync(filters))))
                    else:
                        searcher = (
                            HybridRetriever(retriever, candidate_k=args.candidate_k)
                            if args.mode == "hybrid"
                            else retriever
                        )
                        hits = searcher.search(
                            args.query,
                            k=args.rerank_k if args.rerank else args.k,
                            filters=filters,
                            exact=not args.approximate,
                        )
                        if args.rerank and hits:
                            hits = list(
                                rerank(
                                    args.query,
                                    hits,
                                    _load_reranker(settings),
                                    retriever.catalog,
                                    index.collection,
                                    k=args.k,
                                )
                            )
                        _print_search(args, settings, hits, index.collection)
                return 0
            finally:
                client.close()
        if args.command == "db-upgrade":
            upgrade_database(engine)
            logger.info("database_upgraded")
            return 0
        repository = PostgresDocumentRepository(engine)
        if args.command == "ingest" and prepared is not None:
            result = repository.save(prepared)
            print(json.dumps(asdict(result), default=str))
            logger.info(
                "document_ingested",
                extra={"fields": {"document_id": str(result.document_id), "status": result.status}},
            )
            return 0
        document = repository.get(args.document_id, revision_id=args.revision)
        if document is None:
            logger.error("document_not_found")
            return 1
        print(json.dumps(asdict(document), ensure_ascii=False, default=str))
        return 0
    finally:
        engine.dispose()


def main(argv: Sequence[str] | None = None) -> int:
    """Validate external inputs and keep infrastructure errors out of logs."""
    args = build_parser().parse_args(argv)
    try:
        settings = Settings()
    except ValidationError as exc:
        configure_logging("ERROR")
        # Report field names and error types without echoing external input.
        errors = [
            {"field": ".".join(str(part) for part in error["loc"]), "type": error["type"]}
            for error in exc.errors(include_url=False, include_context=False, include_input=False)
        ]
        logger.error("configuration_invalid", extra={"fields": {"errors": errors}})
        return 2

    configure_logging(settings.log_level)
    if args.command is not None:
        try:
            return _execute(args, settings)
        except (DocumentInputError, OSError, ValueError) as exc:
            logger.error("input_invalid", extra={"fields": {"error_type": type(exc).__name__}})
            return 2
        except (SQLAlchemyError, CommandError) as exc:
            # Driver messages may contain connection credentials or document text.
            logger.error("database_failed", extra={"fields": {"error_type": type(exc).__name__}})
            return 1
        except LLMError as exc:
            logger.error("generation_failed", extra={"fields": {"error_type": type(exc).__name__}})
            return 1
        except (UnexpectedResponse, ResponseHandlingException, RuntimeError) as exc:
            logger.error("retrieval_failed", extra={"fields": {"error_type": type(exc).__name__}})
            return 1
    logger.info(
        "application_ready",
        extra={
            "fields": {
                "environment": settings.environment,
                "data_dir": str(settings.data_dir),
            }
        },
    )
    return 0


def _load_embeddings(settings: Settings) -> E5EmbeddingProvider:
    return E5EmbeddingProvider(
        settings.data_dir / "models",
        model_id=settings.embedding_model,
        revision=settings.embedding_revision,
        batch_size=settings.embedding_batch_size,
        threads=settings.embedding_threads,
        local_files_only=settings.model_local_files_only,
    )


def _load_reranker(settings: Settings) -> CrossEncoderProvider:
    return CrossEncoderProvider(
        settings.data_dir / "models",
        model_id=settings.reranking_model,
        revision=settings.reranking_revision,
        batch_size=settings.reranking_batch_size,
        threads=settings.reranking_threads,
        local_files_only=settings.model_local_files_only,
    )


def _print_search(
    args: argparse.Namespace,
    settings: Settings,
    hits: Sequence[SearchHit],
    collection: str | None,
) -> None:
    payload: dict[str, object] = {"mode": args.mode, "reranked": args.rerank}
    if collection is not None:
        payload["collection"] = collection
    if args.command == "ask":
        with _generation_provider(args.llm, settings) as llm:
            result = answer(
                args.query,
                hits,
                llm,
                max_prompt_bytes=settings.llm_max_prompt_bytes,
                max_tokens=settings.llm_max_tokens,
            )
        payload["llm_provider"] = args.llm
        payload["answer"] = asdict(result)
    else:
        payload["hits"] = [asdict(hit) for hit in hits]
    print(json.dumps(payload, ensure_ascii=False, default=str))


@contextmanager
def _generation_provider(provider: str, settings: Settings) -> Iterator[LLM]:
    if provider == "fake":
        yield FakeLLM()
        return
    if not settings.llm_model or not settings.llm_model.strip():
        raise ValueError("Set RAG_LLM_MODEL for compatible generation")
    headers = {}
    if settings.llm_api_key:
        headers["Authorization"] = f"Bearer {settings.llm_api_key.get_secret_value()}"
    with httpx.Client(
        base_url=settings.llm_base_url.rstrip("/") + "/",
        headers=headers,
        timeout=settings.llm_timeout_seconds,
        follow_redirects=False,
        trust_env=False,
    ) as client:
        yield CompatibleLLM(
            client, settings.llm_model, reasoning_enabled=settings.llm_reasoning_enabled
        )


def _write_new_json(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        output.write(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n")


def _evaluate_rag(args: argparse.Namespace, settings: Settings) -> int:
    from agentic_rag.evaluation.rag_dataset import load_rag_dataset
    from agentic_rag.evaluation.rag_runner import run_rag_evaluation

    if args.output.exists():
        raise ValueError("Output already exists; choose a new report path")
    if not 1 <= args.k <= 100 or not args.k <= args.candidate_k <= 100:
        raise ValueError("Require 1 <= k <= candidate-k <= 100")
    if not 1 <= args.rerank_k <= 100 or (args.rerank and args.rerank_k < args.k):
        raise ValueError("Invalid rerank-k")
    if args.rerank and args.mode == "hybrid" and args.rerank_k > args.candidate_k:
        raise ValueError("Hybrid candidate-k must cover rerank-k")
    if args.mode == "bm25" and args.collection_prefix:
        raise ValueError("BM25 does not use a vector collection")
    dataset, documents = load_rag_dataset(args.dataset)
    if settings.database_url is None:
        raise ValueError("RAG_DATABASE_URL required")
    with ExitStack() as stack:
        llm = stack.enter_context(_generation_provider(args.llm, settings))
        engine = create_database_engine(settings.database_url.get_secret_value())
        stack.callback(engine.dispose)
        catalog = PostgresIndexCatalog(engine)
        writer = PostgresDocumentRepository(engine)
        for document in documents.values():
            writer.save(document)
        filters = SearchFilter(tuple(d.document_id for d in documents.values()))
        config: dict[str, object] = {
            "mode": args.mode,
            "bm25": asdict(BM25Config()),
            "tokenizer": TOKENIZER_VERSION,
            "rrf_rank_constant": 60,
            "exact_search": True,
            "candidate_k": args.candidate_k,
            "rerank_k": args.rerank_k,
            "llm_model": settings.llm_model if args.llm == "compatible" else "fake-extractive-v1",
            "llm_timeout_seconds": settings.llm_timeout_seconds,
            "temperature": 0,
            "reasoning_enabled_requested": settings.llm_reasoning_enabled,
            "chunking": {"max_chars": dataset.max_chars, "overlap": dataset.overlap},
        }
        dense = None
        collection = None
        if args.mode != "bm25":
            embeddings = _load_embeddings(settings)
            client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key.get_secret_value()
                if settings.qdrant_api_key
                else None,
                timeout=30,
            )
            stack.callback(client.close)
            index = QdrantVectorIndex(
                client, embeddings.spec, args.collection_prefix or settings.collection_prefix
            )
            collection = index.collection
            dense = DenseRetriever(embeddings, index, catalog)
            dense.sync(filters)
            config.update({"embeddings": asdict(embeddings.spec), "collection": collection})
        ranking = _load_reranker(settings) if args.rerank else None
        config["reranker"] = asdict(ranking.spec) if ranking else None

        def check_revisions() -> None:
            actual = (
                {d.revision_id for d in catalog.current_documents(filters)}
                if collection is None
                else set(catalog.searchable_revisions(collection, filters))
            )
            if actual != {d.revision_id for d in documents.values()}:
                raise RuntimeError("Evaluation dataset changed or is not fully indexed")

        def search(query: str) -> list[SearchHit]:
            check_revisions()
            count = args.rerank_k if ranking else args.k
            if dense is None:
                hits = SparseRetriever(catalog).search(query, k=count, filters=filters)
            else:
                retriever = (
                    HybridRetriever(dense, candidate_k=args.candidate_k)
                    if args.mode == "hybrid"
                    else dense
                )
                hits = retriever.search(query, k=count, filters=filters, exact=True)
            if ranking:
                hits = list(rerank(query, hits, ranking, catalog, collection, k=args.k))
            return hits

        check_revisions()
        report = run_rag_evaluation(
            args.dataset,
            search,
            llm,
            provider=args.llm,
            config=config,
            k=args.k,
            max_prompt_bytes=settings.llm_max_prompt_bytes,
            max_tokens=settings.llm_max_tokens,
        )
        check_revisions()
        _write_new_json(args.output, report)
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "metrics": report["metrics"],
                    "errors": report["errors"],
                }
            )
        )
        return 0
