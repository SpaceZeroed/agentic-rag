"""Application entry point: startup, text ingestion, and database migrations."""

import argparse
import json
import logging
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from uuid import UUID

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
from agentic_rag.reranking.cross_encoder import CrossEncoderProvider
from agentic_rag.reranking.service import rerank
from agentic_rag.retrieval.hybrid import HybridRetriever
from agentic_rag.retrieval.models import SearchFilter
from agentic_rag.retrieval.qdrant import QdrantVectorIndex
from agentic_rag.retrieval.service import DenseRetriever
from agentic_rag.retrieval.sparse import SparseRetriever
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
    for name in ("index", "search", "evaluate"):
        command = commands.add_parser(name, help=f"{name.capitalize()} dense retrieval")
        command.add_argument("--collection-prefix")
        if name in {"search", "evaluate"}:
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
        if name == "search":
            command.add_argument("query")
            command.add_argument("--mode", choices=["dense", "bm25", "hybrid"], default="dense")
            command.add_argument("--candidate-k", type=int, default=20)
            command.add_argument("--k", type=int, default=5)
            command.add_argument("--approximate", action="store_true")
    show = commands.add_parser("show", help="Read a stored document and its chunks as JSON")
    show.add_argument("document_id", type=UUID)
    show.add_argument(
        "--revision", type=UUID, help="Read a retained revision instead of the current one"
    )
    return parser


def _execute(args: argparse.Namespace, settings: Settings) -> int:
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
    if args.command in {"search", "evaluate"}:
        if not 1 <= args.rerank_k <= 100:
            raise ValueError("rerank-k must be 1..100")
        if args.rerank and args.rerank_k < (args.k if args.command == "search" else 5):
            raise ValueError("rerank-k must cover final k")
    if args.command == "search":
        if not args.query.strip() or not 1 <= args.k <= 100:
            raise ValueError("Query must be nonempty; k must be 1..100")
        if args.rerank and args.mode == "hybrid" and args.rerank_k > args.candidate_k:
            raise ValueError("Hybrid candidate-k must cover rerank-k")
    if args.command == "evaluate" and args.rerank and args.compare:
        raise ValueError("Use --compare or --rerank, not both")
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
        if args.command == "search" and args.mode == "bm25":
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
            print(
                json.dumps(
                    {
                        "reranked": args.rerank,
                        "mode": "bm25",
                        "hits": [asdict(hit) for hit in hits],
                    },
                    ensure_ascii=False,
                    default=str,
                )
            )
            return 0
        if args.command in {"index", "search", "evaluate"}:
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
                        print(
                            json.dumps(
                                {
                                    "collection": index.collection,
                                    "mode": args.mode,
                                    "reranked": args.rerank,
                                    "hits": [asdict(hit) for hit in hits],
                                },
                                ensure_ascii=False,
                                default=str,
                            )
                        )
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
