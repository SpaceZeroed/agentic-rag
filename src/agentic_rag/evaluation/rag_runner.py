"""Run RAG without exposing reference answers to the generation provider."""

import hashlib
import json
import platform
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Literal, cast

from agentic_rag.evaluation.rag_dataset import load_rag_dataset
from agentic_rag.evaluation.rag_metrics import evidence_metrics
from agentic_rag.llm.base import LLM, Completion, IncompleteCompletionError, LLMError, Message
from agentic_rag.llm.fake import FakeLLM
from agentic_rag.rag.context import SYSTEM, build_context
from agentic_rag.rag.service import answer
from agentic_rag.retrieval.models import SearchHit
from agentic_rag.retrieval.sparse import TOKENIZER_VERSION, BM25Config, BM25Index


def fingerprint(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, default=str, allow_nan=False).encode()
    ).hexdigest()


class _RecordingLLM:
    def __init__(self, delegate: LLM) -> None:
        self.delegate = delegate
        self.completion: Completion | None = None

    def complete(self, messages: tuple[Message, ...], *, max_tokens: int) -> Completion:
        try:
            self.completion = self.delegate.complete(messages, max_tokens=max_tokens)
        except IncompleteCompletionError as exc:
            self.completion = exc.completion
            raise
        return self.completion


def run_rag_evaluation(
    path: Path,
    search: Callable[[str], list[SearchHit]],
    llm: LLM,
    *,
    provider: Literal["fake", "compatible"],
    config: dict[str, object],
    k: int = 5,
    max_prompt_bytes: int = 24000,
    max_tokens: int = 512,
) -> dict[str, object]:
    if not 1 <= k <= 100 or max_tokens < 1 or max_prompt_bytes < 1:
        raise ValueError("Invalid RAG evaluation budgets")
    dataset_bytes = path.read_bytes()
    dataset, documents = load_rag_dataset(path)
    # Validate all questions before making the first external generation request.
    for question in dataset.questions:
        build_context(question.query, [], max_prompt_bytes=max_prompt_bytes)
    canonical = {c.id: (d, c) for d in documents.values() for c in d.chunks}
    cases: list[dict[str, object]] = []
    values_by_metric: dict[str, list[float]] = {}
    for question in dataset.questions:
        started = perf_counter()
        hits = search(question.query)
        retrieval_seconds = perf_counter() - started
        if len(hits) > k or len({h.chunk_id for h in hits}) != len(hits):
            raise ValueError("Search returned duplicate hits or exceeded k")
        for hit in hits:
            expected = canonical.get(hit.chunk_id)
            if expected is None:
                raise ValueError("Search returned a chunk outside the dataset")
            doc, chunk = expected
            if (
                hit.revision_id,
                hit.document_id,
                hit.text,
                hit.source_uri,
                hit.start_char,
                hit.end_char,
                hit.start_line,
                hit.end_line,
            ) != (
                doc.revision_id,
                doc.document_id,
                chunk.text,
                doc.source_uri,
                chunk.start_char,
                chunk.end_char,
                chunk.start_line,
                chunk.end_line,
            ):
                raise ValueError("Search returned changed dataset provenance")
        context = build_context(question.query, hits, max_prompt_bytes=max_prompt_bytes)
        metrics = evidence_metrics(
            question, documents, hits, [s.source for s in context.sources], k
        )
        for name, value in metrics.items():
            values_by_metric.setdefault(name, [])
            if value is not None:
                values_by_metric[name].append(value)
        result = None
        error = None
        recording = _RecordingLLM(llm)
        generation_started = perf_counter()
        try:
            result = answer(
                question.query,
                hits,
                recording,
                max_prompt_bytes=max_prompt_bytes,
                max_tokens=max_tokens,
            )
        except LLMError as exc:
            # Do not hide failures by dropping the case or echoing transport bodies.
            error = type(exc).__name__
        cases.append(
            {
                "id": question.id,
                "language": question.language,
                "query": question.query,
                "answerable": question.answerable,
                "reference_answer": question.reference_answer,
                "evidence": [e.model_dump() for e in question.evidence],
                "hits": [asdict(h) for h in hits],
                "context": asdict(context),
                "answer": asdict(result) if result else None,
                "error": error,
                "raw_completion": asdict(recording.completion) if recording.completion else None,
                "metrics": metrics,
                "retrieval_seconds": retrieval_seconds,
                "generation_workflow_seconds": perf_counter() - generation_started,
                "total_seconds": perf_counter() - started,
            }
        )
    aggregates = {
        name: {"mean": sum(values) / len(values) if values else None, "count": len(values)}
        for name, values in values_by_metric.items()
    }
    if path.read_bytes() != dataset_bytes:
        raise ValueError("Dataset changed during evaluation")
    report: dict[str, object] = {
        "schema_version": 1,
        "dataset": dataset.name,
        "dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
        "label_policy": dataset.label_policy,
        "timestamp": datetime.now(UTC).isoformat(),
        "provider": provider,
        "run_kind": "contract_smoke" if provider == "fake" else "model_run",
        "config": {
            **config,
            "k": k,
            "max_prompt_bytes": max_prompt_bytes,
            "max_tokens": max_tokens,
            "system_prompt_sha256": fingerprint(SYSTEM),
        },
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "documents": {
            key: {"revision_id": str(d.revision_id), "sha256": d.raw_sha256}
            for key, d in documents.items()
        },
        "metrics": aggregates,
        "errors": sum(c["error"] is not None for c in cases),
        "answer_quality": None,
        "cases": cases,
    }
    # Convert UUIDs to JSON-native values; the hash also binds timings and answers.
    return cast(dict[str, object], json.loads(json.dumps(report, default=str)))


def run_offline_bm25_evaluation(
    path: Path,
    *,
    k: int = 5,
    max_prompt_bytes: int = 24000,
    max_tokens: int = 512,
) -> dict[str, object]:
    """Evaluate frozen sources without PostgreSQL, Qdrant, or an external LLM."""
    _, documents = load_rag_dataset(path)
    bm25 = BM25Index(list(documents.values()))
    canonical = {
        chunk.id: (document, chunk) for document in documents.values() for chunk in document.chunks
    }

    def search(query: str) -> list[SearchHit]:
        hits = []
        for candidate in bm25.query(query, k):
            document, chunk = canonical[candidate.chunk_id]
            hits.append(
                SearchHit(
                    chunk.id,
                    document.revision_id,
                    document.document_id,
                    candidate.score,
                    chunk.text,
                    document.source_uri,
                    document.title,
                    chunk.start_char,
                    chunk.end_char,
                    chunk.start_line,
                    chunk.end_line,
                )
            )
        return hits

    return run_rag_evaluation(
        path,
        search,
        FakeLLM(),
        provider="fake",
        config={
            "mode": "offline_bm25",
            "bm25": asdict(BM25Config()),
            "tokenizer": TOKENIZER_VERSION,
            "corpus_policy": "all frozen dataset chunks; one in-memory index per run",
            "llm_model": "fake-extractive-v1",
            "temperature": 0,
            "external_services": False,
        },
        k=k,
        max_prompt_bytes=max_prompt_bytes,
        max_tokens=max_tokens,
    )
