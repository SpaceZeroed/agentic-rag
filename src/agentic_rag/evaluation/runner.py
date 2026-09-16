"""A reproducible, explicitly labeled development set; fixed judgments, no evaluation-time judge."""

import hashlib
import platform
from collections import defaultdict
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

from pydantic import BaseModel, ConfigDict, Field

from agentic_rag.evaluation.metrics import retrieval_metrics
from agentic_rag.ingestion.models import ChunkingConfig, DocumentWriter, PreparedDocument
from agentic_rag.ingestion.service import prepare_document
from agentic_rag.retrieval.models import SearchFilter
from agentic_rag.retrieval.service import DenseRetriever


class Source(BaseModel):
    model_config = ConfigDict(extra="forbid")
    key: str
    path: str
    source_uri: str
    sha256: str


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    document: str
    text: str = Field(min_length=1)
    grade: int = Field(default=1, ge=1, le=3)


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    language: str
    query: str = Field(min_length=1)
    evidence: list[Evidence] = Field(min_length=1)


class Dataset(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    label_policy: str
    max_chars: int
    overlap: int
    documents: list[Source] = Field(min_length=1)
    questions: list[Question] = Field(min_length=1)


def load_dataset(
    path: Path,
) -> tuple[Dataset, dict[str, PreparedDocument], dict[str, dict[str, int]]]:
    dataset = Dataset.model_validate_json(path.read_text(encoding="utf-8"))
    if len({doc.key for doc in dataset.documents}) != len(dataset.documents) or len(
        {q.id for q in dataset.questions}
    ) != len(dataset.questions):
        raise ValueError("Dataset keys and question IDs must be unique")
    documents = {}
    for source in dataset.documents:
        document = prepare_document(
            path.parent / source.path,
            ChunkingConfig(dataset.max_chars, dataset.overlap),
            source_uri=source.source_uri,
        )
        if document.raw_sha256 != source.sha256:
            raise ValueError("Dataset source checksum changed; create a new dataset version")
        documents[source.key] = document
    if len({doc.document_id for doc in documents.values()}) != len(documents):
        raise ValueError("Dataset source URIs must be unique")
    labels: dict[str, dict[str, int]] = {}
    for question in dataset.questions:
        qrels: dict[str, int] = {}
        for evidence in question.evidence:
            if evidence.document not in documents:
                raise ValueError("Evidence refers to an unknown document")
            document = documents[evidence.document]
            # Each annotated passage must fit entirely in a labeled chunk.
            matching = [chunk for chunk in document.chunks if evidence.text in chunk.text]
            if not matching:
                raise ValueError("Evidence is absent or split across chunk boundaries")
            for chunk in matching:
                qrels[str(chunk.id)] = max(qrels.get(str(chunk.id), 0), evidence.grade)
        labels[question.id] = qrels
    return dataset, documents, labels


def evaluate(path: Path, retriever: DenseRetriever, writer: DocumentWriter) -> dict[str, object]:
    dataset, documents, labels = load_dataset(path)
    for document in documents.values():
        writer.save(document)
    filters = SearchFilter(tuple(document.document_id for document in documents.values()))
    started = perf_counter()
    indexed = retriever.sync(filters)
    indexing_seconds = perf_counter() - started
    allowed = set(retriever.catalog.searchable_revisions(retriever.index.collection, filters))
    if allowed != {document.revision_id for document in documents.values()}:
        raise RuntimeError("Evaluation requires the complete dataset to be current and indexed")
    aggregates: dict[str, list[float]] = defaultdict(list)
    by_language: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    cases: list[dict[str, object]] = []
    for question in dataset.questions:
        started = perf_counter()
        hits = retriever.search(question.query, k=5, filters=filters, exact=True)
        elapsed = perf_counter() - started
        ranked = [str(hit.chunk_id) for hit in hits]
        metrics = {}
        for k in (1, 3, 5):
            metrics.update(retrieval_metrics(ranked, labels[question.id], k))
        for key, value in metrics.items():
            aggregates[key].append(value)
            by_language[question.language][key].append(value)
        cases.append(
            {
                "id": question.id,
                "language": question.language,
                "query": question.query,
                "qrels": labels[question.id],
                "hits": [
                    asdict(hit)
                    | {
                        "chunk_id": str(hit.chunk_id),
                        "document_id": str(hit.document_id),
                        "revision_id": str(hit.revision_id),
                    }
                    for hit in hits
                ],
                "metrics": metrics,
                "search_seconds": elapsed,
            }
        )
    return {
        "dataset": dataset.name,
        "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "label_policy": dataset.label_policy,
        "timestamp": datetime.now(UTC).isoformat(),
        "documents": len(documents),
        "chunks": sum(len(doc.chunks) for doc in documents.values()),
        "questions": len(cases),
        "model": asdict(retriever.embeddings.spec),
        "collection": retriever.index.collection,
        "exact_search": True,
        "chunking": {"max_chars": dataset.max_chars, "overlap": dataset.overlap},
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                name: version(name) for name in ("torch", "transformers", "qdrant-client")
            },
        },
        "indexing": asdict(indexed) | {"seconds": indexing_seconds},
        "metrics": {key: sum(values) / len(values) for key, values in aggregates.items()},
        "metrics_by_language": {
            language: {key: sum(values) / len(values) for key, values in values_by_key.items()}
            for language, values_by_key in by_language.items()
        },
        "cases": cases,
    }
