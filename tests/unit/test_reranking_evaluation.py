from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from uuid import UUID

import pytest

from agentic_rag.embeddings.base import EmbeddingSpec
from agentic_rag.evaluation import runner
from agentic_rag.ingestion.models import ChunkingConfig, DocumentWriter
from agentic_rag.ingestion.service import prepare_document
from agentic_rag.reranking.base import RerankingSpec
from agentic_rag.retrieval.models import Candidate, SearchFilter, SearchHit
from agentic_rag.retrieval.service import DenseRetriever


def test_paired_evaluation_uses_same_pool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "doc.txt"
    path.write_text("evidence")
    doc = prepare_document(path, ChunkingConfig(100, 0))
    relevant = str(doc.chunks[0].id)
    hits = [
        SearchHit(
            UUID(int=1),
            doc.revision_id,
            doc.document_id,
            10,
            "wrong",
            doc.source_uri,
            "title",
            0,
            5,
            1,
            1,
        ),
        SearchHit(
            doc.chunks[0].id,
            doc.revision_id,
            doc.document_id,
            1,
            "evidence",
            doc.source_uri,
            "title",
            0,
            8,
            1,
            1,
        ),
    ]
    dataset = runner.Dataset(
        name="fake",
        label_policy="test",
        max_chars=100,
        overlap=0,
        documents=[
            runner.Source(key="d", path="doc.txt", source_uri=doc.source_uri, sha256=doc.raw_sha256)
        ],
        questions=[
            runner.Question(
                id="q",
                language="en",
                query="query",
                evidence=[runner.Evidence(document="d", text="evidence")],
            )
        ],
    )
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(dataset.model_dump_json())

    class Catalog:
        def searchable_revisions(self, collection: str, filters: SearchFilter) -> list[UUID]:
            return [doc.revision_id]

        def hydrate(self, collection: str, candidates: Sequence[Candidate]) -> list[SearchHit]:
            by_id = {h.chunk_id: h for h in hits}
            return [replace(by_id[c.chunk_id], score=c.score) for c in candidates]

    class Hybrid:
        calls = 0

        def __init__(self, retriever: object, *, candidate_k: int) -> None:
            assert candidate_k == 20

        def search(
            self, query: str, *, k: int, filters: SearchFilter, exact: bool
        ) -> list[SearchHit]:
            Hybrid.calls += 1
            assert k == 20 and exact
            return hits

    class Provider:
        spec = RerankingSpec("fake", "a" * 40)

        def score(self, query: str, passages: Sequence[str]) -> list[float]:
            assert list(passages) == ["wrong", "evidence"]
            return [-2, 4]

    @dataclass
    class Sync:
        indexed_revisions: int = 0

    retriever = cast(
        DenseRetriever,
        SimpleNamespace(
            catalog=Catalog(),
            index=SimpleNamespace(collection="fake"),
            embeddings=SimpleNamespace(spec=EmbeddingSpec("fake", "a" * 40, 2, 512)),
            sync=lambda filters: Sync(),
        ),
    )
    writer = cast(DocumentWriter, SimpleNamespace(save=lambda document: None))
    monkeypatch.setattr(runner, "HybridRetriever", Hybrid)
    monkeypatch.setattr(runner, "version", lambda name: "test")
    report = runner.compare_reranking(dataset_path, retriever, writer, Provider())
    assert Hybrid.calls == 1
    metrics = cast(dict[str, dict[str, float]], report["metrics"])
    assert metrics["hybrid"]["mrr@5"] == 0.5
    assert metrics["hybrid_reranked"]["mrr@5"] == 1
    run = cast(dict[str, object], report["run"])
    cases = cast(list[dict[str, object]], run["cases"])
    assert cases[0]["candidate_ids"] == [str(UUID(int=1)), relevant]
    assert cases[0]["candidate_recall"] == 1
