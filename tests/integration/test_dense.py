from collections.abc import Iterator, Sequence
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import pytest
from qdrant_client import QdrantClient, models
from sqlalchemy import Engine

from agentic_rag.embeddings.base import EmbeddingSpec
from agentic_rag.ingestion.models import ChunkingConfig, PreparedDocument
from agentic_rag.ingestion.service import prepare_document
from agentic_rag.retrieval.models import SearchFilter
from agentic_rag.retrieval.qdrant import QdrantVectorIndex
from agentic_rag.retrieval.service import DenseRetriever
from agentic_rag.storage.repository import PostgresDocumentRepository
from agentic_rag.storage.retrieval import PostgresIndexCatalog

pytestmark = pytest.mark.postgres


class DeterministicEmbeddings:
    """Known unit vectors for infrastructure assertions, never quality measurements."""

    spec = EmbeddingSpec("test-only", "a" * 40, 2, 512)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, 0.0] if "alpha" in text else [0.6, 0.8] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


@pytest.fixture
def index(request: pytest.FixtureRequest) -> Iterator[QdrantVectorIndex]:
    url = request.config.getoption("--qdrant-url")
    if not url:
        pytest.skip("Use --qdrant-url with --postgres-url for real vector integration")
    client = QdrantClient(url=str(url), timeout=10)
    index = QdrantVectorIndex(client, DeterministicEmbeddings.spec, f"test_{uuid4().hex}")
    try:
        yield index
    finally:
        if client.collection_exists(index.collection):
            client.delete_collection(index.collection)
        client.close()


def document(tmp_path: Path, text: str = "alpha", name: str = "a.md") -> PreparedDocument:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return prepare_document(path, ChunkingConfig(40, 5), source_uri=f"https://example.org/{name}")


def retriever(database: Engine, index: QdrantVectorIndex) -> DenseRetriever:
    return DenseRetriever(DeterministicEmbeddings(), index, PostgresIndexCatalog(database))


def test_exact_cosine_filters_and_idempotency(
    database: Engine, index: QdrantVectorIndex, tmp_path: Path
) -> None:
    docs = [document(tmp_path), document(tmp_path, "beta", "b.txt")]
    repo = PostgresDocumentRepository(database)
    for doc in docs:
        repo.save(doc)
    search = retriever(database, index)
    assert search.sync().indexed_revisions == 2
    assert search.sync().skipped_revisions == 2
    hits = search.search("query", k=2)
    assert [hit.document_id for hit in hits] == [doc.document_id for doc in docs]
    assert [hit.score for hit in hits] == pytest.approx([1.0, 0.6])
    for filters in (
        SearchFilter(document_ids=(docs[1].document_id,)),
        SearchFilter(source_uri=docs[1].source_uri),
        SearchFilter(media_type="text/plain"),
    ):
        selected = search.search("query", k=1, filters=filters)
        assert len(selected) == 1 and selected[0].document_id == docs[1].document_id
    assert search.search("query", filters=SearchFilter(document_ids=(uuid4(),))) == []
    assert search.search("query", k=1, exact=False)[0].document_id == docs[0].document_id


def test_revision_switch_and_post_search_recheck(
    database: Engine, index: QdrantVectorIndex, tmp_path: Path
) -> None:
    repo = PostgresDocumentRepository(database)
    old = document(tmp_path)
    repo.save(old)
    search = retriever(database, index)
    search.sync()
    candidates = index.query([1.0, 0.0], [old.revision_id], 5, exact=True)
    new = document(tmp_path, "beta revision")
    repo.save(new)
    assert search.search("query") == []
    assert search.catalog.hydrate(index.collection, candidates) == []
    search.sync()
    assert {hit.revision_id for hit in search.search("query")} == {new.revision_id}
    repo.save(old)
    assert search.sync().skipped_revisions == 1
    assert {hit.revision_id for hit in search.search("query")} == {old.revision_id}


def test_partial_write_failure_retry_and_pending_recovery(
    database: Engine, index: QdrantVectorIndex, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    doc = document(tmp_path, "alpha beta paragraph. " * 9)
    PostgresDocumentRepository(database).save(doc)
    search = retriever(database, index)
    original = index.upsert

    def fail(document: PreparedDocument, vectors: Sequence[Sequence[float]]) -> None:
        original(replace(document, chunks=document.chunks[:1]), vectors[:1])
        raise RuntimeError("simulated lost connection after partial write")

    monkeypatch.setattr(index, "upsert", fail)
    with pytest.raises(RuntimeError, match="simulated"):
        search.sync()
    assert index.count_revision(doc.revision_id) == 1
    assert search.search("query") == []
    monkeypatch.setattr(index, "upsert", original)
    assert search.sync().indexed_revisions == 1
    assert index.count_revision(doc.revision_id) == len(doc.chunks)
    token = search.catalog.begin(index.collection, doc.revision_id, len(doc.chunks))
    assert search.search("query") == []
    assert search.sync().indexed_revisions == 1
    assert not search.catalog.finish(index.collection, doc.revision_id, token, "failed")
    assert search.search("query")


def test_lost_points_and_collection_are_rebuilt(
    database: Engine, index: QdrantVectorIndex, tmp_path: Path
) -> None:
    doc = document(tmp_path, "alpha text. " * 8)
    PostgresDocumentRepository(database).save(doc)
    search = retriever(database, index)
    search.sync()
    index.client.delete(index.collection, [str(doc.chunks[0].id)], wait=True)
    assert search.sync().indexed_revisions == 1
    assert index.count_revision(doc.revision_id) == len(doc.chunks)
    index.client.delete_collection(index.collection)
    assert search.sync().indexed_revisions == 1
    assert index.count_revision(doc.revision_id) == len(doc.chunks)


def test_incompatible_collection_is_rejected(index: QdrantVectorIndex) -> None:
    index.client.create_collection(
        index.collection,
        vectors_config=models.VectorParams(size=3, distance=models.Distance.COSINE),
    )
    with pytest.raises(ValueError, match="incompatible"):
        index.ensure(index.spec)
    other = QdrantVectorIndex(index.client, replace(index.spec, revision="b" * 40))
    assert other.collection != index.collection
