from collections.abc import AsyncIterator, Sequence
from contextlib import ExitStack, asynccontextmanager
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

import httpx
import pytest
from pydantic import SecretStr
from sqlalchemy import Engine

from agentic_rag.api import runtime as runtime_module
from agentic_rag.api.app import create_app
from agentic_rag.api.runtime import PostgresBackend, Runtime, open_runtime
from agentic_rag.core.config import Settings
from agentic_rag.embeddings.base import EmbeddingSpec
from agentic_rag.ingestion.models import ChunkingConfig
from agentic_rag.ingestion.service import prepare_document
from agentic_rag.llm.async_client import AsyncCompatibleLLM, AsyncFakeLLM
from agentic_rag.storage.database import create_database_engine
from agentic_rag.storage.repository import PostgresDocumentRepository

pytestmark = [pytest.mark.postgres, pytest.mark.anyio]


class Embeddings:
    spec = EmbeddingSpec("api-test-only", "a" * 40, 2, 512)

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid"])
async def test_upload_query_revision_retry_and_filters(
    database: Engine,
    tmp_path: Path,
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    mode: Literal["bm25", "dense", "hybrid"],
) -> None:
    url = request.config.getoption("--qdrant-url")
    if mode != "bm25" and not url:
        pytest.skip("Use --qdrant-url for vector API integration")
    monkeypatch.setattr(runtime_module, "E5EmbeddingProvider", Embeddings)
    settings = Settings(
        database_url=SecretStr(database.url.render_as_string(hide_password=False)),
        api_retrieval_mode=mode,
        qdrant_url=str(url or "http://unused"),
        collection_prefix=f"api_{uuid4().hex}",
    )
    app = create_app(settings)
    payload = {
        "filename": "paper.md",
        "source_uri": "https://example.org/paper",
        "content": "PagedAttention uses blocks.\r\n",
        "max_chars": 100,
        "overlap": 0,
    }
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        backend: PostgresBackend = app.state.runtime.backend
        try:
            created = await client.post("/documents", json=payload)
            assert created.status_code == 201
            data = created.json()
            assert data["index_status"] == ("not_required" if mode == "bm25" else "ready")
            # HTTP and CLI must produce identical revisions, including original bytes.
            path = tmp_path / "paper.md"
            path.write_bytes(str(payload["content"]).encode())
            expected = prepare_document(
                path, ChunkingConfig(100, 0), source_uri=str(payload["source_uri"])
            )
            assert data["document"]["revision_id"] == str(expected.revision_id)
            repeated = await client.post("/documents", json=payload)
            assert repeated.status_code == 200
            assert repeated.json()["document"]["status"] == "unchanged"
            updated = await client.post(
                "/documents",
                json={**payload, "content": "PagedAttention allocates blocks on demand."},
            )
            assert updated.json()["document"]["status"] == "updated"
            answer = await client.post(
                "/query",
                json={"query": "PagedAttention", "document_ids": [data["document"]["document_id"]]},
            )
            assert answer.status_code == 200
            citation = answer.json()["citations"][0]["source"]
            assert citation["revision_id"] == updated.json()["document"]["revision_id"]
            assert "on demand" in citation["text"]
            missing = await client.post(
                "/query", json={"query": "PagedAttention", "document_ids": [str(uuid4())]}
            )
            assert missing.json()["completion"] is None
            assert (await client.get("/health")).status_code == 200
        finally:
            if backend.qdrant and backend.dense:
                backend.qdrant.delete_collection(backend.dense.index.collection)


async def test_saved_document_survives_index_failure_and_retry(
    database: Engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(database_url=SecretStr(database.url.render_as_string(hide_password=False)))
    attempts = 0

    class FailingIndex:
        def sync(self, filters: object) -> None:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("secret-service-url")

    @asynccontextmanager
    async def factory(config: Settings) -> AsyncIterator[Runtime]:
        with ExitStack() as resources:
            backend = PostgresBackend(config, resources)
            monkeypatch.setattr(backend, "dense", FailingIndex())
            yield Runtime(config, backend, AsyncFakeLLM())

    app = create_app(settings, runtime_factory=factory)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        payload = {
            "filename": "a.txt",
            "source_uri": "https://example.org/a",
            "content": "Document",
        }
        failed = await client.post("/documents", json=payload)
        assert failed.status_code == 503
        assert failed.json()["index_status"] == "failed"
        assert "secret-service" not in failed.text
        doc_id = UUID(failed.json()["document"]["document_id"])
        assert PostgresDocumentRepository(database).get(doc_id) is not None
        retried = await client.post("/documents", json=payload)
        assert retried.status_code == 200
        assert retried.json()["index_status"] == "ready"
        assert retried.json()["document"]["status"] == "unchanged"


@pytest.mark.parametrize(
    "change",
    [
        {"filename": "../../private.txt"},
        {"filename": "a.pdf"},
        {"content": "  "},
        {"content": "bad\u0000text"},
        {"source_uri": "relative"},
        {"max_chars": 100, "overlap": 100},
    ],
)
async def test_document_input_validation(database: Engine, change: dict[str, object]) -> None:
    settings = Settings(database_url=SecretStr(database.url.render_as_string(hide_password=False)))
    app = create_app(settings)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        response = await client.post(
            "/documents",
            json={
                "filename": "a.txt",
                "source_uri": "https://example.org/a",
                "content": "Valid text",
                **change,
            },
        )
        assert response.status_code == 422


async def test_lifespan_closes_http_and_database_on_failure(
    database: Engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engines: list[Engine] = []
    original = create_database_engine

    def make_engine(url: str) -> Engine:
        engine = original(url)
        engines.append(engine)
        return engine

    monkeypatch.setattr(runtime_module, "create_database_engine", make_engine)
    settings = Settings(
        database_url=SecretStr(database.url.render_as_string(hide_password=False)),
        api_llm_provider="compatible",
        llm_model="test",
    )
    with pytest.raises(RuntimeError, match="test failure"):
        async with open_runtime(settings) as runtime:
            llm = runtime.llm
            assert isinstance(llm, AsyncCompatibleLLM)
            client = llm.client
            assert not client.is_closed
            old_pool = engines[0].pool
            raise RuntimeError("test failure")
    assert client.is_closed
    assert engines[0].pool is not old_pool
