from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr
from sqlalchemy import Engine

from agentic_rag.api.app import create_app
from agentic_rag.core.config import Settings
from agentic_rag.ingestion.models import ChunkingConfig
from agentic_rag.ingestion.service import prepare_document
from agentic_rag.storage.catalog import document_catalog
from agentic_rag.storage.repository import PostgresDocumentRepository
from agentic_rag.tools.models import CatalogInput

pytestmark = pytest.mark.postgres


def test_catalog_current_revisions_pagination_and_literal_filters(
    database: Engine, tmp_path: Path
) -> None:
    repository = PostgresDocumentRepository(database)
    path = tmp_path / "100%_paper.md"
    path.write_text("Old text.", encoding="utf-8")
    first = repository.save(prepare_document(path, ChunkingConfig(50, 0)))
    path.write_text("New text.\n" * 10, encoding="utf-8")
    revised = repository.save(prepare_document(path, ChunkingConfig(50, 0)))
    second_path = tmp_path / "another.md"
    second_path.write_text("Other.", encoding="utf-8")
    second = repository.save(prepare_document(second_path, ChunkingConfig(50, 0)))
    result = document_catalog(database, CatalogInput(limit=1))
    next_page = document_catalog(database, CatalogInput(limit=1, offset=1))
    assert result.has_more and not next_page.has_more
    assert {result.documents[0].document_id, next_page.documents[0].document_id} == {
        first.document_id,
        second.document_id,
    }
    filtered = document_catalog(database, CatalogInput(title_contains="%_"))
    assert len(filtered.documents) == 1
    assert filtered.documents[0].revision_id == revised.revision_id != first.revision_id
    assert filtered.documents[0].chunk_count == revised.chunk_count
    assert not document_catalog(database, CatalogInput(title_contains="' OR 1=1 --")).documents
    assert not document_catalog(database, CatalogInput(offset=100)).documents


@pytest.mark.anyio
async def test_agent_real_storage_catalog_and_document_search(database: Engine) -> None:
    settings = Settings(
        database_url=SecretStr(database.url.render_as_string(hide_password=False)),
        api_llm_provider="fake",
    )
    app = create_app(settings)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        uploaded = await client.post(
            "/documents",
            json={
                "filename": "paper.md",
                "source_uri": "https://example.org/agent",
                "content": "PagedAttention stores cache in blocks.",
            },
        )
        assert uploaded.status_code == 201
        identity = uploaded.json()["document"]
        catalog = await client.post("/agent", json={"query": "/catalog"})
        assert catalog.status_code == 200
        item = catalog.json()["observations"][0]["data"]["documents"][0]
        assert item["document_id"] == identity["document_id"]
        assert item["revision_id"] == identity["revision_id"]
        answer = await client.post("/agent", json={"query": "/search PagedAttention"})
        assert answer.status_code == 200
        citation = answer.json()["citations"][0]["source"]
        assert citation["document_id"] == identity["document_id"]
        assert citation["revision_id"] == identity["revision_id"]
        assert citation["text"] == "PagedAttention stores cache in blocks."
        missing = await client.post("/agent", json={"query": "/search NonexistentUniqueWord"})
        assert missing.status_code == 200
        assert missing.json()["status"] == "insufficient_evidence"
