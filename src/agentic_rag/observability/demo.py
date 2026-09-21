"""Offline ASGI trace demo: real BM25/context flow, fixture reranker and models."""

import argparse
import asyncio
import json
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import cast

import httpx

from agentic_rag.agents.fake import FakeToolLLM
from agentic_rag.api.app import create_app
from agentic_rag.api.models import DocumentRequest, DocumentResponse, QueryRequest
from agentic_rag.api.runtime import PostgresBackend, Runtime
from agentic_rag.core.config import Settings
from agentic_rag.ingestion.models import ChunkingConfig, PreparedDocument
from agentic_rag.ingestion.parsing import parse_bytes
from agentic_rag.ingestion.service import prepare_parsed
from agentic_rag.llm.async_client import AsyncFakeLLM
from agentic_rag.observability.tracing import Tracer
from agentic_rag.rag.context import Context
from agentic_rag.reranking.base import RerankingSpec
from agentic_rag.retrieval.models import Candidate, SearchFilter, SearchHit
from agentic_rag.retrieval.sparse import SparseRetriever
from agentic_rag.storage.retrieval import PostgresIndexCatalog
from agentic_rag.tools.models import CatalogInput, CatalogOutput


class DemoCatalog:
    def __init__(self) -> None:
        self.document = prepare_parsed(
            parse_bytes(b"PagedAttention stores KV cache in blocks.", filename="demo.txt"),
            ChunkingConfig(400, 60),
            source_uri="https://example.org/trace-demo",
        )

    def current_documents(self, filters: SearchFilter) -> list[PreparedDocument]:
        return [self.document]

    def hydrate(self, collection: str | None, candidates: Sequence[Candidate]) -> list[SearchHit]:
        doc = self.document
        chunks = {c.id: c for c in doc.chunks}
        return [
            SearchHit(
                c.chunk_id,
                doc.revision_id,
                doc.document_id,
                c.score,
                chunks[c.chunk_id].text,
                doc.source_uri,
                doc.title,
                chunks[c.chunk_id].start_char,
                chunks[c.chunk_id].end_char,
                chunks[c.chunk_id].start_line,
                chunks[c.chunk_id].end_line,
            )
            for c in candidates
        ]


class DemoReranker:
    spec = RerankingSpec("fixture-only", "0" * 40)

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        return [1.0] * len(passages)


class DemoBackend(PostgresBackend):
    """Reuse the production context orchestration with explicit in-memory dependencies."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.catalog = cast(PostgresIndexCatalog, DemoCatalog())
        self.sparse = SparseRetriever(self.catalog)
        self.dense = None
        self.qdrant = None
        self.reranker = DemoReranker()

    def context(self, request: QueryRequest) -> Context:
        if request.query == "fail":
            raise RuntimeError("private fixture error; must not enter traces")
        return super().context(request)

    def ingest(self, request: DocumentRequest) -> DocumentResponse:
        raise ValueError("Read-only demo")

    def health(self) -> dict[str, str]:
        return {"fixture": "ok"}

    def catalog_documents(self, arguments: CatalogInput) -> CatalogOutput:
        return CatalogOutput(documents=(), offset=arguments.offset, has_more=False)


async def run_demo(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, object]] = []
    tracer = Tracer(on_end=records.append)
    # Explicit settings; .env and provider configuration are not used by this demo.
    settings = Settings(_env_file=None, api_llm_provider="fake", api_retrieval_mode="bm25")

    @asynccontextmanager
    async def factory(config: Settings) -> AsyncIterator[Runtime]:
        yield Runtime(config, DemoBackend(config), AsyncFakeLLM(), FakeToolLLM())

    app = create_app(settings, runtime_factory=factory, tracer=tracer)
    responses = []
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://demo") as client,
    ):
        for route, body in (
            ("/query", {"query": "PagedAttention"}),
            ("/agent", {"query": "/search PagedAttention"}),
            ("/agent", {"query": "/calc 6 / 8 * 100"}),
            ("/query", {"query": "PagedAttention", "stream": True}),
            ("/query", {"query": "fail", "stream": True}),
        ):
            response = await client.post(route, json=body)
            responses.append(
                {
                    "route": route,
                    "status": response.status_code,
                    "trace_id": response.headers["x-trace-id"],
                }
            )
        metrics = (await client.get("/metrics")).text
    (output / "spans.json").write_text(json.dumps(records, indent=2) + "\n")
    (output / "requests.json").write_text(json.dumps(responses, indent=2) + "\n")
    (output / "metrics.txt").write_text(metrics)
    print(f"Saved {len(records)} spans in {output}; fixture models, no external requests")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(run_demo(args.output))


if __name__ == "__main__":
    main()
