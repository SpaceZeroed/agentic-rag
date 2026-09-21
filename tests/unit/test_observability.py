import asyncio
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import cast
from uuid import uuid4

import anyio
import httpx
import pytest
from langfuse import Langfuse
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from agentic_rag.agents.fake import FakeToolLLM
from agentic_rag.api.app import create_app
from agentic_rag.api.models import DocumentRequest, DocumentResponse, QueryRequest
from agentic_rag.api.runtime import Runtime
from agentic_rag.core.config import Settings
from agentic_rag.core.logging import JsonFormatter
from agentic_rag.llm.async_client import AsyncFakeLLM
from agentic_rag.llm.base import Completion
from agentic_rag.observability.langfuse import LangfuseSink, configured_tracer
from agentic_rag.observability.tracing import Tracer, correlation, span
from agentic_rag.rag.context import Context, build_context
from agentic_rag.tools.models import CatalogInput, CatalogOutput


class Backend:
    def context(self, request: QueryRequest) -> Context:
        with span("retrieval"):
            if request.query == "fail":
                raise RuntimeError("SECRET DATABASE URL")
        return build_context(request.query, [])

    def ingest(self, request: DocumentRequest) -> DocumentResponse:
        raise NotImplementedError

    def health(self) -> dict[str, str]:
        return {"fixture": "ok"}

    def catalog_documents(self, arguments: CatalogInput) -> CatalogOutput:
        return CatalogOutput(documents=(), offset=0, has_more=False)


@asynccontextmanager
async def factory(settings: Settings) -> AsyncIterator[Runtime]:
    yield Runtime(settings, Backend(), AsyncFakeLLM(), FakeToolLLM())


@pytest.mark.anyio
async def test_concurrent_requests_worker_context_errors_and_metrics() -> None:
    records: list[dict[str, object]] = []
    tracer = Tracer(on_end=records.append)
    app = create_app(Settings(), runtime_factory=factory, tracer=tracer)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client,
    ):
        a, b = await asyncio.gather(
            client.post("/query", json={"query": "SECRET query"}),
            client.post("/query", json={"query": "fail"}),
        )
        assert (a.status_code, b.status_code) == (200, 503)
        assert a.headers["x-trace-id"] != b.headers["x-trace-id"]
        for response in (a, b):
            trace = [r for r in records if r["trace_id"] == response.headers["x-trace-id"]]
            by_name = {r["name"]: r for r in trace}
            assert by_name["retrieval"]["parent_id"] == by_name["context"]["span_id"]
            assert by_name["context"]["parent_id"] == by_name["http.request"]["span_id"]
        failed = [r for r in records if r["trace_id"] == b.headers["x-trace-id"]]
        assert all(r["status"] == "error" for r in failed)
        response = await client.post("/agent", json={"query": "/calc 6 / 8 * 100"})
        assert response.status_code == 200
        assert any(r["name"] == "tool.execute" for r in records)
        text = (await client.get("/metrics")).text
        assert "rag_llm_usage_unknown_total 2" in text
        assert 'name="http.request",status="error"' in text
    assert correlation() == {}
    assert "SECRET" not in json.dumps(records)


def test_log_correlation_usage_and_cardinality() -> None:
    records: list[dict[str, object]] = []
    tracer = Tracer(on_end=records.append)
    with tracer.activate(), span("http.request") as root, span("llm.generate") as child:
        child.completion(Completion("SECRET answer", "fake", "stop", 11, 7))
        child.set(prompt="SECRET prompt")
        entry = json.loads(
            JsonFormatter().format(
                logging.LogRecord("test", logging.INFO, __file__, 1, "safe", (), None)
            )
        )
        assert entry["trace_id"] == root.trace_id
        assert entry["span_id"] == child.span_id
    assert "SECRET" not in json.dumps(records)
    text = tracer.metrics.render()
    assert 'kind="prompt_tokens"} 11' in text
    assert 'kind="completion_tokens"} 7' in text
    assert root.trace_id not in text
    with pytest.raises(ValueError, match="Unknown span"), tracer.span("user-controlled-name"):
        pass


@pytest.mark.anyio
async def test_cancellation_closes_spans_and_clears_context() -> None:
    records: list[dict[str, object]] = []
    tracer = Tracer(on_end=records.append)
    with anyio.move_on_after(0.01), tracer.activate(), span("http.request"), span("llm.stream"):
        await anyio.sleep_forever()
    assert len(records) == 2
    assert all(r["status"] == "cancelled" for r in records)
    assert correlation() == {}


class BrokenSink:
    def start(self, name: str, trace_id: str, parent_id: str | None) -> None:
        raise RuntimeError("SECRET exporter key")

    def close(self) -> None:
        raise RuntimeError("SECRET exporter key")


def test_exporter_failure_does_not_break_request(caplog: pytest.LogCaptureFixture) -> None:
    from agentic_rag.observability.tracing import Sink

    tracer = Tracer(sink=cast(Sink, BrokenSink()))
    with tracer.activate(), span("http.request"):
        pass
    tracer.close()
    assert "SECRET" not in caplog.text
    assert "trace_export_start_failed" in caplog.text


def test_real_langfuse_sdk_exports_parentage_and_usage_without_content() -> None:
    exporter = InMemorySpanExporter()
    client = Langfuse(
        public_key="pk-lf-test-" + uuid4().hex,
        secret_key="sk-lf-test",
        base_url="http://127.0.0.1:1",
        tracer_provider=TracerProvider(),
        span_exporter=exporter,
    )
    tracer = Tracer(sink=LangfuseSink(client))
    try:
        with tracer.activate(), span("http.request") as root, span("llm.generate") as child:
            child.completion(Completion("SECRET", "fake", "stop", 10, 5))
        client.flush()
        exported = exporter.get_finished_spans()
        assert len(exported) == 2
        parent = next(s for s in exported if s.name == "http.request")
        generation = next(s for s in exported if s.name == "llm.generate")
        assert parent.context and generation.context and generation.parent
        assert parent.parent is None
        assert f"{parent.context.trace_id:032x}" == root.trace_id
        assert generation.parent.span_id == parent.context.span_id
        assert "SECRET" not in str([dict(s.attributes or {}) for s in exported])
        assert '"input": 10' in str(generation.attributes)
    finally:
        tracer.close()


def test_disabled_export_needs_no_credentials() -> None:
    tracer = configured_tracer(Settings())
    assert tracer.sink is None and not tracer.enabled
    with pytest.raises(ValueError):
        configured_tracer(Settings(langfuse_enabled=True))


@pytest.mark.anyio
async def test_complete_demo_and_stream_error_after_http_200(tmp_path: Path) -> None:
    from agentic_rag.observability.demo import run_demo

    output = tmp_path / "traces"
    await run_demo(output)
    records = json.loads((output / "spans.json").read_text())
    requests = json.loads((output / "requests.json").read_text())
    assert len(records) == 24 and len(requests) == 5
    first = [r for r in records if r["trace_id"] == requests[0]["trace_id"]]
    by_name = {r["name"]: r for r in first}
    assert by_name["retrieval"]["parent_id"] == by_name["context"]["span_id"]
    assert by_name["reranking"]["parent_id"] == by_name["context"]["span_id"]
    assert by_name["llm.generate"]["parent_id"] == by_name["http.request"]["span_id"]
    assert requests[-1]["status"] == 200
    assert records[-1]["status"] == "error"
    assert "private fixture" not in json.dumps(records)
    assert "rag_llm_usage_unknown_total 6" in (output / "metrics.txt").read_text()
