import json
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from uuid import UUID

import anyio
import httpx
import pytest
from fastapi import FastAPI
from starlette.types import Message as ASGIMessage

from agentic_rag.api.app import create_app
from agentic_rag.api.models import DocumentRequest, DocumentResponse, QueryRequest
from agentic_rag.api.runtime import Runtime
from agentic_rag.core.config import Settings
from agentic_rag.ingestion.models import IngestResult
from agentic_rag.llm.async_client import AsyncFakeLLM, AsyncLLM, TextDelta
from agentic_rag.llm.base import Completion, LLMError, Message
from agentic_rag.rag.context import Context, build_context
from agentic_rag.retrieval.models import SearchHit

pytestmark = pytest.mark.anyio


@asynccontextmanager
async def api_client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        yield client


class Backend:
    def __init__(self) -> None:
        hit = SearchHit(
            UUID(int=1),
            UUID(int=2),
            UUID(int=3),
            1.0,
            "Факт",
            "https://example.org/a",
            "A",
            0,
            4,
            1,
            1,
        )
        self.result = build_context("Вопрос", [hit])
        self.error: Exception | None = None
        self.calls = 0

    def context(self, request: QueryRequest) -> Context:
        self.calls += 1
        if self.error:
            raise self.error
        return self.result

    def ingest(self, request: DocumentRequest) -> DocumentResponse:
        return DocumentResponse(
            document=IngestResult(UUID(int=3), UUID(int=2), "created", 1),
            index_status="not_required",
        )

    def health(self) -> dict[str, str]:
        return {"postgres": "ok", "llm": "not_probed"}


class Factory:
    def __init__(self, backend: Backend | None = None, llm: AsyncLLM | None = None) -> None:
        self.backend = backend or Backend()
        self.llm = llm or AsyncFakeLLM()
        self.opened = self.closed = 0

    @asynccontextmanager
    async def __call__(self, settings: Settings) -> AsyncIterator[Runtime]:
        self.opened += 1
        try:
            yield Runtime(settings, self.backend, self.llm)
        finally:
            self.closed += 1


def events(text: str) -> list[tuple[str, dict[str, object]]]:
    return [
        (lines[0][7:], json.loads(lines[1][6:]))
        for block in text.strip().split("\n\n")
        if (lines := block.splitlines())
    ]


async def test_json_stream_equivalence_and_lifespan() -> None:
    factory = Factory()
    app = create_app(Settings(), runtime_factory=factory)
    assert factory.opened == 0
    async with api_client(app) as client:
        result = await client.post("/query", json={"query": "Вопрос"})
        assert result.status_code == 200
        assert "context" not in result.json()
        assert result.json()["citations"][0]["source"]["revision_id"] == str(UUID(int=2))
        stream = await client.post("/query", json={"query": "Вопрос", "stream": True})
        parsed = events(stream.text)
        assert parsed[0][0] == "sources"
        assert parsed[-1] == ("result", result.json())
        assert all(item[1]["provisional"] is True for item in parsed if item[0] == "delta")
        assert (await client.get("/health")).json()["checks"]["llm"] == "not_probed"
        assert (await client.get("/openapi.json")).status_code == 200
        assert factory.opened == 1
    assert factory.closed == 1


@pytest.mark.parametrize(
    "body",
    [
        {"query": " "},
        {"query": "q", "k": 0},
        {"query": "q", "k": True},
        {"query": "q", "stream": "true"},
        {"query": "q", "document_ids": ["bad"]},
        {"query": "q", "api_key": "private"},
    ],
)
async def test_validation_does_not_call_backend_or_echo_input(body: dict[str, object]) -> None:
    factory = Factory()
    async with api_client(create_app(Settings(), runtime_factory=factory)) as client:
        response = await client.post("/query", json=body)
        assert response.status_code == 422
        assert "private" not in response.text
        assert factory.backend.calls == 0


@pytest.mark.parametrize(
    "error,status",
    [
        (ValueError("private"), 422),
        (RuntimeError("postgresql://secret"), 503),
        (LLMError("private"), 502),
        (TimeoutError("private"), 504),
    ],
)
@pytest.mark.parametrize("stream", [False, True])
async def test_safe_failures(error: Exception, status: int, stream: bool) -> None:
    factory = Factory()
    factory.backend.error = error
    async with api_client(create_app(Settings(), runtime_factory=factory)) as client:
        response = await client.post("/query", json={"query": "q", "stream": stream})
        assert "private" not in response.text and "secret" not in response.text
        if stream:
            assert response.status_code == 200
            assert events(response.text)[-1][1]["status"] == status
            assert "event: result" not in response.text
        else:
            assert response.status_code == status


class WrongCitation(AsyncFakeLLM):
    async def complete(self, messages: tuple[Message, ...], *, max_tokens: int) -> Completion:
        return Completion("Unsupported [C999]", "test", "stop")


@pytest.mark.parametrize("stream", [False, True])
async def test_invalid_citations_never_become_final(stream: bool) -> None:
    async with api_client(
        create_app(Settings(), runtime_factory=Factory(llm=WrongCitation()))
    ) as client:
        response = await client.post("/query", json={"query": "q", "stream": stream})
    if stream:
        assert "event: delta" in response.text
        assert events(response.text)[-1] == (
            "error",
            {"error": "generation_failed", "status": 502, "discard_draft": True},
        )
        assert "event: result" not in response.text
    else:
        assert response.status_code == 502


class MustNotCall(AsyncFakeLLM):
    async def complete(self, messages: tuple[Message, ...], *, max_tokens: int) -> Completion:
        raise AssertionError("No generation without context")


@pytest.mark.parametrize("stream", [False, True])
async def test_empty_context_does_not_generate(stream: bool) -> None:
    factory = Factory(llm=MustNotCall())
    factory.backend.result = build_context("q", [])
    async with api_client(create_app(Settings(), runtime_factory=factory)) as client:
        response = await client.post("/query", json={"query": "q", "stream": stream})
        result = events(response.text)[-1][1] if stream else response.json()
        assert result["status"] == "insufficient_evidence"
        assert result["completion"] is None


class BlockingLLM(AsyncFakeLLM):
    def __init__(self) -> None:
        self.started = anyio.Event()
        self.cancelled = False

    async def complete(self, messages: tuple[Message, ...], *, max_tokens: int) -> Completion:
        self.started.set()
        try:
            await anyio.sleep_forever()
        finally:
            self.cancelled = True
        raise AssertionError("unreachable")

    async def stream(
        self, messages: tuple[Message, ...], *, max_tokens: int
    ) -> AsyncGenerator[TextDelta | Completion, None]:
        yield TextDelta("draft")
        yield await self.complete(messages, max_tokens=max_tokens)


@pytest.mark.parametrize("stream", [False, True])
async def test_deadline_and_busy_server(stream: bool) -> None:
    llm = BlockingLLM()
    app = create_app(
        Settings(api_request_timeout_seconds=0.15, api_max_concurrent_requests=1),
        runtime_factory=Factory(llm=llm),
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        results: list[httpx.Response] = []

        async def first() -> None:
            results.append(await client.post("/query", json={"query": "q", "stream": stream}))

        async with anyio.create_task_group() as group:
            group.start_soon(first)
            await llm.started.wait()
            busy = await client.post("/query", json={"query": "q"})
            assert busy.status_code == 503 and busy.json()["error"] == "server_busy"
            assert (await client.get("/health")).status_code == 200
        assert llm.cancelled
        assert (
            events(results[0].text)[-1][1]["status"] if stream else results[0].status_code
        ) == 504
        assert app.state.runtime.requests.borrowed_tokens == 0


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("spec", ["2.3", "2.4"])
async def test_real_asgi_disconnect_cancels_generation(stream: bool, spec: str) -> None:
    llm = BlockingLLM()
    app = create_app(Settings(), runtime_factory=Factory(llm=llm))
    disconnected = anyio.Event()
    request_sent = False
    sent: list[ASGIMessage] = []

    async def receive() -> ASGIMessage:
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {
                "type": "http.request",
                "body": json.dumps({"query": "q", "stream": stream}).encode(),
            }
        await disconnected.wait()
        return {"type": "http.disconnect"}

    async def send(message: ASGIMessage) -> None:
        sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": spec},
        "method": "POST",
        "path": "/query",
        "raw_path": b"/query",
        "query_string": b"",
        "scheme": "http",
        "server": ("test", 80),
        "client": ("test", 1),
        "headers": [(b"content-type", b"application/json")],
        "root_path": "",
        "http_version": "1.1",
    }
    async with app.router.lifespan_context(app):
        with anyio.fail_after(2):
            async with anyio.create_task_group() as group:
                group.start_soon(app, scope, receive, send)
                await llm.started.wait()
                disconnected.set()
        assert llm.cancelled
        assert app.state.runtime.requests.borrowed_tokens == 0
        assert not any(b"event: result" in message.get("body", b"") for message in sent)


async def test_chunked_body_limit() -> None:
    factory = Factory()
    app = create_app(Settings(api_max_body_bytes=32), runtime_factory=factory)

    async def body() -> AsyncIterator[bytes]:
        yield b'{"query":"'
        yield b"x" * 100
        yield b'"}'

    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        response = await client.post(
            "/query", content=body(), headers={"content-type": "application/json"}
        )
        assert response.status_code == 413
        assert factory.backend.calls == 0


async def test_disconnect_waits_for_started_sync_work_then_skips_llm() -> None:
    import threading

    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    class SlowBackend(Backend):
        def context(self, request: QueryRequest) -> Context:
            started.set()
            assert release.wait(timeout=3)
            finished.set()
            return self.result

    llm = BlockingLLM()
    app = create_app(Settings(), runtime_factory=Factory(backend=SlowBackend(), llm=llm))
    disconnected = anyio.Event()
    delivered = False

    async def receive() -> ASGIMessage:
        nonlocal delivered
        if not delivered:
            delivered = True
            return {"type": "http.request", "body": b'{"query":"q"}'}
        await disconnected.wait()
        return {"type": "http.disconnect"}

    async def send(message: ASGIMessage) -> None:
        pass

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.4"},
        "method": "POST",
        "path": "/query",
        "raw_path": b"/query",
        "query_string": b"",
        "scheme": "http",
        "server": ("test", 80),
        "client": ("test", 1),
        "headers": [(b"content-type", b"application/json")],
        "root_path": "",
        "http_version": "1.1",
    }
    async with app.router.lifespan_context(app):
        with anyio.fail_after(2):
            async with anyio.create_task_group() as group:
                group.start_soon(app, scope, receive, send)
                while not started.is_set():
                    await anyio.sleep(0.001)
                disconnected.set()
                await anyio.sleep(0.01)
                assert not finished.is_set()
                release.set()
        assert finished.is_set()
        assert not llm.started.is_set()
        assert app.state.runtime.requests.borrowed_tokens == 0


async def test_chunk_work_bound_and_health_failure() -> None:
    class Unhealthy(Backend):
        def health(self) -> dict[str, str]:
            return {"postgres": "unavailable", "llm": "not_probed"}

    app = create_app(Settings(), runtime_factory=Factory(backend=Unhealthy()))
    async with api_client(app) as client:
        response = await client.post(
            "/documents",
            json={
                "filename": "a.txt",
                "source_uri": "https://example.org/a",
                "content": "a" * 20000,
                "max_chars": 10,
                "overlap": 9,
            },
        )
        assert response.status_code == 422
        health = await client.get("/health")
        assert health.status_code == 503
        assert health.json()["status"] == "degraded"
