import json
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from uuid import UUID

import anyio
import httpx
import pytest
from fastapi import FastAPI
from starlette.types import Message as ASGIMessage

from agentic_rag.agents.fake import FakeToolLLM
from agentic_rag.api.app import create_app
from agentic_rag.api.models import DocumentRequest, DocumentResponse, QueryRequest
from agentic_rag.api.runtime import Runtime
from agentic_rag.core.config import Settings
from agentic_rag.ingestion.models import IngestResult
from agentic_rag.llm.async_client import AsyncFakeLLM, AsyncLLM, TextDelta
from agentic_rag.llm.base import Completion, LLMError, Message
from agentic_rag.llm.tool_client import CompatibleToolLLM, ToolLLM, ToolTurn
from agentic_rag.rag.context import Context, build_context
from agentic_rag.retrieval.models import SearchHit
from agentic_rag.tools.models import CatalogInput, CatalogOutput

pytestmark = pytest.mark.anyio


@asynccontextmanager
async def api_client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        yield client


class Backend:
    def catalog_documents(self, arguments: CatalogInput) -> CatalogOutput:
        return CatalogOutput(documents=(), offset=arguments.offset, has_more=False)

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
    def __init__(
        self,
        backend: Backend | None = None,
        llm: AsyncLLM | None = None,
        tool_llm: ToolLLM | None = None,
    ) -> None:
        self.backend = backend or Backend()
        self.llm = llm or AsyncFakeLLM()
        self.tool_llm = tool_llm
        self.opened = self.closed = 0

    @asynccontextmanager
    async def __call__(self, settings: Settings) -> AsyncIterator[Runtime]:
        self.opened += 1
        try:
            yield Runtime(settings, self.backend, self.llm, self.tool_llm)
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


class BlockingToolLLM:
    def __init__(
        self, blocker: BlockingLLM, *, search_first: bool = False, repair_first: bool = False
    ) -> None:
        self.blocker = blocker
        self.search_first = search_first
        self.repair_first = repair_first

    async def complete(
        self,
        messages: tuple[dict[str, object], ...],
        tools: list[dict[str, object]],
        *,
        max_tokens: int,
    ) -> ToolTurn:
        if self.repair_first and tools:
            return ToolTurn(Completion("Invalid [C1]", "test", "stop"))
        if self.search_first and messages[-1]["role"] == "user":
            return await FakeToolLLM().complete(messages, tools, max_tokens=max_tokens)
        await self.blocker.complete((), max_tokens=max_tokens)
        raise AssertionError("unreachable")


@pytest.mark.parametrize(
    "path,stream,repair",
    [
        ("/query", False, False),
        ("/query", True, False),
        ("/agent", False, False),
        ("/agent", False, True),
    ],
)
async def test_deadline_and_busy_server(path: str, stream: bool, repair: bool) -> None:
    llm = BlockingLLM()
    app = create_app(
        Settings(api_request_timeout_seconds=0.15, api_max_concurrent_requests=1),
        runtime_factory=Factory(llm=llm, tool_llm=BlockingToolLLM(llm, repair_first=repair)),
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        results: list[httpx.Response] = []

        async def first() -> None:
            body: dict[str, object] = {"query": "q"}
            if path == "/query":
                body["stream"] = stream
            results.append(await client.post(path, json=body))

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


@pytest.mark.parametrize(
    "path,stream,repair",
    [
        ("/query", False, False),
        ("/query", True, False),
        ("/agent", False, False),
        ("/agent", False, True),
    ],
)
@pytest.mark.parametrize("spec", ["2.3", "2.4"])
async def test_real_asgi_disconnect_cancels_generation(
    path: str,
    stream: bool,
    spec: str,
    repair: bool,
) -> None:
    llm = BlockingLLM()
    app = create_app(
        Settings(),
        runtime_factory=Factory(
            llm=llm,
            tool_llm=BlockingToolLLM(llm, repair_first=repair),
        ),
    )
    disconnected = anyio.Event()
    request_sent = False
    sent: list[ASGIMessage] = []

    async def receive() -> ASGIMessage:
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {
                "type": "http.request",
                "body": json.dumps(
                    {"query": "q", **({"stream": stream} if path == "/query" else {})}
                ).encode(),
            }
        await disconnected.wait()
        return {"type": "http.disconnect"}

    async def send(message: ASGIMessage) -> None:
        sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": spec},
        "method": "POST",
        "path": path,
        "raw_path": path.encode(),
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


@pytest.mark.parametrize("path", ["/query", "/agent"])
async def test_disconnect_waits_for_started_sync_work_then_skips_llm(path: str) -> None:
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
    app = create_app(
        Settings(),
        runtime_factory=Factory(
            backend=SlowBackend(),
            llm=llm,
            tool_llm=BlockingToolLLM(llm, search_first=True),
        ),
    )
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
        "path": path,
        "raw_path": path.encode(),
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


@pytest.mark.parametrize(
    "query,basis",
    [
        ("/calc 6/8*100", "tools"),
        ("/catalog", "tools"),
        ("/search Факт", "tools"),
        ("/direct", "model"),
    ],
)
async def test_agent_endpoint_contract_and_fake_routes(query: str, basis: str) -> None:
    factory = Factory(tool_llm=FakeToolLLM())
    async with api_client(create_app(Settings(), runtime_factory=factory)) as client:
        response = await client.post("/agent", json={"query": query})
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "answered" and data["basis"] == basis
        assert data["llm_provider"] == "fake"
        assert data["model_calls"] == (1 if basis == "model" else 2)
        if query.startswith("/search"):
            assert data["citations"][0]["source"]["text"] == "Факт"
        if query.startswith("/calc"):
            assert data["observations"][0]["data"]["value"] == "75.00"
        assert all(c["text"] == "" for c in data["completions"])
    assert factory.closed == 1


async def test_agent_limits_and_validation_are_visible_http_errors() -> None:
    async with api_client(
        create_app(
            Settings(agent_max_model_calls=1), runtime_factory=Factory(tool_llm=FakeToolLLM())
        )
    ) as client:
        response = await client.post("/agent", json={"query": "/calc 1+1"})
        assert response.status_code == 422
        assert response.json()["status"] == "limit_reached"
        assert response.json()["tool_calls"] == 0
        bad = await client.post("/agent", json={"query": " ", "max_steps": 9999})
        assert bad.status_code == 422
        assert bad.json()["error"] == "invalid_request"


@pytest.mark.parametrize("corrected,status", [("No documents [T1]", 200), ("Still uncited", 502)])
async def test_agent_citation_repair_wire_and_public_provenance(
    corrected: str, status: int
) -> None:
    requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        data = json.loads(request.content)
        requests.append(data)
        if len(requests) == 1:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "a",
                        "type": "function",
                        "function": {"name": "document_catalog", "arguments": "{}"},
                    }
                ],
            }
            finish = "tool_calls"
        else:
            message = {
                "role": "assistant",
                "content": "No documents" if len(requests) == 2 else corrected,
            }
            finish = "stop"
        if len(requests) == 3:
            assert data["tool_choice"] == "none" and "tools" not in data
        assert len(requests) <= 3
        return httpx.Response(
            200, json={"model": "test", "choices": [{"message": message, "finish_reason": finish}]}
        )

    async with httpx.AsyncClient(
        base_url="https://test/", transport=httpx.MockTransport(handler)
    ) as upstream:
        factory = Factory(tool_llm=CompatibleToolLLM(upstream, "test"))
        async with api_client(
            create_app(Settings(api_llm_provider="compatible"), runtime_factory=factory)
        ) as client:
            response = await client.post("/agent", json={"query": "Which documents are available?"})
    assert response.status_code == status
    result = response.json()
    assert result["repair_attempts"] == 1 and result["model_calls"] == 3
    assert result["tool_calls"] == 1
    assert result["citation_failures"][0] == {"model_call": 2, "error": "invalid_citations"}
    assert len(result["citation_failures"]) == (1 if status == 200 else 2)
    assert result["text"] == (corrected if status == 200 else "")


async def test_agent_malformed_arguments_recover_through_compatible_wire() -> None:
    requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        data = json.loads(request.content)
        requests.append(data)
        # Emulate a provider that rejects malformed historical tool arguments.
        for message in data["messages"]:
            for call in message.get("tool_calls", []):
                assert isinstance(json.loads(call["function"]["arguments"]), dict)
        step = len(requests)
        assert step <= 3
        if step < 3:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"call_{step}",
                        "type": "function",
                        "function": {
                            "name": "document_catalog",
                            "arguments": '{"limit": 10, "offset": 10'
                            if step == 1
                            else '{"limit": 10, "offset": 10}',
                        },
                    }
                ],
            }
            finish = "tool_calls"
        else:
            message = {"role": "assistant", "content": "This page is empty [T2]"}
            finish = "stop"
        return httpx.Response(
            200, json={"model": "test", "choices": [{"message": message, "finish_reason": finish}]}
        )

    async with httpx.AsyncClient(
        base_url="https://test/", transport=httpx.MockTransport(handler)
    ) as upstream:
        factory = Factory(tool_llm=CompatibleToolLLM(upstream, "test"))
        async with api_client(
            create_app(Settings(api_llm_provider="compatible"), runtime_factory=factory)
        ) as client:
            response = await client.post("/agent", json={"query": "List documents"})
    assert response.status_code == 200
    result = response.json()
    assert result["model_calls"] == 3 and result["tool_calls"] == 2
    assert result["observations"][0]["error"] == "invalid_arguments"
    assert result["observations"][1]["data"]["offset"] == 10
    assert result["repair_attempts"] == 0
