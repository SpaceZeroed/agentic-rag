"""Application factory: bounded HTTP requests, safe failures and SSE drafts/final."""

import json
import logging
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, aclosing, asynccontextmanager
from dataclasses import asdict
from functools import partial
from typing import cast

import anyio
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from agentic_rag.agents.models import AgentResult
from agentic_rag.api.models import (
    AgentRequest,
    DocumentRequest,
    DocumentResponse,
    QueryRequest,
    QueryResponse,
)
from agentic_rag.api.runtime import Runtime, open_runtime
from agentic_rag.core.config import Settings
from agentic_rag.llm.async_client import TextDelta
from agentic_rag.llm.base import Completion, LLMError
from agentic_rag.rag.service import finalize_answer

logger = logging.getLogger(__name__)
type RuntimeFactory = Callable[[Settings], AbstractAsyncContextManager[Runtime]]


class BodyLimit:
    """Enforce actual bytes before JSON decoding, including chunked request bodies."""

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] != "POST":
            await self.app(scope, receive, send)
            return
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body.extend(message.get("body", b""))
            if len(body) > self.max_bytes:
                await JSONResponse({"error": "body_too_large"}, status_code=413)(
                    scope, receive, send
                )
                return
            if not message.get("more_body", False):
                break
        delivered = False

        async def replay() -> Message:
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        await self.app(scope, replay, send)


class CancellableStream(StreamingResponse):
    """Watch disconnect even while upstream stalls, for both ASGI 2.3 and 2.4+."""

    def __init__(self, source: AsyncGenerator[str, None]) -> None:
        super().__init__(
            source,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )
        self.source = source

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        async with anyio.create_task_group() as group:

            async def emit() -> None:
                try:
                    async with aclosing(self.source):
                        await self.stream_response(send)
                except OSError:
                    pass  # Client socket closed during send.
                finally:
                    group.cancel_scope.cancel()

            group.start_soon(emit)
            await self.listen_for_disconnect(receive)
            group.cancel_scope.cancel()


def failure(exc: Exception) -> tuple[int, str]:
    if isinstance(exc, anyio.WouldBlock):
        return 503, "server_busy"
    if isinstance(exc, TimeoutError):
        return 504, "request_timeout"
    if isinstance(exc, LLMError):
        return 502, "generation_failed"
    if isinstance(exc, ValueError):
        return 422, "invalid_request"
    # Don't serialize exception strings: drivers may include URLs or query data.
    logger.warning("api_dependency_failed", extra={"fields": {"type": type(exc).__name__}})
    return 503, "dependency_unavailable"


@asynccontextmanager
async def request_budget(runtime: Runtime) -> AsyncIterator[None]:
    runtime.requests.acquire_nowait()
    try:
        with anyio.fail_after(runtime.settings.api_request_timeout_seconds):
            yield
    finally:
        runtime.requests.release()


async def connected[T](request: Request, operation: Callable[[], Awaitable[T]]) -> T | None:
    result: T | None = None
    async with anyio.create_task_group() as group:

        async def watch() -> None:
            while True:
                message = await request.receive()
                if message["type"] == "http.disconnect":
                    group.cancel_scope.cancel()
                    return

        group.start_soon(watch)
        try:
            result = await operation()
        finally:
            group.cancel_scope.cancel()
    return result


def event(name: str, payload: object) -> str:
    return f"event: {name}\ndata: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n"


async def stream_query(runtime: Runtime, query: QueryRequest) -> AsyncGenerator[str, None]:
    try:
        async with request_budget(runtime):
            context = await runtime.run_sync(partial(runtime.backend.context, query))
            yield event("sources", {"sources": [asdict(source) for source in context.sources]})
            completion: Completion | None = None
            if context.sources:
                async with aclosing(
                    runtime.llm.stream(context.messages, max_tokens=runtime.settings.llm_max_tokens)
                ) as stream:
                    async for item in stream:
                        if isinstance(item, TextDelta):
                            yield event("delta", {"text": item.text, "provisional": True})
                        else:
                            completion = item
            answer = finalize_answer(context, completion)
            result = QueryResponse.from_answer(answer, runtime.settings.api_llm_provider)
            yield event("result", result.model_dump(mode="json"))
    except Exception as exc:
        status, code = failure(exc)
        yield event("error", {"error": code, "status": status, "discard_draft": True})


def create_app(
    settings: Settings | None = None, *, runtime_factory: RuntimeFactory = open_runtime
) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with runtime_factory(settings) as runtime:
            app.state.runtime = runtime
            yield

    app = FastAPI(title="Agentic RAG", version="0.1.0", lifespan=lifespan)
    app.add_middleware(BodyLimit, max_bytes=settings.api_max_body_bytes)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        # Default errors echo input values, which can include uploaded private text.
        return JSONResponse(
            {
                "error": "invalid_request",
                "issues": [{"loc": error["loc"], "type": error["type"]} for error in exc.errors()],
            },
            status_code=422,
        )

    @app.post("/documents", response_model=DocumentResponse)
    async def documents(document: DocumentRequest, request: Request) -> JSONResponse:
        runtime = cast(Runtime, request.app.state.runtime)

        async def operation() -> JSONResponse:
            try:
                async with request_budget(runtime):
                    result = await runtime.run_sync(partial(runtime.backend.ingest, document))
                status = (
                    503
                    if result.index_status == "failed"
                    else (201 if result.document.status == "created" else 200)
                )
                return JSONResponse(result.model_dump(mode="json"), status_code=status)
            except Exception as exc:
                status, code = failure(exc)
                return JSONResponse({"error": code}, status_code=status)

        return await connected(request, operation) or JSONResponse(
            {"error": "client_disconnected"}, status_code=499
        )

    @app.post(
        "/query",
        response_model=QueryResponse,
        responses={
            200: {"content": {"text/event-stream": {"schema": {"type": "string"}}}},
        },
    )
    async def query(body: QueryRequest, request: Request) -> JSONResponse | StreamingResponse:
        runtime = cast(Runtime, request.app.state.runtime)
        if body.stream:
            return CancellableStream(stream_query(runtime, body))

        async def operation() -> JSONResponse:
            try:
                async with request_budget(runtime):
                    context = await runtime.run_sync(partial(runtime.backend.context, body))
                    completion = (
                        await runtime.llm.complete(
                            context.messages, max_tokens=settings.llm_max_tokens
                        )
                        if context.sources
                        else None
                    )
                    result = QueryResponse.from_answer(
                        finalize_answer(context, completion), settings.api_llm_provider
                    )
                return JSONResponse(result.model_dump(mode="json"))
            except Exception as exc:
                status, code = failure(exc)
                return JSONResponse({"error": code}, status_code=status)

        return await connected(request, operation) or JSONResponse(
            {"error": "client_disconnected"}, status_code=499
        )

    @app.post("/agent", response_model=AgentResult)
    async def agent(body: AgentRequest, request: Request) -> JSONResponse:
        runtime = cast(Runtime, request.app.state.runtime)

        async def operation() -> JSONResponse:
            try:
                async with request_budget(runtime):
                    if runtime.agent is None:
                        return JSONResponse({"error": "agent_unavailable"}, status_code=503)
                    result = await runtime.agent.run(body.query)
                status = 200
                if result.status == "failed":
                    status = 504 if result.error == "llm_timeout" else 502
                elif result.status == "limit_reached":
                    status = 422
                return JSONResponse(result.model_dump(mode="json"), status_code=status)
            except Exception as exc:
                status, code = failure(exc)
                return JSONResponse({"error": code}, status_code=status)

        return await connected(request, operation) or JSONResponse(
            {"error": "client_disconnected"}, status_code=499
        )

    @app.get("/health")
    async def health(request: Request) -> JSONResponse:
        runtime = cast(Runtime, request.app.state.runtime)
        try:
            with anyio.fail_after(10):
                # Independent of model-work limiter; health shouldn't queue behind inference.
                checks = await anyio.to_thread.run_sync(runtime.backend.health)
                await anyio.lowlevel.checkpoint()
            ready = "unavailable" not in checks.values()
            return JSONResponse(
                {"status": "ok" if ready else "degraded", "checks": checks},
                status_code=200 if ready else 503,
            )
        except Exception as exc:
            status, code = failure(exc)
            return JSONResponse({"error": code}, status_code=status)

    return app
