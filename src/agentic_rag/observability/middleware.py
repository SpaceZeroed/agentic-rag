"""ASGI scope includes streaming and disconnect; no request body is inspected."""

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from agentic_rag.observability.tracing import Tracer


class TraceRequests:
    def __init__(self, app: ASGIApp, tracer: Tracer) -> None:
        self.app, self.tracer = app, tracer

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"] == "/metrics" or not self.tracer.enabled:
            await self.app(scope, receive, send)
            return
        with self.tracer.activate(), self.tracer.span("http.request") as request_span:
            route = (
                scope["path"]
                if scope["path"] in ("/query", "/agent", "/documents", "/health", "/ready", "/live")
                else "other"
            )
            method = scope["method"] if scope["method"] in ("POST", "GET") else "other"
            request_span.set(route=route, method=method)
            finished = False

            async def traced_send(message: Message) -> None:
                nonlocal finished
                if message["type"] == "http.response.start":
                    message = dict(message)
                    message["headers"] = [
                        *message.get("headers", []),
                        (b"x-trace-id", request_span.trace_id.encode("ascii")),
                    ]
                    request_span.set(http_status=message["status"])
                    if message["status"] == 499:
                        request_span.status = "cancelled"
                    elif message["status"] >= 400:
                        request_span.fail("http_error")
                await send(message)
                if message["type"] == "http.response.body" and not message.get("more_body", False):
                    finished = True

            async def traced_receive() -> Message:
                message = await receive()
                if message["type"] == "http.disconnect" and not finished:
                    request_span.status = "cancelled"
                return message

            try:
                await self.app(scope, traced_receive, traced_send)
            finally:
                request_span.set(stream_completed=finished)
