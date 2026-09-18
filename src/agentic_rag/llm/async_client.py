"""Cancellable generation and bounded Chat Completions SSE parsing."""

import json
from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import dataclass
from typing import Literal, Protocol

import anyio
import httpx
from pydantic import Field, ValidationError

from agentic_rag.llm.base import Completion, IncompleteCompletionError, LLMError, Message
from agentic_rag.llm.compatible import _Usage, _WireModel, parse_completion, request_payload
from agentic_rag.llm.fake import FakeLLM

MAX_OUTPUT_BYTES = 1024 * 1024
MAX_EVENT_BYTES = 128 * 1024


@dataclass(frozen=True)
class TextDelta:
    text: str


class AsyncLLM(Protocol):
    async def complete(self, messages: tuple[Message, ...], *, max_tokens: int) -> Completion: ...

    def stream(
        self, messages: tuple[Message, ...], *, max_tokens: int
    ) -> AsyncGenerator[TextDelta | Completion, None]: ...


class AsyncFakeLLM:
    async def complete(self, messages: tuple[Message, ...], *, max_tokens: int) -> Completion:
        await anyio.lowlevel.checkpoint()
        return FakeLLM().complete(messages, max_tokens=max_tokens)

    async def stream(
        self, messages: tuple[Message, ...], *, max_tokens: int
    ) -> AsyncGenerator[TextDelta | Completion, None]:
        completion = await self.complete(messages, max_tokens=max_tokens)
        # Deliberately synthetic streaming for contract tests only.
        for offset in range(0, len(completion.text), 32):
            await anyio.lowlevel.checkpoint()
            yield TextDelta(completion.text[offset : offset + 32])
        yield completion


class _Delta(_WireModel):
    role: Literal["assistant"] | None = None
    content: str | None = None
    reasoning: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[object] | None = None
    refusal: str | None = None


class _StreamChoice(_WireModel):
    index: Literal[0]
    delta: _Delta
    finish_reason: str | None = None


class _StreamResponse(_WireModel):
    model: str
    choices: list[_StreamChoice] = Field(max_length=1)
    usage: _Usage | None = None


async def _events(response: httpx.Response) -> AsyncIterator[str]:
    """SSE data fields, including multiline events; bound incomplete lines/events."""
    buffer = b""
    lines: list[bytes] = []
    event_size = 0
    async for chunk in response.aiter_bytes():
        # Slice even a large transport chunk before buffering it.
        for offset in range(0, len(chunk), 4096):
            buffer += chunk[offset : offset + 4096]
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                line = line.removesuffix(b"\r")
                event_size += len(line)
                if event_size > MAX_EVENT_BYTES:
                    raise LLMError("LLM stream event exceeds limit")
                if not line:
                    if lines:
                        yield b"\n".join(lines).decode("utf-8")
                    lines = []
                    event_size = 0
                elif line.startswith(b"data:"):
                    lines.append(line[5:].removeprefix(b" "))
            if len(buffer) + event_size > MAX_EVENT_BYTES:
                raise LLMError("LLM stream event exceeds limit")
    if buffer or lines:
        raise LLMError("Incomplete LLM stream event")


class AsyncCompatibleLLM:
    def __init__(
        self, client: httpx.AsyncClient, model: str, *, reasoning_enabled: bool | None = None
    ) -> None:
        if not model.strip():
            raise ValueError("LLM model is required")
        self.client = client
        self.model = model
        self.reasoning_enabled = reasoning_enabled

    async def complete(self, messages: tuple[Message, ...], *, max_tokens: int) -> Completion:
        payload = request_payload(
            self.model, messages, max_tokens, self.reasoning_enabled, stream=False
        )
        try:
            async with self.client.stream("POST", "chat/completions", json=payload) as response:
                response.raise_for_status()
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    body.extend(chunk)
                    if len(body) > MAX_OUTPUT_BYTES:
                        raise LLMError("LLM response exceeds limit")
                return parse_completion(json.loads(body))
        except httpx.TimeoutException as exc:
            raise TimeoutError("LLM timeout") from exc
        except (httpx.HTTPError, ValidationError, ValueError) as exc:
            raise LLMError("LLM transport or response failure") from exc

    async def stream(
        self, messages: tuple[Message, ...], *, max_tokens: int
    ) -> AsyncGenerator[TextDelta | Completion, None]:
        payload = request_payload(
            self.model, messages, max_tokens, self.reasoning_enabled, stream=True
        )
        parts: list[str] = []
        size = reasoning_count = 0
        finish: str | None = None
        model: str | None = None
        usage: _Usage | None = None
        try:
            async with self.client.stream("POST", "chat/completions", json=payload) as response:
                response.raise_for_status()
                if response.headers.get("content-type", "").split(";")[0] != "text/event-stream":
                    raise LLMError("Expected LLM event stream")
                async for event in _events(response):
                    if event == "[DONE]":
                        if finish is None or model is None:
                            raise LLMError("LLM stream ended without finish metadata")
                        completion = Completion(
                            "".join(parts),
                            model,
                            finish,
                            usage.prompt_tokens if usage else None,
                            usage.completion_tokens if usage else None,
                            usage.cost if usage else None,
                            reasoning_count,
                        )
                        if finish != "stop" or not completion.text.strip():
                            raise IncompleteCompletionError(completion)
                        yield completion
                        return
                    item = _StreamResponse.model_validate_json(event)
                    if model is not None and model != item.model:
                        raise LLMError("LLM stream model changed")
                    model = item.model
                    if item.usage is not None:
                        usage = item.usage
                    for choice in item.choices:
                        delta = choice.delta
                        if finish is not None:
                            # Some compatible gateways repeat the empty terminal choice.
                            # Only an identical finish with no new payload is idempotent.
                            if choice.finish_reason == finish and not any(
                                (
                                    delta.content,
                                    delta.reasoning,
                                    delta.reasoning_content,
                                    delta.tool_calls,
                                    delta.refusal,
                                )
                            ):
                                continue
                            raise LLMError("LLM stream continued after finish")
                        if delta.tool_calls or delta.refusal:
                            raise LLMError("Unsupported LLM stream response")
                        reasoning_count += len(delta.reasoning_content or delta.reasoning or "")
                        if delta.content:
                            size += len(delta.content.encode("utf-8"))
                            if size > MAX_OUTPUT_BYTES:
                                raise LLMError("LLM output exceeds limit")
                            parts.append(delta.content)
                            yield TextDelta(delta.content)
                        finish = choice.finish_reason
                raise LLMError("LLM stream ended without DONE")
        except httpx.TimeoutException as exc:
            raise TimeoutError("LLM timeout") from exc
        except (httpx.HTTPError, ValidationError, ValueError) as exc:
            raise LLMError("LLM transport or response failure") from exc
