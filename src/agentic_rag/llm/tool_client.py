"""Explicit nonstream Chat Completions tool-calling transport."""

import json
from dataclasses import dataclass
from typing import Literal, Protocol

import httpx
from pydantic import Field, ValidationError

from agentic_rag.llm.async_client import MAX_OUTPUT_BYTES
from agentic_rag.llm.base import Completion, LLMError
from agentic_rag.llm.compatible import _Usage, _WireModel


class FunctionCall(_WireModel):
    name: str = Field(min_length=1, max_length=100)
    arguments: str = Field(max_length=8192)


class ToolCall(_WireModel):
    id: str = Field(min_length=1, max_length=200)
    type: Literal["function"] = "function"
    function: FunctionCall


@dataclass(frozen=True)
class ToolTurn:
    completion: Completion
    calls: tuple[ToolCall, ...] = ()

    def assistant_message(self) -> dict[str, object]:
        message: dict[str, object] = {
            "role": "assistant",
            "content": self.completion.text or None,
        }
        if self.calls:
            message["tool_calls"] = [call.model_dump() for call in self.calls]
        return message


class ToolLLM(Protocol):
    async def complete(
        self,
        messages: tuple[dict[str, object], ...],
        tools: list[dict[str, object]],
        *,
        max_tokens: int,
    ) -> ToolTurn: ...


class _ToolMessage(_WireModel):
    role: Literal["assistant"]
    content: str | None = None
    tool_calls: list[ToolCall] | None = Field(default=None, max_length=4)
    reasoning: str | None = None
    reasoning_content: str | None = None
    refusal: str | None = None


class _ToolChoice(_WireModel):
    message: _ToolMessage
    finish_reason: str


class _ToolResponse(_WireModel):
    model: str
    choices: list[_ToolChoice] = Field(min_length=1, max_length=1)
    usage: _Usage | None = None


def parse_tool_turn(payload: object) -> ToolTurn:
    result = _ToolResponse.model_validate(payload)
    choice = result.choices[0]
    message = choice.message
    calls = tuple(message.tool_calls or ())
    if message.refusal or len({call.id for call in calls}) != len(calls):
        raise LLMError("Invalid tool completion")
    if choice.finish_reason == "tool_calls":
        if not calls:
            raise LLMError("Missing tool calls")
    elif choice.finish_reason != "stop" or calls or not (message.content or "").strip():
        raise LLMError("Incomplete tool completion")
    usage = result.usage
    return ToolTurn(
        Completion(
            message.content or "",
            result.model,
            choice.finish_reason,
            usage.prompt_tokens if usage else None,
            usage.completion_tokens if usage else None,
            usage.cost if usage else None,
            len(message.reasoning_content or message.reasoning or ""),
        ),
        calls,
    )


class CompatibleToolLLM:
    def __init__(
        self, client: httpx.AsyncClient, model: str, *, reasoning_enabled: bool | None = None
    ) -> None:
        if not model.strip():
            raise ValueError("Model is required")
        self.client = client
        self.model = model
        self.reasoning_enabled = reasoning_enabled

    async def complete(
        self,
        messages: tuple[dict[str, object], ...],
        tools: list[dict[str, object]],
        *,
        max_tokens: int,
    ) -> ToolTurn:
        payload: dict[str, object] = {
            "model": self.model,
            "messages": list(messages),
            "tools": tools,
            "tool_choice": "auto",
            "parallel_tool_calls": False,
            "temperature": 0,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if self.reasoning_enabled is not None:
            payload["reasoning"] = {"enabled": self.reasoning_enabled}
        try:
            async with self.client.stream("POST", "chat/completions", json=payload) as response:
                response.raise_for_status()
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    body.extend(chunk)
                    if len(body) > MAX_OUTPUT_BYTES:
                        raise LLMError("Tool response exceeds limit")
            return parse_tool_turn(json.loads(body))
        except httpx.TimeoutException as exc:
            raise TimeoutError("LLM timeout") from exc
        except (httpx.HTTPError, ValidationError, ValueError) as exc:
            raise LLMError("Tool transport or response failure") from exc
