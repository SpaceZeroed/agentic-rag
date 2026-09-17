"""Minimal Chat Completions subset, usable with local inference servers."""

from dataclasses import asdict
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agentic_rag.llm.base import Completion, IncompleteCompletionError, LLMError, Message


class _WireModel(BaseModel):
    model_config = ConfigDict(strict=True)


class _Message(_WireModel):
    role: Literal["assistant"]
    content: str | None
    reasoning: str | None = None
    reasoning_content: str | None = None


class _Choice(_WireModel):
    message: _Message
    finish_reason: str


class _Usage(_WireModel):
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    cost: float | None = Field(default=None, ge=0, allow_inf_nan=False)


class _Response(_WireModel):
    model: str
    choices: list[_Choice] = Field(min_length=1, max_length=1)
    usage: _Usage | None = None


class CompatibleLLM:
    """Caller owns the HTTP client lifecycle; no retries or silent fake fallback."""

    def __init__(
        self, client: httpx.Client, model: str, *, reasoning_enabled: bool | None = None
    ) -> None:
        if not model.strip():
            raise ValueError("LLM model is required")
        self.client = client
        self.model = model
        self.reasoning_enabled = reasoning_enabled

    def complete(self, messages: tuple[Message, ...], *, max_tokens: int) -> Completion:
        if not messages or max_tokens < 1:
            raise ValueError("Messages and positive max_tokens required")
        try:
            response = self.client.post(
                "chat/completions",
                json={
                    "model": self.model,
                    "messages": [asdict(message) for message in messages],
                    "max_tokens": max_tokens,
                    "temperature": 0,
                    "stream": False,
                    **(
                        {"reasoning": {"enabled": self.reasoning_enabled}}
                        if self.reasoning_enabled is not None
                        else {}
                    ),
                },
            )
            response.raise_for_status()
            result = _Response.model_validate(response.json())
        except (httpx.HTTPError, ValidationError, ValueError) as exc:
            raise LLMError("LLM transport or response failure") from exc
        choice = result.choices[0]
        completion = Completion(
            choice.message.content or "",
            result.model,
            choice.finish_reason,
            result.usage.prompt_tokens if result.usage else None,
            result.usage.completion_tokens if result.usage else None,
            result.usage.cost if result.usage else None,
            len(choice.message.reasoning_content or choice.message.reasoning or ""),
        )
        if choice.finish_reason != "stop" or not completion.text.strip():
            raise IncompleteCompletionError(completion)
        return completion
