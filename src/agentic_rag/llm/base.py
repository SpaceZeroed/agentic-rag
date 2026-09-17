from dataclasses import dataclass
from typing import Literal, Protocol


class LLMError(RuntimeError):
    """Safe public error: never include prompts, keys or server response bodies."""


@dataclass(frozen=True)
class Message:
    role: Literal["system", "user"]
    content: str


@dataclass(frozen=True)
class Completion:
    text: str
    model: str
    finish_reason: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    provider_cost: float | None = None
    reasoning_characters: int | None = None


class IncompleteCompletionError(LLMError):
    """Retain safe accounting/finish metadata without accepting a partial answer."""

    def __init__(self, completion: Completion) -> None:
        super().__init__("LLM did not return a complete text answer")
        self.completion = completion


class LLM(Protocol):
    def complete(self, messages: tuple[Message, ...], *, max_tokens: int) -> Completion: ...
