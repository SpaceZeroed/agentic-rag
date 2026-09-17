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


class LLM(Protocol):
    def complete(self, messages: tuple[Message, ...], *, max_tokens: int) -> Completion: ...
