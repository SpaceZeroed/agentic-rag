import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from agentic_rag.llm.base import LLM, Completion, LLMError
from agentic_rag.rag.context import Citation, Context, build_context
from agentic_rag.retrieval.models import SearchHit


class CitationError(LLMError):
    """Generated citation syntax or identity does not match the supplied context."""


@dataclass(frozen=True)
class Answer:
    status: Literal["answered", "insufficient_evidence"]
    text: str
    citations: tuple[Citation, ...]
    context: Context
    completion: Completion | None


def answer(
    query: str,
    hits: Sequence[SearchHit],
    llm: LLM,
    *,
    max_prompt_bytes: int = 24000,
    max_tokens: int = 512,
) -> Answer:
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    context = build_context(query, hits, max_prompt_bytes=max_prompt_bytes)
    completion = llm.complete(context.messages, max_tokens=max_tokens) if context.sources else None
    return finalize_answer(context, completion)


def finalize_answer(context: Context, completion: Completion | None) -> Answer:
    """One final validation contract for CLI, async HTTP and streamed generation."""
    abstention = "Недостаточно данных в переданном контексте."
    if not context.sources:
        return Answer("insufficient_evidence", abstention, (), context, None)
    if completion is None:
        raise LLMError("Missing completion for nonempty context")
    if completion.finish_reason != "stop" or not completion.text.strip():
        raise LLMError("LLM did not return a complete text answer")
    text = completion.text.strip()
    if text == "INSUFFICIENT_EVIDENCE":
        return Answer("insufficient_evidence", abstention, (), context, completion)
    ids = re.findall(r"\[(C[1-9][0-9]*)\]", text)
    remainder = re.sub(r"\[C[1-9][0-9]*\]", "", text)
    available = {source.id: source for source in context.sources}
    if not ids or "[C" in remainder or any(key not in available for key in ids):
        raise CitationError("Missing, malformed or unknown citation")
    citations = tuple(available[key] for key in dict.fromkeys(ids))
    return Answer("answered", text, citations, context, completion)
