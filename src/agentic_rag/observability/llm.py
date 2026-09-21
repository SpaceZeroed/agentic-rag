"""Trace model calls without serializing messages, tools, deltas or answers."""

from collections.abc import AsyncGenerator
from contextlib import aclosing

from agentic_rag.llm.async_client import AsyncLLM, TextDelta
from agentic_rag.llm.base import Completion, IncompleteCompletionError, Message
from agentic_rag.llm.tool_client import ToolLLM, ToolTurn
from agentic_rag.observability.tracing import span


class TracedLLM:
    def __init__(self, wrapped: AsyncLLM) -> None:
        self.wrapped = wrapped

    async def complete(self, messages: tuple[Message, ...], *, max_tokens: int) -> Completion:
        with span("llm.generate") as current:
            try:
                result = await self.wrapped.complete(messages, max_tokens=max_tokens)
            except IncompleteCompletionError as exc:
                current.completion(exc.completion)
                raise
            current.completion(result)
            return result

    async def stream(
        self, messages: tuple[Message, ...], *, max_tokens: int
    ) -> AsyncGenerator[TextDelta | Completion, None]:
        with span("llm.stream") as current:
            try:
                async with aclosing(self.wrapped.stream(messages, max_tokens=max_tokens)) as source:
                    async for item in source:
                        if isinstance(item, Completion):
                            current.completion(item)
                        yield item
            except IncompleteCompletionError as exc:
                current.completion(exc.completion)
                raise


class TracedToolLLM:
    def __init__(self, wrapped: ToolLLM) -> None:
        self.wrapped = wrapped

    async def complete(
        self,
        messages: tuple[dict[str, object], ...],
        tools: list[dict[str, object]],
        *,
        max_tokens: int,
    ) -> ToolTurn:
        with span("llm.tools") as current:
            current.set(repair=not tools)
            result = await self.wrapped.complete(messages, tools, max_tokens=max_tokens)
            current.completion(result.completion)
            current.set(proposals=len(result.calls))
            return result
