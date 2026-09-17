"""Deterministic fixture generator, not a language model or quality baseline."""

import json

from agentic_rag.llm.base import Completion, Message


class FakeLLM:
    def complete(self, messages: tuple[Message, ...], *, max_tokens: int) -> Completion:
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        # Understands only our RAG prompt schema. No fabricated token accounting.
        sources = json.loads(messages[-1].content)["sources"]
        text = "INSUFFICIENT_EVIDENCE"
        if sources:
            source = sources[0]
            excerpt = source["text"].replace("[C", "［C")
            text = f"Mock: фрагмент источника (не ответ LLM):\n{excerpt} [{source['id']}]"
        return Completion(text, "fake-extractive-v1", "stop")
