import asyncio
from collections.abc import AsyncGenerator

from agentic_rag.evaluation.inference import measure, run_cell
from agentic_rag.llm.async_client import TextDelta
from agentic_rag.llm.base import Completion, IncompleteCompletionError, Message


class FixtureLLM:
    def __init__(self, mode: str = "success") -> None:
        self.mode = mode
        self.active = self.peak = 0

    async def complete(self, messages: tuple[Message, ...], *, max_tokens: int) -> Completion:
        raise NotImplementedError

    async def stream(
        self, messages: tuple[Message, ...], *, max_tokens: int
    ) -> AsyncGenerator[TextDelta | Completion, None]:
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            yield TextDelta("private answer")
            await asyncio.sleep(0.01)
            if self.mode == "missing_terminal":
                return
            completion = Completion("private answer", "fixture", "stop")
            if self.mode == "length":
                raise IncompleteCompletionError(
                    Completion("private answer", "fixture", "length", 10, 5)
                )
            yield completion
        finally:
            self.active -= 1


def test_bounded_concurrency_unknown_usage_and_privacy() -> None:
    llm = FixtureLLM()
    result = asyncio.run(run_cell(llm, (), concurrency=2, requests=5, max_tokens=10, timeout=1))
    assert llm.peak == 2
    assert llm.active == 0
    assert result["successes"] == 5
    assert result["success_usage_coverage"] == 0
    assert "private answer" not in str(result)


def test_incomplete_and_timeout_are_not_success() -> None:
    for mode in ("length", "missing_terminal"):
        sample = asyncio.run(measure(FixtureLLM(mode), (), max_tokens=10, timeout=1))
        assert sample.status == "incomplete"
        assert sample.first_text_seconds is not None
    llm = FixtureLLM()
    sample = asyncio.run(measure(llm, (), max_tokens=10, timeout=0.001))
    assert sample.status == "timeout"
    assert llm.active == 0
