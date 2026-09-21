"""Client-observed streaming benchmark; chunks are not tokens."""

import asyncio
import statistics
import time
from dataclasses import asdict, dataclass

from agentic_rag.llm.async_client import AsyncLLM, TextDelta
from agentic_rag.llm.base import Completion, IncompleteCompletionError, LLMError, Message


@dataclass
class Sample:
    status: str
    elapsed_seconds: float
    first_text_seconds: float | None
    chunk_gaps_seconds: list[float]
    completion: dict[str, str | int | float | None] | None


async def measure(
    llm: AsyncLLM, messages: tuple[Message, ...], *, max_tokens: int, timeout: float
) -> Sample:
    start = time.perf_counter()
    first: float | None = None
    previous: float | None = None
    gaps: list[float] = []
    completion: Completion | None = None
    status = "incomplete"
    try:
        async with asyncio.timeout(timeout):
            async for event in llm.stream(messages, max_tokens=max_tokens):
                now = time.perf_counter()
                if isinstance(event, TextDelta):
                    if not event.text:
                        continue
                    if first is None:
                        first = now - start
                    if previous is not None:
                        gaps.append(now - previous)
                    previous = now
                else:
                    completion = event
                    status = "success"
    except IncompleteCompletionError as exc:
        completion = exc.completion
        status = "incomplete"
    except TimeoutError:
        status = "timeout"
    except LLMError:
        status = "transport_or_protocol_error"
    metadata = None
    if completion is not None:
        metadata = {k: v for k, v in asdict(completion).items() if k != "text"}
    return Sample(status, time.perf_counter() - start, first, gaps, metadata)


async def run_cell(
    llm: AsyncLLM,
    messages: tuple[Message, ...],
    *,
    concurrency: int,
    requests: int,
    max_tokens: int,
    timeout: float,
) -> dict[str, object]:
    if min(concurrency, requests, max_tokens) < 1 or timeout <= 0:
        raise ValueError("Benchmark limits must be positive")
    queue = list(range(requests))
    samples: list[Sample] = []

    async def worker() -> None:
        while queue:
            queue.pop()
            samples.append(await measure(llm, messages, max_tokens=max_tokens, timeout=timeout))

    start = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(min(concurrency, requests))))
    elapsed = time.perf_counter() - start
    successful = [s for s in samples if s.status == "success"]
    tokens = [
        s.completion["completion_tokens"]
        for s in successful
        if s.completion is not None and s.completion["completion_tokens"] is not None
    ]
    firsts = [s.first_text_seconds for s in successful if s.first_text_seconds is not None]
    return {
        "concurrency": concurrency,
        "requests": requests,
        "wall_seconds": elapsed,
        "successes": len(successful),
        "successful_requests_per_second": len(successful) / elapsed,
        "known_success_completion_tokens_per_second": sum(float(t) for t in tokens) / elapsed,
        "success_usage_coverage": len(tokens) / len(successful) if successful else None,
        "median_first_text_seconds": statistics.median(firsts) if firsts else None,
        "median_success_seconds": statistics.median(s.elapsed_seconds for s in successful)
        if successful
        else None,
        "samples": [asdict(s) for s in samples],
    }
