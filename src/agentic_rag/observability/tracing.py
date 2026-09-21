"""Context-local spans shared by async tasks and AnyIO worker threads."""

import logging
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from threading import Lock
from time import perf_counter
from typing import Protocol
from uuid import uuid4

from agentic_rag.llm.base import Completion

logger = logging.getLogger(__name__)
NAMES = frozenset(
    {
        "http.request",
        "agent.run",
        "llm.generate",
        "llm.stream",
        "llm.tools",
        "retrieval",
        "reranking",
        "context",
        "tool.execute",
        "ingest",
        "health",
    }
)
FIELDS = frozenset(
    {
        "route",
        "method",
        "http_status",
        "mode",
        "candidate_count",
        "source_count",
        "tool",
        "cached",
        "error_type",
        "error_code",
        "model_calls",
        "tool_calls",
        "repair_attempts",
        "finish_reason",
        "prompt_tokens",
        "completion_tokens",
        "usage_known",
        "proposals",
        "repair",
        "outcome",
        "stream_completed",
    }
)
_current: ContextVar["Span | None"] = ContextVar("rag_span", default=None)
_tracer: ContextVar["Tracer | None"] = ContextVar("rag_tracer", default=None)


class RemoteSpan(Protocol):
    @property
    def id(self) -> str: ...
    @property
    def trace_id(self) -> str: ...
    def finish(self, fields: dict[str, object], status: str) -> None: ...


class Sink(Protocol):
    def start(self, name: str, trace_id: str, parent_id: str | None) -> RemoteSpan: ...
    def close(self) -> None: ...


@dataclass
class Span:
    name: str
    trace_id: str
    span_id: str
    parent_id: str | None
    status: str = "ok"
    fields: dict[str, object] = field(default_factory=dict)

    def set(self, **values: str | int | float | bool | None) -> None:
        # No prompts, tool arguments, source text, model outputs or arbitrary dicts.
        self.fields.update({key: value for key, value in values.items() if key in FIELDS})

    def fail(self, code: str) -> None:
        self.status = "error"
        self.set(error_code=code)

    def completion(self, result: Completion) -> None:
        self.set(
            finish_reason=result.finish_reason,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            usage_known=result.prompt_tokens is not None and result.completion_tokens is not None,
        )


def correlation() -> dict[str, str]:
    current = _current.get()
    return {"trace_id": current.trace_id, "span_id": current.span_id} if current else {}


def mark_failure(code: str) -> None:
    current = _current.get()
    if current:
        current.fail(code)


class Metrics:
    """Per-process Prometheus counters/histograms; finite names/statuses, no request labels."""

    bounds = (0.01, 0.1, 0.5, 1.0, 5.0, 30.0, 120.0)

    def __init__(self) -> None:
        self.lock = Lock()
        self.counts: Counter[tuple[str, str]] = Counter()
        self.seconds: dict[tuple[str, str], float] = defaultdict(float)
        self.buckets: Counter[tuple[str, str, float]] = Counter()
        self.tokens: Counter[str] = Counter()
        self.unknown_usage = 0

    def record(self, span: Span, elapsed: float) -> None:
        key = (span.name, span.status)
        with self.lock:
            self.counts[key] += 1
            self.seconds[key] += elapsed
            for bound in self.bounds:
                self.buckets[(*key, bound)] += int(elapsed <= bound)
            if span.name.startswith("llm."):
                for field_name in ("prompt_tokens", "completion_tokens"):
                    value = span.fields.get(field_name)
                    if type(value) is int and value >= 0:
                        self.tokens[field_name] += value
                self.unknown_usage += int(not span.fields.get("usage_known", False))

    def render(self) -> str:
        lines = ["# TYPE rag_span_duration_seconds histogram"]
        with self.lock:
            for (name, status), count in sorted(self.counts.items()):
                labels = f'name="{name}",status="{status}"'
                prefix = "rag_span_duration_seconds"
                for bound in self.bounds:
                    lines.append(
                        f'{prefix}_bucket{{{labels},le="{bound}"}} '
                        f"{self.buckets[(name, status, bound)]}"
                    )
                lines.extend(
                    [
                        f'{prefix}_bucket{{{labels},le="+Inf"}} {count}',
                        f"{prefix}_count{{{labels}}} {count}",
                        f"{prefix}_sum{{{labels}}} {self.seconds[(name, status)]}",
                    ]
                )
            lines.append("# TYPE rag_llm_tokens_total counter")
            for name in ("prompt_tokens", "completion_tokens"):
                lines.append(f'rag_llm_tokens_total{{kind="{name}"}} {self.tokens[name]}')
            lines.extend(
                [
                    "# TYPE rag_llm_usage_unknown_total counter",
                    f"rag_llm_usage_unknown_total {self.unknown_usage}",
                ]
            )
        return "\n".join(lines) + "\n"


class Tracer:
    def __init__(
        self,
        *,
        enabled: bool = True,
        sink: Sink | None = None,
        on_end: Callable[[dict[str, object]], None] | None = None,
    ) -> None:
        self.enabled, self.sink, self.on_end = enabled, sink, on_end
        self.metrics = Metrics()

    @contextmanager
    def activate(self) -> Iterator[None]:
        token = _tracer.set(self)
        try:
            yield
        finally:
            _tracer.reset(token)

    @contextmanager
    def span(self, name: str) -> Iterator[Span]:
        if name not in NAMES:
            raise ValueError("Unknown span name")
        parent = _current.get()
        current = Span(
            name,
            parent.trace_id if parent else uuid4().hex,
            uuid4().hex[:16],
            parent.span_id if parent else None,
        )
        if not self.enabled:
            yield current
            return
        remote = None
        if self.sink:
            try:
                remote = self.sink.start(name, current.trace_id, current.parent_id)
                current.span_id = remote.id
                current.trace_id = remote.trace_id
            except Exception:
                logger.warning("trace_export_start_failed")
        token = _current.set(current)
        start = perf_counter()
        try:
            yield current
        except BaseException as exc:
            current.status = "error" if isinstance(exc, Exception) else "cancelled"
            current.set(error_type=type(exc).__name__)
            raise
        finally:
            elapsed = perf_counter() - start
            record: dict[str, object] = {
                "name": name,
                "trace_id": current.trace_id,
                "span_id": current.span_id,
                "parent_id": current.parent_id,
                "status": current.status,
                "duration_seconds": elapsed,
                **current.fields,
            }
            try:
                self.metrics.record(current, elapsed)
                logger.info("span_finished", extra={"fields": record})
                if self.on_end:
                    self.on_end(record)
            except Exception:
                logger.warning("trace_local_record_failed")
            finally:
                try:
                    if remote:
                        remote.finish(current.fields, current.status)
                except Exception:
                    logger.warning("trace_export_finish_failed")
                finally:
                    _current.reset(token)

    def close(self) -> None:
        if self.sink:
            try:
                self.sink.close()
            except Exception:
                logger.warning("trace_export_shutdown_failed")


_disabled = Tracer(enabled=False)


@contextmanager
def span(name: str) -> Iterator[Span]:
    with (_tracer.get() or _disabled).span(name) as current:
        yield current
