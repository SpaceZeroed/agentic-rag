"""Manual SDK adapter; no callback/autoinstrumentation or prompt capture."""

from langfuse import Langfuse, LangfuseGeneration, LangfuseSpan
from langfuse.types import TraceContext
from opentelemetry.context import Context as OtelContext
from opentelemetry.context import attach, detach
from opentelemetry.sdk.trace import TracerProvider

from agentic_rag.core.config import Settings
from agentic_rag.observability.tracing import Tracer


class LangfuseObservation:
    def __init__(self, observation: LangfuseSpan | LangfuseGeneration) -> None:
        self.observation = observation

    @property
    def id(self) -> str:
        return self.observation.id

    @property
    def trace_id(self) -> str:
        return self.observation.trace_id

    def finish(self, fields: dict[str, object], status: str) -> None:
        usage: dict[str, int] = {
            target: value
            for source, target in (("prompt_tokens", "input"), ("completion_tokens", "output"))
            if type(value := fields.get(source)) is int and value >= 0
        }
        try:
            self.observation.update(
                metadata={**fields, "status": status},
                level="DEFAULT" if status == "ok" else "ERROR",
                usage_details=usage or None,
            )
        finally:
            self.observation.end()


class LangfuseSink:
    def __init__(self, client: Langfuse) -> None:
        self.client = client

    def start(self, name: str, trace_id: str, parent_id: str | None) -> LangfuseObservation:
        if parent_id is None:
            # A trace_context with only trace_id creates an unrecorded synthetic
            # parent in SDK 4.15.4. Start a true root and adopt its generated ID.
            token = attach(OtelContext())
            try:
                if name.startswith("llm."):
                    return LangfuseObservation(
                        self.client.start_observation(
                            name=name,
                            as_type="generation",
                        )
                    )
                return LangfuseObservation(self.client.start_observation(name=name, as_type="span"))
            finally:
                detach(token)
        context: TraceContext = {"trace_id": trace_id}
        if parent_id is not None:
            context["parent_span_id"] = parent_id
        if name.startswith("llm."):
            return LangfuseObservation(
                self.client.start_observation(
                    name=name,
                    as_type="generation",
                    trace_context=context,
                )
            )
        return LangfuseObservation(
            self.client.start_observation(
                name=name,
                as_type="span",
                trace_context=context,
            )
        )

    def close(self) -> None:
        self.client.shutdown()


def configured_tracer(settings: Settings) -> Tracer:
    if not settings.langfuse_enabled:
        return Tracer(enabled=settings.observability_enabled)
    if not settings.observability_enabled or not (
        settings.langfuse_public_key and settings.langfuse_secret_key
    ):
        raise ValueError("Langfuse requires observability and explicit public/secret keys")
    # Private provider: do not attach our exporter to other libraries' global spans.
    client = Langfuse(
        public_key=settings.langfuse_public_key.get_secret_value(),
        secret_key=settings.langfuse_secret_key.get_secret_value(),
        base_url=settings.langfuse_base_url,
        environment=settings.environment,
        timeout=5,
        tracer_provider=TracerProvider(),
    )
    return Tracer(sink=LangfuseSink(client))
