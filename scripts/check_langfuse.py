"""Send fake-model application traces to local Langfuse and verify stored observations."""

import argparse
import asyncio
import json
from collections import Counter
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from time import monotonic

import httpx

from agentic_rag.agents.fake import FakeToolLLM
from agentic_rag.api.app import create_app
from agentic_rag.api.runtime import Runtime
from agentic_rag.core.config import Settings
from agentic_rag.llm.async_client import AsyncFakeLLM
from agentic_rag.observability.demo import DemoBackend
from agentic_rag.observability.langfuse import configured_tracer

ROOT = Path(__file__).resolve().parents[1]


async def run(output: Path) -> None:
    started = datetime.now(UTC) - timedelta(minutes=1)
    settings = Settings(
        _env_file=ROOT / ".env.langfuse",
        api_llm_provider="fake",
        api_retrieval_mode="bm25",
        observability_enabled=True,
        langfuse_enabled=True,
        langfuse_base_url="http://127.0.0.1:3000",
    )
    assert settings.langfuse_public_key and settings.langfuse_secret_key
    output.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, object]] = []
    tracer = configured_tracer(settings)
    tracer.on_end = records.append

    @asynccontextmanager
    async def factory(config: Settings) -> AsyncIterator[Runtime]:
        yield Runtime(config, DemoBackend(config), AsyncFakeLLM(), FakeToolLLM())

    app = create_app(settings, runtime_factory=factory, tracer=tracer)
    trace_ids = []
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://demo") as api,
    ):
        for route, query in (
            ("/query", "PagedAttention"),
            ("/agent", "/search PagedAttention"),
            ("/query", "fail"),
        ):
            response = await api.post(route, json={"query": query})
            assert response.status_code == (503 if query == "fail" else 200)
            trace_ids.append(response.headers["x-trace-id"])
    (output / "local_spans.json").write_text(json.dumps(records, indent=2) + "\n")
    verified = []
    async with httpx.AsyncClient(
        base_url=settings.langfuse_base_url,
        auth=(
            settings.langfuse_public_key.get_secret_value(),
            settings.langfuse_secret_key.get_secret_value(),
        ),
        timeout=15,
        trust_env=False,
    ) as client:
        for trace_id in trace_ids:
            expected = [r for r in records if r["trace_id"] == trace_id]
            deadline = monotonic() + 120
            while True:
                response = await client.get(
                    "/api/public/v2/observations",
                    params={
                        "traceId": trace_id,
                        "fromStartTime": started.isoformat(),
                        "toStartTime": (datetime.now(UTC) + timedelta(seconds=10)).isoformat(),
                        "fields": "core,basic,io",
                        "limit": 100,
                    },
                )
                data = response.json() if response.status_code == 200 else {}
                observations = data.get("data", [])
                if Counter(o["name"] for o in observations) == Counter(r["name"] for r in expected):
                    break
                if monotonic() >= deadline:
                    raise RuntimeError(
                        f"Trace not fully ingested: {trace_id}, HTTP {response.status_code}"
                    )
                await asyncio.sleep(2)
            by_id = {o["id"]: o for o in observations}
            for local in expected:
                remote = by_id[str(local["span_id"])]
                assert remote.get("parentObservationId") == local["parent_id"]
                assert remote.get("input") is None and remote.get("output") is None
                if local["status"] == "error":
                    assert remote["level"] == "ERROR"
            verified.append(
                {
                    "trace_id": trace_id,
                    "observations": len(observations),
                    "url": f"http://localhost:3000/project/retrieval-rag/traces/{trace_id}",
                }
            )
    (output / "verified.json").write_text(json.dumps(verified, indent=2) + "\n")
    print(json.dumps(verified, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(run(parser.parse_args().output))
