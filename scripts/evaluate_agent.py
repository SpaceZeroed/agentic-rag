"""Validate or run frozen agent cases; paid execution requires explicit CLI opt-in."""

import argparse
import asyncio
import json
import os
from pathlib import Path
from urllib.parse import urlsplit

# Set before importing clients/graphs. Reports are local; no external tracing.
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

import httpx  # noqa: E402

from agentic_rag.agents.fake import FakeToolLLM  # noqa: E402
from agentic_rag.core.config import Settings  # noqa: E402
from agentic_rag.evaluation.agent_dataset import load_agent_dataset  # noqa: E402
from agentic_rag.evaluation.agent_runner import (  # noqa: E402
    TraceTransport,
    fingerprint_bytes,
    run_agent_suite,
    source_state,
)
from agentic_rag.llm.tool_client import CompatibleToolLLM  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "benchmarks/agent_v1/cases.json")
    parser.add_argument("--mode", choices=("validate", "mock", "live"), default="validate")
    parser.add_argument(
        "--cases", nargs="+", help="Case IDs; live mode requires an explicit selection"
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--allow-paid", action="store_true")
    parser.add_argument("--max-live-calls", type=int)
    args = parser.parse_args()
    dataset, documents = load_agent_dataset(args.dataset)
    by_id = {c.id: c for c in dataset.cases}
    selected = args.cases or list(by_id)
    if len(set(selected)) != len(selected) or not set(selected) <= by_id.keys():
        parser.error("Select unique known case IDs")
    cases = [by_id[key] for key in selected]
    if args.mode == "validate":
        print(
            json.dumps(
                {
                    "validated_cases": [c.id for c in cases],
                    "documents": len(documents),
                    "paid_calls": 0,
                },
                ensure_ascii=False,
            )
        )
        return
    if args.output_dir is None:
        parser.error("--output-dir is required for runs")
    if args.mode == "live" and (
        not args.allow_paid or not args.cases or not args.max_live_calls or args.max_live_calls < 1
    ):
        parser.error("Live mode requires --allow-paid, --cases and positive --max-live-calls")
    extra: dict[str, object] = {
        "repository": source_state(ROOT),
        "cases_sha256": fingerprint_bytes(args.dataset.read_bytes()),
        "corpus_manifest_sha256": dataset.corpus.sha256,
    }
    if args.mode == "mock":
        report = await run_agent_suite(
            dataset,
            documents,
            cases,
            FakeToolLLM(),
            args.output_dir,
            mode="mock",
            model="fake-tools",
            max_calls=len(cases) * dataset.run_policy.model_calls_per_case,
            manifest_extra=extra,
        )
    else:
        settings = Settings()
        if not settings.llm_model or not settings.llm_api_key:
            parser.error("Configure RAG_LLM_MODEL and RAG_LLM_API_KEY for a live run")
        if settings.llm_reasoning_enabled != dataset.run_policy.reasoning_enabled:
            parser.error("Configured reasoning must match the frozen policy (false)")
        secret = settings.llm_api_key.get_secret_value()
        trace = TraceTransport(httpx.AsyncHTTPTransport(retries=0))
        extra["llm_timeout_seconds"] = settings.llm_timeout_seconds
        endpoint = urlsplit(settings.llm_base_url)
        extra["provider_origin"] = {
            "scheme": endpoint.scheme,
            "host": endpoint.hostname,
            "port": endpoint.port,
        }
        async with httpx.AsyncClient(
            base_url=settings.llm_base_url.rstrip("/") + "/",
            headers={"Authorization": "Bearer " + secret, "Accept-Encoding": "identity"},
            transport=trace,
            timeout=settings.llm_timeout_seconds,
            follow_redirects=False,
            trust_env=False,
        ) as client:
            report = await run_agent_suite(
                dataset,
                documents,
                cases,
                CompatibleToolLLM(client, settings.llm_model, reasoning_enabled=False),
                args.output_dir,
                mode="live",
                model=settings.llm_model,
                max_calls=args.max_live_calls,
                manifest_extra=extra,
                trace=trace,
                secrets=(secret,),
            )
    print(
        json.dumps(
            {
                "output": str(args.output_dir),
                "completed_cases": len(report["cases"]) if isinstance(report["cases"], list) else 0,
                "stop_reason": report["stop_reason"],
                "usage": report["usage"],
                "quality_scores": None,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (ValueError, OSError) as exc:
        # Configuration/driver exceptions can contain secrets; never print their values.
        raise SystemExit(f"Evaluation setup failed ({type(exc).__name__}); no retry.") from None
