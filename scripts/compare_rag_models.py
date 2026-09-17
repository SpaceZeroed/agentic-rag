"""Replay frozen hits through two models; no search rerun or reference leakage."""

import argparse
import json
from pathlib import Path
from uuid import UUID

import httpx

from agentic_rag.core.config import Settings
from agentic_rag.evaluation.rag_dataset import load_rag_dataset
from agentic_rag.evaluation.rag_runner import fingerprint, run_rag_evaluation
from agentic_rag.llm.compatible import CompatibleLLM
from agentic_rag.rag.context import build_context
from agentic_rag.retrieval.models import SearchHit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--qwen-disable-reasoning", action="store_true")
    args = parser.parse_args()
    settings = Settings()
    if not settings.llm_api_key:
        raise ValueError("LLM API key is required")
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    dataset, _ = load_rag_dataset(args.dataset)
    frozen = {}
    for case in baseline["cases"]:
        hits = []
        for raw in case["hits"]:
            fields = dict(raw)
            for name in ("chunk_id", "document_id", "revision_id"):
                fields[name] = UUID(fields[name])
            hits.append(SearchHit(**fields))
        context = build_context(case["query"], hits, max_prompt_bytes=24000)
        # Only the shared system instruction changes. User content and sources stay identical.
        assert context.messages[1].content == case["context"]["messages"][1]["content"]
        frozen[case["query"]] = hits
    assert {q.query for q in dataset.questions} == set(frozen)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    variants = [
        ("qwen", "qwen/qwen3.6-35b-a3b", False if args.qwen_disable_reasoning else None),
        ("haiku", "anthropic/claude-haiku-4.5", None),
    ]
    for name, _, _ in variants:
        if (args.output_dir / f"{name}.json").exists():
            raise ValueError("Output exists; choose a new directory")
    for name, model, reasoning in variants:
        print(f"Starting {name}: 10 requests, max_tokens=4096", flush=True)

        def frozen_search(query: str, variant: str = name) -> list[SearchHit]:
            case_id = next(q.id for q in dataset.questions if q.query == query)
            print(f"{variant}: {case_id}", flush=True)
            return frozen[query]

        with httpx.Client(
            base_url=settings.llm_base_url.rstrip("/") + "/",
            headers={"Authorization": "Bearer " + settings.llm_api_key.get_secret_value()},
            timeout=120,
            follow_redirects=False,
            trust_env=False,
        ) as client:
            report = run_rag_evaluation(
                args.dataset,
                frozen_search,
                CompatibleLLM(client, model, reasoning_enabled=reasoning),
                provider="compatible",
                k=5,
                max_tokens=4096,
                max_prompt_bytes=24000,
                config={
                    "mode": "frozen_bm25_hits",
                    "llm_model": model,
                    "temperature": 0,
                    "reasoning_enabled_requested": reasoning,
                    "llm_timeout_seconds": 120,
                    "baseline_report_sha256": fingerprint(baseline),
                    "retrieval_timing_policy": "In-memory replay, not retrieval latency",
                    "comparison_policy": (
                        "Same user messages/context, same system prompt; serial Qwen then Haiku"
                    ),
                },
            )
        serialized = json.dumps(report, ensure_ascii=False, indent=2)
        assert settings.llm_api_key.get_secret_value() not in serialized
        with (args.output_dir / f"{name}.json").open("x", encoding="utf-8") as out:
            out.write(serialized + "\n")
        print(
            json.dumps(
                {
                    "model": model,
                    "errors": report["errors"],
                    "output": str(args.output_dir / f"{name}.json"),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
