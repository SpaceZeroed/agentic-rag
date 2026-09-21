"""Bounded live benchmark using configured credentials; never saves answer text."""

import argparse
import asyncio
import hashlib
import json
import platform
import random
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import httpx

from agentic_rag.core.config import Settings
from agentic_rag.evaluation.inference import measure, run_cell
from agentic_rag.llm.async_client import AsyncCompatibleLLM
from agentic_rag.llm.base import Message


async def run(output: Path) -> None:
    settings = Settings()
    if not settings.llm_api_key or not settings.llm_model:
        raise ValueError("Configure llm_api_key and llm_model")
    output.mkdir(parents=True, exist_ok=False)
    report: dict[str, object] = {
        "started_at": datetime.now(UTC).isoformat(),
        "model": settings.llm_model,
        "reasoning_enabled_requested": settings.llm_reasoning_enabled,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), Path("src/agentic_rag/evaluation/inference.py"))
        },
        "client_platform": platform.platform(),
        "max_tokens": 512,
        "timeout_seconds": 90,
        "requests_per_cell": 4,
        "policy": "Closed loop; shared HTTP pool; no retries; one warmup per cell excluded. "
        "Identical synthetic prompts may hit provider caches. Random cell order seed=12. "
        "First text and chunk gaps include network/buffering; no exact TTFT/TPOT. "
        "Token throughput counts reported successful usage only, possibly reasoning tokens. "
        "No GPU telemetry or provider configuration control; small exploratory sample.",
    }
    cells: list[dict[str, object]] = []
    report["cells"] = cells
    schedule = [(size, c) for size in (8, 128) for c in (1, 2, 4)]
    random.Random(12).shuffle(schedule)
    async with httpx.AsyncClient(
        base_url=settings.llm_base_url.rstrip("/") + "/",
        headers={"Authorization": "Bearer " + settings.llm_api_key.get_secret_value()},
        timeout=90,
        limits=httpx.Limits(max_connections=4, max_keepalive_connections=4),
        trust_env=False,
        follow_redirects=False,
    ) as client:
        llm = AsyncCompatibleLLM(
            client, settings.llm_model, reasoning_enabled=settings.llm_reasoning_enabled
        )
        for size, concurrency in schedule:
            context = "\n".join(
                f"Service {i}: requests use a 30 second timeout and at most 2 retries."
                for i in range(size)
            )
            messages = (
                Message("system", "Answer briefly in Russian. Use only the supplied context."),
                Message(
                    "user", context + "\nSummarize the timeout and retry policy in 3 sentences."
                ),
            )
            warmup = await measure(llm, messages, max_tokens=512, timeout=90)
            cell = await run_cell(
                llm, messages, concurrency=concurrency, requests=4, max_tokens=512, timeout=90
            )
            cell.update(
                context_records=size,
                prompt_bytes=sum(len(m.content.encode()) for m in messages),
                prompt_sha256=hashlib.sha256(repr(messages).encode()).hexdigest(),
                warmup=asdict(warmup),
            )
            cells.append(cell)
            serialized = json.dumps(report, ensure_ascii=False, indent=2)
            assert settings.llm_api_key.get_secret_value() not in serialized
            (output / "report.json").write_text(serialized + "\n", encoding="utf-8")
            print(
                json.dumps({k: v for k, v in cell.items() if k not in {"samples", "warmup"}}),
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(run(args.output_dir))


if __name__ == "__main__":
    main()
