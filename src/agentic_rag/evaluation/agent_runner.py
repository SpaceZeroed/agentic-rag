"""Bounded agent runs with explicit replay provenance and immutable local reports."""

import asyncio
import hashlib
import json
import math
import platform
import subprocess
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from decimal import Decimal
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import Literal

import anyio
import httpx

from agentic_rag.agents.service import Agent
from agentic_rag.evaluation.agent_backend import SnapshotBackend
from agentic_rag.evaluation.agent_dataset import AgentCase, AgentDataset
from agentic_rag.ingestion.models import PreparedDocument
from agentic_rag.llm.async_client import MAX_OUTPUT_BYTES
from agentic_rag.llm.tool_client import ToolLLM, ToolTurn
from agentic_rag.retrieval.sparse import TOKENIZER_VERSION, BM25Config


def fingerprint_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_report(path: Path, value: object, secrets: tuple[str, ...] = ()) -> str:
    text = json.dumps(value, ensure_ascii=False, indent=2, default=str, allow_nan=False) + "\n"
    for secret in secrets:
        if secret:
            text = text.replace(secret, "[REDACTED]")
    with path.open("x", encoding="utf-8") as output:
        output.write(text)
    return fingerprint_bytes(text.encode())


def source_state(root: Path) -> dict[str, object]:
    """Fingerprint tracked code plus new source files, excluding docs/config/artifacts."""

    def git(*args: str) -> bytes:
        return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.DEVNULL)

    try:
        paths = git(
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
            "--",
            "src",
            "tests",
            "scripts",
            "benchmarks",
            "pyproject.toml",
            "uv.lock",
        )
        hashes = {
            p: fingerprint_bytes((root / p).read_bytes()) if (root / p).is_file() else None
            for p in sorted(set(paths.decode().split("\0")) - {""})
        }
        return {
            "commit": git("rev-parse", "HEAD").decode().strip(),
            "dirty_diff_sha256": fingerprint_bytes(
                git(
                    "diff",
                    "HEAD",
                    "--binary",
                    "--",
                    "src",
                    "tests",
                    "scripts",
                    "benchmarks",
                    "pyproject.toml",
                    "uv.lock",
                )
            ),
            "working_file_sha256": hashes,
        }
    except (OSError, subprocess.CalledProcessError):
        raise ValueError("Cannot fingerprint repository state") from None


def safe_response(raw: bytes) -> dict[str, object]:
    """Only allow listed fields; provider reasoning and arbitrary errors are excluded."""
    try:
        data = json.loads(raw)
    except ValueError:
        return {"invalid_json": True}
    if not isinstance(data, dict):
        return {"invalid_shape": True}
    usage = data.get("usage")
    clean_usage: dict[str, object] = {}
    if isinstance(usage, dict):
        for key in ("prompt_tokens", "completion_tokens", "total_tokens", "cost"):
            value = usage.get(key)
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                and value >= 0
            ):
                clean_usage[key] = value
    choices = []
    for choice in data.get("choices", []) if isinstance(data.get("choices"), list) else []:
        if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
            continue
        message = choice["message"]
        calls = []
        raw_calls = message.get("tool_calls")
        for call in raw_calls if isinstance(raw_calls, list) else []:
            if isinstance(call, dict) and isinstance(call.get("function"), dict):
                calls.append(
                    {
                        "id": call.get("id"),
                        "type": call.get("type"),
                        "function": {
                            "name": call["function"].get("name"),
                            "arguments": call["function"].get("arguments"),
                        },
                    }
                )
        reasoning = message.get("reasoning_content") or message.get("reasoning")
        choices.append(
            {
                "finish_reason": choice.get("finish_reason"),
                "message": {
                    "role": message.get("role"),
                    "content": message.get("content"),
                    "tool_calls": calls,
                },
                "reasoning_characters": len(reasoning) if isinstance(reasoning, str) else 0,
            }
        )
    return {"usage": clean_usage, "choices": choices}


class TraceTransport(httpx.AsyncBaseTransport):
    """Capture even HTTP error bodies, bounded to the same limit as the LLM parser."""

    def __init__(self, wrapped: httpx.AsyncBaseTransport) -> None:
        self.wrapped = wrapped
        self.records: list[dict[str, object]] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        record: dict[str, object] = {"request": json.loads(request.content)}
        self.records.append(record)
        try:
            response = await self.wrapped.handle_async_request(request)
            record["http_status"] = response.status_code
            body = bytearray()
            try:
                async for chunk in response.aiter_bytes():
                    body.extend(chunk)
                    if len(body) > MAX_OUTPUT_BYTES:
                        record["response_too_large"] = True
                        raise httpx.DecodingError("Response exceeds capture limit")
            finally:
                await response.aclose()
            record["response"] = safe_response(bytes(body))
            headers = dict(response.headers)
            # aiter_bytes has already decoded any content encoding.
            headers.pop("content-encoding", None)
            headers.pop("content-length", None)
            return httpx.Response(
                response.status_code,
                headers=headers,
                content=bytes(body),
                extensions=response.extensions,
            )
        except asyncio.CancelledError:
            record["error"] = "cancelled"
            raise
        except Exception as exc:
            record["error"] = type(exc).__name__
            raise

    async def aclose(self) -> None:
        await self.wrapped.aclose()


class RunCallLimit(Exception):
    pass


class CallBudget:
    def __init__(self, maximum: int) -> None:
        if maximum < 1:
            raise ValueError("Call budget must be positive")
        self.maximum = maximum
        self.used = 0


class RecordedModel:
    def __init__(self, llm: ToolLLM, case: AgentCase, budget: CallBudget, *, live: bool) -> None:
        self.llm, self.budget, self.live = llm, budget, live
        self.prefix = tuple(t.turn() for t in case.fault.prefix) if case.fault else ()
        self.records: list[dict[str, object]] = []

    async def complete(
        self,
        messages: tuple[dict[str, object], ...],
        tools: list[dict[str, object]],
        *,
        max_tokens: int,
    ) -> ToolTurn:
        index = len(self.records)
        replayed = index < len(self.prefix)
        if not replayed and self.budget.used >= self.budget.maximum:
            raise RunCallLimit
        record: dict[str, object] = {
            "origin": "replayed" if replayed else "live" if self.live else "mock",
            "messages": messages,
            "tools": tools,
            "max_tokens": max_tokens,
        }
        self.records.append(record)
        try:
            if replayed:
                turn = self.prefix[index]
            else:
                self.budget.used += 1
                turn = await self.llm.complete(messages, tools, max_tokens=max_tokens)
            record["assistant_message"] = turn.assistant_message()
            # Only parsed usage and assistant output: ToolTurn never contains raw reasoning.
            record["completion"] = asdict(turn.completion)
            return turn
        except asyncio.CancelledError:
            record["error"] = "cancelled"
            raise
        except Exception as exc:
            record["error"] = type(exc).__name__
            raise


def usage_summary(records: list[dict[str, object]]) -> dict[str, object]:
    prompt = output = known_usage = known_cost = 0
    cost = Decimal(0)
    for record in records:
        response = record.get("response")
        usage = response.get("usage") if isinstance(response, dict) else None
        if not isinstance(usage, dict):
            continue
        if isinstance(usage.get("prompt_tokens"), int) and isinstance(
            usage.get("completion_tokens"), int
        ):
            known_usage += 1
            prompt += usage["prompt_tokens"]
            output += usage["completion_tokens"]
        if usage.get("cost") is not None:
            known_cost += 1
            cost += Decimal(str(usage["cost"]))
    return {
        "actual_http_calls": len(records),
        "prompt_tokens_known": prompt,
        "completion_tokens_known": output,
        "usage_missing_calls": len(records) - known_usage,
        "provider_cost_known": str(cost),
        "cost_missing_calls": len(records) - known_cost,
        "cost_unit": "provider-reported; no currency conversion",
    }


async def run_agent_suite(
    dataset: AgentDataset,
    documents: tuple[PreparedDocument, ...],
    cases: list[AgentCase],
    llm: ToolLLM,
    output: Path,
    *,
    mode: Literal["mock", "live"],
    model: str,
    max_calls: int,
    manifest_extra: dict[str, object],
    trace: TraceTransport | None = None,
    secrets: tuple[str, ...] = (),
) -> dict[str, object]:
    if not cases or len({c.id for c in cases}) != len(cases):
        raise ValueError("Select unique nonempty cases")
    if mode == "live" and trace is None:
        raise ValueError("Live runs require HTTP capture")
    budget = CallBudget(max_calls)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        **manifest_extra,
        "dataset": dataset.name,
        "mode": mode,
        "model": model,
        "started_at": datetime.now(UTC).isoformat(),
        "selected_cases": [c.id for c in cases],
        "policy": dataset.run_policy.model_dump(),
        "max_new_model_calls": max_calls,
        "backend": "isolated in-memory corpus; real BM25, catalog snapshot; not PostgreSQL/API",
        "runtime_versions": {
            "python": platform.python_version(),
            **{name: version(name) for name in ("langgraph", "httpx", "pydantic", "anyio")},
        },
        "bm25": asdict(BM25Config()),
        "tokenizer": TOKENIZER_VERSION,
        "documents": [
            {
                "document_id": d.document_id,
                "revision_id": d.revision_id,
                "raw_sha256": d.raw_sha256,
                "chunk_ids": [c.id for c in d.chunks],
            }
            for d in documents
        ],
        "quality_scores": None,
        "review_required": True,
    }
    manifest_hash = write_report(output / "manifest.json", manifest, secrets)
    summaries: list[dict[str, object]] = []
    stop_reason: str | None = None
    review_cases: list[dict[str, object]] = []
    try:
        for case in cases:
            if budget.used >= budget.maximum:
                stop_reason = "run_call_limit"
                break
            backend = SnapshotBackend(documents, case.setup)
            recorded = RecordedModel(llm, case, budget, live=mode == "live")
            start_record = len(trace.records) if trace else 0
            start = perf_counter()
            result = None
            error = None
            interrupted = False
            try:
                with anyio.fail_after(dataset.run_policy.deadline_seconds_per_case):
                    result = await Agent(
                        recorded,
                        backend,
                        dataset.run_policy.limits(),
                        provider="compatible" if mode == "live" else "fake",
                    ).run(case.query)
            except RunCallLimit:
                error = "run_call_limit"
            except TimeoutError:
                error = "request_timeout"
            except asyncio.CancelledError:
                error, interrupted = "cancelled", True
            except Exception as exc:
                error = "runner_error:" + type(exc).__name__
            rounds = trace.records[start_record:] if trace else []
            summary: dict[str, object] = {
                "id": case.id,
                "group": case.mode,
                "status": result.status
                if result
                else "interrupted"
                if interrupted
                else "runner_failed",
                "error": error or (result.error if result else None),
                "elapsed_seconds": perf_counter() - start,
                "new_model_calls": sum(r["origin"] != "replayed" for r in recorded.records),
                "replayed_model_calls": sum(r["origin"] == "replayed" for r in recorded.records),
                "controlled_fault_exercised": (
                    sum(r["origin"] == "replayed" for r in recorded.records) == len(recorded.prefix)
                    if case.setup == "scripted_prefix"
                    else any(e["fault_injected"] for e in backend.events)
                )
                if case.mode == "controlled"
                else None,
                "model_calls": result.model_calls if result else len(recorded.records),
                "tool_calls": result.tool_calls if result else None,
                "repair_attempts": result.repair_attempts if result else None,
                "tool_errors": [o.error for o in result.observations if o.status == "error"]
                if result
                else None,
                "usage": usage_summary(rounds),
                "quality_scores": None,
            }
            report = {
                "summary": summary,
                "query": case.query,
                "setup": case.setup,
                "manifest_sha256": manifest_hash,
                "result": result.model_dump(mode="json") if result else None,
                "model_turns": recorded.records,
                "backend_events": backend.events,
                "http_rounds": rounds,
            }
            report_hash = write_report(output / f"{case.id}.json", report, secrets)
            summaries.append(summary)
            review_cases.append(
                {
                    "id": case.id,
                    "report_sha256": report_hash,
                    "structural_validation_passed": result.status == "answered" if result else None,
                    "dimensions": {
                        d: {"score": None, "applicable": None, "rationale": ""}
                        for d in dataset.review_dimensions
                    },
                    "claim_reviews": [],
                }
            )
            if interrupted:
                stop_reason = "cancelled"
                raise asyncio.CancelledError
            if error or (result and result.error in ("generation_failed", "llm_timeout")):
                stop_reason = str(summary["error"])
                break
    finally:
        finished = {s["id"] for s in summaries}
        summary_report: dict[str, object] = {
            "mode": mode,
            "cases": summaries,
            "stop_reason": stop_reason,
            "skipped_cases": [c.id for c in cases if c.id not in finished],
            "new_model_calls": budget.used,
            "usage": usage_summary(trace.records if trace else []),
            "quality_scores": None,
            "groups": {
                group: [s for s in summaries if s["group"] == group]
                for group in ("natural", "controlled")
            },
            "status_counts_by_group": {
                group: dict(Counter(str(s["status"]) for s in summaries if s["group"] == group))
                for group in ("natural", "controlled")
            },
        }
        write_report(output / "summary.json", summary_report, secrets)
        write_report(
            output / "review.json",
            {
                "manifest_sha256": manifest_hash,
                "reviewer": None,
                "reviewer_type": None,
                "cases": review_cases,
            },
            secrets,
        )
    return summary_report
