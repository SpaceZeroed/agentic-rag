import asyncio
import gzip
import json
from pathlib import Path

import anyio
import httpx
import pytest

from agentic_rag.evaluation.agent_backend import SnapshotBackend
from agentic_rag.evaluation.agent_dataset import AgentDataset, load_agent_dataset
from agentic_rag.evaluation.agent_runner import (
    TraceTransport,
    fingerprint_bytes,
    run_agent_suite,
    write_report,
)
from agentic_rag.ingestion.models import PreparedDocument
from agentic_rag.llm.base import Completion
from agentic_rag.llm.tool_client import CompatibleToolLLM, FunctionCall, ToolCall, ToolTurn
from agentic_rag.tools.models import CatalogInput, SearchInput

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "benchmarks/agent_v1/cases.json"
pytestmark = pytest.mark.anyio


@pytest.fixture
def data() -> tuple[AgentDataset, tuple[PreparedDocument, ...]]:
    return load_agent_dataset(DATASET)


class Scripted:
    def __init__(self, *turns: ToolTurn) -> None:
        self.turns = iter(turns)
        self.requests: list[tuple[dict[str, object], ...]] = []

    async def complete(
        self,
        messages: tuple[dict[str, object], ...],
        tools: list[dict[str, object]],
        *,
        max_tokens: int,
    ) -> ToolTurn:
        self.requests.append(messages)
        return next(self.turns)


def final(text: str) -> ToolTurn:
    return ToolTurn(Completion(text, "fixture", "stop"))


def call(arguments: str, id: str = "fixed") -> ToolTurn:
    return ToolTurn(
        Completion("", "fixture", "tool_calls"),
        (ToolCall(id=id, function=FunctionCall(name="document_catalog", arguments=arguments)),),
    )


def read(directory: Path, name: str) -> dict[str, object]:
    value = json.loads((directory / name).read_text())
    assert isinstance(value, dict)
    return value


async def test_pinned_snapshot_catalog_pagination_and_real_bm25(
    data: tuple[AgentDataset, tuple[PreparedDocument, ...]],
) -> None:
    _, documents = data
    backend = SnapshotBackend(documents, "standard")
    first = await backend.catalog(CatalogInput(limit=3))
    rest = await backend.catalog(CatalogInput(limit=20, offset=3))
    assert first.has_more and not rest.has_more
    assert len(first.documents) == 3 and len(rest.documents) == 7
    assert len({d.document_id for d in (*first.documents, *rest.documents)}) == 10
    filtered = await backend.catalog(CatalogInput(title_contains="paged_attention"))
    assert [d.title for d in filtered.documents] == ["paged_attention"]
    assert not (await backend.catalog(CatalogInput(title_contains="%"))).documents
    hits = await backend.search(SearchInput(query="PagedAttention KV cache", k=3))
    assert hits and hits[0].title == "paged_attention"
    assert all(h.document_id in {d.document_id for d in documents} for h in hits)


@pytest.mark.parametrize("mutation", ["hash", "duplicate", "evidence", "fault", "unsafe_id"])
async def test_invalid_dataset_rejected_before_run(tmp_path: Path, mutation: str) -> None:
    raw = json.loads(DATASET.read_text())
    raw["corpus"]["manifest"] = str(ROOT / "benchmarks/rag_v1/dataset.json")
    if mutation == "hash":
        raw["corpus"]["sha256"] = "0" * 64
    elif mutation == "duplicate":
        raw["cases"][1]["id"] = raw["cases"][0]["id"]
    elif mutation == "evidence":
        raw["cases"][3]["evidence"][0]["text"] = "NOT IN CORPUS"
    elif mutation == "fault":
        raw["cases"][0]["setup"] = "empty_search"
    else:
        raw["cases"][0]["id"] = "../outside"
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        load_agent_dataset(path)


async def test_controlled_repair_has_replay_provenance_and_no_label_leak(
    tmp_path: Path, data: tuple[AgentDataset, tuple[PreparedDocument, ...]]
) -> None:
    dataset, documents = data
    case = next(c for c in dataset.cases if c.id == "uncited_catalog_answer")
    llm = Scripted(final("Catalog [T1]"))
    output = tmp_path / "run"
    summary = await run_agent_suite(
        dataset,
        documents,
        [case],
        llm,
        output,
        mode="mock",
        model="fixture",
        max_calls=1,
        manifest_extra={},
    )
    assert summary["new_model_calls"] == 1
    report = read(output, case.id + ".json")
    assert isinstance(report["summary"], dict)
    assert report["summary"]["controlled_fault_exercised"] is True
    result = report["result"]
    assert isinstance(result, dict)
    assert result["status"] == "answered" and result["repair_attempts"] == 1
    assert result["model_calls"] == 3
    assert len(llm.requests) == 1
    serialized = json.dumps(llm.requests, ensure_ascii=False)
    assert all(criterion not in serialized for criterion in case.criteria)
    assert "expected_statuses" not in serialized
    turns = report["model_turns"]
    assert isinstance(turns, list)
    assert [r["origin"] for r in turns] == ["replayed", "replayed", "mock"]
    assert turns[-1]["tools"] == []
    assert report["manifest_sha256"] == fingerprint_bytes((output / "manifest.json").read_bytes())
    review = read(output, "review.json")["cases"]
    assert isinstance(review, list)
    assert review[0]["report_sha256"] == fingerprint_bytes(
        (output / (case.id + ".json")).read_bytes()
    )
    assert all(d["score"] is None for d in review[0]["dimensions"].values())
    with pytest.raises(FileExistsError):
        await run_agent_suite(
            dataset,
            documents,
            [case],
            llm,
            output,
            mode="mock",
            model="fixture",
            max_calls=1,
            manifest_extra={},
        )
    assert len(llm.requests) == 1


async def test_malformed_prefix_does_not_execute_until_corrected(
    tmp_path: Path, data: tuple[AgentDataset, tuple[PreparedDocument, ...]]
) -> None:
    dataset, documents = data
    case = next(c for c in dataset.cases if c.id == "malformed_catalog_arguments")
    llm = Scripted(call('{"limit":3,"offset":0}'), final("Catalog [T2]"))
    await run_agent_suite(
        dataset,
        documents,
        [case],
        llm,
        tmp_path / "run",
        mode="mock",
        model="fixture",
        max_calls=2,
        manifest_extra={},
    )
    report = read(tmp_path / "run", case.id + ".json")
    assert isinstance(report["backend_events"], list) and len(report["backend_events"]) == 1
    result = report["result"]
    assert isinstance(result, dict)
    assert result["observations"][0]["error"] == "invalid_arguments"
    assert result["status"] == "answered"


async def test_faults_persist_and_identical_unavailable_calls_are_bounded(
    tmp_path: Path, data: tuple[AgentDataset, tuple[PreparedDocument, ...]]
) -> None:
    dataset, documents = data
    empty = SnapshotBackend(documents, "empty_search")
    assert await empty.search(SearchInput(query="PagedAttention")) == ()
    assert await empty.search(SearchInput(query="LoRA")) == ()
    case = next(c for c in dataset.cases if c.id == "catalog_unavailable")
    llm = Scripted(*(call("{}", str(i)) for i in range(3)), final("INSUFFICIENT_EVIDENCE"))
    await run_agent_suite(
        dataset,
        documents,
        [case],
        llm,
        tmp_path / "run",
        mode="mock",
        model="fixture",
        max_calls=4,
        manifest_extra={},
    )
    report = read(tmp_path / "run", case.id + ".json")
    assert isinstance(report["backend_events"], list) and len(report["backend_events"]) == 2
    result = report["result"]
    assert isinstance(result, dict)
    assert [o["error"] for o in result["observations"]] == [
        "tool_unavailable",
        "tool_unavailable",
        "retry_exhausted",
    ]


async def test_run_budget_stops_continuation_and_skips_remaining_cases(
    tmp_path: Path, data: tuple[AgentDataset, tuple[PreparedDocument, ...]]
) -> None:
    dataset, documents = data
    cases = dataset.cases[:2]
    llm = Scripted(call("{}"))
    summary = await run_agent_suite(
        dataset,
        documents,
        cases,
        llm,
        tmp_path / "run",
        mode="mock",
        model="fixture",
        max_calls=1,
        manifest_extra={},
    )
    assert len(llm.requests) == 1
    assert summary["stop_reason"] == "run_call_limit"
    assert summary["skipped_cases"] == [cases[1].id]
    assert read(tmp_path / "run", cases[0].id + ".json")["result"] is None


@pytest.mark.parametrize("cancel", [False, True])
async def test_timeout_and_cancellation_preserve_partial_reports(
    tmp_path: Path, data: tuple[AgentDataset, tuple[PreparedDocument, ...]], cancel: bool
) -> None:
    dataset, documents = data
    dataset = dataset.model_copy(
        update={
            "run_policy": dataset.run_policy.model_copy(update={"deadline_seconds_per_case": 0.05})
        }
    )

    entered = anyio.Event()

    class Waiting:
        async def complete(
            self,
            messages: tuple[dict[str, object], ...],
            tools: list[dict[str, object]],
            *,
            max_tokens: int,
        ) -> ToolTurn:
            entered.set()
            await anyio.sleep_forever()
            raise AssertionError

    run = run_agent_suite(
        dataset,
        documents,
        dataset.cases[:2],
        Waiting(),
        tmp_path / "run",
        mode="mock",
        model="fixture",
        max_calls=2,
        manifest_extra={},
    )
    if cancel:
        task = asyncio.create_task(run)
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        await run
    summary = read(tmp_path / "run", "summary.json")
    assert summary["stop_reason"] == ("cancelled" if cancel else "request_timeout")
    assert summary["skipped_cases"] == [dataset.cases[1].id]
    assert (tmp_path / "run" / (dataset.cases[0].id + ".json")).exists()


@pytest.mark.parametrize("status", [200, 400])
async def test_http_capture_sanitization_error_usage_and_no_retry(
    tmp_path: Path, data: tuple[AgentDataset, tuple[PreparedDocument, ...]], status: int
) -> None:
    dataset, documents = data
    secret = "TOKEN_MUST_NOT_PERSIST"

    def handler(request: httpx.Request) -> httpx.Response:
        body = {
            "model": "test",
            "usage": {"prompt_tokens": 2, "completion_tokens": 3, "cost": 0.01},
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "Hello " + secret,
                        "reasoning_content": "PRIVATE_THOUGHT",
                    },
                }
            ],
            "error": {"message": "PRIVATE_ERROR " + secret},
        }
        return httpx.Response(
            status,
            content=gzip.compress(json.dumps(body).encode()),
            headers={"content-encoding": "gzip"},
        )

    trace = TraceTransport(httpx.MockTransport(handler))
    async with httpx.AsyncClient(
        base_url="https://fixture/", transport=trace, headers={"Authorization": "Bearer " + secret}
    ) as client:
        summary = await run_agent_suite(
            dataset,
            documents,
            dataset.cases[:2],
            CompatibleToolLLM(client, "test"),
            tmp_path / "run",
            mode="live",
            model="test",
            max_calls=1,
            manifest_extra={},
            trace=trace,
            secrets=(secret,),
        )
    assert len(trace.records) == 1
    usage = summary["usage"]
    assert isinstance(usage, dict)
    assert usage["provider_cost_known"] == "0.01" and usage["cost_missing_calls"] == 0
    text = "".join(p.read_text() for p in (tmp_path / "run").glob("*.json"))
    assert secret not in text and "PRIVATE_THOUGHT" not in text and "PRIVATE_ERROR" not in text
    assert '"reasoning_characters": 15' in text
    if status == 400:
        assert summary["stop_reason"] == "generation_failed"


async def test_report_redaction_and_exclusive_write(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    write_report(path, {"text": "secret"}, ("secret",))
    assert "secret" not in path.read_text()
    with pytest.raises(FileExistsError):
        write_report(path, {})


@pytest.mark.parametrize("oversized", [False, True])
async def test_bad_provider_response_has_unknown_cost_and_is_not_retried(
    tmp_path: Path, data: tuple[AgentDataset, tuple[PreparedDocument, ...]], oversized: bool
) -> None:
    dataset, documents = data
    from agentic_rag.llm.async_client import MAX_OUTPUT_BYTES

    trace = TraceTransport(
        httpx.MockTransport(
            lambda request: httpx.Response(
                200, content=b"x" * (MAX_OUTPUT_BYTES + 1) if oversized else b"not JSON"
            )
        )
    )
    async with httpx.AsyncClient(base_url="https://fixture/", transport=trace) as client:
        summary = await run_agent_suite(
            dataset,
            documents,
            dataset.cases[:2],
            CompatibleToolLLM(client, "test"),
            tmp_path / "run",
            mode="live",
            model="test",
            max_calls=3,
            manifest_extra={},
            trace=trace,
        )
    assert len(trace.records) == 1
    assert summary["stop_reason"] == "generation_failed"
    usage = summary["usage"]
    assert isinstance(usage, dict)
    assert usage["cost_missing_calls"] == usage["usage_missing_calls"] == 1


@pytest.mark.parametrize(
    "arguments,code",
    [([], 0), (["--cases", "unknown_case"], 2), (["--mode", "live", "--output-dir", "unused"], 2)],
)
async def test_cli_validation_and_paid_guard(arguments: list[str], code: int) -> None:
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/evaluate_agent.py"), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == code
    if code == 0:
        assert json.loads(result.stdout)["paid_calls"] == 0
    else:
        assert "error:" in result.stderr
    assert not Path("unused").exists()


async def test_unexercised_fault_is_not_reported_as_tested(
    tmp_path: Path, data: tuple[AgentDataset, tuple[PreparedDocument, ...]]
) -> None:
    dataset, documents = data
    case = next(c for c in dataset.cases if c.id == "catalog_unavailable")
    await run_agent_suite(
        dataset,
        documents,
        [case],
        Scripted(final("No tool used")),
        tmp_path / "run",
        mode="mock",
        model="fixture",
        max_calls=1,
        manifest_extra={},
    )
    report = read(tmp_path / "run", case.id + ".json")
    assert isinstance(report["summary"], dict)
    assert report["summary"]["controlled_fault_exercised"] is False
    assert report["backend_events"] == []
