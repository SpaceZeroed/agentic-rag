import json
from uuid import UUID

import anyio
import pytest

from agentic_rag.agents.fake import FakeToolLLM
from agentic_rag.agents.models import AgentLimits
from agentic_rag.agents.service import Agent
from agentic_rag.llm.base import Completion, LLMError
from agentic_rag.llm.tool_client import FunctionCall, ToolCall, ToolTurn
from agentic_rag.retrieval.models import SearchHit
from agentic_rag.tools.models import CatalogInput, CatalogOutput, SearchInput

pytestmark = pytest.mark.anyio


def call(name: str, arguments: object, id: str = "a") -> ToolCall:
    return ToolCall(id=id, function=FunctionCall(name=name, arguments=json.dumps(arguments)))


def tools(*calls: ToolCall) -> ToolTurn:
    return ToolTurn(Completion("", "scripted", "tool_calls", 10, 5), calls)


def final(text: str = "Answer [C1]") -> ToolTurn:
    return ToolTurn(Completion(text, "scripted", "stop", 10, 5))


class Scripted:
    def __init__(self, *turns: ToolTurn | Exception) -> None:
        self.turns = iter(turns)
        self.histories: list[tuple[dict[str, object], ...]] = []

    async def complete(
        self,
        messages: tuple[dict[str, object], ...],
        definitions: list[dict[str, object]],
        *,
        max_tokens: int,
    ) -> ToolTurn:
        self.histories.append(messages)
        turn = next(self.turns)
        if isinstance(turn, Exception):
            raise turn
        return turn


class Backend:
    def __init__(self) -> None:
        self.searches = 0
        self.catalogs = 0
        self.error = False
        self.hits: tuple[SearchHit, ...] = (
            SearchHit(
                UUID(int=1),
                UUID(int=2),
                UUID(int=3),
                1,
                "Blocks",
                "https://example.org",
                "Paper",
                0,
                6,
                1,
                1,
            ),
        )

    async def search(self, arguments: SearchInput) -> tuple[SearchHit, ...]:
        self.searches += 1
        if self.error:
            raise RuntimeError("private database URL")
        return self.hits

    async def catalog(self, arguments: CatalogInput) -> CatalogOutput:
        self.catalogs += 1
        return CatalogOutput(documents=(), offset=arguments.offset, has_more=False)


async def test_multi_step_search_calculate_and_exact_sources() -> None:
    backend = Backend()
    llm = Scripted(
        tools(call("search_documents", {"query": "blocks"})),
        tools(call("calculate", {"expression": "6/8*100"}, "b")),
        final("Blocks [C1]; 75% [T2]"),
    )
    result = await Agent(llm, backend).run("Explain and compute")
    assert result.status == "answered"
    assert result.basis == "tools"
    assert result.model_calls == 3 and result.tool_calls == 2
    assert result.citations[0].source == backend.hits[0]
    assert result.tool_references == ("T2",)
    assert result.observations[1].data["value"] == "75.00"
    assert [m["role"] for m in llm.histories[-1]] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
    ]
    assert llm.histories[-1][-1]["tool_call_id"] == "b"
    assert all(c.text == "" for c in result.completions)


async def test_invalid_arguments_are_observations_then_corrected() -> None:
    llm = Scripted(
        tools(call("calculate", {"expression": 123})),
        tools(call("calculate", {"expression": "2+2"}, "b")),
        final("4 [T2]"),
    )
    result = await Agent(llm, Backend()).run("compute")
    assert result.status == "answered"
    assert result.observations[0].error == "invalid_arguments"
    assert '"error":"invalid_arguments"' in str(llm.histories[1][-1]["content"])


async def test_unknown_tools_never_execute_backend() -> None:
    backend = Backend()
    result = await Agent(
        Scripted(tools(call("run_shell", {"command": "x"})), final("INSUFFICIENT_EVIDENCE")),
        backend,
    ).run("x")
    assert result.status == "insufficient_evidence"
    assert result.observations[0].error == "unknown_tool"
    assert backend.searches == backend.catalogs == 0


async def test_repeated_success_reuses_snapshot_and_ids() -> None:
    backend = Backend()
    result = await Agent(
        Scripted(
            tools(call("search_documents", {"query": "blocks"})),
            tools(call("search_documents", {"k": 5, "query": "blocks"}, "b")),
            final(),
        ),
        backend,
    ).run("x")
    assert result.status == "answered"
    assert backend.searches == 1
    assert result.observations[1].cached
    assert result.observations[1].call_id == "b"
    assert len(result.citations) == 1


async def test_tool_failure_is_safe_and_retry_bounded() -> None:
    backend = Backend()
    backend.error = True
    result = await Agent(
        Scripted(
            *(tools(call("search_documents", {"query": "x"}, str(i))) for i in range(3)),
            final("INSUFFICIENT_EVIDENCE"),
        ),
        backend,
    ).run("x")
    assert backend.searches == 2
    assert [o.error for o in result.observations] == [
        "tool_unavailable",
        "tool_unavailable",
        "retry_exhausted",
    ]
    assert "private" not in result.model_dump_json()


@pytest.mark.parametrize(
    "limits,expected,turns",
    [
        (AgentLimits(max_model_calls=1), "model_call_limit", [tools(call("document_catalog", {}))]),
        (
            AgentLimits(max_tool_calls=1),
            "tool_call_limit",
            [tools(call("document_catalog", {}), call("calculate", {"expression": "1"}, "b"))],
        ),
        (AgentLimits(max_prompt_bytes=1), "prompt_budget", []),
    ],
)
async def test_limits_stop_without_executing_unusable_tools(
    limits: AgentLimits, expected: str, turns: list[ToolTurn]
) -> None:
    backend = Backend()
    result = await Agent(Scripted(*turns), backend, limits).run("x")
    assert result.status == "limit_reached"
    assert result.error == expected
    assert backend.catalogs == backend.searches == 0
    assert result.text == ""


@pytest.mark.parametrize(
    "answer", ["Uncited", "Wrong [C2]", "Grouped [C1, C2]", "Not a passage [T1]"]
)
async def test_search_answer_citations_fail_closed(answer: str) -> None:
    result = await Agent(
        Scripted(tools(call("search_documents", {"query": "x"})), final(answer)), Backend()
    ).run("x")
    assert result.status == "failed" and result.error == "invalid_citations"
    assert result.text == ""


async def test_direct_answer_is_explicitly_unverified() -> None:
    backend = Backend()
    result = await Agent(Scripted(final("Hello")), backend).run("Hello")
    assert result.status == "answered" and result.basis == "model"
    assert not result.citations and not result.observations
    assert backend.searches == 0
    bad = await Agent(Scripted(final()), backend).run("x")
    assert bad.error == "invalid_citations"


async def test_no_hits_and_observation_overflow_do_not_create_sources() -> None:
    backend = Backend()
    result = await Agent(
        Scripted(tools(call("search_documents", {"query": "x"})), final()),
        backend,
        AgentLimits(max_observation_bytes=256),
    ).run("x")
    assert result.error == "invalid_citations"
    backend.hits = ()
    result = await Agent(
        Scripted(tools(call("search_documents", {"query": "x"})), final("INSUFFICIENT_EVIDENCE")),
        backend,
    ).run("x")
    assert result.status == "insufficient_evidence" and not result.citations


async def test_reused_call_id_and_llm_errors_do_not_repeat_execution() -> None:
    backend = Backend()
    result = await Agent(
        Scripted(tools(call("document_catalog", {})), tools(call("document_catalog", {}))), backend
    ).run("x")
    assert result.error == "reused_call_id" and backend.catalogs == 1
    llm = Scripted(LLMError("private failure"))
    result = await Agent(llm, backend).run("x")
    assert result.error == "generation_failed"
    assert len(llm.histories) == 1 and result.model_calls == 1
    assert "private" not in result.model_dump_json()


async def test_request_state_is_isolated_under_concurrency() -> None:
    agent = Agent(FakeToolLLM(), Backend())
    results: dict[str, str] = {}

    async def run(query: str) -> None:
        results[query] = (await agent.run(query)).text

    async with anyio.create_task_group() as group:
        group.start_soon(run, "/calc 2+2")
        group.start_soon(run, "/calc 3+3")
    assert '"value": "4"' in results["/calc 2+2"]
    assert '"value": "6"' in results["/calc 3+3"]


async def test_cancellation_during_model_wait_never_calls_tools() -> None:
    entered = anyio.Event()
    closed = anyio.Event()

    class Waiting:
        async def complete(
            self,
            messages: tuple[dict[str, object], ...],
            definitions: list[dict[str, object]],
            *,
            max_tokens: int,
        ) -> ToolTurn:
            try:
                entered.set()
                await anyio.sleep_forever()
            finally:
                closed.set()
            raise AssertionError("unreachable")

    backend = Backend()
    async with anyio.create_task_group() as group:
        group.start_soon(Agent(Waiting(), backend).run, "x")
        await entered.wait()
        group.cancel_scope.cancel()
    assert closed.is_set() and backend.searches == 0
