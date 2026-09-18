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
        self.definitions: list[list[dict[str, object]]] = []

    async def complete(
        self,
        messages: tuple[dict[str, object], ...],
        definitions: list[dict[str, object]],
        *,
        max_tokens: int,
    ) -> ToolTurn:
        self.histories.append(messages)
        self.definitions.append(definitions)
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


@pytest.mark.parametrize("raw", ['{"limit": 10, "offset": 10', "[]", "null", '{"x": NaN}'])
async def test_malformed_arguments_recover_without_poisoning_wire_history(raw: str) -> None:
    backend = Backend()
    llm = Scripted(
        tools(call("document_catalog", {"limit": 10}, "first")),
        tools(ToolCall(id="bad", function=FunctionCall(name="document_catalog", arguments=raw))),
        tools(call("document_catalog", {"limit": 10, "offset": 10}, "corrected")),
        final("Catalog checked [T3]"),
    )
    result = await Agent(llm, backend).run("catalog")
    assert result.status == "answered"
    assert result.model_calls == 4 and result.tool_calls == 3
    assert backend.catalogs == 2
    assert result.observations[1].error == "invalid_arguments"
    assert result.repair_attempts == 0
    record = json.loads(str(llm.histories[2][-2]["content"]))
    assert record["rejected_proposal"]["tool_calls"][0]["function"]["arguments"] == raw
    # Every native call retains its paired result; malformed proposals are text.
    for history in llm.histories:
        pending: set[str] = set()
        for message in history:
            for proposal in json.loads(json.dumps(message.get("tool_calls", []))):
                assert isinstance(json.loads(proposal["function"]["arguments"]), dict)
                pending.add(proposal["id"])
            if message["role"] == "tool":
                pending.remove(str(message["tool_call_id"]))
        assert not pending


async def test_malformed_batch_executes_no_siblings_and_can_be_resubmitted() -> None:
    backend = Backend()
    llm = Scripted(
        tools(
            call("document_catalog", {}, "valid"),
            ToolCall(id="bad", function=FunctionCall(name="calculate", arguments="{")),
        ),
        tools(call("document_catalog", {}, "new")),
        final("Catalog checked [T3]"),
    )
    result = await Agent(llm, backend).run("catalog")
    assert result.status == "answered"
    assert backend.catalogs == 1
    assert [o.error for o in result.observations] == ["batch_rejected", "invalid_arguments", None]
    assert not result.observations[-1].cached
    assert "No tool in that batch was executed" in str(llm.histories[1][-1]["content"])


@pytest.mark.parametrize("tool_limit", [1, 8])
async def test_malformed_retries_obey_shared_limits(tool_limit: int) -> None:
    backend = Backend()
    llm = Scripted(
        *(
            tools(
                ToolCall(id=str(i), function=FunctionCall(name="document_catalog", arguments="{"))
            )
            for i in range(3)
        )
    )
    result = await Agent(
        llm, backend, AgentLimits(max_model_calls=3, max_tool_calls=tool_limit)
    ).run("catalog")
    assert result.status == "limit_reached"
    assert result.error == ("tool_call_limit" if tool_limit == 1 else "model_call_limit")
    assert result.tool_calls == min(tool_limit, 2)
    assert backend.catalogs == 0


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
        Scripted(tools(call("search_documents", {"query": "x"})), final(answer), final(answer)),
        Backend(),
    ).run("x")
    assert result.status == "failed" and result.error == "invalid_citations"
    assert result.text == ""
    assert result.repair_attempts == 1 and result.model_calls == 3
    assert [failure.model_call for failure in result.citation_failures] == [2, 3]


async def test_direct_answer_is_explicitly_unverified() -> None:
    backend = Backend()
    result = await Agent(Scripted(final("Hello")), backend).run("Hello")
    assert result.status == "answered" and result.basis == "model"
    assert not result.citations and not result.observations
    assert backend.searches == 0
    bad = await Agent(Scripted(final(), final()), backend).run("x")
    assert bad.error == "invalid_citations"


async def test_no_hits_and_observation_overflow_do_not_create_sources() -> None:
    backend = Backend()
    result = await Agent(
        Scripted(tools(call("search_documents", {"query": "x"})), final(), final()),
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


@pytest.mark.parametrize(
    "name,arguments,repaired,reference",
    [
        ("document_catalog", {}, "No documents [T1]", "T1"),
        ("search_documents", {"query": "blocks"}, "Blocks [C1]", "C1"),
    ],
)
async def test_citation_repair_uses_existing_evidence_once(
    name: str,
    arguments: dict[str, object],
    repaired: str,
    reference: str,
) -> None:
    backend = Backend()
    llm = Scripted(tools(call(name, arguments)), final("Uncited draft"), final(repaired))
    result = await Agent(llm, backend).run("x")
    assert result.status == "answered" and result.text == repaired
    assert result.model_calls == 3 and result.tool_calls == 1 and result.repair_attempts == 1
    assert len(result.citation_failures) == 1
    assert result.citation_failures[0].model_call == 2
    assert result.citation_failures[0].error == "invalid_citations"
    assert backend.searches + backend.catalogs == 1
    assert llm.definitions[-1] == []
    assert llm.histories[-1][-2] == {"role": "assistant", "content": "Uncited draft"}
    assert llm.histories[-1][-1]["role"] == "system"
    assert reference in str(llm.histories[-1][-1]["content"])
    assert sum(c.prompt_tokens or 0 for c in result.completions) == 30
    assert all(c.text == "" for c in result.completions)


async def test_repair_may_abstain_without_evidence() -> None:
    backend = Backend()
    backend.hits = ()
    llm = Scripted(
        tools(call("search_documents", {"query": "x"})), final(), final("INSUFFICIENT_EVIDENCE")
    )
    result = await Agent(llm, backend).run("x")
    assert result.status == "insufficient_evidence"
    assert result.repair_attempts == 1 and len(result.citation_failures) == 1
    assert not result.citations and not result.tool_references
    assert "Available IDs: []" in str(llm.histories[-1][-1]["content"])


@pytest.mark.parametrize(
    "draft,expected,absent",
    [
        ("Catalog list", "no valid citation markers", "unavailable"),
        ("Catalog [T99]", 'These cited IDs are unavailable: ["T99"]', "no valid citation"),
        ("Catalog [T1, T2]", "malformed or grouped", "These cited IDs are unavailable"),
    ],
)
async def test_repair_explains_the_actual_validation_error(
    draft: str, expected: str, absent: str
) -> None:
    llm = Scripted(tools(call("document_catalog", {})), final(draft), final("Empty page [T1]"))
    result = await Agent(llm, Backend()).run("catalog")
    assert result.status == "answered" and result.repair_attempts == 1
    prompt = str(llm.histories[-1][-1]["content"])
    diagnostics = " ".join(
        json.loads(prompt.split("Specific validation errors: ", 1)[1].split(". Make ONE", 1)[0])
    )
    assert expected in diagnostics and absent not in diagnostics
    assert '"T1": "document_catalog result: one catalog page, not a total corpus count"' in prompt
    assert "remove unsupported claims" in prompt
    assert "cite each supported item" in prompt


async def test_repair_describes_only_successful_evidence_without_promoting_source_text() -> None:
    backend = Backend()
    llm = Scripted(
        tools(call("document_catalog", {"limit": "invalid"})),
        tools(call("search_documents", {"query": "blocks"}, "b")),
        tools(call("calculate", {"expression": "2+2"}, "c")),
        final("Uncited"),
        final("Blocks [C1]; 4 [T3]"),
    )
    result = await Agent(llm, backend).run("x")
    assert result.status == "answered"
    prompt = str(llm.histories[-1][-1]["content"])
    evidence = json.loads(prompt.split("Evidence descriptions: ", 1)[1])
    assert evidence == {
        "C1": "retrieved document passage",
        "T3": "calculate result: the supplied expression and computed value",
    }
    assert backend.hits[0].text not in prompt
    assert backend.hits[0].source_uri not in prompt


@pytest.mark.parametrize("limit", [1, 2])
async def test_repair_does_not_exceed_model_budget(limit: int) -> None:
    turns = [final()] if limit == 1 else [tools(call("document_catalog", {})), final("Uncited")]
    llm = Scripted(*turns)
    result = await Agent(llm, Backend(), AgentLimits(max_model_calls=limit)).run("x")
    assert result.status == "limit_reached" and result.error == "model_call_limit"
    assert result.model_calls == len(llm.histories) == limit
    assert result.repair_attempts == 0
    assert result.citation_failures[0].model_call == limit
    assert result.text == ""


async def test_repair_checks_full_prompt_budget_before_call() -> None:
    # The initial small query fits; the long rejected final answer does not.
    llm = Scripted(final("x" * 10000 + " [C1]"))
    result = await Agent(llm, Backend(), AgentLimits(max_prompt_bytes=8000)).run("x")
    assert result.status == "limit_reached" and result.error == "prompt_budget"
    assert result.model_calls == 1 and len(llm.histories) == 1
    assert result.repair_attempts == 0 and len(result.citation_failures) == 1


async def test_repair_cannot_route_back_to_tools() -> None:
    backend = Backend()
    llm = Scripted(
        tools(call("document_catalog", {})),
        final("Uncited"),
        tools(call("search_documents", {"query": "x"}, "b")),
    )
    result = await Agent(llm, backend).run("x")
    assert result.status == "failed" and result.error == "repair_requested_tools"
    assert result.repair_attempts == 1 and result.model_calls == 3
    assert backend.catalogs == 1 and backend.searches == 0 and result.tool_calls == 1
    assert len(result.citation_failures) == 1 and len(result.completions) == 3


@pytest.mark.parametrize(
    "error,code",
    [(LLMError("private"), "generation_failed"), (TimeoutError("private"), "llm_timeout")],
)
async def test_repair_transport_failure_is_not_retried(error: Exception, code: str) -> None:
    llm = Scripted(final(), error)
    result = await Agent(llm, Backend()).run("x")
    assert result.error == code and result.status == "failed"
    assert result.model_calls == 2 and result.repair_attempts == 1
    assert len(llm.histories) == 2 and len(result.citation_failures) == 1
    assert "private" not in result.model_dump_json()
