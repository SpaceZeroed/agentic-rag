from collections.abc import Awaitable, Callable

import anyio
import pytest
from mcp import Client
from mcp.types import CallToolResult, ListToolsResult, TextContent, Tool

from agentic_rag.mcp.client import MCPCalculator, MCPContractError, MCPLimits
from agentic_rag.mcp.server import create_server
from agentic_rag.tools.models import CalculateInput, CalculateOutput

pytestmark = pytest.mark.anyio


def definition() -> Tool:
    return Tool(
        name="calculate",
        input_schema=CalculateInput.model_json_schema(),
        output_schema=CalculateOutput.model_json_schema(),
    )


class Session:
    def __init__(self) -> None:
        self.page = ListToolsResult(tools=[definition()])
        self.result = CallToolResult(
            content=[], structured_content={"expression": "1+2", "value": "3", "precision": 28}
        )
        self.calls = 0
        self.wait: Callable[[], Awaitable[None]] | None = None

    async def list_tools(self, *, cursor: str | None = None) -> ListToolsResult:
        return self.page

    async def call_tool(
        self, name: str, arguments: dict[str, object] | None = None
    ) -> CallToolResult:
        self.calls += 1
        assert name == "calculate"
        if self.wait:
            await self.wait()
        return self.result


async def test_sdk_server_discovery_and_invocation() -> None:
    async with Client(create_server()) as session:
        calculator = MCPCalculator(session, MCPLimits())
        await calculator.discover()
        result = await calculator.calculate(CalculateInput(expression="6 / 8 * 100"))
        assert result.value == "75.00"
        with pytest.raises(MCPContractError):
            await calculator.calculate(CalculateInput(expression="1/0"))
        bad = await session.call_tool("calculate", {"expression": "1" * 257})
        assert bad.is_error
        assert (await session.call_tool("missing", {})).is_error


async def test_discovery_required() -> None:
    session = Session()
    with pytest.raises(MCPContractError, match="discovery required"):
        await MCPCalculator(session, MCPLimits()).calculate(CalculateInput(expression="1+2"))
    assert session.calls == 0


@pytest.mark.parametrize(
    "fault", ["missing", "duplicate", "input", "output", "cursor", "bytes", "count", "pages"]
)
async def test_discovery_rejects_bad_contract_and_budgets(fault: str) -> None:
    session = Session()
    limits = MCPLimits()
    if fault == "missing":
        session.page.tools = []
    elif fault == "duplicate":
        session.page.tools.append(definition())
    elif fault == "input":
        session.page.tools[0].input_schema = {"type": "object"}
    elif fault == "output":
        session.page.tools[0].output_schema = None
    elif fault in {"cursor", "pages"}:
        session.page = ListToolsResult(tools=[], next_cursor="same")
        if fault == "pages":
            limits = MCPLimits(max_pages=1)
    elif fault == "bytes":
        limits = MCPLimits(max_discovery_bytes=10)
    elif fault == "count":
        session.page.tools.append(Tool(name="unapproved", input_schema={}))
        limits = MCPLimits(max_tools=1)
    calculator = MCPCalculator(session, limits)
    with pytest.raises(MCPContractError):
        await calculator.discover()
    assert not calculator.ready


@pytest.mark.parametrize(
    "fault", ["error", "missing", "wrong_type", "expression", "nan", "precision", "extra", "bytes"]
)
async def test_invalid_results(fault: str) -> None:
    session = Session()
    if fault == "error":
        session.result.is_error = True
    elif fault == "missing":
        session.result.structured_content = None
    elif fault == "bytes":
        session.result.content = [TextContent(type="text", text="x" * 5000)]
    else:
        assert session.result.structured_content is not None
        key, value = {
            "wrong_type": ("value", 3),
            "expression": ("expression", "2+2"),
            "nan": ("value", "NaN"),
            "precision": ("precision", 10),
            "extra": ("secret", "x"),
        }[fault]
        session.result.structured_content[key] = value
    calculator = MCPCalculator(session, MCPLimits())
    await calculator.discover()
    with pytest.raises(MCPContractError):
        await calculator.calculate(CalculateInput(expression="1+2"))


async def test_timeout_and_cancellation_do_not_retry() -> None:
    session = Session()
    session.wait = anyio.sleep_forever
    calculator = MCPCalculator(session, MCPLimits(timeout_seconds=0.02))
    await calculator.discover()
    with pytest.raises(TimeoutError):
        await calculator.calculate(CalculateInput(expression="1+2"))
    with anyio.move_on_after(0.005) as scope:
        await calculator.calculate(CalculateInput(expression="1+2"))
    assert scope.cancel_called
    assert session.calls == 2


async def test_discovery_pagination_and_unapproved_tools() -> None:
    class Paginated(Session):
        async def list_tools(self, *, cursor: str | None = None) -> ListToolsResult:
            if cursor is None:
                return ListToolsResult(
                    tools=[Tool(name="execute_shell", input_schema={})], next_cursor="second"
                )
            assert cursor == "second"
            return ListToolsResult(tools=[definition()])

    session = Paginated()
    calculator = MCPCalculator(session, MCPLimits())
    await calculator.discover()
    assert (await calculator.calculate(CalculateInput(expression="1+2"))).value == "3"
    assert session.calls == 1


async def test_transport_failure_reaches_agent_as_sanitized_observation() -> None:
    from agentic_rag.agents.fake import FakeToolLLM
    from agentic_rag.agents.service import Agent
    from agentic_rag.mcp.demo import UnavailableBackend

    class Broken(Session):
        async def call_tool(
            self, name: str, arguments: dict[str, object] | None = None
        ) -> CallToolResult:
            raise RuntimeError("private remote error")

    calculator = MCPCalculator(Broken(), MCPLimits())
    await calculator.discover()
    result = await Agent(FakeToolLLM(), UnavailableBackend(), calculator=calculator).run(
        "/calc 1+2"
    )
    assert result.observations[0].error == "tool_unavailable"
    assert "private" not in result.model_dump_json()
