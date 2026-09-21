import sys
from pathlib import Path

import anyio
import pytest
from mcp import StdioServerParameters
from mcp.shared.exceptions import MCPError

from agentic_rag.agents.fake import FakeToolLLM
from agentic_rag.agents.service import Agent
from agentic_rag.mcp.client import MCPContractError, MCPLimits, connect_calculator
from agentic_rag.mcp.demo import UnavailableBackend
from agentic_rag.tools.models import CalculateInput

pytestmark = pytest.mark.anyio


async def test_real_stdio_agent_roundtrip() -> None:
    parameters = StdioServerParameters(
        command=sys.executable, args=["-m", "agentic_rag.mcp.server"]
    )
    async with connect_calculator(parameters) as calculator:
        agent = Agent(FakeToolLLM(), UnavailableBackend(), calculator=calculator)
        result = await agent.run("/calc 6 / 8 * 100")
        assert result.status == "answered"
        assert result.tool_calls == 1
        assert result.observations[0].data["value"] == "75.00"
        assert result.tool_references == ("T1",)
        error = await agent.run("/calc 1 / 0")
        assert error.status == "insufficient_evidence"
        assert error.observations[0].error == "tool_unavailable"
    with pytest.raises(MCPContractError, match="discovery required"):
        await calculator.calculate(CalculateInput(expression="1+2"))


async def test_external_server_and_cancelled_call(tmp_path: Path) -> None:
    # Independent SDK server, with no imports from the application's server module.
    script = tmp_path / "external.py"
    script.write_text(
        """
from typing import Annotated
import anyio
from mcp.server import MCPServer
from pydantic import BaseModel, Field

server = MCPServer("external-fixture", log_level="CRITICAL")

class Result(BaseModel):
    expression: str
    value: str
    precision: int = 28

@server.tool()
async def calculate(expression: Annotated[str, Field(min_length=1, max_length=256)]) -> Result:
    if expression == "0":
        await anyio.sleep_forever()
    return Result(expression=expression, value="42")

@server.tool()
def unapproved() -> str:
    raise AssertionError("Must never be invoked")

server.run()
""",
        encoding="utf-8",
    )
    parameters = StdioServerParameters(command=sys.executable, args=[str(script)])
    with anyio.fail_after(15):
        async with connect_calculator(parameters) as calculator:
            result = await Agent(FakeToolLLM(), UnavailableBackend(), calculator=calculator).run(
                "/calc 40+2"
            )
            assert result.observations[0].data["value"] == "42"
            with anyio.move_on_after(0.1) as cancelled:
                await calculator.calculate(CalculateInput(expression="0"))
            assert cancelled.cancel_called
            assert (await calculator.calculate(CalculateInput(expression="40+2"))).value == "42"


async def test_unresponsive_server_startup_has_deadline(tmp_path: Path) -> None:
    script = tmp_path / "silent.py"
    script.write_text("import time\ntime.sleep(60)\n", encoding="utf-8")
    parameters = StdioServerParameters(command=sys.executable, args=[str(script)])
    # The SDK may wrap handshake errors in its transport's ExceptionGroup.
    with anyio.fail_after(15):
        with pytest.raises((MCPError, ExceptionGroup)):
            async with connect_calculator(parameters, MCPLimits(timeout_seconds=0.1)):
                pytest.fail("Silent process must not connect")
