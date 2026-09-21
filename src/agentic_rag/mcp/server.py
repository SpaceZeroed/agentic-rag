"""Expose the existing bounded calculator over stdio; stdout is protocol-only."""

from typing import Annotated

from mcp.server import MCPServer
from pydantic import Field

from agentic_rag.tools.calculator import calculate as evaluate
from agentic_rag.tools.models import DESCRIPTIONS, CalculateInput, CalculateOutput


def create_server() -> MCPServer[None]:
    server: MCPServer[None] = MCPServer("agentic-rag-calculator", log_level="WARNING")

    @server.tool(description=DESCRIPTIONS["calculate"])
    def calculate(
        expression: Annotated[str, Field(min_length=1, max_length=256)],
    ) -> CalculateOutput:
        return evaluate(CalculateInput(expression=expression))

    return server


def main() -> None:
    create_server().run(transport="stdio")


if __name__ == "__main__":
    main()
