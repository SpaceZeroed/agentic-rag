"""Offline agent/MCP integration demo, including operator-selected external servers."""

import argparse
import sys

import anyio
from mcp import StdioServerParameters

from agentic_rag.agents.fake import FakeToolLLM
from agentic_rag.agents.service import Agent
from agentic_rag.mcp.client import connect_calculator
from agentic_rag.retrieval.models import SearchHit
from agentic_rag.tools.models import CatalogInput, CatalogOutput, SearchInput


class UnavailableBackend:
    async def search(self, arguments: SearchInput) -> tuple[SearchHit, ...]:
        raise RuntimeError("Search not configured in calculator demo")

    async def catalog(self, arguments: CatalogInput) -> CatalogOutput:
        raise RuntimeError("Catalog not configured in calculator demo")


async def run(expression: str, command: list[str]) -> None:
    parameters = StdioServerParameters(command=command[0], args=command[1:])
    async with connect_calculator(parameters) as calculator:
        agent = Agent(FakeToolLLM(), UnavailableBackend(), calculator=calculator)
        result = await agent.run("/calc " + expression)
        print(result.model_dump_json(indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("expression", nargs="?", default="6 / 8 * 100")
    parser.add_argument(
        "--server",
        nargs=argparse.REMAINDER,
        help="Trusted operator command and arguments; must come last. No shell expansion.",
    )
    args = parser.parse_args()
    command = args.server or [sys.executable, "-m", "agentic_rag.mcp.server"]
    anyio.run(run, args.expression, command)


if __name__ == "__main__":
    main()
