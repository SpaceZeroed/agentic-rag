"""Explicitly approved calculator contract, not an arbitrary remote-tool registry."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Protocol

import anyio
from mcp import Client, StdioServerParameters
from mcp.types import CallToolResult, ListToolsResult
from pydantic import ValidationError

from agentic_rag.tools.models import CalculateInput, CalculateOutput


class MCPContractError(RuntimeError):
    """Sanitized remote failure, safe for application logs."""


class Session(Protocol):
    async def list_tools(self, *, cursor: str | None = None) -> ListToolsResult: ...
    async def call_tool(
        self, name: str, arguments: dict[str, object] | None = None
    ) -> CallToolResult: ...


@dataclass(frozen=True)
class MCPLimits:
    timeout_seconds: float = 5.0
    max_pages: int = 4
    max_tools: int = 32
    max_discovery_bytes: int = 32768
    max_result_bytes: int = 4096

    def __post_init__(self) -> None:
        if not 0 < self.timeout_seconds <= 60:
            raise ValueError("MCP timeout must be in (0, 60]")
        if any(
            value <= 0
            for value in (
                self.max_pages,
                self.max_tools,
                self.max_discovery_bytes,
                self.max_result_bytes,
            )
        ):
            raise ValueError("MCP budgets must be positive")


def without_titles(value: object) -> object:
    """Ignore documentation titles only; retain all schema constraints."""
    if isinstance(value, dict):
        return {key: without_titles(item) for key, item in value.items() if key != "title"}
    if isinstance(value, list):
        return [without_titles(item) for item in value]
    return value


class MCPCalculator:
    def __init__(self, session: Session, limits: MCPLimits) -> None:
        self.session = session
        self.limits = limits
        self.ready = False

    async def discover(self) -> None:
        self.ready = False
        cursor = None
        cursors: set[str] = set()
        names: set[str] = set()
        size = 0
        found = False
        with anyio.fail_after(self.limits.timeout_seconds):
            for _ in range(self.limits.max_pages):
                page = await self.session.list_tools(cursor=cursor)
                size += len(page.model_dump_json().encode("utf-8"))
                if size > self.limits.max_discovery_bytes:
                    raise MCPContractError("MCP discovery budget exceeded")
                for tool in page.tools:
                    if tool.name in names or len(names) >= self.limits.max_tools:
                        raise MCPContractError("MCP duplicate tool or tool count exceeded")
                    names.add(tool.name)
                    if tool.name != "calculate":
                        continue
                    expected = CalculateInput.model_json_schema()
                    # SDK derives a function schema without additionalProperties.
                    expected.pop("additionalProperties")
                    actual = dict(tool.input_schema)
                    if actual.get("additionalProperties") is False:
                        actual.pop("additionalProperties")
                    if without_titles(actual) != without_titles(expected):
                        raise MCPContractError("MCP calculator input contract mismatch")
                    if without_titles(tool.output_schema) != without_titles(
                        CalculateOutput.model_json_schema()
                    ):
                        raise MCPContractError("MCP calculator output contract mismatch")
                    found = True
                cursor = page.next_cursor
                if cursor is None:
                    if not found:
                        raise MCPContractError("MCP calculator not discovered")
                    self.ready = True
                    return
                if cursor in cursors:
                    raise MCPContractError("MCP repeated discovery cursor")
                cursors.add(cursor)
        raise MCPContractError("MCP discovery page budget exceeded")

    async def calculate(self, arguments: CalculateInput) -> CalculateOutput:
        if not self.ready:
            raise MCPContractError("MCP discovery required")
        try:
            with anyio.fail_after(self.limits.timeout_seconds):
                result = await self.session.call_tool("calculate", arguments.model_dump())
            if len(result.model_dump_json().encode("utf-8")) > self.limits.max_result_bytes:
                raise MCPContractError("MCP result budget exceeded")
            if result.is_error or result.structured_content is None:
                raise MCPContractError("MCP tool failed or missing structured result")
            raw = result.structured_content
            if set(raw) != {"expression", "value", "precision"}:
                raise MCPContractError("MCP result fields mismatch")
            output = CalculateOutput.model_validate(raw, strict=True)
            number = Decimal(output.value)
            if (
                output.expression != arguments.expression.strip()
                or output.precision != 28
                or not number.is_finite()
                or (number and not -100 <= number.adjusted() <= 100)
            ):
                raise MCPContractError("MCP calculator result contract mismatch")
            return output
        except (ValidationError, InvalidOperation, ValueError) as exc:
            raise MCPContractError("MCP invalid structured result") from exc


@asynccontextmanager
async def connect_calculator(
    parameters: StdioServerParameters, limits: MCPLimits | None = None
) -> AsyncIterator[MCPCalculator]:
    """Caller owns the command/args. Never build them from model output.

    Enter and exit in the same task: SDK transports own AnyIO task groups.
    Limits cap accepted decoded payloads, not subprocess memory or wire bytes.
    """
    bounds = limits or MCPLimits()
    async with Client(parameters, read_timeout_seconds=bounds.timeout_seconds) as session:
        calculator = MCPCalculator(session, bounds)
        await calculator.discover()
        try:
            yield calculator
        finally:
            calculator.ready = False
