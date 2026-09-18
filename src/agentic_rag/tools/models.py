from typing import Literal, Protocol
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agentic_rag.retrieval.models import SearchHit


class ToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class SearchInput(ToolInput):
    query: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=5, ge=1, le=10)

    @field_validator("query")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Query cannot be blank")
        return value


class CalculateInput(ToolInput):
    expression: str = Field(min_length=1, max_length=256)


class CatalogInput(ToolInput):
    title_contains: str = Field(default="", max_length=200)
    limit: int = Field(default=10, ge=1, le=20)
    offset: int = Field(default=0, ge=0, le=10000)


class CalculateOutput(BaseModel):
    expression: str
    value: str
    precision: int = 28


class CatalogItem(BaseModel):
    document_id: UUID
    revision_id: UUID
    source_uri: str
    title: str
    media_type: str
    chunk_count: int


class CatalogOutput(BaseModel):
    documents: tuple[CatalogItem, ...]
    offset: int
    has_more: bool


class ToolBackend(Protocol):
    async def search(self, arguments: SearchInput) -> tuple[SearchHit, ...]: ...
    async def catalog(self, arguments: CatalogInput) -> CatalogOutput: ...


class Observation(BaseModel):
    """Public execution record; identifiers are assigned by the application."""

    call_id: str
    name: str
    status: Literal["ok", "error"]
    data: dict[str, object] = Field(default_factory=dict)
    error: str | None = None
    cached: bool = False
    reference: str | None = None


TOOL_INPUTS: dict[str, type[ToolInput]] = {
    "search_documents": SearchInput,
    "calculate": CalculateInput,
    "document_catalog": CatalogInput,
}
DESCRIPTIONS = {
    "search_documents": "Search indexed technical documents. Returns passages with [C...] IDs. "
    "Use these sources to answer document questions; no hits means no evidence.",
    "calculate": "Evaluate arithmetic using decimal numbers, parentheses and + - * /. "
    "No powers, functions, variables or code. Decimal precision is 28 digits.",
    "document_catalog": "Read a page of current document metadata and chunk counts from "
    "PostgreSQL. Optional literal, case-sensitive title substring. No SQL input. "
    "has_more indicates pagination; a page is not a total corpus count.",
}


def tool_definitions() -> list[dict[str, object]]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": DESCRIPTIONS[name],
                "parameters": schema.model_json_schema(),
            },
        }
        for name, schema in TOOL_INPUTS.items()
    ]
