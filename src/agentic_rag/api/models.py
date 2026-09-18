from typing import Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from agentic_rag.ingestion.models import IngestResult
from agentic_rag.llm.base import Completion
from agentic_rag.rag.context import Citation
from agentic_rag.rag.service import Answer


class InputModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class QueryRequest(InputModel):
    query: str = Field(min_length=1, max_length=8000)
    k: int = Field(default=5, ge=1, le=100, strict=True)
    document_ids: list[UUID] = Field(default_factory=list, max_length=100)
    source_uri: str | None = Field(default=None, max_length=2048)
    media_type: Literal["text/plain", "text/markdown"] | None = None
    stream: bool = Field(default=False, strict=True)

    @field_validator("query")
    @classmethod
    def nonempty_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Query must contain non-whitespace text")
        return value


class DocumentRequest(InputModel):
    filename: str = Field(min_length=1, max_length=255)
    source_uri: str = Field(min_length=1, max_length=2048)
    content: str = Field(min_length=1, max_length=1_000_000)
    max_chars: int = Field(default=1200, ge=1, le=10000, strict=True)
    overlap: int = Field(default=200, ge=0, strict=True)

    @field_validator("filename")
    @classmethod
    def simple_filename(cls, value: str) -> str:
        if "/" in value or "\\" in value or "\x00" in value:
            raise ValueError("Filename must be a plain .md or .txt name")
        return value

    @model_validator(mode="after")
    def valid_overlap(self) -> Self:
        if self.overlap >= self.max_chars:
            raise ValueError("Overlap must be smaller than max_chars")
        min_step = max(1, self.max_chars // 2 - self.overlap)
        if 1 + len(self.content) // min_step > 10000:
            raise ValueError("Document and overlap exceed the API chunk-work limit")
        return self


class DocumentResponse(BaseModel):
    document: IngestResult
    index_status: Literal["not_required", "ready", "failed"]
    error: str | None = None


class QueryResponse(BaseModel):
    status: Literal["answered", "insufficient_evidence"]
    text: str
    citations: tuple[Citation, ...]
    completion: Completion | None
    prompt_bytes: int
    context_chunk_count: int
    llm_provider: Literal["fake", "compatible"]

    @classmethod
    def from_answer(cls, answer: Answer, provider: Literal["fake", "compatible"]) -> Self:
        return cls(
            status=answer.status,
            text=answer.text,
            citations=answer.citations,
            completion=answer.completion,
            prompt_bytes=answer.context.prompt_bytes,
            context_chunk_count=len(answer.context.sources),
            llm_provider=provider,
        )
