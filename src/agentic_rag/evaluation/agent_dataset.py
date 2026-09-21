"""Validated agent development cases; labels are never generation input."""

import hashlib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agentic_rag.agents.models import AgentLimits
from agentic_rag.evaluation.rag_dataset import load_rag_dataset
from agentic_rag.evaluation.runner import Evidence
from agentic_rag.ingestion.models import PreparedDocument
from agentic_rag.llm.tool_client import ToolCall, ToolTurn, parse_tool_turn


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PrefixTurn(StrictModel):
    finish_reason: Literal["stop", "tool_calls"]
    content: str | None
    tool_calls: list[ToolCall]

    def turn(self) -> ToolTurn:
        return parse_tool_turn(
            {
                "model": "replayed-fixture",
                "choices": [
                    {
                        "finish_reason": self.finish_reason,
                        "message": {
                            "role": "assistant",
                            "content": self.content,
                            "tool_calls": [c.model_dump() for c in self.tool_calls],
                        },
                    }
                ],
            }
        )


class Fault(StrictModel):
    target: Literal["ToolBackend.search", "ToolBackend.catalog", "ToolLLM.complete"]
    behavior: str | None = None
    prefix: list[PrefixTurn] = Field(default_factory=list)
    continuation: str | None = None


class AgentCase(StrictModel):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,79}$")
    category: str
    mode: Literal["natural", "controlled"]
    query: str = Field(min_length=1, max_length=8000)
    setup: Literal["standard", "empty_search", "unavailable_catalog", "scripted_prefix"]
    fault: Fault | None
    expected_statuses: list[Literal["answered", "insufficient_evidence", "failed", "limit_reached"]]
    required_successful_tools: list[Literal["search_documents", "calculate", "document_catalog"]]
    criteria: list[str] = Field(min_length=1)
    evidence: list[Evidence]

    @model_validator(mode="after")
    def coherent_fault(self) -> "AgentCase":
        if not self.query.strip():
            raise ValueError("Empty query")
        targets = {
            "empty_search": "ToolBackend.search",
            "unavailable_catalog": "ToolBackend.catalog",
            "scripted_prefix": "ToolLLM.complete",
        }
        if self.setup == "standard":
            if self.mode != "natural" or self.fault is not None:
                raise ValueError("Natural case cannot inject a fault")
        elif (
            self.mode != "controlled"
            or self.fault is None
            or self.fault.target != targets[self.setup]
        ):
            raise ValueError("Fault target does not match setup")
        if self.fault:
            if bool(self.fault.prefix) != (self.setup == "scripted_prefix"):
                raise ValueError("Only scripted cases require a prefix")
            for turn in self.fault.prefix:
                turn.turn()
        return self


class CorpusSpec(StrictModel):
    manifest: str
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    documents: str
    isolation: str


class RunPolicy(StrictModel):
    model_calls_per_case: int = Field(ge=1, le=20)
    tool_calls_per_case: int = Field(ge=1, le=40)
    max_prompt_bytes: int = Field(ge=1, le=1_000_000)
    max_observation_bytes: int = Field(ge=256, le=100000)
    max_tokens_per_call: int = Field(ge=1)
    deadline_seconds_per_case: float = Field(gt=0, allow_inf_nan=False)
    temperature: Literal[0]
    reasoning_enabled: Literal[False]
    automatic_http_retries: Literal[0]
    retrieval: Literal["bm25"]
    rerank: Literal[False]
    live_run_authorization: str
    freeze: str

    def limits(self) -> AgentLimits:
        return AgentLimits(
            max_model_calls=self.model_calls_per_case,
            max_tool_calls=self.tool_calls_per_case,
            max_prompt_bytes=self.max_prompt_bytes,
            max_observation_bytes=self.max_observation_bytes,
            max_tokens=self.max_tokens_per_call,
        )


class AgentDataset(StrictModel):
    schema_version: Literal[1]
    name: str
    label_policy: str
    corpus: CorpusSpec
    run_policy: RunPolicy
    review_dimensions: list[str]
    cases: list[AgentCase] = Field(min_length=1)


def load_agent_dataset(path: Path) -> tuple[AgentDataset, tuple[PreparedDocument, ...]]:
    data = AgentDataset.model_validate_json(path.read_text(encoding="utf-8"))
    manifest = path.parent / data.corpus.manifest
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != data.corpus.sha256:
        raise ValueError("Corpus manifest checksum changed")
    _, documents = load_rag_dataset(manifest)
    if len({c.id for c in data.cases}) != len(data.cases):
        raise ValueError("Duplicate case IDs")
    for case in data.cases:
        if case.fault and len(case.fault.prefix) >= data.run_policy.model_calls_per_case:
            raise ValueError("No model budget for continuation")
        for evidence in case.evidence:
            if evidence.document not in documents or not any(
                evidence.text in chunk.text for chunk in documents[evidence.document].chunks
            ):
                raise ValueError("Case evidence absent or split across chunks")
    return data, tuple(documents.values())
