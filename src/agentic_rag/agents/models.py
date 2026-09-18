from typing import Literal

from pydantic import BaseModel, Field

from agentic_rag.llm.base import Completion
from agentic_rag.rag.context import Citation
from agentic_rag.tools.models import Observation


class AgentResult(BaseModel):
    status: Literal["answered", "insufficient_evidence", "limit_reached", "failed"]
    text: str = ""
    basis: Literal["model", "tools"] = "model"
    citations: tuple[Citation, ...] = ()
    tool_references: tuple[str, ...] = ()
    observations: tuple[Observation, ...] = ()
    completions: tuple[Completion, ...] = ()
    model_calls: int = 0
    tool_calls: int = 0
    error: str | None = None
    llm_provider: Literal["fake", "compatible"] = "fake"


class AgentLimits(BaseModel):
    max_model_calls: int = Field(default=6, ge=1, le=20)
    max_tool_calls: int = Field(default=8, ge=1, le=40)
    max_prompt_bytes: int = Field(default=48000, ge=1, le=1_000_000)
    max_observation_bytes: int = Field(default=12000, ge=256, le=100000)
    max_tokens: int = Field(default=512, ge=1)
