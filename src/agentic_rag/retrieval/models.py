from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol
from uuid import UUID

from agentic_rag.embeddings.base import EmbeddingSpec
from agentic_rag.ingestion.models import PreparedDocument


@dataclass(frozen=True)
class SearchFilter:
    document_ids: tuple[UUID, ...] = ()
    source_uri: str | None = None
    media_type: str | None = None


@dataclass(frozen=True)
class Candidate:
    chunk_id: UUID
    revision_id: UUID
    score: float


@dataclass(frozen=True)
class SearchHit:
    chunk_id: UUID
    revision_id: UUID
    document_id: UUID
    score: float
    text: str
    source_uri: str
    title: str
    start_char: int
    end_char: int
    start_line: int
    end_line: int


class VectorIndex(Protocol):
    @property
    def collection(self) -> str: ...

    def ensure(self, spec: EmbeddingSpec) -> bool:
        """Validate/create collection; return True only if newly created."""
        ...

    def count_revision(self, revision_id: UUID) -> int: ...

    def upsert(self, document: PreparedDocument, vectors: Sequence[Sequence[float]]) -> None: ...

    def query(
        self, vector: list[float], revisions: Sequence[UUID], k: int, *, exact: bool
    ) -> list[Candidate]: ...


class IndexCatalog(Protocol):
    def current_documents(self, filters: SearchFilter) -> list[PreparedDocument]: ...

    def invalidate(self, collection: str) -> None: ...

    def is_ready(self, collection: str, revision_id: UUID, count: int) -> bool: ...

    def begin(self, collection: str, revision_id: UUID, count: int) -> UUID: ...

    def finish(
        self,
        collection: str,
        revision_id: UUID,
        token: UUID,
        state: Literal["ready", "failed"],
        error_type: str | None = None,
    ) -> bool: ...

    def searchable_revisions(self, collection: str, filters: SearchFilter) -> list[UUID]: ...

    def hydrate(self, collection: str, candidates: Sequence[Candidate]) -> list[SearchHit]: ...
