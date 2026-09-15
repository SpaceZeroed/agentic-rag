"""Infrastructure-independent values used by ingestion and persistence."""

from dataclasses import dataclass
from typing import Literal, Protocol
from uuid import UUID


@dataclass(frozen=True)
class ChunkingConfig:
    max_chars: int = 1200
    overlap: int = 200

    def __post_init__(self) -> None:
        if self.max_chars < 1:
            raise ValueError("max_chars must be positive")
        if not 0 <= self.overlap < self.max_chars:
            raise ValueError("overlap must satisfy 0 <= overlap < max_chars")


@dataclass(frozen=True)
class Chunk:
    id: UUID
    position: int
    text: str
    start_char: int
    end_char: int
    start_line: int
    end_line: int


@dataclass(frozen=True)
class PreparedDocument:
    document_id: UUID
    revision_id: UUID
    source_uri: str
    title: str
    media_type: str
    raw_sha256: str
    content_sha256: str
    text: str
    parser_version: str
    chunker_version: str
    config: ChunkingConfig
    chunks: tuple[Chunk, ...]


@dataclass(frozen=True)
class IngestResult:
    document_id: UUID
    revision_id: UUID
    status: Literal["created", "updated", "unchanged", "reactivated"]
    chunk_count: int


class DocumentWriter(Protocol):
    def save(self, document: PreparedDocument) -> IngestResult:
        """Atomically persist a complete revision and make it current."""
        ...
