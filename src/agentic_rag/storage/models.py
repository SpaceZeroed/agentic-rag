"""Relational schema: stable sources, retained revisions, and exact text slices."""

from datetime import datetime
from uuid import UUID

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class DocumentRow(Base):
    __tablename__ = "documents"

    id: Mapped[UUID] = mapped_column(primary_key=True)
    source_uri: Mapped[str] = mapped_column(Text, unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class RevisionRow(Base):
    __tablename__ = "document_revisions"
    __table_args__ = (
        CheckConstraint("max_chars > 0", name="ck_revision_max_chars"),
        CheckConstraint("overlap >= 0 AND overlap < max_chars", name="ck_revision_overlap"),
        Index("ix_revision_document", "document_id"),
        Index(
            "uq_revision_current", "document_id", unique=True, postgresql_where=text("is_current")
        ),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True)
    document_id: Mapped[UUID] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    is_current: Mapped[bool] = mapped_column(Boolean, server_default=text("false"))
    title: Mapped[str] = mapped_column(Text)
    media_type: Mapped[str] = mapped_column(String(64))
    raw_sha256: Mapped[str] = mapped_column(String(64))
    content_sha256: Mapped[str] = mapped_column(String(64))
    text: Mapped[str] = mapped_column(Text)
    parser_version: Mapped[str] = mapped_column(String(64))
    chunker_version: Mapped[str] = mapped_column(String(64))
    max_chars: Mapped[int]
    overlap: Mapped[int]
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ChunkRow(Base):
    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("revision_id", "position", name="uq_chunk_position"),
        CheckConstraint("position >= 0", name="ck_chunk_position"),
        CheckConstraint("start_char >= 0 AND end_char > start_char", name="ck_chunk_offsets"),
        CheckConstraint("char_length(text) = end_char - start_char", name="ck_chunk_length"),
        CheckConstraint("start_line >= 1 AND end_line >= start_line", name="ck_chunk_lines"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True)
    revision_id: Mapped[UUID] = mapped_column(
        ForeignKey("document_revisions.id", ondelete="CASCADE")
    )
    position: Mapped[int]
    text: Mapped[str] = mapped_column(Text)
    start_char: Mapped[int]
    end_char: Mapped[int]
    start_line: Mapped[int]
    end_line: Mapped[int]


class VectorSyncRow(Base):
    __tablename__ = "vector_sync_states"
    __table_args__ = (
        CheckConstraint("state IN ('pending', 'ready', 'failed')", name="ck_vector_sync_state"),
        CheckConstraint("chunk_count > 0", name="ck_vector_sync_count"),
    )

    collection: Mapped[str] = mapped_column(String(128), primary_key=True)
    revision_id: Mapped[UUID] = mapped_column(
        ForeignKey("document_revisions.id", ondelete="CASCADE"), primary_key=True
    )
    state: Mapped[str] = mapped_column(String(16))
    attempt: Mapped[UUID]
    chunk_count: Mapped[int]
    error_type: Mapped[str | None] = mapped_column(String(128))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
