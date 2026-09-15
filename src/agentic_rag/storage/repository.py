"""Atomic ingestion and snapshot reads using PostgreSQL row-level locking."""

from dataclasses import asdict
from typing import Literal
from uuid import UUID

from sqlalchemy import Engine, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from agentic_rag.ingestion.models import Chunk, ChunkingConfig, IngestResult, PreparedDocument
from agentic_rag.storage.models import ChunkRow, DocumentRow, RevisionRow


class PostgresDocumentRepository:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def save(self, document: PreparedDocument) -> IngestResult:
        with Session(self.engine) as session, session.begin():
            # Concurrent first inserts wait on the same unique source identity.
            session.execute(
                insert(DocumentRow)
                .values(id=document.document_id, source_uri=document.source_uri)
                .on_conflict_do_nothing(index_elements=[DocumentRow.id])
            )
            session.execute(
                select(DocumentRow.id)
                .where(DocumentRow.id == document.document_id)
                .with_for_update()
            ).scalar_one()
            current = session.scalar(
                select(RevisionRow.id).where(
                    RevisionRow.document_id == document.document_id, RevisionRow.is_current
                )
            )
            if current == document.revision_id:
                return IngestResult(
                    document.document_id, document.revision_id, "unchanged", len(document.chunks)
                )
            existing = session.get(RevisionRow, document.revision_id)
            status: Literal["created", "updated", "reactivated"]
            # Demotion and promotion remain invisible until the transaction commits.
            session.execute(
                update(RevisionRow)
                .where(RevisionRow.document_id == document.document_id, RevisionRow.is_current)
                .values(is_current=False)
            )
            if existing is not None:
                existing.is_current = True
                status = "reactivated"
            else:
                session.add(
                    RevisionRow(
                        id=document.revision_id,
                        document_id=document.document_id,
                        is_current=True,
                        title=document.title,
                        media_type=document.media_type,
                        raw_sha256=document.raw_sha256,
                        content_sha256=document.content_sha256,
                        text=document.text,
                        parser_version=document.parser_version,
                        chunker_version=document.chunker_version,
                        max_chars=document.config.max_chars,
                        overlap=document.config.overlap,
                    )
                )
                session.flush()
                session.add_all(
                    ChunkRow(revision_id=document.revision_id, **asdict(chunk))
                    for chunk in document.chunks
                )
                status = "created" if current is None else "updated"
            # Explicitly flush so database constraint errors occur before returning.
            session.flush()
            return IngestResult(
                document.document_id,
                document.revision_id,
                status,
                len(document.chunks),
            )

    def get(self, document_id: UUID, *, revision_id: UUID | None = None) -> PreparedDocument | None:
        # Pin an immutable revision first; never reread "current" between queries.
        # Avoid repeating the full document text once per chunk in a large JOIN.
        statement = (
            select(DocumentRow, RevisionRow)
            .join(RevisionRow, RevisionRow.document_id == DocumentRow.id)
            .where(DocumentRow.id == document_id)
        )
        statement = statement.where(
            RevisionRow.is_current if revision_id is None else RevisionRow.id == revision_id
        )
        with Session(self.engine) as session:
            row = session.execute(statement).one_or_none()
            if row is None:
                return None
            source, revision = row
            chunks = session.scalars(
                select(ChunkRow)
                .where(ChunkRow.revision_id == revision.id)
                .order_by(ChunkRow.position)
            ).all()
            return PreparedDocument(
                document_id=source.id,
                revision_id=revision.id,
                source_uri=source.source_uri,
                title=revision.title,
                media_type=revision.media_type,
                raw_sha256=revision.raw_sha256,
                content_sha256=revision.content_sha256,
                text=revision.text,
                parser_version=revision.parser_version,
                chunker_version=revision.chunker_version,
                config=ChunkingConfig(revision.max_chars, revision.overlap),
                chunks=tuple(
                    Chunk(
                        chunk.id,
                        chunk.position,
                        chunk.text,
                        chunk.start_char,
                        chunk.end_char,
                        chunk.start_line,
                        chunk.end_line,
                    )
                    for chunk in chunks
                ),
            )
