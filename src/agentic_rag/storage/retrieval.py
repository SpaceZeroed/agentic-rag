from collections.abc import Sequence
from typing import Literal
from uuid import UUID, uuid4

from sqlalchemy import Engine, Select, func, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from agentic_rag.ingestion.models import PreparedDocument
from agentic_rag.retrieval.models import Candidate, SearchFilter, SearchHit
from agentic_rag.storage.models import ChunkRow, DocumentRow, RevisionRow, VectorSyncRow
from agentic_rag.storage.repository import PostgresDocumentRepository


class PostgresIndexCatalog:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    @staticmethod
    def _current(filters: SearchFilter) -> Select[tuple[UUID, UUID]]:
        statement = (
            select(DocumentRow.id, RevisionRow.id)
            .join(RevisionRow, RevisionRow.document_id == DocumentRow.id)
            .where(RevisionRow.is_current)
        )
        if filters.document_ids:
            statement = statement.where(DocumentRow.id.in_(filters.document_ids))
        if filters.source_uri is not None:
            statement = statement.where(DocumentRow.source_uri == filters.source_uri)
        if filters.media_type is not None:
            statement = statement.where(RevisionRow.media_type == filters.media_type)
        return statement.order_by(DocumentRow.id)

    def current_documents(self, filters: SearchFilter) -> list[PreparedDocument]:
        with Session(self.engine) as session:
            identities = session.execute(self._current(filters)).all()
        repository = PostgresDocumentRepository(self.engine)
        documents = []
        for document_id, revision_id in identities:
            document = repository.get(document_id, revision_id=revision_id)
            if document is not None:
                documents.append(document)
        return documents

    def invalidate(self, collection: str) -> None:
        with Session(self.engine) as session, session.begin():
            session.execute(
                update(VectorSyncRow)
                .where(VectorSyncRow.collection == collection)
                .values(state="pending", attempt=uuid4(), error_type=None, updated_at=func.now())
            )

    def is_ready(self, collection: str, revision_id: UUID, count: int) -> bool:
        with Session(self.engine) as session:
            row = session.get(VectorSyncRow, (collection, revision_id))
            return row is not None and row.state == "ready" and row.chunk_count == count

    def begin(self, collection: str, revision_id: UUID, count: int) -> UUID:
        token = uuid4()
        with Session(self.engine) as session, session.begin():
            statement = insert(VectorSyncRow).values(
                collection=collection,
                revision_id=revision_id,
                state="pending",
                attempt=token,
                chunk_count=count,
            )
            session.execute(
                statement.on_conflict_do_update(
                    index_elements=[VectorSyncRow.collection, VectorSyncRow.revision_id],
                    set_={
                        "state": "pending",
                        "attempt": token,
                        "chunk_count": count,
                        "error_type": None,
                        "updated_at": func.now(),
                    },
                )
            )
        return token

    def finish(
        self,
        collection: str,
        revision_id: UUID,
        token: UUID,
        state: Literal["ready", "failed"],
        error_type: str | None = None,
    ) -> bool:
        with Session(self.engine) as session, session.begin():
            changed = session.scalar(
                update(VectorSyncRow)
                .where(
                    VectorSyncRow.collection == collection,
                    VectorSyncRow.revision_id == revision_id,
                    VectorSyncRow.attempt == token,
                )
                .values(state=state, error_type=error_type, updated_at=func.now())
                .returning(VectorSyncRow.revision_id)
            )
            return changed is not None

    def searchable_revisions(self, collection: str, filters: SearchFilter) -> list[UUID]:
        statement = (
            self._current(filters)
            .join(VectorSyncRow, VectorSyncRow.revision_id == RevisionRow.id)
            .where(VectorSyncRow.collection == collection, VectorSyncRow.state == "ready")
        )
        with Session(self.engine) as session:
            return [revision_id for _, revision_id in session.execute(statement)]

    def hydrate(self, collection: str | None, candidates: Sequence[Candidate]) -> list[SearchHit]:
        if not candidates:
            return []
        # None is lexical-only: current canonical text needs no vector-ready status.
        statement = (
            select(ChunkRow, RevisionRow, DocumentRow)
            .join(RevisionRow, ChunkRow.revision_id == RevisionRow.id)
            .join(DocumentRow, DocumentRow.id == RevisionRow.document_id)
            .where(
                ChunkRow.id.in_([candidate.chunk_id for candidate in candidates]),
                RevisionRow.is_current,
            )
        )
        if collection is not None:
            statement = statement.join(
                VectorSyncRow, VectorSyncRow.revision_id == RevisionRow.id
            ).where(VectorSyncRow.collection == collection, VectorSyncRow.state == "ready")
        with Session(self.engine) as session:
            rows = {
                chunk.id: (chunk, revision, document)
                for chunk, revision, document in session.execute(statement)
            }
            hits = []
            for candidate in candidates:
                row = rows.get(candidate.chunk_id)
                if row is None or row[1].id != candidate.revision_id:
                    continue
                chunk, revision, document = row
                hits.append(
                    SearchHit(
                        chunk.id,
                        revision.id,
                        document.id,
                        candidate.score,
                        chunk.text,
                        document.source_uri,
                        revision.title,
                        chunk.start_char,
                        chunk.end_char,
                        chunk.start_line,
                        chunk.end_line,
                    )
                )
            return hits
