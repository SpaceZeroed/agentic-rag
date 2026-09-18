"""A fixed, parameterized read-only query for the agent's metadata tool."""

from sqlalchemy import Engine, func, select

from agentic_rag.storage.models import ChunkRow, DocumentRow, RevisionRow
from agentic_rag.tools.models import CatalogInput, CatalogItem, CatalogOutput


def document_catalog(engine: Engine, arguments: CatalogInput) -> CatalogOutput:
    chunk_count = (
        select(func.count(ChunkRow.id))
        .where(ChunkRow.revision_id == RevisionRow.id)
        .correlate(RevisionRow)
        .scalar_subquery()
    )
    statement = (
        select(
            DocumentRow.id.label("document_id"),
            RevisionRow.id.label("revision_id"),
            DocumentRow.source_uri,
            RevisionRow.title,
            RevisionRow.media_type,
            chunk_count.label("chunk_count"),
        )
        .join(RevisionRow, RevisionRow.document_id == DocumentRow.id)
        .where(RevisionRow.is_current.is_(True))
        .order_by(DocumentRow.id)
        .offset(arguments.offset)
        .limit(arguments.limit + 1)
    )
    if arguments.title_contains:
        statement = statement.where(
            RevisionRow.title.contains(arguments.title_contains, autoescape=True)
        )
    with engine.connect() as connection:
        rows = connection.execute(statement).mappings().all()
    return CatalogOutput(
        documents=tuple(CatalogItem.model_validate(row) for row in rows[: arguments.limit]),
        offset=arguments.offset,
        has_more=len(rows) > arguments.limit,
    )
