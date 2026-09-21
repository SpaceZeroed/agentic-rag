"""Isolated immutable corpus adapter; no writes or access to the user's database."""

from dataclasses import asdict

import anyio

from agentic_rag.ingestion.models import PreparedDocument
from agentic_rag.retrieval.models import SearchHit
from agentic_rag.retrieval.sparse import BM25Index
from agentic_rag.tools.models import CatalogInput, CatalogItem, CatalogOutput, SearchInput


class SnapshotBackend:
    def __init__(self, documents: tuple[PreparedDocument, ...], setup: str) -> None:
        if len({d.document_id for d in documents}) != len(documents):
            raise ValueError("Snapshot has duplicate document identities")
        self.documents = tuple(sorted(documents, key=lambda d: d.document_id))
        self.index = BM25Index(self.documents)
        self.chunks = {c.id: (d, c) for d in self.documents for c in d.chunks}
        self.setup = setup
        self.events: list[dict[str, object]] = []

    async def search(self, arguments: SearchInput) -> tuple[SearchHit, ...]:
        await anyio.lowlevel.checkpoint()
        hits = []
        if self.setup != "empty_search":
            for candidate in self.index.query(arguments.query, arguments.k):
                doc, chunk = self.chunks[candidate.chunk_id]
                hits.append(
                    SearchHit(
                        chunk.id,
                        doc.revision_id,
                        doc.document_id,
                        candidate.score,
                        chunk.text,
                        doc.source_uri,
                        doc.title,
                        chunk.start_char,
                        chunk.end_char,
                        chunk.start_line,
                        chunk.end_line,
                    )
                )
        self.events.append(
            {
                "name": "search_documents",
                "arguments": arguments.model_dump(),
                "fault_injected": self.setup == "empty_search",
                "result": [asdict(h) for h in hits],
            }
        )
        return tuple(hits)

    async def catalog(self, arguments: CatalogInput) -> CatalogOutput:
        await anyio.lowlevel.checkpoint()
        event: dict[str, object] = {
            "name": "document_catalog",
            "arguments": arguments.model_dump(),
            "fault_injected": self.setup == "unavailable_catalog",
        }
        self.events.append(event)
        if self.setup == "unavailable_catalog":
            event["error"] = "injected_catalog_unavailable"
            raise RuntimeError("injected_catalog_unavailable")
        matched = [d for d in self.documents if arguments.title_contains in d.title]
        page = matched[arguments.offset : arguments.offset + arguments.limit]
        result = CatalogOutput(
            documents=tuple(
                CatalogItem(
                    document_id=d.document_id,
                    revision_id=d.revision_id,
                    source_uri=d.source_uri,
                    title=d.title,
                    media_type=d.media_type,
                    chunk_count=len(d.chunks),
                )
                for d in page
            ),
            offset=arguments.offset,
            has_more=arguments.offset + len(page) < len(matched),
        )
        event["result"] = result.model_dump(mode="json")
        return result
