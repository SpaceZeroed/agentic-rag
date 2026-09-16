from dataclasses import dataclass

from agentic_rag.embeddings.base import EmbeddingProvider, validate_vectors
from agentic_rag.retrieval.models import IndexCatalog, SearchFilter, SearchHit, VectorIndex


@dataclass(frozen=True)
class IndexResult:
    collection: str
    indexed_revisions: int
    skipped_revisions: int
    chunk_count: int


class DenseRetriever:
    def __init__(
        self, embeddings: EmbeddingProvider, index: VectorIndex, catalog: IndexCatalog
    ) -> None:
        self.embeddings = embeddings
        self.index = index
        self.catalog = catalog

    def sync(self, filters: SearchFilter | None = None) -> IndexResult:
        filters = filters or SearchFilter()
        if self.index.ensure(self.embeddings.spec):
            self.catalog.invalidate(self.index.collection)
        indexed = skipped = chunk_count = 0
        for document in self.catalog.current_documents(filters):
            count = len(document.chunks)
            chunk_count += count
            if (
                self.catalog.is_ready(self.index.collection, document.revision_id, count)
                and self.index.count_revision(document.revision_id) == count
            ):
                skipped += 1
                continue
            token = self.catalog.begin(self.index.collection, document.revision_id, count)
            try:
                vectors = self.embeddings.embed_documents([chunk.text for chunk in document.chunks])
                validate_vectors(vectors, count, self.embeddings.spec.dimension)
                self.index.upsert(document, vectors)
                if self.index.count_revision(document.revision_id) != count:
                    raise RuntimeError("Qdrant point count differs from canonical chunk count")
                if not self.catalog.finish(
                    self.index.collection, document.revision_id, token, "ready"
                ):
                    raise RuntimeError("Index attempt superseded; run sync again")
            except Exception as exc:
                self.catalog.finish(
                    self.index.collection, document.revision_id, token, "failed", type(exc).__name__
                )
                raise
            indexed += 1
        return IndexResult(self.index.collection, indexed, skipped, chunk_count)

    def search(
        self, query: str, *, k: int = 5, filters: SearchFilter | None = None, exact: bool = True
    ) -> list[SearchHit]:
        if not query.strip() or not 1 <= k <= 100:
            raise ValueError("Query must be nonempty; k must be between 1 and 100")
        revisions = self.catalog.searchable_revisions(
            self.index.collection, filters or SearchFilter()
        )
        if not revisions:
            return []
        vector = self.embeddings.embed_query(query)
        candidates = self.index.query(vector, revisions, k, exact=exact)
        return self.catalog.hydrate(self.index.collection, candidates)
