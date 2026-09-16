"""Explicit small-corpus BM25 with an inverted index; no model or vector service."""

import math
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID

from agentic_rag.ingestion.models import PreparedDocument
from agentic_rag.retrieval.models import Candidate, IndexCatalog, SearchFilter, SearchHit

TOKENIZER_VERSION = "nfkc-casefold-unicode-alnum-v1"


def tokenize(text: str) -> list[str]:
    return re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", text).casefold())


@dataclass(frozen=True)
class BM25Config:
    k1: float = 1.2
    b: float = 0.75

    def __post_init__(self) -> None:
        if not math.isfinite(self.k1) or self.k1 <= 0:
            raise ValueError("k1 must be finite and positive")
        if not math.isfinite(self.b) or not 0 <= self.b <= 1:
            raise ValueError("b must be between zero and one")


class BM25Index:
    def __init__(
        self, documents: Sequence[PreparedDocument], config: BM25Config | None = None
    ) -> None:
        self.config = config or BM25Config()
        self.identities: dict[UUID, UUID] = {}
        self.lengths: dict[UUID, int] = {}
        self.postings: dict[str, dict[UUID, int]] = defaultdict(dict)
        for document in documents:
            for chunk in document.chunks:
                if chunk.id in self.identities:
                    raise ValueError("Duplicate chunk in BM25 corpus")
                self.identities[chunk.id] = document.revision_id
                terms = Counter(tokenize(chunk.text))
                self.lengths[chunk.id] = sum(terms.values())
                for term, frequency in terms.items():
                    self.postings[term][chunk.id] = frequency
        self.size = len(self.lengths)
        self.average_length = sum(self.lengths.values()) / self.size if self.size else 0.0

    def query(self, text: str, k: int) -> list[Candidate]:
        if not text.strip() or not 1 <= k <= 100:
            raise ValueError("Query must be nonempty; k must be between 1 and 100")
        scores: dict[UUID, float] = defaultdict(float)
        if not self.average_length:
            return []
        # Repeated query terms count once; no stemming, stopword removal or translation.
        for term in sorted(set(tokenize(text))):
            posting = self.postings.get(term, {})
            df = len(posting)
            idf = math.log1p((self.size - df + 0.5) / (df + 0.5))
            for chunk_id, tf in posting.items():
                norm = (
                    1
                    - self.config.b
                    + self.config.b * (self.lengths[chunk_id] / self.average_length)
                )
                scores[chunk_id] += idf * tf * (self.config.k1 + 1) / (tf + self.config.k1 * norm)
        ranked = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], str(chunk_id)))[:k]
        return [Candidate(cid, self.identities[cid], scores[cid]) for cid in ranked]


class SparseRetriever:
    def __init__(
        self,
        catalog: IndexCatalog,
        *,
        collection: str | None = None,
        config: BM25Config | None = None,
    ) -> None:
        self.catalog = catalog
        self.collection = collection
        self.config = config or BM25Config()

    def search(
        self, query: str, *, k: int = 5, filters: SearchFilter | None = None
    ) -> list[SearchHit]:
        if not query.strip() or not 1 <= k <= 100:
            raise ValueError("Query must be nonempty; k must be between 1 and 100")
        filters = filters or SearchFilter()
        documents = self.catalog.current_documents(filters)
        if self.collection is not None:
            ready = set(self.catalog.searchable_revisions(self.collection, filters))
            documents = [doc for doc in documents if doc.revision_id in ready]
        # Deliberate first baseline: rebuild from canonical current revisions per request.
        # Statistics belong to the filtered eligible corpus, not historical revisions.
        candidates = BM25Index(documents, self.config).query(query, k)
        return self.catalog.hydrate(self.collection, candidates)
