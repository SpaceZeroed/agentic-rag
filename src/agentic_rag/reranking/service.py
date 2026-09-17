"""Rerank hydrated candidates, preserving provenance and rechecking visibility."""

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass

from agentic_rag.reranking.base import RerankingProvider
from agentic_rag.retrieval.models import Candidate, IndexCatalog, SearchHit


@dataclass(frozen=True)
class RerankedHit(SearchHit):
    retrieval_score: float
    retrieval_rank: int


def rerank(
    query: str,
    hits: Sequence[SearchHit],
    provider: RerankingProvider,
    catalog: IndexCatalog,
    collection: str | None,
    *,
    k: int = 5,
) -> list[RerankedHit]:
    if not query.strip() or not 1 <= k <= 100:
        raise ValueError("Query must be nonempty and k must be 1..100")
    if len({hit.chunk_id for hit in hits}) != len(hits):
        raise ValueError("Duplicate reranking candidates")
    if not hits:
        return []
    scores = provider.score(query, [hit.text for hit in hits])
    if len(scores) != len(hits) or not all(math.isfinite(s) for s in scores):
        raise RuntimeError("Invalid reranker scores")
    original = {hit.chunk_id: (rank, hit.score) for rank, hit in enumerate(hits, 1)}
    candidates = sorted(
        [
            Candidate(hit.chunk_id, hit.revision_id, score)
            for hit, score in zip(hits, scores, strict=True)
        ],
        key=lambda c: (-c.score, str(c.chunk_id)),
    )
    # Rehydrate all scored candidates before slicing, filling gaps left by stale revisions.
    return [
        RerankedHit(
            **asdict(hit),
            retrieval_rank=original[hit.chunk_id][0],
            retrieval_score=original[hit.chunk_id][1],
        )
        for hit in catalog.hydrate(collection, candidates)[:k]
    ]
