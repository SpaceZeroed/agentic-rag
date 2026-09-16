"""Rank fusion uses ranks rather than incomparable dense and lexical scores."""

from collections import defaultdict
from collections.abc import Sequence
from uuid import UUID

from agentic_rag.retrieval.models import Candidate, SearchFilter, SearchHit
from agentic_rag.retrieval.service import DenseRetriever
from agentic_rag.retrieval.sparse import SparseRetriever


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[Candidate]], *, k: int, rank_constant: int = 60
) -> list[Candidate]:
    if not 1 <= k <= 100 or rank_constant < 1:
        raise ValueError("k must be 1..100 and rank_constant must be positive")
    scores: dict[UUID, float] = defaultdict(float)
    revisions: dict[UUID, UUID] = {}
    for ranking in rankings:
        seen: set[UUID] = set()
        for rank, candidate in enumerate(ranking, start=1):
            if candidate.chunk_id in seen:
                raise ValueError("Duplicate chunk in a ranking")
            if (
                candidate.chunk_id in revisions
                and revisions[candidate.chunk_id] != candidate.revision_id
            ):
                raise ValueError("Conflicting revision for a chunk")
            seen.add(candidate.chunk_id)
            revisions[candidate.chunk_id] = candidate.revision_id
            scores[candidate.chunk_id] += 1 / (rank_constant + rank)
    ordered = sorted(scores, key=lambda cid: (-scores[cid], str(cid)))[:k]
    return [Candidate(cid, revisions[cid], scores[cid]) for cid in ordered]


class HybridRetriever:
    def __init__(
        self, dense: DenseRetriever, *, candidate_k: int = 20, rank_constant: int = 60
    ) -> None:
        if not 1 <= candidate_k <= 100 or rank_constant < 1:
            raise ValueError("Invalid hybrid candidate depth or rank constant")
        self.dense = dense
        self.sparse = SparseRetriever(dense.catalog, collection=dense.index.collection)
        self.candidate_k = candidate_k
        self.rank_constant = rank_constant

    def search(
        self, query: str, *, k: int = 5, filters: SearchFilter | None = None, exact: bool = True
    ) -> list[SearchHit]:
        if not query.strip() or not 1 <= k <= self.candidate_k:
            raise ValueError("Query must be nonempty; k must be within candidate depth")
        dense_hits = self.dense.search(query, k=self.candidate_k, filters=filters, exact=exact)
        sparse_hits = self.sparse.search(query, k=self.candidate_k, filters=filters)
        rankings = [
            [Candidate(hit.chunk_id, hit.revision_id, hit.score) for hit in hits]
            for hits in (dense_hits, sparse_hits)
        ]
        fused = reciprocal_rank_fusion(
            rankings, k=self.candidate_k, rank_constant=self.rank_constant
        )
        # Recheck after both branches; never return an obsolete intermediate hit.
        return self.dense.catalog.hydrate(self.dense.index.collection, fused)[:k]
