"""Rank-based metrics; missing ranks count as nonrelevant, grades are nonnegative."""

import math
from collections.abc import Mapping, Sequence


def retrieval_metrics(
    ranked: Sequence[str], relevance: Mapping[str, int], k: int
) -> dict[str, float]:
    if k < 1 or len(set(ranked)) != len(ranked):
        raise ValueError("k must be positive and ranked IDs must be unique")
    if any(grade < 0 for grade in relevance.values()):
        raise ValueError("Relevance grades must be nonnegative")
    relevant = {item for item, grade in relevance.items() if grade > 0}
    if not relevant:
        raise ValueError("Metrics require at least one labeled relevant chunk")
    top = ranked[:k]
    hits = sum(item in relevant for item in top)
    reciprocal = next((1.0 / rank for rank, item in enumerate(top, 1) if item in relevant), 0.0)
    dcg = sum(
        (2 ** relevance.get(item, 0) - 1) / math.log2(rank + 1) for rank, item in enumerate(top, 1)
    )
    ideal = sum(
        (2**grade - 1) / math.log2(rank + 1)
        for rank, grade in enumerate(sorted(relevance.values(), reverse=True)[:k], 1)
    )
    return {
        f"recall@{k}": hits / len(relevant),
        f"precision@{k}": hits / k,
        f"mrr@{k}": reciprocal,
        f"ndcg@{k}": dcg / ideal,
    }
