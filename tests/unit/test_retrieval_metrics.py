import math
from dataclasses import replace

import pytest

from agentic_rag.embeddings.base import EmbeddingSpec, validate_vectors
from agentic_rag.evaluation.metrics import retrieval_metrics


def test_metrics_against_hand_computed_graded_example() -> None:
    values = retrieval_metrics(["b", "x", "a"], {"a": 3, "b": 1, "c": 1}, 3)
    assert values["recall@3"] == pytest.approx(2 / 3)
    assert values["precision@3"] == pytest.approx(2 / 3)
    assert values["mrr@3"] == 1
    assert values["ndcg@3"] == pytest.approx((1 + 7 / 2) / (7 + 1 / math.log2(3) + 1 / 2))


def test_missing_ranks_still_use_k_for_precision_and_mrr_is_truncated() -> None:
    assert retrieval_metrics(["a"], {"a": 1}, 5)["precision@5"] == 0.2
    assert retrieval_metrics(["x", "a"], {"a": 1}, 1)["mrr@1"] == 0
    assert retrieval_metrics(["x", "a"], {"a": 1}, 2)["mrr@2"] == 0.5
    assert all(value == 0 for value in retrieval_metrics([], {"a": 1}, 3).values())


@pytest.mark.parametrize(
    "ranking,labels,k",
    [(["a", "a"], {"a": 1}, 2), ([], {}, 1), ([], {"a": -1}, 1), ([], {"a": 1}, 0)],
)
def test_undefined_metrics_fail_explicitly(
    ranking: list[str], labels: dict[str, int], k: int
) -> None:
    with pytest.raises(ValueError):
        retrieval_metrics(ranking, labels, k)


@pytest.mark.parametrize(
    "vectors,count,dimension",
    [
        ([[0.0, 0.0]], 1, 2),
        ([[float("nan"), 1.0]], 1, 2),
        ([[1.0]], 1, 2),
        ([[1.0, 0.0]], 2, 2),
        ([[2.0, 0.0]], 1, 2),
    ],
)
def test_invalid_provider_output_cannot_enter_index(
    vectors: list[list[float]], count: int, dimension: int
) -> None:
    with pytest.raises(ValueError):
        validate_vectors(vectors, count, dimension)


def test_embedding_profile_changes_for_same_dimension_but_different_model_or_prefix() -> None:
    spec = EmbeddingSpec("test", "a" * 40, 2, 512)
    assert spec.fingerprint == replace(spec).fingerprint
    assert spec.fingerprint != replace(spec, revision="b" * 40).fingerprint
    assert spec.fingerprint != replace(spec, query_prefix="different: ").fingerprint
