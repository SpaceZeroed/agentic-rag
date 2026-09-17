from collections.abc import Sequence
from dataclasses import replace
from typing import cast
from uuid import UUID

import pytest

from agentic_rag.reranking.base import RerankingSpec
from agentic_rag.reranking.service import rerank
from agentic_rag.retrieval.models import Candidate, IndexCatalog, SearchHit


def hit(i: int) -> SearchHit:
    return SearchHit(
        UUID(int=i),
        UUID(int=10),
        UUID(int=20),
        1 / i,
        f"passage {i}",
        "https://example.org",
        "title",
        0,
        9,
        1,
        1,
    )


class Scores:
    spec = RerankingSpec("fake", "a" * 40)

    def __init__(self, values: list[float]) -> None:
        self.values = values
        self.calls = 0

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        self.calls += 1
        assert query == "query"
        assert list(passages) == ["passage 3", "passage 2", "passage 1"]
        return self.values


class Catalog:
    def hydrate(self, collection: str | None, candidates: Sequence[Candidate]) -> list[SearchHit]:
        assert collection == "collection"
        # Highest score became stale during inference; fill the slot with a valid candidate.
        return [
            replace(hit(c.chunk_id.int), score=c.score) for c in candidates if c.chunk_id.int != 3
        ]


def test_sort_ties_provenance_and_stale_gap() -> None:
    result = rerank(
        "query",
        [hit(3), hit(2), hit(1)],
        Scores([10, -2, -2]),
        cast(IndexCatalog, Catalog()),
        "collection",
        k=2,
    )
    assert [h.chunk_id.int for h in result] == [1, 2]
    assert result[0].retrieval_rank == 3
    assert result[0].retrieval_score == 1
    assert result[0].score == -2
    assert result[0].text == hit(1).text


@pytest.mark.parametrize("values", [[1], [1, 2, float("nan")], [1, float("inf"), 2]])
def test_invalid_scores(values: list[float]) -> None:
    with pytest.raises(RuntimeError, match="scores"):
        rerank(
            "query",
            [hit(3), hit(2), hit(1)],
            Scores(values),
            cast(IndexCatalog, Catalog()),
            "collection",
        )


def test_empty_duplicate_and_invalid_input() -> None:
    provider = Scores([])
    catalog = cast(IndexCatalog, Catalog())
    assert rerank("query", [], provider, catalog, "collection") == []
    assert provider.calls == 0
    with pytest.raises(ValueError, match="Duplicate"):
        rerank("query", [hit(1), hit(1)], provider, catalog, "collection")
    for query, k in [(" ", 5), ("query", 0), ("query", 101)]:
        with pytest.raises(ValueError):
            rerank(query, [], provider, catalog, "collection", k=k)
