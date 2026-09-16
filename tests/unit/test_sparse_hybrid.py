import math
from pathlib import Path
from uuid import UUID

import pytest

from agentic_rag.ingestion.models import ChunkingConfig, PreparedDocument
from agentic_rag.ingestion.service import prepare_document
from agentic_rag.retrieval.hybrid import reciprocal_rank_fusion
from agentic_rag.retrieval.models import Candidate
from agentic_rag.retrieval.sparse import BM25Config, BM25Index, tokenize


def corpus(tmp_path: Path, texts: list[str]) -> list[PreparedDocument]:
    result = []
    for i, text in enumerate(texts):
        path = tmp_path / f"{i}.txt"
        path.write_text(text)
        result.append(prepare_document(path, ChunkingConfig(1000, 0)))
    return result


def test_tokenizer_preserves_languages_numbers_and_normalizes() -> None:
    assert tokenize("ＬｏＲＡ KV_cache 4-bit ЁЖ ёж Straße") == [
        "lora",
        "kv",
        "cache",
        "4",
        "bit",
        "ёж",
        "ёж",
        "strasse",
    ]
    assert tokenize("!!!") == []
    assert tokenize("key value") != tokenize("ключ значение")


def test_bm25_matches_hand_calculation_and_query_terms_are_unique(tmp_path: Path) -> None:
    docs = corpus(tmp_path, ["alpha alpha beta", "beta"])
    index = BM25Index(docs)
    hits = index.query("alpha", 5)
    # N=2, df=1, tf=2, dl=3, avgdl=2, k1=1.2, b=.75.
    expected = math.log(2) * 2 * 2.2 / (2 + 1.2 * (0.25 + 0.75 * 3 / 2))
    assert len(hits) == 1
    assert hits[0].chunk_id == docs[0].chunks[0].id
    assert hits[0].score == pytest.approx(expected)
    assert index.query("alpha alpha", 5) == hits
    assert index.query("unknown", 5) == []
    assert index.query("!!!", 5) == []


def test_length_normalization_empty_corpus_and_tie_order(tmp_path: Path) -> None:
    docs = corpus(tmp_path, ["alpha", "alpha beta beta beta"])
    assert BM25Index(docs).query("alpha", 2)[0].chunk_id == docs[0].chunks[0].id
    unnormalized = BM25Index(docs, BM25Config(b=0)).query("alpha", 2)
    assert unnormalized[0].score == unnormalized[1].score
    assert [str(c.chunk_id) for c in unnormalized] == sorted(str(c.chunk_id) for c in unnormalized)
    assert BM25Index([]).query("alpha", 5) == []
    assert BM25Index(corpus(tmp_path, ["!!!"])).query("alpha", 5) == []
    with pytest.raises(ValueError, match="Duplicate"):
        BM25Index([docs[0], docs[0]])


@pytest.mark.parametrize("k1,b", [(0, 0.75), (float("nan"), 0.75), (1.2, -1), (1.2, 2)])
def test_invalid_bm25_parameters(k1: float, b: float) -> None:
    with pytest.raises(ValueError):
        BM25Config(k1, b)


def test_rrf_uses_ranks_and_deduplicates_across_branches() -> None:
    a = Candidate(UUID(int=1), UUID(int=10), 1000)
    b = Candidate(UUID(int=2), UUID(int=20), 0.2)
    c = Candidate(UUID(int=3), UUID(int=30), 9999)
    fused = reciprocal_rank_fusion([[a, b], [c, b]], k=3)
    assert [hit.chunk_id for hit in fused] == [b.chunk_id, a.chunk_id, c.chunk_id]
    assert fused[0].score == pytest.approx(2 / 62)
    assert fused[1].score == pytest.approx(1 / 61)
    assert reciprocal_rank_fusion([[], []], k=5) == []
    assert reciprocal_rank_fusion([[a, b], []], k=1)[0].chunk_id == a.chunk_id
    with pytest.raises(ValueError, match="Duplicate"):
        reciprocal_rank_fusion([[a, a]], k=2)
    with pytest.raises(ValueError, match="Conflicting"):
        reciprocal_rank_fusion([[a], [Candidate(a.chunk_id, c.revision_id, 0)]], k=2)
