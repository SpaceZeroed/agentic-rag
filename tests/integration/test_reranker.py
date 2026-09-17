from pathlib import Path

import pytest

from agentic_rag.reranking.cross_encoder import CrossEncoderProvider


def test_real_reranker_pairs_padding_limit(request: pytest.FixtureRequest) -> None:
    cache = request.config.getoption("--reranker-cache")
    if not cache:
        pytest.skip("Use --reranker-cache for the offline CPU reranker test")
    provider = CrossEncoderProvider(Path(cache), batch_size=2, local_files_only=True)
    query = "What is the capital of France?"
    passages = [
        "Paris is the capital of France.",
        "Bananas are yellow fruit. " * 10,
        "The capital city of France is Paris.",
    ]
    scores = provider.score(query, passages)
    assert len(scores) == 3
    assert scores[0] > scores[1]
    assert scores[0] == pytest.approx(provider.score(query, passages[:1])[0], abs=1e-4)
    assert provider.score(query, []) == []
    with pytest.raises(ValueError, match="512 tokens"):
        provider.score(query, ["retrieval " * 600])
    with pytest.raises(ValueError, match="512 tokens"):
        provider.score("retrieval " * 600, passages[:1])
    with pytest.raises(ValueError, match="nonempty"):
        provider.score(" ", passages)
