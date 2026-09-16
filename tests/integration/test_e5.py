from pathlib import Path

import pytest

from agentic_rag.embeddings.base import validate_vectors
from agentic_rag.embeddings.e5 import E5EmbeddingProvider


def test_real_e5_padding_normalization_and_token_limit(request: pytest.FixtureRequest) -> None:
    cache = request.config.getoption("--model-cache")
    if not cache:
        pytest.skip("Use --model-cache with downloaded E5 weights for the offline CPU test")
    provider = E5EmbeddingProvider(Path(cache), local_files_only=True)
    text = "An embedding represents a passage as a vector."
    single = provider.embed_documents([text])[0]
    batch = provider.embed_documents([text, "A longer discussion of retrieval. " * 10])
    validate_vectors(batch, 2, 384)
    assert batch[0] == pytest.approx(single, abs=2e-6)
    query = provider.embed_query(text)
    validate_vectors([query], 1, 384)
    assert query != single
    with pytest.raises(ValueError, match="512 tokens"):
        provider.embed_query("retrieval " * 600)
    with pytest.raises(ValueError, match="whitespace"):
        provider.embed_query("   ")
