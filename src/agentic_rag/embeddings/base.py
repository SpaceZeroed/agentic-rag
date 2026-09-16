import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Protocol


@dataclass(frozen=True)
class EmbeddingSpec:
    model_id: str
    revision: str
    dimension: int
    max_tokens: int
    query_prefix: str = "query: "
    passage_prefix: str = "passage: "
    algorithm: str = "masked-mean-l2-float32-v1"

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


class EmbeddingProvider(Protocol):
    @property
    def spec(self) -> EmbeddingSpec: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def validate_vectors(vectors: Sequence[Sequence[float]], count: int, dimension: int) -> None:
    if len(vectors) != count:
        raise ValueError("Embedding count differs from text count")
    for vector in vectors:
        if len(vector) != dimension or not all(math.isfinite(value) for value in vector):
            raise ValueError("Embedding dimension or values are invalid")
        if not math.isclose(sum(value * value for value in vector), 1.0, abs_tol=1e-4):
            raise ValueError("Embeddings must have unit L2 norm")
