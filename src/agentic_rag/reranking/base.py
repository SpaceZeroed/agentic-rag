from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class RerankingSpec:
    model_id: str
    revision: str
    max_tokens: int = 512
    truncation: str = "reject"
    score: str = "raw_logit"
    device: str = "cpu"
    dtype: str = "float32"
    batch_size: int = 4
    threads: int = 4


class RerankingProvider(Protocol):
    @property
    def spec(self) -> RerankingSpec: ...

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        """Return one finite relevance logit per passage, in input order."""
        ...
