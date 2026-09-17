"""Pinned multilingual MiniLM, explicit pair tokenization and raw logits."""

import importlib
import math
import re
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from agentic_rag.reranking.base import RerankingSpec

MODEL_ID = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
REVISION = "1427fd652930e4ba29e8149678df786c240d8825"


class CrossEncoderProvider:
    def __init__(
        self,
        cache_dir: Path,
        *,
        model_id: str = MODEL_ID,
        revision: str = REVISION,
        batch_size: int = 4,
        threads: int = 4,
        local_files_only: bool = False,
    ) -> None:
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("Pin a Hugging Face model commit SHA")
        if batch_size < 1 or threads < 1:
            raise ValueError("Batch size and threads must be positive")
        try:
            self._torch = importlib.import_module("torch")
            transformers = importlib.import_module("transformers")
        except ImportError as exc:
            raise ValueError("Install model dependencies with uv sync --extra embeddings") from exc
        self._torch.set_num_threads(threads)
        kwargs = dict(
            revision=revision,
            cache_dir=str(cache_dir),
            trust_remote_code=False,
            local_files_only=local_files_only,
        )
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, **kwargs)
        self._model = (
            transformers.AutoModelForSequenceClassification.from_pretrained(
                model_id, use_safetensors=True, **kwargs
            )
            .to(device="cpu", dtype=self._torch.float32)
            .eval()
        )
        if self._model.config.num_labels != 1:
            raise ValueError("This provider requires a single-logit ranking head")
        self._spec = RerankingSpec(model_id, revision, batch_size=batch_size, threads=threads)

    @property
    def spec(self) -> RerankingSpec:
        return self._spec

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not query.strip() or any(not passage.strip() for passage in passages):
            raise ValueError("Query and passages must be nonempty")
        scores: list[float] = []
        for start in range(0, len(passages), self.spec.batch_size):
            batch = list(passages[start : start + self.spec.batch_size])
            tokens = self._tokenizer(
                [query] * len(batch), batch, padding=True, truncation=False, return_tensors="pt"
            )
            if tokens["input_ids"].shape[1] > self.spec.max_tokens:
                raise ValueError("Pair exceeds 512 tokens; shorten query or rechunk passage")
            with self._torch.inference_mode():
                logits = self._model(**tokens).logits
            if tuple(logits.shape) != (len(batch), 1):
                raise RuntimeError("Unexpected reranker output shape")
            scores.extend(cast(list[float], logits[:, 0].cpu().tolist()))
        if len(scores) != len(passages) or not all(math.isfinite(s) for s in scores):
            raise RuntimeError("Invalid reranker scores")
        return scores
