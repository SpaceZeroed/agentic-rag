"""Hugging Face E5 on CPU with visible pooling and no silent truncation."""

import importlib
import re
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from agentic_rag.embeddings.base import EmbeddingSpec, validate_vectors

E5_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"


class E5EmbeddingProvider:
    def __init__(
        self,
        cache_dir: Path,
        *,
        model_id: str = "intfloat/multilingual-e5-small",
        revision: str = E5_REVISION,
        batch_size: int = 8,
        threads: int = 4,
        local_files_only: bool = False,
    ) -> None:
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("Pin a Hugging Face model commit SHA, not a mutable branch")
        if batch_size < 1 or threads < 1:
            raise ValueError("Batch size and CPU thread count must be positive")
        try:
            self._torch = importlib.import_module("torch")
            transformers = importlib.import_module("transformers")
        except ImportError as exc:
            raise ValueError("Install model dependencies with uv sync --extra embeddings") from exc
        self._torch.set_num_threads(threads)
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            cache_dir=str(cache_dir),
            trust_remote_code=False,
            local_files_only=local_files_only,
        )
        self._model = (
            transformers.AutoModel.from_pretrained(
                model_id,
                revision=revision,
                cache_dir=str(cache_dir),
                trust_remote_code=False,
                use_safetensors=True,
                local_files_only=local_files_only,
            )
            .to("cpu")
            .eval()
        )
        self._spec = EmbeddingSpec(model_id, revision, int(self._model.config.hidden_size), 512)
        self._batch_size = batch_size

    @property
    def spec(self) -> EmbeddingSpec:
        return self._spec

    def _encode(self, texts: Sequence[str], prefix: str) -> list[list[float]]:
        result: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            if any(not text.strip() for text in batch):
                raise ValueError("Cannot embed empty/whitespace text")
            tokens = self._tokenizer(
                [prefix + text for text in batch],
                padding=True,
                truncation=False,
                return_tensors="pt",
            )
            if tokens["input_ids"].shape[1] > self.spec.max_tokens:
                raise ValueError("Input exceeds 512 tokens including prefix; rechunk or shorten it")
            with self._torch.inference_mode():
                hidden = self._model(**tokens).last_hidden_state
                mask = tokens["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
                vectors = self._torch.nn.functional.normalize(pooled, p=2, dim=1)
                result.extend(cast(list[list[float]], vectors.cpu().tolist()))
        validate_vectors(result, len(texts), self.spec.dimension)
        return result

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._encode(texts, self.spec.passage_prefix)

    def embed_query(self, text: str) -> list[float]:
        return self._encode([text], self.spec.query_prefix)[0]
