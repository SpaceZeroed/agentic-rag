"""Qdrant SDK types stay inside this adapter."""

import math
import re
from collections.abc import Sequence
from uuid import UUID

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from agentic_rag.embeddings.base import EmbeddingSpec, validate_vectors
from agentic_rag.ingestion.models import PreparedDocument
from agentic_rag.retrieval.models import Candidate


class QdrantVectorIndex:
    def __init__(
        self, client: QdrantClient, spec: EmbeddingSpec, prefix: str = "rag_dense"
    ) -> None:
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,40}", prefix):
            raise ValueError("Collection prefix must contain 1-40 letters, digits, '_' or '-'")
        self.client = client
        self.spec = spec
        self._collection = f"{prefix}_{spec.fingerprint}"

    @property
    def collection(self) -> str:
        return self._collection

    def ensure(self, spec: EmbeddingSpec) -> bool:
        if spec != self.spec:
            raise ValueError("Embedding/index profiles differ")
        created = False
        if not self.client.collection_exists(self.collection):
            try:
                self.client.create_collection(
                    self.collection,
                    vectors_config=models.VectorParams(
                        size=spec.dimension, distance=models.Distance.COSINE
                    ),
                )
                created = True
            except UnexpectedResponse as exc:
                if exc.status_code != 409:
                    raise
        vectors = self.client.get_collection(self.collection).config.params.vectors
        if (
            not isinstance(vectors, models.VectorParams)
            or vectors.size != spec.dimension
            or vectors.distance != models.Distance.COSINE
        ):
            raise ValueError("Existing Qdrant collection has incompatible vector configuration")
        return created

    @staticmethod
    def _filter(revisions: Sequence[UUID]) -> models.Filter:
        return models.Filter(
            must=[
                models.FieldCondition(
                    key="revision_id", match=models.MatchAny(any=[str(item) for item in revisions])
                )
            ]
        )

    def count_revision(self, revision_id: UUID) -> int:
        return self.client.count(
            self.collection, count_filter=self._filter([revision_id]), exact=True
        ).count

    def upsert(self, document: PreparedDocument, vectors: Sequence[Sequence[float]]) -> None:
        validate_vectors(vectors, len(document.chunks), self.spec.dimension)
        for start in range(0, len(vectors), 64):
            points = [
                models.PointStruct(
                    id=str(chunk.id),
                    vector=list(vector),
                    payload={
                        "revision_id": str(document.revision_id),
                        "document_id": str(document.document_id),
                    },
                )
                for chunk, vector in zip(
                    document.chunks[start : start + 64], vectors[start : start + 64], strict=True
                )
            ]
            self.client.upsert(self.collection, points=points, wait=True)

    def query(
        self, vector: list[float], revisions: Sequence[UUID], k: int, *, exact: bool
    ) -> list[Candidate]:
        if not revisions:
            return []
        validate_vectors([vector], 1, self.spec.dimension)
        points = self.client.query_points(
            self.collection,
            query=vector,
            query_filter=self._filter(revisions),
            search_params=models.SearchParams(exact=exact),
            limit=k,
            with_payload=True,
            with_vectors=False,
        ).points
        result = []
        for point in points:
            if not math.isfinite(point.score) or not point.payload:
                raise ValueError("Invalid vector search result")
            result.append(
                Candidate(UUID(str(point.id)), UUID(str(point.payload["revision_id"])), point.score)
            )
        return result
