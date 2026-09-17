"""Versioned RAG labels, separate from the frozen retrieval benchmark."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agentic_rag.evaluation.runner import Evidence, Source
from agentic_rag.ingestion.models import ChunkingConfig, PreparedDocument
from agentic_rag.ingestion.service import prepare_document


class RAGQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    id: str = Field(min_length=1)
    language: str = Field(min_length=1)
    query: str = Field(min_length=1)
    answerable: bool
    reference_answer: str = Field(min_length=1)
    evidence: list[Evidence]

    @model_validator(mode="after")
    def consistent_labels(self) -> RAGQuestion:
        if self.answerable != bool(self.evidence):
            raise ValueError("Answerable cases require evidence; unanswerable cases have none")
        keys = [(e.document, e.text) for e in self.evidence]
        if len(set(keys)) != len(keys):
            raise ValueError("Duplicate evidence")
        return self


class RAGDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: int = Field(ge=1, le=1)
    name: str
    label_policy: str
    max_chars: int
    overlap: int
    documents: list[Source] = Field(min_length=1)
    questions: list[RAGQuestion] = Field(min_length=1)


def load_rag_dataset(path: Path) -> tuple[RAGDataset, dict[str, PreparedDocument]]:
    data = RAGDataset.model_validate_json(path.read_text(encoding="utf-8"))
    if len({d.key for d in data.documents}) != len(data.documents) or len(
        {q.id for q in data.questions}
    ) != len(data.questions):
        raise ValueError("Duplicate document/question IDs")
    documents = {}
    for source in data.documents:
        doc = prepare_document(
            path.parent / source.path,
            ChunkingConfig(data.max_chars, data.overlap),
            source_uri=source.source_uri,
        )
        if doc.raw_sha256 != source.sha256:
            raise ValueError("Source checksum changed; create a new dataset version")
        documents[source.key] = doc
    if len({d.document_id for d in documents.values()}) != len(documents):
        raise ValueError("Duplicate source identities")
    for question in data.questions:
        for evidence in question.evidence:
            if evidence.document not in documents or not any(
                evidence.text in c.text for c in documents[evidence.document].chunks
            ):
                raise ValueError("Evidence absent, unknown or split across chunks")
    return data, documents
