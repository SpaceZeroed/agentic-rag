"""Label-based evidence metrics, not semantic judgments of answer quality."""

from collections.abc import Sequence

from agentic_rag.evaluation.metrics import retrieval_metrics
from agentic_rag.evaluation.rag_dataset import RAGQuestion
from agentic_rag.ingestion.models import PreparedDocument
from agentic_rag.retrieval.models import SearchHit


def evidence_metrics(
    question: RAGQuestion,
    documents: dict[str, PreparedDocument],
    retrieved: Sequence[SearchHit],
    context: Sequence[SearchHit],
    k: int,
) -> dict[str, float | None]:
    if not question.answerable:
        return {
            f"recall@{k}": None,
            f"precision@{k}": None,
            f"mrr@{k}": None,
            f"ndcg@{k}": None,
            "context_evidence_coverage": None,
            "context_relevance": None,
        }
    qrels: dict[str, int] = {}
    for evidence in question.evidence:
        for chunk in documents[evidence.document].chunks:
            if evidence.text in chunk.text:
                key = str(chunk.id)
                qrels[key] = max(qrels.get(key, 0), evidence.grade)
    metrics: dict[str, float | None] = dict(
        retrieval_metrics([str(h.chunk_id) for h in retrieved], qrels, k)
    )
    covered = sum(
        any(
            h.revision_id == documents[e.document].revision_id and e.text in h.text for h in context
        )
        for e in question.evidence
    )
    metrics["context_evidence_coverage"] = covered / len(question.evidence)
    metrics["context_relevance"] = (
        sum(str(h.chunk_id) in qrels for h in context) / len(context) if context else None
    )
    return metrics
