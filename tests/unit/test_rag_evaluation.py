import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from agentic_rag.evaluation.rag_dataset import load_rag_dataset
from agentic_rag.evaluation.rag_metrics import evidence_metrics
from agentic_rag.evaluation.rag_review import Review, review_template, score_review
from agentic_rag.evaluation.rag_runner import run_rag_evaluation
from agentic_rag.llm.base import Completion, LLMError, Message
from agentic_rag.llm.fake import FakeLLM
from agentic_rag.rag.context import build_context
from agentic_rag.retrieval.models import SearchHit

DATASET = Path(__file__).resolve().parents[2] / "benchmarks/rag_v1/dataset.json"


def source_hits() -> list[SearchHit]:
    _, documents = load_rag_dataset(DATASET)
    return [
        SearchHit(
            c.id,
            d.revision_id,
            d.document_id,
            1.0,
            c.text,
            d.source_uri,
            d.title,
            c.start_char,
            c.end_char,
            c.start_line,
            c.end_line,
        )
        for d in documents.values()
        for c in d.chunks
    ]


def test_dataset_has_frozen_sources_multisource_and_unanswerable_cases() -> None:
    dataset, docs = load_rag_dataset(DATASET)
    assert len(docs) == 10
    assert len(dataset.questions) == 10
    assert sum(not q.answerable for q in dataset.questions) == 2
    assert {q.language for q in dataset.questions} == {"ru", "en"}
    assert any(len(q.evidence) > 1 for q in dataset.questions)


@pytest.mark.parametrize(
    "damage",
    [
        "checksum",
        "evidence",
        "unknown",
        "duplicate_question",
        "duplicate_source",
        "answerability",
        "duplicate_evidence",
    ],
)
def test_bad_dataset_rejected(tmp_path: Path, damage: str) -> None:
    data = json.loads(DATASET.read_text())
    for doc in data["documents"]:
        doc["path"] = str((DATASET.parent / doc["path"]).resolve())
    if damage == "checksum":
        data["documents"][0]["sha256"] = "0" * 64
    elif damage == "evidence":
        data["questions"][0]["evidence"][0]["text"] = "absent text"
    elif damage == "unknown":
        data["questions"][0]["evidence"][0]["document"] = "unknown"
    elif damage == "duplicate_question":
        data["questions"].append(data["questions"][0])
    elif damage == "duplicate_source":
        data["documents"][0]["source_uri"] = data["documents"][1]["source_uri"]
    elif damage == "answerability":
        data["questions"][0]["answerable"] = False
    else:
        data["questions"][0]["evidence"] *= 2
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_rag_dataset(path)


def test_retrieval_and_context_metrics_have_distinct_denominators() -> None:
    data, docs = load_rag_dataset(DATASET)
    question = next(q for q in data.questions if q.id == "compare_en")
    hits = [h for h in source_hits() if any(e.text in h.text for e in question.evidence)]
    assert len(hits) == 2
    metrics = evidence_metrics(question, docs, hits, hits[:1], 5)
    assert metrics["recall@5"] == 1
    assert metrics["precision@5"] == 0.4
    assert metrics["context_evidence_coverage"] == 0.5
    assert metrics["context_relevance"] == 1
    empty = evidence_metrics(question, docs, hits, [], 5)
    assert empty["context_evidence_coverage"] == 0
    assert empty["context_relevance"] is None
    assert all(
        v is None for v in evidence_metrics(data.questions[-1], docs, hits, hits, 5).values()
    )


def test_runner_records_failures_and_never_sends_reference_answers() -> None:
    hits = source_hits()[:1]

    class FailingLLM:
        def complete(self, messages: tuple[Message, ...], *, max_tokens: int) -> Completion:
            payload = json.loads(messages[-1].content)
            assert set(payload) == {"question", "sources"}
            assert "reference_answer" not in payload
            raise LLMError("secret-server-body")

    report: Any = run_rag_evaluation(
        DATASET, lambda q: hits, FailingLLM(), provider="compatible", config={}
    )
    assert report["errors"] == 10
    assert report["answer_quality"] is None
    assert "secret-server-body" not in json.dumps(report)
    assert report["metrics"]["recall@5"]["count"] == 8


def test_fake_and_empty_context_do_not_become_quality_measurements() -> None:
    report = run_rag_evaluation(DATASET, lambda q: [], FakeLLM(), provider="fake", config={})
    assert report["run_kind"] == "contract_smoke"
    assert report["errors"] == 0
    assert report["answer_quality"] is None
    template = Review.model_validate(review_template(report))
    with pytest.raises(ValueError, match="Fake"):
        score_review(report, template)


def test_budget_loss_and_invalid_citation_are_recorded() -> None:
    data, _ = load_rag_dataset(DATASET)
    hits = source_hits()[:1]
    budget = max(build_context(q.query, []).prompt_bytes for q in data.questions)
    empty: Any = run_rag_evaluation(
        DATASET, lambda q: hits, FakeLLM(), provider="fake", config={}, max_prompt_bytes=budget
    )
    assert all(not c["context"]["sources"] for c in empty["cases"])

    class InvalidCitation:
        def complete(self, messages: tuple[Message, ...], *, max_tokens: int) -> Completion:
            return Completion("false assertion [C999]", "stub", "stop")

    result: Any = run_rag_evaluation(
        DATASET, lambda q: hits, InvalidCitation(), provider="compatible", config={}
    )
    assert result["errors"] == 10
    assert result["cases"][0]["raw_completion"]["text"] == "false assertion [C999]"


@pytest.mark.parametrize("damage", ["duplicate", "text", "revision", "limit"])
def test_changed_search_results_fail_before_generation(damage: str) -> None:
    from uuid import uuid4

    hits = source_hits()[:1]
    if damage == "duplicate":
        hits *= 2
    elif damage == "text":
        hits = [replace(hits[0], text="wrong")]
    elif damage == "revision":
        hits = [replace(hits[0], revision_id=uuid4())]
    else:
        hits = source_hits()[:6]
    with pytest.raises(ValueError):
        run_rag_evaluation(DATASET, lambda q: hits, FakeLLM(), provider="fake", config={})


def review_report() -> dict[str, object]:
    return {
        "schema_version": 1,
        "provider": "compatible",
        "cases": [
            {
                "id": "a",
                "answerable": True,
                "answer": {"status": "answered", "text": "A. B."},
                "error": None,
            },
            {
                "id": "b",
                "answerable": False,
                "answer": {"status": "insufficient_evidence", "text": "No"},
                "error": None,
            },
            {"id": "c", "answerable": True, "answer": None, "error": "LLMError"},
        ],
    }


def annotations(report: dict[str, object]) -> Review:
    template: Any = review_template(report)
    template["reviewer"] = "test fixture, not a real benchmark assessment"
    template["cases"][0].update(
        correctness=1,
        rationale="One of two facts is correct",
        claims=[
            {"text": "A", "verdict": "supported", "rationale": "A is in the context"},
            {"text": "B", "verdict": "contradicted", "rationale": "Context says not B"},
        ],
    )
    template["cases"][1].update(correctness=2, rationale="No answer in corpus")
    return Review.model_validate(template)


def test_review_math_errors_abstentions_and_claim_denominators() -> None:
    report = review_report()
    summary = score_review(report, annotations(report))
    assert summary["correctness_all_cases_errors_zero"] == 0.5
    assert summary["faithfulness_micro"] == 0.5
    assert summary["unsupported_claim_rate"] == 0.5
    assert summary["contradiction_rate"] == 0.5
    assert summary["abstention_accuracy_all_cases_errors_zero"] == pytest.approx(2 / 3)
    assert summary["reviewed_count"] == 2
    assert summary["errors"] == 1


@pytest.mark.parametrize(
    "damage",
    [
        "hash",
        "missing",
        "duplicate",
        "unfinished",
        "no_claims",
        "abstention_claims",
        "reviewer",
        "score",
    ],
)
def test_review_validation(damage: str) -> None:
    report = review_report()
    data = annotations(report).model_dump()
    if damage == "hash":
        data["report_sha256"] = "wrong"
    elif damage == "missing":
        data["cases"].pop()
    elif damage == "duplicate":
        data["cases"].append(data["cases"][0])
    elif damage == "unfinished":
        data["cases"][0]["correctness"] = None
    elif damage == "no_claims":
        data["cases"][0]["claims"] = []
    elif damage == "abstention_claims":
        data["cases"][1]["claims"] = data["cases"][0]["claims"]
    elif damage == "reviewer":
        data["reviewer"] = " "
    else:
        data["cases"][0]["correctness"] = 3
    with pytest.raises(ValueError):
        score_review(report, Review.model_validate(data))


def test_runner_preserves_incomplete_completion_accounting() -> None:
    from agentic_rag.llm.base import IncompleteCompletionError

    class Truncated:
        def complete(self, messages: tuple[Message, ...], *, max_tokens: int) -> Completion:
            raise IncompleteCompletionError(Completion("", "stub", "length", 50, 512, 0.1, 123))

    report: Any = run_rag_evaluation(
        DATASET, lambda q: source_hits()[:1], Truncated(), provider="compatible", config={}
    )
    assert report["errors"] == 10
    assert all(c["raw_completion"]["completion_tokens"] == 512 for c in report["cases"])
    assert all(c["answer"] is None for c in report["cases"])


def test_assistant_review_is_explicitly_labeled() -> None:
    report = review_report()
    data = annotations(report).model_dump()
    data["review_method"] = "assistant"
    summary = score_review(report, Review.model_validate(data))
    assert summary["review_method"] == "assistant"
