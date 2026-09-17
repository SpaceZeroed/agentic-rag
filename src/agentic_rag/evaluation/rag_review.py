"""Human semantic review tied to the exact saved report, never a fake quality score."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agentic_rag.evaluation.rag_runner import fingerprint


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)


class Claim(StrictModel):
    text: str = Field(min_length=1)
    verdict: Literal["supported", "unsupported", "contradicted"]
    rationale: str = Field(min_length=1)


class CaseReview(StrictModel):
    id: str
    correctness: int | None = Field(default=None, ge=0, le=2)
    rationale: str = ""
    claims: list[Claim] = Field(default_factory=list)


class Review(StrictModel):
    schema_version: Literal[1]
    report_sha256: str
    reviewer: str
    review_method: Literal["human", "assistant"] = "human"
    cases: list[CaseReview]

    @model_validator(mode="after")
    def unique_cases(self) -> Review:
        if len({c.id for c in self.cases}) != len(self.cases):
            raise ValueError("Duplicate review case IDs")
        return self


class _Answer(BaseModel):
    model_config = ConfigDict(strict=True)
    status: Literal["answered", "insufficient_evidence"]
    text: str


class _Case(BaseModel):
    model_config = ConfigDict(strict=True)
    id: str
    answerable: bool
    answer: _Answer | None
    error: str | None


class _Report(BaseModel):
    model_config = ConfigDict(strict=True)
    schema_version: Literal[1]
    provider: Literal["fake", "compatible"]
    cases: list[_Case] = Field(min_length=1)

    @model_validator(mode="after")
    def valid_cases(self) -> _Report:
        if len({c.id for c in self.cases}) != len(self.cases):
            raise ValueError("Duplicate report case IDs")
        if any((c.answer is None) != (c.error is not None) for c in self.cases):
            raise ValueError("Each case must have either an answer or an error")
        return self


def review_template(report: dict[str, object]) -> dict[str, object]:
    parsed = _Report.model_validate(report)
    return Review(
        schema_version=1,
        report_sha256=fingerprint(report),
        reviewer="",
        cases=[CaseReview(id=c.id) for c in parsed.cases if c.error is None],
    ).model_dump()


def score_review(report: dict[str, object], review: Review) -> dict[str, object]:
    parsed = _Report.model_validate(report)
    if parsed.provider == "fake":
        raise ValueError("Fake runs are contract tests, not answer-quality measurements")
    if review.report_sha256 != fingerprint(report):
        raise ValueError("Review belongs to a different report")
    if not review.reviewer.strip():
        raise ValueError("Reviewer identity is required")
    successful = {c.id: c for c in parsed.cases if c.error is None}
    if set(successful) != {c.id for c in review.cases}:
        raise ValueError("Review must cover every non-error case exactly once")
    supported = contradicted = total_claims = correctness_sum = 0
    per_case: list[dict[str, object]] = []
    for annotation in review.cases:
        case = successful[annotation.id]
        if (
            annotation.correctness is None
            or not annotation.rationale.strip()
            or case.answer is None
        ):
            raise ValueError("Review is incomplete")
        answered = case.answer.status == "answered"
        if answered != bool(annotation.claims):
            raise ValueError("Answered cases require claim review; abstentions have no claims")
        texts = [claim.text for claim in annotation.claims]
        if len(set(texts)) != len(texts) or any(text not in case.answer.text for text in texts):
            raise ValueError("Claims must be unique verbatim excerpts of the answer")
        supported_count = sum(c.verdict == "supported" for c in annotation.claims)
        contradicted_count = sum(c.verdict == "contradicted" for c in annotation.claims)
        supported += supported_count
        contradicted += contradicted_count
        total_claims += len(annotation.claims)
        correctness_sum += annotation.correctness
        per_case.append(
            {
                "id": case.id,
                "correctness": annotation.correctness / 2,
                "faithfulness": supported_count / len(annotation.claims)
                if annotation.claims
                else None,
            }
        )
    errors = sum(c.error is not None for c in parsed.cases)
    abstention_correct = sum(
        c.answer is not None
        and c.error is None
        and (c.answer.status == "insufficient_evidence") == (not c.answerable)
        for c in parsed.cases
    )
    return {
        "report_sha256": review.report_sha256,
        "reviewer": review.reviewer,
        "review_method": review.review_method,
        "case_count": len(parsed.cases),
        "reviewed_count": len(review.cases),
        "errors": errors,
        "correctness_all_cases_errors_zero": correctness_sum / (2 * len(parsed.cases)),
        "abstention_accuracy_all_cases_errors_zero": abstention_correct / len(parsed.cases),
        "claim_count": total_claims,
        "faithfulness_micro": supported / total_claims if total_claims else None,
        "unsupported_claim_rate": (total_claims - supported) / total_claims
        if total_claims
        else None,
        "contradiction_rate": contradicted / total_claims if total_claims else None,
        "cases": per_case,
    }
