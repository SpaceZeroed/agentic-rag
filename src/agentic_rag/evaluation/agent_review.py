"""Offline, hash-bound tool-use review. No model calls or automatic semantic judge."""

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

POLICY: Literal["agent-tool-review-v1"] = "agent-tool-review-v1"
DIMENSIONS = ("task_completion", "claim_support", "citation_placement", "failure_handling")


class ReviewModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Judgment(ReviewModel):
    value: bool | None = None
    rationale: str = ""

    def checked(self) -> bool:
        if self.value is None or not self.rationale.strip():
            raise ValueError("Every judgment needs a boolean and a written rationale")
        return self.value


class ProposalReview(ReviewModel):
    turn: int = Field(ge=0)
    call: int = Field(ge=0)
    call_id: str
    name: str
    arguments: str
    origin: Literal["live", "mock", "replayed"]
    selection: Judgment = Field(default_factory=Judgment)
    arguments_correct: Judgment = Field(default_factory=Judgment)
    unnecessary: Judgment = Field(default_factory=Judgment)


class CaseReview(ReviewModel):
    id: str
    report_sha256: str
    # Includes missing required tools and the correct choice to use no tools.
    selection_complete: Judgment = Field(default_factory=Judgment)
    proposals: list[ProposalReview]


class ToolReview(ReviewModel):
    policy: Literal["agent-tool-review-v1"] = POLICY
    manifest_sha256: str
    reviewer: str | None = None
    reviewer_type: Literal["human", "assistant"] | None = None
    cases: list[CaseReview]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def proposals(report: dict[str, Any]) -> list[ProposalReview]:
    result = []
    for turn_index, turn in enumerate(report["model_turns"]):
        for call_index, call in enumerate(turn.get("assistant_message", {}).get("tool_calls", [])):
            result.append(
                ProposalReview(
                    turn=turn_index,
                    call=call_index,
                    call_id=call["id"],
                    name=call["function"]["name"],
                    arguments=call["function"]["arguments"],
                    origin=turn["origin"],
                )
            )
    return result


def load_run(directory: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    manifest = read_object(directory / "manifest.json")
    summary = read_object(directory / "summary.json")
    reports = {}
    for item in summary["cases"]:
        case_id = item["id"]
        if not isinstance(case_id, str) or Path(case_id).name != case_id or case_id in reports:
            raise ValueError("Invalid or duplicate case ID")
        report = read_object(directory / f"{case_id}.json")
        if report["manifest_sha256"] != digest(directory / "manifest.json"):
            raise ValueError("Report manifest hash mismatch")
        if report["summary"] != item:
            raise ValueError("Summary disagrees with case report")
        reports[case_id] = report
    selected = manifest["selected_cases"]
    skipped = summary["skipped_cases"]
    if (
        len(set(selected)) != len(selected)
        or len(set(skipped)) != len(skipped)
        or set(reports) & set(skipped)
        or set(reports) | set(skipped) != set(selected)
    ):
        raise ValueError("Attempted/skipped coverage disagrees with manifest")
    return manifest, reports


def make_template(directory: Path) -> ToolReview:
    _, reports = load_run(directory)
    return ToolReview(
        manifest_sha256=digest(directory / "manifest.json"),
        cases=[
            CaseReview(
                id=case_id,
                report_sha256=digest(directory / f"{case_id}.json"),
                proposals=proposals(report),
            )
            for case_id, report in reports.items()
        ],
    )


def fraction(values: list[bool]) -> dict[str, int | float | None]:
    return {
        "numerator": sum(values),
        "denominator": len(values),
        "rate": sum(values) / len(values) if values else None,
    }


def distribution(values: list[float]) -> dict[str, int | float | None]:
    ordered = sorted(values)
    return {
        "n": len(values),
        "mean": mean(values) if values else None,
        "p50_nearest_rank": ordered[math.ceil(len(values) * 0.5) - 1] if values else None,
        "p95_nearest_rank": ordered[math.ceil(len(values) * 0.95) - 1] if values else None,
    }


def checked_quality(review: dict[str, Any], template: ToolReview) -> dict[str, Any]:
    if (
        review.get("manifest_sha256") != template.manifest_sha256
        or not str(review.get("reviewer") or "").strip()
        or review.get("reviewer_type") not in ("human", "assistant")
    ):
        raise ValueError("Quality review needs matching manifest and named reviewer")
    cases = review["cases"]
    by_id = {case["id"]: case for case in cases}
    if len(by_id) != len(cases) or set(by_id) != {case.id for case in template.cases}:
        raise ValueError("Quality review must cover exactly the attempted cases")
    for case in template.cases:
        quality = by_id[case.id]
        if quality["report_sha256"] != case.report_sha256:
            raise ValueError("Quality report hash mismatch")
        if set(quality["dimensions"]) != set(DIMENSIONS):
            raise ValueError("Unexpected quality dimensions")
        for dimension in quality["dimensions"].values():
            applicable, score = dimension["applicable"], dimension["score"]
            if not isinstance(applicable, bool) or not dimension["rationale"].strip():
                raise ValueError("Quality applicability and rationale required")
            if (applicable and (type(score) is not int or score not in (0, 1, 2))) or (
                not applicable and score is not None
            ):
                raise ValueError("Invalid quality score or N/A")
    return by_id


def summarize_reviews(
    directories: list[Path], tool_files: list[Path], quality_files: list[Path]
) -> dict[str, Any]:
    if not directories or not len(directories) == len(tool_files) == len(quality_files):
        raise ValueError("Supply one tool review and quality review per run")
    rows: list[dict[str, Any]] = []
    sources = []
    seen: set[str] = set()
    comparison = None
    skipped_cases: list[str] = []
    for directory, tool_file, quality_file in zip(
        directories, tool_files, quality_files, strict=True
    ):
        manifest, reports = load_run(directory)
        # Combining disjoint subsets requires the same experimental configuration.
        config: dict[str, Any] = {
            key: manifest[key]
            for key in (
                "cases_sha256",
                "corpus_manifest_sha256",
                "mode",
                "model",
                "policy",
                "backend",
                "provider_origin",
                "bm25",
                "tokenizer",
                "documents",
                "runtime_versions",
                "llm_timeout_seconds",
            )
            if key in manifest
        }
        # README/result prose can change between disjoint subsets. Compare executable
        # code, dependencies and machine-readable labels, retaining manifest hashes.
        config["working_file_sha256"] = {
            path: value
            for path, value in manifest.get("repository", {}).get("working_file_sha256", {}).items()
            if path.startswith(("src/", "scripts/", "tests/"))
            or path in ("pyproject.toml", "uv.lock")
            or (path.startswith("benchmarks/") and not path.endswith(".md"))
        }
        if comparison is not None and config != comparison:
            raise ValueError("Cannot combine different experimental configurations")
        comparison = config
        template = make_template(directory)
        review = ToolReview.model_validate_json(tool_file.read_bytes())
        if review.manifest_sha256 != template.manifest_sha256:
            raise ValueError("Tool review manifest hash mismatch")
        if not review.reviewer or not review.reviewer.strip() or review.reviewer_type is None:
            raise ValueError("Tool review needs a named reviewer")
        expected = {case.id: case for case in template.cases}
        if len(review.cases) != len(expected) or {c.id for c in review.cases} != set(expected):
            raise ValueError("Tool review must cover exactly the attempted cases")
        quality_raw = read_object(quality_file)
        quality = checked_quality(quality_raw, template)
        for case in review.cases:
            if case.id in seen:
                raise ValueError("Duplicate case across runs")
            seen.add(case.id)
            original = expected[case.id]
            if case.report_sha256 != original.report_sha256:
                raise ValueError("Tool report hash mismatch")
            identities = [
                p.model_dump(exclude={"selection", "arguments_correct", "unnecessary"})
                for p in case.proposals
            ]
            if identities != [
                p.model_dump(exclude={"selection", "arguments_correct", "unnecessary"})
                for p in original.proposals
            ]:
                raise ValueError("Review proposal coverage/identity mismatch")
            selection_complete = case.selection_complete.checked()
            for proposal in case.proposals:
                proposal.selection.checked()
                proposal.arguments_correct.checked()
                proposal.unnecessary.checked()
            report = reports[case.id]
            summary = report["summary"]
            elapsed = summary["elapsed_seconds"]
            if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
                raise ValueError("Invalid elapsed time")
            turns = report["model_turns"]
            if summary["model_calls"] != len(turns):
                raise ValueError("Model call count disagrees with trace")
            observations = report["result"]["observations"] if report["result"] else None
            if observations is not None and summary["tool_calls"] != len(observations):
                raise ValueError("Tool observation count disagrees with trace")
            rows.append(
                {
                    "id": case.id,
                    "group": summary["group"],
                    "status": summary["status"],
                    "selection_complete": selection_complete,
                    "proposals": [p.model_dump() for p in case.proposals],
                    "model_calls": len(turns),
                    "replayed_model_calls": sum(t["origin"] == "replayed" for t in turns),
                    "tool_observations": len(observations) if observations is not None else None,
                    "observation_errors": dict(
                        Counter(o["error"] for o in observations or [] if o["status"] == "error")
                    ),
                    "repair_attempts": summary["repair_attempts"],
                    "elapsed_seconds": elapsed,
                    "dimensions": quality[case.id]["dimensions"],
                    "report_sha256": case.report_sha256,
                }
            )
        skipped_cases.extend(read_object(directory / "summary.json")["skipped_cases"])
        sources.append(
            {
                "directory": str(directory),
                "manifest_sha256": template.manifest_sha256,
                "tool_review_sha256": digest(tool_file),
                "quality_review_sha256": digest(quality_file),
                "tool_reviewer": review.reviewer,
                "tool_reviewer_type": review.reviewer_type,
                "quality_reviewer": quality_raw["reviewer"],
                "quality_reviewer_type": quality_raw["reviewer_type"],
            }
        )
    groups = {}
    for group in ("natural", "controlled"):
        selected = [row for row in rows if row["group"] == group]
        origin_metrics = {}
        for origin in ("live", "mock", "replayed"):
            calls = [p for row in selected for p in row["proposals"] if p["origin"] == origin]
            origin_metrics[origin] = {
                key: fraction([p[key]["value"] for p in calls])
                for key in ("selection", "arguments_correct", "unnecessary")
            }
        dimensions = {}
        for name in DIMENSIONS:
            scores = [
                row["dimensions"][name]["score"]
                for row in selected
                if row["dimensions"][name]["applicable"]
            ]
            dimensions[name] = {
                "applicable_cases": len(scores),
                "mean_score": mean(scores) if scores else None,
                "full_score": fraction([score == 2 for score in scores]),
            }
        groups[group] = {
            "cases": len(selected),
            "status_counts": dict(Counter(row["status"] for row in selected)),
            "selection_complete": fraction([row["selection_complete"] for row in selected]),
            "proposal_metrics_by_origin": origin_metrics,
            "quality": dimensions,
            "model_calls": distribution([row["model_calls"] for row in selected]),
            "tool_observations": distribution(
                [
                    row["tool_observations"]
                    for row in selected
                    if row["tool_observations"] is not None
                ]
            ),
            "latency_seconds_by_replay": {
                kind: distribution(
                    [
                        row["elapsed_seconds"]
                        for row in selected
                        if bool(row["replayed_model_calls"]) == replay
                    ]
                )
                for kind, replay in (("fresh", False), ("contains_replay", True))
            },
        }
    return {
        "policy": POLICY,
        "evaluator_sha256": digest(Path(__file__)),
        "scope": "Descriptive development-set review; semantic judgments are reviewer-supplied. "
        "Tool proposals include blocked calls; observations include rejected/cached calls. "
        "Neither is a count of backend executions. Latency percentiles are descriptive only.",
        "sources": sources,
        "configuration": comparison,
        "skipped_cases": sorted(set(skipped_cases) - seen),
        "cases": rows,
        "groups": groups,
    }
