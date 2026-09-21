import json
from pathlib import Path
from typing import Any

import pytest

from agentic_rag.evaluation.agent_review import (
    DIMENSIONS,
    digest,
    make_template,
    summarize_reviews,
    write_json,
)


def fixture_run(root: Path) -> tuple[Path, Path, Path]:
    run = root / "run"
    run.mkdir()
    write_json(run / "manifest.json", {"selected_cases": ["case"], "model": "fake"})
    summary = {
        "id": "case",
        "group": "controlled",
        "status": "limit_reached",
        "model_calls": 2,
        "tool_calls": 1,
        "repair_attempts": 0,
        "elapsed_seconds": 1.5,
    }
    turns = [
        {
            "origin": origin,
            "assistant_message": {
                "tool_calls": [
                    {
                        "id": "same_id",
                        "function": {"name": "document_catalog", "arguments": arguments},
                    }
                ]
            },
        }
        for origin, arguments in (("replayed", "{"), ("live", '{"limit": 3}'))
    ]
    write_json(
        run / "case.json",
        {
            "summary": summary,
            "manifest_sha256": digest(run / "manifest.json"),
            "model_turns": turns,
            "result": {"observations": [{"status": "error", "error": "invalid_arguments"}]},
        },
    )
    write_json(run / "summary.json", {"cases": [summary], "skipped_cases": []})
    review = make_template(run).model_dump()
    review.update(reviewer="test reviewer", reviewer_type="human")
    case = review["cases"][0]
    case["selection_complete"] = {"value": True, "rationale": "Required catalog proposed."}
    for proposal in case["proposals"]:
        for dimension in ("selection", "arguments_correct", "unnecessary"):
            proposal[dimension] = {
                "value": dimension == "selection"
                or (dimension == "arguments_correct" and proposal["origin"] == "live"),
                "rationale": "Explicit test judgment.",
            }
    tool = root / "tool.json"
    write_json(tool, review)
    quality = root / "quality.json"
    write_json(
        quality,
        {
            "manifest_sha256": digest(run / "manifest.json"),
            "reviewer": "test reviewer",
            "reviewer_type": "human",
            "cases": [
                {
                    "id": "case",
                    "report_sha256": digest(run / "case.json"),
                    "dimensions": {
                        name: {
                            "applicable": name != "claim_support",
                            "score": None if name == "claim_support" else 1,
                            "rationale": "Explicit test judgment.",
                        }
                        for name in DIMENSIONS
                    },
                }
            ],
        },
    )
    return run, tool, quality


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())  # type: ignore[no-any-return]


def replace(path: Path, value: object) -> None:
    path.write_text(json.dumps(value))


def test_replay_blocked_proposals_and_na_are_not_conflated(tmp_path: Path) -> None:
    run, tool, quality = fixture_run(tmp_path)
    result = summarize_reviews([run], [tool], [quality])
    group = result["groups"]["controlled"]
    assert group["proposal_metrics_by_origin"]["live"]["arguments_correct"]["rate"] == 1
    assert group["proposal_metrics_by_origin"]["replayed"]["arguments_correct"]["rate"] == 0
    assert group["proposal_metrics_by_origin"]["mock"]["selection"]["rate"] is None
    assert group["tool_observations"]["mean"] == 1
    assert len(result["cases"][0]["proposals"]) == 2
    assert group["quality"]["claim_support"]["mean_score"] is None
    assert group["latency_seconds_by_replay"]["fresh"]["n"] == 0
    assert group["latency_seconds_by_replay"]["contains_replay"]["n"] == 1
    assert result["groups"]["natural"]["selection_complete"]["rate"] is None


@pytest.mark.parametrize("mutation", ["hash", "missing", "duplicate", "identity", "unscored"])
def test_rejects_incomplete_or_stale_tool_review(tmp_path: Path, mutation: str) -> None:
    run, tool, quality = fixture_run(tmp_path)
    review = read(tool)
    case = review["cases"][0]
    if mutation == "hash":
        case["report_sha256"] = "bad"
    elif mutation == "missing":
        case["proposals"].pop()
    elif mutation == "duplicate":
        review["cases"].append(case)
    elif mutation == "identity":
        case["proposals"][1]["arguments"] = "{}"
    else:
        case["proposals"][1]["selection"]["value"] = None
    replace(tool, review)
    with pytest.raises(ValueError):
        summarize_reviews([run], [tool], [quality])


@pytest.mark.parametrize("mutation", ["hash", "na", "bool_score", "missing", "reviewer"])
def test_quality_review_validation(tmp_path: Path, mutation: str) -> None:
    run, tool, quality = fixture_run(tmp_path)
    review = read(quality)
    if mutation == "hash":
        review["cases"][0]["report_sha256"] = "bad"
    elif mutation == "na":
        review["cases"][0]["dimensions"]["claim_support"]["score"] = 0
    elif mutation == "bool_score":
        review["cases"][0]["dimensions"]["task_completion"]["score"] = True
    elif mutation == "missing":
        review["cases"] = []
    else:
        review["reviewer"] = None
    replace(quality, review)
    with pytest.raises(ValueError):
        summarize_reviews([run], [tool], [quality])


def test_duplicate_runs_and_changed_reports_fail(tmp_path: Path) -> None:
    run, tool, quality = fixture_run(tmp_path)
    with pytest.raises(ValueError, match="Duplicate case"):
        summarize_reviews([run, run], [tool, tool], [quality, quality])
    report = read(run / "case.json")
    report["extra"] = "changed"
    replace(run / "case.json", report)
    with pytest.raises(ValueError, match="hash mismatch"):
        summarize_reviews([run], [tool], [quality])


def test_output_cannot_overwrite_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "output.json"
    write_json(path, {"original": True})
    with pytest.raises(FileExistsError):
        write_json(path, {"original": False})
    assert read(path) == {"original": True}


def test_template_has_no_invented_scores(tmp_path: Path) -> None:
    run, _, _ = fixture_run(tmp_path)
    review = make_template(run)
    assert review.reviewer is None
    assert review.cases[0].selection_complete.value is None
    assert all(p.selection.value is None for p in review.cases[0].proposals)


def test_no_proposals_does_not_hide_missing_required_tool(tmp_path: Path) -> None:
    run, tool, quality = fixture_run(tmp_path)
    report = read(run / "case.json")
    for turn in report["model_turns"]:
        turn["assistant_message"] = {"content": "No tools"}
    replace(run / "case.json", report)
    review = read(tool)
    review["cases"][0]["proposals"] = []
    review["cases"][0]["selection_complete"] = {
        "value": False,
        "rationale": "Required catalog omitted.",
    }
    review["cases"][0]["report_sha256"] = digest(run / "case.json")
    replace(tool, review)
    prior = read(quality)
    prior["cases"][0]["report_sha256"] = digest(run / "case.json")
    replace(quality, prior)
    result = summarize_reviews([run], [tool], [quality])["groups"]["controlled"]
    assert result["selection_complete"]["rate"] == 0
    assert result["proposal_metrics_by_origin"]["live"]["selection"]["rate"] is None


def test_different_configuration_cannot_be_pooled(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    run1, tool1, quality1 = fixture_run(first)
    run2, tool2, quality2 = fixture_run(second)
    manifest = read(run2 / "manifest.json")
    manifest["model"] = "different"
    replace(run2 / "manifest.json", manifest)
    report = read(run2 / "case.json")
    report["manifest_sha256"] = digest(run2 / "manifest.json")
    replace(run2 / "case.json", report)
    with pytest.raises(ValueError, match="different experimental"):
        summarize_reviews([run1, run2], [tool1, tool2], [quality1, quality2])
