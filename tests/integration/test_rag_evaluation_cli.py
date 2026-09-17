import json
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import Engine

from agentic_rag.cli import main

DATASET = Path(__file__).resolve().parents[2] / "benchmarks/rag_v1/dataset.json"


@pytest.mark.postgres
def test_evaluation_bm25_and_review_template(
    database: Engine,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("RAG_DATABASE_URL", database.url.render_as_string(hide_password=False))
    path = tmp_path / "run.json"
    assert main(["evaluate-rag", str(DATASET), "--output", str(path)]) == 0
    report: Any = json.loads(path.read_text())
    assert report["provider"] == "fake"
    assert len(report["cases"]) == 10
    assert len(report["documents"]) == 10
    assert report["errors"] == 0
    assert report["answer_quality"] is None
    before = path.read_bytes()
    assert main(["evaluate-rag", str(DATASET), "--output", str(path)]) == 2
    assert path.read_bytes() == before
    template = tmp_path / "review.json"
    assert main(["review-rag", str(path), "--output", str(template)]) == 0
    assert len(json.loads(template.read_text())["cases"]) == 10
    assert (
        main(
            [
                "review-rag",
                str(path),
                "--annotations",
                str(template),
                "--output",
                str(tmp_path / "quality.json"),
            ]
        )
        == 2
    )
    capsys.readouterr()


@pytest.mark.parametrize(
    "args",
    [
        ["--k", "0"],
        ["--candidate-k", "1"],
        ["--rerank", "--rerank-k", "1"],
        ["--llm", "compatible"],
    ],
)
def test_invalid_evaluation_config(tmp_path: Path, args: list[str]) -> None:
    assert main(["evaluate-rag", str(DATASET), "--output", str(tmp_path / "run.json"), *args]) == 2


def test_review_scoring_cli_without_database(tmp_path: Path) -> None:
    from agentic_rag.evaluation.rag_review import review_template

    report: dict[str, object] = {
        "schema_version": 1,
        "provider": "compatible",
        "cases": [
            {
                "id": "absent",
                "answerable": False,
                "answer": {"status": "insufficient_evidence", "text": "No evidence"},
                "error": None,
            },
        ],
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report))
    template: Any = review_template(report)
    template["reviewer"] = "test-only"
    template["cases"][0].update(correctness=2, rationale="Absent from corpus")
    annotation_path = tmp_path / "review.json"
    annotation_path.write_text(json.dumps(template))
    output = tmp_path / "score.json"
    assert (
        main(
            [
                "review-rag",
                str(report_path),
                "--annotations",
                str(annotation_path),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    summary = json.loads(output.read_text())
    assert summary["correctness_all_cases_errors_zero"] == 1
    assert summary["faithfulness_micro"] is None
