import json
import subprocess
import sys
from pathlib import Path

import pytest


def run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "agentic_rag", *arguments],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )


def test_preview_works_without_database_and_exposes_traceable_slices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAG_DATABASE_URL", "not-a-database")
    path = tmp_path / "paper.md"
    path.write_bytes("# Пример\r\n\r\nText with a source.\r\n".encode())
    result = run_cli("preview", str(path), "--max-chars", "15", "--overlap", "3")
    assert result.returncode == 0, result.stderr
    document = json.loads(result.stdout)
    assert document["source_uri"] == path.as_uri()
    for chunk in document["chunks"]:
        assert document["text"][chunk["start_char"] : chunk["end_char"]] == chunk["text"]


@pytest.mark.parametrize(
    "arguments", [("--max-chars", "0"), ("--overlap", "1200"), ("--source-uri", "relative")]
)
def test_cli_rejects_invalid_ingestion_inputs(tmp_path: Path, arguments: tuple[str, str]) -> None:
    path = tmp_path / "paper.txt"
    path.write_text("Text", encoding="utf-8")
    result = run_cli("ingest", str(path), *arguments)
    assert result.returncode == 2
    assert json.loads(result.stderr)["message"] == "input_invalid"


def test_database_commands_require_explicit_configuration() -> None:
    result = run_cli("db-upgrade")
    assert result.returncode == 2
    assert json.loads(result.stderr)["message"] == "database_url_required"


@pytest.mark.parametrize(
    "url",
    [
        "sqlite://user:secret-value@example/db",
        "secret-value-not-a-url",
        "postgresql+psycopg://user:secret-value@localhost:invalid/db",
    ],
)
def test_invalid_database_url_does_not_echo_credentials(
    monkeypatch: pytest.MonkeyPatch, url: str
) -> None:
    monkeypatch.setenv("RAG_DATABASE_URL", url)
    result = run_cli("db-upgrade")
    assert result.returncode == 2
    assert "secret-value" not in result.stdout + result.stderr
