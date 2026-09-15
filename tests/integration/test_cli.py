import json
import shutil
import subprocess
import sys

import pytest


@pytest.mark.parametrize("entry_point", ["module", "console"])
def test_installed_entry_points_work_outside_repository(entry_point: str) -> None:
    executable = shutil.which("agentic-rag")
    assert executable is not None, "Install the project with uv sync before testing."
    command = [sys.executable, "-m", "agentic_rag"] if entry_point == "module" else [executable]

    result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=10)

    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    event = json.loads(result.stderr)
    assert event["message"] == "application_ready"
    assert event["fields"]["environment"] == "local"


def test_cli_uses_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_ENVIRONMENT", "test")

    result = subprocess.run(
        [sys.executable, "-m", "agentic_rag"],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )

    assert result.returncode == 0
    assert json.loads(result.stderr)["fields"]["environment"] == "test"


def test_cli_configuration_failure_has_nonzero_status_without_echoing_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invalid_value = "invalid-sensitive-value"
    monkeypatch.setenv("RAG_LOG_LEVEL", invalid_value)

    result = subprocess.run(
        [sys.executable, "-m", "agentic_rag"],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )

    assert result.returncode == 2
    assert invalid_value not in result.stdout + result.stderr
    event = json.loads(result.stderr)
    assert event["message"] == "configuration_invalid"
    assert event["fields"]["errors"] == [{"field": "log_level", "type": "literal_error"}]


def test_log_level_suppresses_info_events(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_LOG_LEVEL", "WARNING")

    result = subprocess.run(
        [sys.executable, "-m", "agentic_rag"],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )

    assert result.returncode == 0
    assert result.stdout == result.stderr == ""


def test_logging_reconfiguration_does_not_duplicate_events() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import logging\n"
            "from agentic_rag.core.logging import configure_logging\n"
            "configure_logging()\n"
            "configure_logging()\n"
            "logging.getLogger('agentic_rag').info('once')\n",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )

    assert result.returncode == 0
    assert len(result.stderr.splitlines()) == 1
    assert json.loads(result.stderr)["message"] == "once"


def test_import_does_not_load_settings_or_configure_logging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAG_LOG_LEVEL", "invalid")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import logging\nimport agentic_rag.cli\nassert not logging.getLogger().handlers\n",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == result.stderr == ""
