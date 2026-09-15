"""Keep tests independent of the developer's environment and local .env file."""

import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolated_configuration(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for key in os.environ:
        if key.upper().startswith("RAG_"):
            monkeypatch.delenv(key)
    monkeypatch.chdir(tmp_path)
