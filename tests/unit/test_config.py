from pathlib import Path

import pytest
from pydantic import ValidationError

from agentic_rag.core.config import Settings


def test_defaults_do_not_create_data_directory() -> None:
    settings = Settings()

    assert settings.environment == "local"
    assert settings.log_level == "INFO"
    assert settings.data_dir == Path("data")
    assert not settings.data_dir.exists()


def test_environment_values_are_loaded_and_paths_are_typed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_ENVIRONMENT", "test")
    monkeypatch.setenv("RAG_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("RAG_DATA_DIR", "custom/documents")

    settings = Settings()

    assert settings.environment == "test"
    assert settings.log_level == "DEBUG"
    assert settings.data_dir == Path("custom/documents")


def test_dotenv_values_are_loaded_and_unrelated_keys_are_ignored(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text(
        "RAG_ENVIRONMENT=test\nRAG_DATA_DIR=документы\nANOTHER_SERVICE_SETTING=value\n",
        encoding="utf-8",
    )

    settings = Settings()

    assert settings.environment == "test"
    assert settings.data_dir == Path("документы")


def test_environment_overrides_dotenv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("RAG_LOG_LEVEL=DEBUG\n", encoding="utf-8")
    monkeypatch.setenv("RAG_LOG_LEVEL", "ERROR")

    assert Settings().log_level == "ERROR"


def test_constructor_overrides_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_LOG_LEVEL", "ERROR")

    assert Settings(log_level="DEBUG").log_level == "DEBUG"


@pytest.mark.parametrize(
    ("variable", "value", "field"),
    [("RAG_LOG_LEVEL", "VERBOSE", "log_level"), ("RAG_ENVIRONMENT", "prod", "environment")],
)
def test_invalid_configuration_fails_early(
    monkeypatch: pytest.MonkeyPatch, variable: str, value: str, field: str
) -> None:
    monkeypatch.setenv(variable, value)

    with pytest.raises(ValidationError) as exc_info:
        Settings()

    assert exc_info.value.errors()[0]["loc"] == (field,)
