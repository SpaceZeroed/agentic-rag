"""Validated configuration, loaded explicitly at an application boundary."""

from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

type LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class Settings(BaseSettings):
    """Precedence: constructor arguments > environment > .env > defaults.

    The optional .env file and relative paths are resolved from the working
    directory. Constructing settings validates values but creates no directories.
    Unknown dotenv keys are ignored so a file can be shared with future services.
    """

    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    environment: Literal["local", "test", "production"] = "local"
    log_level: LogLevel = "INFO"
    data_dir: Path = Path("data")
    database_url: SecretStr | None = None
