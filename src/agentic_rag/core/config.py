"""Validated configuration, loaded explicitly at an application boundary."""

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
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
    qdrant_url: str = "http://127.0.0.1:6333"
    qdrant_api_key: SecretStr | None = None
    collection_prefix: str = "rag_dense"
    embedding_model: str = "intfloat/multilingual-e5-small"
    embedding_revision: str = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
    embedding_batch_size: int = 8
    embedding_threads: int = 4
    model_local_files_only: bool = False
    reranking_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    reranking_revision: str = "1427fd652930e4ba29e8149678df786c240d8825"
    reranking_batch_size: int = 4
    reranking_threads: int = 4

    llm_base_url: str = "http://127.0.0.1:8000/v1"
    llm_model: str | None = None
    llm_reasoning_enabled: bool | None = None
    llm_api_key: SecretStr | None = None
    llm_timeout_seconds: float = Field(default=60, gt=0, allow_inf_nan=False)
    llm_max_prompt_bytes: int = Field(default=24000, ge=1)
    llm_max_tokens: int = Field(default=512, ge=1)

    api_llm_provider: Literal["fake", "compatible"] = "fake"
    api_retrieval_mode: Literal["bm25", "dense", "hybrid"] = "bm25"
    api_rerank: bool = False
    api_candidate_k: int = Field(default=20, ge=1, le=100)
    api_max_concurrent_requests: int = Field(default=8, ge=1, le=100)
    api_request_timeout_seconds: float = Field(default=180, gt=0, allow_inf_nan=False)
    api_max_body_bytes: int = Field(default=2 * 1024 * 1024, ge=1)
