"""Generate local-only Langfuse credentials once; never print or overwrite secrets."""

import os
import secrets
from pathlib import Path


def main() -> None:
    target = Path(__file__).resolve().parents[1] / ".env.langfuse"
    values = {
        name: secrets.token_hex(32)
        for name in (
            "LF_POSTGRES_PASSWORD",
            "LF_AUTH_SECRET",
            "LF_SALT",
            "LF_ENCRYPTION_KEY",
            "LF_CLICKHOUSE_PASSWORD",
            "LF_REDIS_PASSWORD",
            "LF_S3_PASSWORD",
            "LF_LOGIN_PASSWORD",
        )
    }
    values.update(
        {
            "RAG_OBSERVABILITY_ENABLED": "true",
            "RAG_LANGFUSE_ENABLED": "true",
            "RAG_LANGFUSE_BASE_URL": "http://127.0.0.1:3000",
            "RAG_LANGFUSE_PUBLIC_KEY": "pk-lf-" + secrets.token_hex(16),
            "RAG_LANGFUSE_SECRET_KEY": "sk-lf-" + secrets.token_hex(32),
        }
    )
    with os.fdopen(os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as stream:
        stream.write("# Local credentials. Ignored by Git. Do not share.\n")
        stream.writelines(f"{key}={value}\n" for key, value in values.items())
    print(f"Created {target.name} (mode 600); existing .env unchanged")


if __name__ == "__main__":
    main()
