"""JSON logging using Python's standard logger/handler/formatter pipeline."""

import json
import logging
import sys
from datetime import UTC, datetime

from agentic_rag.core.config import LogLevel


class JsonFormatter(logging.Formatter):
    """Emit one JSON object per line, keeping custom fields in their own namespace."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        fields = getattr(record, "fields", None)
        if fields is not None:
            payload["fields"] = fields
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(level: LogLevel = "INFO") -> None:
    """Configure the process root logger at startup, replacing existing handlers.

    Call only from an executable entry point that owns logging configuration.
    Imported library modules should just use logging.getLogger(__name__).
    """
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter())
    logging.basicConfig(level=level, handlers=[handler], force=True)
