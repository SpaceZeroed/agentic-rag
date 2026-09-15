"""Stage 0 startup check; this does not start a service or run retrieval."""

import logging

from pydantic import ValidationError

from agentic_rag.core.config import Settings
from agentic_rag.core.logging import configure_logging

logger = logging.getLogger(__name__)


def main() -> int:
    """Validate configuration, initialize logs, and report startup status."""
    try:
        settings = Settings()
    except ValidationError as exc:
        configure_logging("ERROR")
        # Report field names and error types without echoing external input.
        errors = [
            {"field": ".".join(str(part) for part in error["loc"]), "type": error["type"]}
            for error in exc.errors(include_url=False, include_context=False, include_input=False)
        ]
        logger.error("configuration_invalid", extra={"fields": {"errors": errors}})
        return 2

    configure_logging(settings.log_level)
    logger.info(
        "application_ready",
        extra={
            "fields": {
                "environment": settings.environment,
                "data_dir": str(settings.data_dir),
            }
        },
    )
    return 0
