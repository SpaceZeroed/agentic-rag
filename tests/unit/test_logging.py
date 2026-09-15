import io
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

from agentic_rag.core.logging import JsonFormatter


def test_json_log_preserves_types_and_protects_standard_fields() -> None:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    logger = logging.Logger("test.logger")
    logger.addHandler(handler)

    logger.info(
        "processed %s documents",
        3,
        extra={"fields": {"count": 3, "cached": False, "level": "custom", "path": Path("data")}},
    )

    event = json.loads(stream.getvalue())
    assert event["level"] == "INFO"
    assert event["logger"] == "test.logger"
    assert event["message"] == "processed 3 documents"
    assert event["fields"] == {"count": 3, "cached": False, "level": "custom", "path": "data"}
    assert datetime.fromisoformat(event["timestamp"]).utcoffset() == timedelta(0)


def test_unicode_and_newlines_still_produce_one_json_line() -> None:
    record = logging.LogRecord("test", logging.INFO, __file__, 1, "Документ\nnext line", (), None)

    encoded = JsonFormatter().format(record)

    assert len(encoded.splitlines()) == 1
    assert json.loads(encoded)["message"] == "Документ\nnext line"


def test_exception_traceback_is_included_in_json() -> None:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    logger = logging.Logger("test.exception")
    logger.addHandler(handler)

    try:
        raise ValueError("invalid document")
    except ValueError:
        logger.exception("processing_failed")

    event = json.loads(stream.getvalue())
    assert event["level"] == "ERROR"
    assert event["message"] == "processing_failed"
    assert "Traceback" in event["exception"]
    assert "ValueError: invalid document" in event["exception"]
    assert len(stream.getvalue().splitlines()) == 1
