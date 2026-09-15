"""Conservative parsing of local UTF-8 Markdown and plain text."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

PARSER_VERSION = "utf8-newlines-v1"
MAX_FILE_BYTES = 10 * 1024 * 1024


class DocumentInputError(ValueError):
    """The source cannot be represented by this text ingestion pipeline."""


@dataclass(frozen=True)
class ParsedText:
    title: str
    media_type: str
    raw_sha256: str
    content_sha256: str
    text: str


def parse_file(path: Path) -> ParsedText:
    media_types = {".md": "text/markdown", ".txt": "text/plain"}
    media_type = media_types.get(path.suffix.lower())
    if media_type is None:
        raise DocumentInputError("Only .md and .txt files are supported")
    if not path.is_file():
        raise DocumentInputError("Source must be a regular file")
    with path.open("rb") as stream:
        raw = stream.read(MAX_FILE_BYTES + 1)
    if len(raw) > MAX_FILE_BYTES:
        raise DocumentInputError("Source exceeds the 10 MiB file limit")
    try:
        text = raw.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    except UnicodeDecodeError as exc:
        raise DocumentInputError("Source must be valid UTF-8") from exc
    if "\x00" in text:
        raise DocumentInputError("NUL characters are not supported")
    if not text.strip():
        raise DocumentInputError("Source contains no non-whitespace text")
    return ParsedText(
        title=path.stem,
        media_type=media_type,
        raw_sha256=sha256(raw).hexdigest(),
        content_sha256=sha256(text.encode("utf-8")).hexdigest(),
        text=text,
    )
