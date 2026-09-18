"""Prepare deterministic revisions before opening a persistence transaction."""

import json
from pathlib import Path
from urllib.parse import urlsplit
from uuid import NAMESPACE_URL, uuid5

from agentic_rag.ingestion.chunking import CHUNKER_VERSION, split_text
from agentic_rag.ingestion.models import (
    ChunkingConfig,
    DocumentWriter,
    IngestResult,
    PreparedDocument,
)
from agentic_rag.ingestion.parsing import PARSER_VERSION, DocumentInputError, ParsedText, parse_file


def prepare_document(
    path: Path, config: ChunkingConfig, *, source_uri: str | None = None
) -> PreparedDocument:
    path = path.resolve()
    uri = path.as_uri() if source_uri is None else source_uri
    return prepare_parsed(parse_file(path), config, source_uri=uri)


def prepare_parsed(
    parsed: ParsedText, config: ChunkingConfig, *, source_uri: str
) -> PreparedDocument:
    uri = source_uri
    if len(uri.encode("utf-8")) > 2048:
        raise DocumentInputError("Source URI exceeds the 2048-byte limit")
    try:
        parsed_uri = urlsplit(uri)
    except ValueError as exc:
        raise DocumentInputError("Invalid source URI") from exc
    if not (
        (parsed_uri.scheme in {"http", "https"} and parsed_uri.netloc)
        or (parsed_uri.scheme == "file" and parsed_uri.path.startswith("/"))
    ) or any(character.isspace() for character in uri):
        raise DocumentInputError("Source URI must be an absolute file, HTTP or HTTPS URI")
    document_id = uuid5(NAMESPACE_URL, uri)
    identity = json.dumps(
        {
            "raw_sha256": parsed.raw_sha256,
            "content_sha256": parsed.content_sha256,
            "title": parsed.title,
            "media_type": parsed.media_type,
            "parser": PARSER_VERSION,
            "chunker": CHUNKER_VERSION,
            "max_chars": config.max_chars,
            "overlap": config.overlap,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    revision_id = uuid5(document_id, identity)
    return PreparedDocument(
        document_id=document_id,
        revision_id=revision_id,
        source_uri=uri,
        title=parsed.title,
        media_type=parsed.media_type,
        raw_sha256=parsed.raw_sha256,
        content_sha256=parsed.content_sha256,
        text=parsed.text,
        parser_version=PARSER_VERSION,
        chunker_version=CHUNKER_VERSION,
        config=config,
        chunks=split_text(parsed.text, revision_id, config),
    )


def ingest_file(
    path: Path,
    writer: DocumentWriter,
    config: ChunkingConfig,
    *,
    source_uri: str | None = None,
) -> IngestResult:
    return writer.save(prepare_document(path, config, source_uri=source_uri))
