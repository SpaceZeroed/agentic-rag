from dataclasses import dataclass, field
from pathlib import Path
from random import Random
from uuid import UUID, uuid4

import pytest

from agentic_rag.ingestion.chunking import split_text
from agentic_rag.ingestion.models import ChunkingConfig, IngestResult, PreparedDocument
from agentic_rag.ingestion.parsing import DocumentInputError, parse_file
from agentic_rag.ingestion.service import ingest_file, prepare_document


@pytest.mark.parametrize("size,overlap", [(0, 0), (-1, 0), (10, -1), (10, 10), (10, 11)])
def test_invalid_chunking_configuration(size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        ChunkingConfig(size, overlap)


def test_chunk_slices_cover_text_with_exact_overlap_and_monotonic_progress() -> None:
    random = Random(42)
    texts = ["", "a", "x" * 173, "Документ\n\n😀e\u0301\t" * 20, " " * 70 + "end"]
    texts += ["".join(random.choices("abcабв😀\n \t", k=200)) for _ in range(10)]
    for text in texts:
        for size in (1, 2, 7, 31, 250):
            for overlap in sorted({0, size // 2, size - 1}):
                config = ChunkingConfig(size, overlap)
                revision = UUID(int=1)
                chunks = split_text(text, revision, config)
                assert chunks == split_text(text, revision, config)
                if not text:
                    assert chunks == ()
                    continue
                assert chunks[0].start_char == 0
                assert chunks[-1].end_char == len(text)
                assert len({chunk.id for chunk in chunks}) == len(chunks)
                rebuilt = chunks[0].text
                for index, chunk in enumerate(chunks):
                    assert chunk.position == index
                    assert 0 < len(chunk.text) <= size
                    assert chunk.text == text[chunk.start_char : chunk.end_char]
                    assert chunk.start_line == text.count("\n", 0, chunk.start_char) + 1
                    assert chunk.end_line == text.count("\n", 0, chunk.end_char - 1) + 1
                    if index:
                        previous = chunks[index - 1]
                        assert chunk.start_char == previous.end_char - overlap
                        assert chunk.start_char > previous.start_char
                        assert chunk.end_char > previous.end_char
                        rebuilt += chunk.text[overlap:]
                assert rebuilt == text


def test_paragraph_boundary_is_preferred_to_later_word_boundary() -> None:
    chunks = split_text("aaaa bbbb\n\ncccc dddd eeee", uuid4(), ChunkingConfig(18, 2))
    assert chunks[0].text == "aaaa bbbb\n\n"


def test_parser_preserves_markdown_indentation_and_normalizes_only_bom_and_newlines(
    tmp_path: Path,
) -> None:
    path = tmp_path / "пример.MD"
    path.write_bytes("\ufeff# Title\r\n\r\n```python\r    print('😀')  \r\n```\r\n".encode())
    parsed = parse_file(path)
    assert parsed.text == "# Title\n\n```python\n    print('😀')  \n```\n"
    assert parsed.title == "пример"
    assert parsed.media_type == "text/markdown"


@pytest.mark.parametrize("raw", [b"", b" \r\n\t", b"\xff", b"hello\x00world"])
def test_invalid_text_is_rejected(tmp_path: Path, raw: bytes) -> None:
    path = tmp_path / "input.txt"
    path.write_bytes(raw)
    with pytest.raises(DocumentInputError):
        parse_file(path)


def test_unsupported_and_non_regular_sources_are_rejected(tmp_path: Path) -> None:
    for path in (tmp_path / "paper.pdf", tmp_path / "missing.txt", tmp_path / "folder.txt"):
        if path.name == "folder.txt":
            path.mkdir()
        with pytest.raises(DocumentInputError):
            parse_file(path)


def test_file_limit_is_checked_before_decoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("agentic_rag.ingestion.parsing.MAX_FILE_BYTES", 10)
    path = tmp_path / "input.txt"
    path.write_bytes(b"a" * 11)
    with pytest.raises(DocumentInputError, match="file limit"):
        parse_file(path)


def test_source_content_and_processing_have_distinct_identities(tmp_path: Path) -> None:
    path = tmp_path / "paper.md"
    path.write_text("# Paper\n\nSome content.\n", encoding="utf-8")
    config = ChunkingConfig(15, 3)
    first = prepare_document(path, config)
    assert prepare_document(path, config) == first
    different_config = prepare_document(path, ChunkingConfig(16, 3))
    assert different_config.document_id == first.document_id
    assert different_config.revision_id != first.revision_id
    path.write_text("# Paper\n\nUpdated content.\n", encoding="utf-8")
    updated = prepare_document(path, config)
    assert updated.document_id == first.document_id
    assert updated.revision_id != first.revision_id
    assert {c.id for c in updated.chunks}.isdisjoint(c.id for c in first.chunks)
    separate = prepare_document(path, config, source_uri="https://example.org/paper")
    assert separate.document_id != updated.document_id


def test_same_canonical_text_can_have_distinct_raw_revisions(tmp_path: Path) -> None:
    path = tmp_path / "input.txt"
    path.write_bytes(b"one\r\ntwo")
    first = prepare_document(path, ChunkingConfig())
    path.write_bytes(b"one\ntwo")
    second = prepare_document(path, ChunkingConfig())
    assert first.content_sha256 == second.content_sha256
    assert first.raw_sha256 != second.raw_sha256
    assert first.revision_id != second.revision_id


@pytest.mark.parametrize("uri", ["", "relative/path", "https:///missing-host", "https://a/b c"])
def test_invalid_source_identity(tmp_path: Path, uri: str) -> None:
    with pytest.raises(DocumentInputError):
        prepare_document(tmp_path / "input.txt", ChunkingConfig(), source_uri=uri)


@dataclass
class RecordingWriter:
    saved: list[PreparedDocument] = field(default_factory=list)

    def save(self, document: PreparedDocument) -> IngestResult:
        self.saved.append(document)
        return IngestResult(
            document.document_id, document.revision_id, "created", len(document.chunks)
        )


def test_workflow_validates_before_calling_persistence(tmp_path: Path) -> None:
    writer = RecordingWriter()
    path = tmp_path / "input.txt"
    with pytest.raises(DocumentInputError):
        ingest_file(path, writer, ChunkingConfig())
    assert writer.saved == []
    path.write_text("Valid content", encoding="utf-8")
    result = ingest_file(path, writer, ChunkingConfig())
    assert result.document_id == writer.saved[0].document_id
    assert result.chunk_count == 1
