"""Character windows with explicit boundary selection and exact overlap."""

from bisect import bisect_left
from uuid import UUID, uuid5

from agentic_rag.ingestion.models import Chunk, ChunkingConfig

CHUNKER_VERSION = "char-boundaries-v1"


def split_text(text: str, revision_id: UUID, config: ChunkingConfig) -> tuple[Chunk, ...]:
    """Cover all text with bounded slices; offsets count Unicode code points.

    Prefer paragraph, line, then word boundaries in the latter part of a window.
    Hard-split when no boundary fits. Exact overlap can split a word at the next
    chunk's start. This is not a Markdown AST or tokenizer-based splitter.
    """
    newlines = [index for index, character in enumerate(text) if character == "\n"]
    chunks: list[Chunk] = []
    start = 0
    while start < len(text):
        end = min(start + config.max_chars, len(text))
        if end < len(text):
            # A chosen chunk must be longer than overlap to guarantee progress.
            lower = start + max(config.overlap + 1, config.max_chars // 2)
            for separator in ("\n\n", "\n", " ", "\t"):
                boundary = text.rfind(separator, lower, end)
                if boundary != -1:
                    end = boundary + len(separator)
                    break
        position = len(chunks)
        chunks.append(
            Chunk(
                id=uuid5(revision_id, f"{position}:{start}:{end}"),
                position=position,
                text=text[start:end],
                start_char=start,
                end_char=end,
                start_line=bisect_left(newlines, start) + 1,
                end_line=bisect_left(newlines, end - 1) + 1,
            )
        )
        if end == len(text):
            break
        start = end - config.overlap
    return tuple(chunks)
