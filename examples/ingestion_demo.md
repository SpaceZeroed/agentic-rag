# Ingestion demonstration

This document was written for this repository to demonstrate text ingestion.
It is a local example, not a research paper or a labeled retrieval benchmark.
Its stable source identity can be supplied explicitly when running the CLI.

## Normalization

The parser reads UTF-8 text and normalizes line endings. It preserves Markdown
headings, links, indentation, and code. Chunk offsets refer to the normalized
text stored with the document revision. They count Unicode characters rather
than UTF-8 bytes: the symbol Ж occupies one character and more than one byte.

```python
text = "Документ"
assert len(text) == 8
```

## Repeated ingestion

Loading the same source with the same content and processing settings returns
the existing revision. Changing its content or chunk settings produces a new
revision while retaining the previous one. Each chunk belongs to one revision.

## Failure handling

A revision and all of its chunks are saved in one database transaction. If any
chunk write fails, the transaction rolls back and the previous current revision
remains available. Concurrent writes for the same source are serialized.
