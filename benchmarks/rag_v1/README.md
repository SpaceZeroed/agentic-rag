# RAG development set v1

10 assistant-authored EN/RU questions (5 each) over the existing 10-note corpus:
8 answerable including two bilingual LoRA/QLoRA comparison cases, and 2 questions
whose requested user-specific facts are absent from the corpus. Chunking is 400/60.

References/evidence were fixed before runs. This is a development set, pending
human label review, not held-out quality evidence. Corpus SHA-256 checksums are
validated; earlier dense_v1 questions and judgments are untouched.

Run and review instructions, metric formulas, rubric, denominators and limitations:
[Stage 6 guide](../../docs/rag_evaluation.md).

Results named bm25_fake*.json measure retrieval/context with real PostgreSQL and
mock generation; they are not LLM quality results. Semantic quality is null until
real generation and explicit human review. Do not overwrite reports or tune these
labels to generated outputs; create a new dataset version for substantive changes.
