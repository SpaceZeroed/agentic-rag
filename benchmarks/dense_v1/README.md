# dense-v1 authored development set

10 English reading notes, 20 chunks (`max_chars=400`, `overlap=60`), 24 questions
(12 English, 12 Russian). Twenty questions target one evidence passage; four ask
for evidence from two notes. Paired translations are not independent examples.

The assistant authored notes, questions and evidence spans together **before**
the first retrieval run. Notes paraphrase the primary papers linked in each file;
they are not paper downloads. URI fragments identify these local notes. Files and
labels were not changed after observing retrieval scores. Human review is pending.

All chunks fully containing an annotated evidence span receive binary relevance.
Unjudged chunks score as nonrelevant, even if another passage could be useful.
There are no unanswerable cases and no held-out split. Use for workflow verification
and a later same-set development comparison, not generalization claims.

`dataset.json` pins source SHA-256 and chunking. Changes require a new dataset
version; never edit labels to improve an existing run. The saved report includes
ranked chunks, scores, qrels, per-case metrics, model commit and package versions.

Run from repository root after configuring PostgreSQL and Qdrant:

```bash
uv run --locked --extra embeddings agentic-rag evaluate benchmarks/dense_v1/dataset.json \
  --collection-prefix rag_dense_v1 --output artifacts/dense-v1-repeat.json
```

Keep the first measured report in `results/e5_cpu.json`; use another output path
for repeats. Repeated evaluation activates these fixed revisions in PostgreSQL
and uses only these documents. It is a data-writing evaluation command.
See [results](../../docs/results.md) and the [guide](../../docs/dense_retrieval.md).
