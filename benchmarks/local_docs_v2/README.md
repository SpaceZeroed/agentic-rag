# local_docs_v2 candidate

This frozen local candidate contains 30 snapshots of project documentation, 60 questions: 45 Russian answerable, 5 English answerable, and 10 Russian intended to be unanswerable. The 50 answerable cases have exact source excerpts. The source files and question labels were authored by the same assistant; there has been **no independent human review**. This set is for workflow development and human annotation review, not a held-out quality claim. Ten documents currently serve as distractors only. Most answerable questions are single-source fact lookup. Five bilingual pairs are dependent and five cases require two source documents; neither subgroup is large enough for a stable separate estimate.

The `corpus/` files are frozen copies, with SHA-256 in `dataset.json`. `source_uri` points to the original file at the pinned Git commit; the benchmark itself reads its frozen `corpus/` copies. Do not silently change the snapshots or labels after observing results; create a new version for substantive changes. `scripts/build_local_docs_v2.py` records the construction. Since it rejects an existing output directory, rerunning requires a new version name.

Before using this as a decision set, a human reviewer should check each evidence excerpt against the whole corpus, identify omitted relevant passages, verify that the 10 refusal cases really have no answer, and rewrite weak or leading questions. A separate held-out split must be annotated independently. The current loader only proves that each excerpt occurs in a chunk; it cannot establish semantic correctness or complete qrels.

Validate without a database:

```bash
.venv/bin/python -c 'from pathlib import Path; from agentic_rag.evaluation.rag_dataset import load_rag_dataset; d, docs = load_rag_dataset(Path("benchmarks/local_docs_v2/dataset.json")); print(len(d.documents), len(d.questions), sum(len(x.chunks) for x in docs.values()))'
```

Get BM25 retrieval/context metrics without PostgreSQL, Qdrant, or an API key:

```bash
uv run --locked python scripts/evaluate_rag_offline.py \
  benchmarks/local_docs_v2/dataset.json \
  --output artifacts/local_docs_v2_bm25.json
```

This uses one in-memory index over all frozen chunks. The regular CLI currently
rebuilds the same BM25 statistics from PostgreSQL for every query, so rankings and
evidence metrics are equivalent; timings are not comparable.

## Preliminary BM25 result (2026-09-24)

The saved `bm25_fake.json` report covers all 60 cases with no workflow errors.
Metrics below use only the 50 answerable questions; the ten unanswerable questions
have no retrieval qrels and are excluded by design.

| Group | Cases | Recall@5 | MRR@5 | nDCG@5 |
| --- | ---: | ---: | ---: | ---: |
| All answerable | 50 | 0.720 | 0.565 | 0.595 |
| Russian answerable | 45 | 0.756 | 0.598 | 0.628 |
| English answerable | 5 | 0.400 | 0.267 | 0.300 |
| Single-source | 45 | 0.733 | 0.574 | 0.613 |
| Multi-source | 5 | 0.600 | 0.483 | 0.428 |

These subgroup samples are small and dependent, especially the five English and
five multi-source cases. Twelve questions had zero labeled recall and four
multi-source questions recovered one of two evidence spans. The ten unanswerable
queries all retrieved five chunks; this is expected for top-k BM25 and does not
measure whether a real model would refuse. The fake generator answered whenever it
received context, so its 0/10 structured refusals are a fixture behavior, not a
generation-quality result.

The label audit also found that four short evidence strings (`Python 3.12+`,
`PostgreSQL.`, `через RRF.`, and `` `Decimal` ``) occur in more than one corpus
document. Their document-specific qrels may therefore treat a substantively useful
duplicate as nonrelevant. This candidate needs exhaustive human qrel review before
the numbers are used for model selection; do not tune against these preliminary
scores.

Run the existing RAG evaluator with a configured PostgreSQL instance:

```bash
uv run --locked --extra embeddings agentic-rag evaluate-rag benchmarks/local_docs_v2/dataset.json --mode bm25 --llm fake --output artifacts/local_docs_v2_smoke.json
```

The fake LLM run checks the workflow only. For answer quality, use a compatible LLM and independently review its output with `review-rag`.

The paired Qwen 3.6 / GLM 5.3 Flash run, including latency, usage, strict refusal
behavior and report fingerprints, is documented in [model_comparison.md](model_comparison.md).
