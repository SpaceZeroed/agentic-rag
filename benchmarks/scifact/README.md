# BEIR SciFact external retrieval check

Source: [official BEIR SciFact](https://github.com/beir-cellar/beir), test qrels. Download the official archive and run:

```bash
curl -L --fail -o /tmp/scifact.zip https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip
.venv/bin/python scripts/evaluate_scifact.py --archive /tmp/scifact.zip --output artifacts/scifact_bm25_v1.json
```

The 2026-09-23 run used archive SHA-256 `536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165`: 5,183 documents and 300 test queries. It evaluated the project's `BM25Index` with `k1=1.2`, `b=0.75`, and its Unicode tokenizer. Each SciFact title and text became **one searchable document**. Thus this checks lexical ranking on an external corpus, but does not exercise application ingestion, chunking, PostgreSQL, Qdrant, RRF, reranking, or answer generation. The full per-query report is in ignored local `artifacts/scifact_bm25_v1.json`.

| Metric | @1 | @3 | @5 | @10 |
| --- | ---: | ---: | ---: | ---: |
| Recall | 0.515 | 0.679 | 0.726 | 0.791 |
| MRR | 0.533 | 0.608 | 0.620 | 0.628 |
| nDCG | 0.533 | 0.620 | 0.639 | 0.662 |

These are averages over 300 queries. BEIR qrels may be incomplete; unjudged results are scored nonrelevant. Do not compare these document-level numbers directly with the project's chunk-level `dense_v1` results.
