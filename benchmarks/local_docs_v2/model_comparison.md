# Qwen 3.6 vs GLM 5.3 Flash

Paired development run on 2026-09-24 through the Rus-GPT Chat Completions API.
This is an operational comparison on an assistant-authored candidate dataset,
not an independently reviewed model-quality benchmark.

Both models received byte-identical user messages built from the same DB-backed
BM25 hits: 60 questions, `k=5`, 24,000 UTF-8 prompt bytes, `max_tokens=4096`,
temperature 0, reasoning requested disabled. Runs were sequential, Qwen first,
with no retries. All 120 HTTP requests returned 200.

| Observed measure | Qwen 3.6 35B A3B | GLM 5.3 Flash |
| --- | ---: | ---: |
| Application errors | 2 | 1 |
| Answered / strict refusal / error | 47 / 11 / 2 | 49 / 10 / 1 |
| Strict refusals on 10 unanswerable cases | 10/10 | 8/10 |
| Answerable cases returned as strict refusal | 1 | 2 |
| Median generation workflow latency | 2.6210 s | 1.9258 s |
| Mean generation workflow latency | 3.1200 s | 2.0588 s |
| P95 generation workflow latency | 5.3543 s | 3.2710 s |
| Provider input tokens | 160,421 | 150,296 |
| Provider output tokens | 3,674 | 4,264 |
| Provider-reported cost | 3.63093 RUB | 2.46764 RUB |
| Returned reasoning characters | 0 | 1,733 |

GLM was 26.5% faster by median, 34.0% faster by mean, and 32.0% cheaper in this
single sequential run. Token counts differ because they are provider/model-tokenizer
usage, even though the messages were identical. Latencies include network/provider
conditions and are not a load test.

The two GLM failures on unanswerable cases were semantically worded refusals, but
they appended cited explanations after `INSUFFICIENT_EVIDENCE`. The strict service
therefore classified them as answered. Both models failed `answer_15` with
`CitationError` after repeating a source fragment containing the literal malformed
marker `` `[C` ``; Qwen also failed `multi_03` for the same reason. This exposes a
collision between documentation about citation syntax and the validator.

No semantic correctness or faithfulness winner is claimed. Report-bound review
templates were generated but have not been completed independently. A safe refusal
after a retrieval miss and an unsupported answer must not be equated using only the
dataset-level `answerable` flag.

Raw reports remain local because they duplicate about 7 MB of frozen source/context
text under ignored `artifacts/`. Their SHA-256 fingerprints are:

- Qwen: `0e5576dd5741a163190c00faffb90d62f48b3dc44c72ff654fe7d26dd0a34b08`
- GLM: `2b8a872a9d9276b28081640d21eff8fa14e15d887591c50f66d175329b8b059f`
- DB-backed fake smoke: `b58b709acd128e88b49076fd4e0559c117673ee9cece36caa64b782739050ca3`

Retrieval metrics were identical by construction: Recall@5 0.720, MRR@5 0.565,
and nDCG@5 0.595 on 50 answerable cases. See the [dataset README](README.md) for
label limitations and subgroup results.
