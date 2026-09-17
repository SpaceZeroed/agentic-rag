# Qwen3.6 vs Haiku 4.5 — controlled development comparison

2026-09-17, Rus-GPT Chat Completions API. Decision for this project: retain
`qwen/qwen3.6-35b-a3b`, requesting `reasoning.enabled=false`.
This is a 10-case authored development experiment, not a model leaderboard.

Both models received byte-identical messages: saved BM25 hits from the original
`../rusgpt_qwen36_bm25.json`, same source order, same user JSON, and the same updated
system instruction requiring separate `[C1] [C2]` markers. Exact equality verified.
Temperature=0, max_tokens=4096, prompt budget=24000 UTF-8 bytes, HTTP timeout=120s.
No retrieval/DB work during replay. Runs were serial, Qwen then Haiku; no retries.
`retrieval_seconds` measures dictionary lookup, not real retrieval latency.
Haiku used provider defaults; Qwen requested reasoning=false. Neither returned
reasoning text in any of the ten responses. That observation does not disclose
unobservable internal computation or guarantee future provider behavior.

| Observed measure | Qwen | Haiku |
| --- | ---: | ---: |
| Application errors | 0/10 | 1/10 |
| Structured abstentions on two unanswerable questions | 2/2 | 0/2 |
| Median generation workflow time | 1.06135 s | 2.10446 s |
| Input tokens (provider usage) | 11002 | 10482 |
| Output tokens (provider usage) | 742 | 1421 |
| Sum of provider usage.cost, RUB for this service | 0.30537 | 1.75870 |

This does NOT mean Haiku fabricated answers to both unanswerable questions: both
texts acknowledge insufficient information. On absent_en it appended an explanation
and a valid citation after INSUFFICIENT_EVIDENCE, so the strict service interpreted
it as answered. On absent_ru its explanation had no citation, yielding CitationError.
These are instruction/output-contract failures. We did not silently strip the text
or normalize the responses into successful abstentions.

## Semantic review provenance

`*.assistant_review.json` contains exact claim excerpts and rationales;
`*.assistant_scores.json` contains computed metrics and report fingerprints.
Review method is explicitly **assistant**: Codex inspected the actual contexts and
answers. The same assistant authored the development labels; judgments are not
independent, blinded or human-validated. They may be wrong. An independent human
review remains necessary before stronger claims. No separate judge API was called.

| Preliminary assistant judgment | Original Qwen | Qwen configured | Haiku |
| --- | ---: | ---: | ---: |
| Correctness, all cases, errors zero | .90 | 1.00 | .90 |
| Claim support, successful reviewed outputs only | 23/23 | 31/31 | 49/50 |
| Structured abstention decision accuracy, errors zero | .90 | 1.00 | .80 |

Correctness judges central reference facts and the meaning of a refusal, independent
of its status flag; therefore Haiku absent_en gets full semantic correctness but
fails structured abstention accuracy. The technical failure absent_ru contributes
zero end-to-end correctness, although its raw text also conveys a refusal.

The one Haiku claim marked unsupported is that LoRA's base model remains in its
original precision: the context says LoRA itself does not quantize, which does not
specify storage dtype or forbid external precision changes. This is insufficient
support, not a demonstrated falsehood. Core LoRA/QLoRA distinctions are correct.
For all models the review permits paraphrase, basic two-source synthesis and the
contextual reading of "LoRA without quantization" as the described base method,
not a universal ban on combining adapters with quantization. These are reviewer
judgments, not mechanically proven entailment. Claims can contain related subfacts;
changing segmentation changes micro-averages. Do not compare 31 vs 50 as answer quality.

## Thinking probes

`thinking_probes.json` preserves three short checks on the same troublesome question:

1. `chat_template_kwargs.enable_thinking=false`: still returned reasoning, length,
   null final content, 1024 output tokens.
2. `enable_thinking=false`: same failure pattern, 1024 output tokens.
3. `reasoning.enabled=false`: stop, final answer, no returned reasoning, 301 output tokens.

The last format was then used successfully on all ten Qwen comparison questions.
The probe prompt was the original prompt; the main comparison uses the same revised
prompt for both models. Compared with the original baseline, multiple settings
changed (thinking request, citation instruction and output cap), so improvements
cannot be attributed causally to only one change.

Total for this follow-up: 3 paid probes + 20 comparison requests. Sum of API-reported
cost for probes .312955 RUB, comparison 2.06407 RUB; total 2.377025 RUB. This excludes
all requests in the earlier baseline task; these are API usage fields, not a balance
audit. Source text and answers are saved, credentials and reasoning text are not.

## Reproduction

With ignored local .env containing the endpoint and key, from the project root:

```bash
uv run --locked --extra embeddings python scripts/compare_rag_models.py \
  --baseline benchmarks/rag_v1/results/rusgpt_qwen36_bm25.json \
  --dataset benchmarks/rag_v1/dataset.json \
  --output-dir artifacts/model_comparison_repeat --qwen-disable-reasoning
```

This makes 20 paid calls. Existing output files are rejected. Repetitions may differ;
we did not rerun failed cases to select favorable results. Model weights/revisions,
backend routing and effective generation settings beyond request/response are not
independently verified. Latencies include provider/network conditions, not only
model compute; no concurrency, confidence intervals or held-out evaluation.
