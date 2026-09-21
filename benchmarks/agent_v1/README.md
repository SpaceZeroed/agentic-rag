# Agent development cases v1

14 fixed cases: **10 natural tasks and 4 controlled fault scenarios**. This is an
assistant-authored development set, pending human review. Some cases derive from
previous smoke failures; it is not a held-out benchmark. All 14 cases have been
attempted across two disjoint live runs and reviewed by the assistant.
`scripts/evaluate_agent.py` validates
the corpus, injects the specified faults and captures reports. Offline mock runs
exercise the runner, not model quality. Sending controlled queries to `/agent`
without fault adapters does not exercise these scenarios.

| Group | Case IDs | Purpose |
| --- | --- | --- |
| Direct | direct_greeting | Avoid unnecessary tools |
| Calculation | calculate_ratio, calculate_increase | Correct arithmetic, denominator, units and T references |
| Retrieval | search_paged, search_compare, search_false_premise | Evidence, comparison and correction of a false premise |
| Catalog | catalog_page, catalog_complete | Page boundaries, completeness, metadata and citation placement |
| Combined | catalog_calculate | Compute using a value actually obtained from a tool |
| Missing evidence | absent_private_fact | Decline an absent user-specific claim |
| Controlled | empty_retrieval, catalog_unavailable | Handle empty results and unavailable tools without invented facts |
| Controlled | malformed_catalog_arguments, uncited_catalog_answer | Recover from explicitly supplied bad model outputs |

## Reproducible setup

`cases.json` pins the SHA-256 of `../rag_v1/dataset.json`, which supplies the ten
source files, their checksums and chunking (400/60). Only its corpus configuration
is reused. Existing retrieval/RAG questions and judgments remain unchanged.
The runner constructs an isolated in-memory snapshot containing exactly the manifest's
documents after validating source checksums and evidence. It uses the existing BM25
implementation and a catalog with UUID ordering, literal case-sensitive substring
filtering and page semantics. It never opens the user's database or a vector store.
This measures the agent loop over the fixed corpus, not PostgreSQL/Qdrant/API
integration or their latency. The backend type and document/revision/chunk IDs are
recorded explicitly in the manifest. Any future database-backed runner must verify
exact active membership; it must not delete user data to prepare this fixture.

The initial policy is six model calls, eight tool observations, 48,000 prompt bytes,
12,000 observation bytes, 4,096 output tokens per call and 180 seconds per case.
Temperature=0, reasoning disabled, no reranker or automatic HTTP retry. Pin the
actual provider/model and effective settings in the run manifest. This does not
promise deterministic provider responses or a monetary ceiling. Confirm the chosen
case subset and spending constraints separately before any paid run.

In live mode, natural cases use real model decisions and read-only snapshot tools. Controlled cases apply
only the specified adapter fault and record it. `scripted_prefix` returns the exact
embedded model responses (fixture usage/cost is null, not actual paid usage). After
the prefix the real model continues. Tool execution in the prefix still uses the
isolated corpus; replayed model calls still consume agent call budgets. The uncited
prefix intentionally contains an incomplete draft: repair must use the actual
catalog observation to produce the requested list. The malformed JSON is intentional
inside a string; the containing case file is valid JSON. Empty-search and unavailable-
catalog faults persist for the entire case. Other tools retain their normal behavior.

Run each case in a fresh request with no shared conversation. Keep expected answers,
evidence excerpts and reviewer criteria out of model input. Do not enforce an exact
tool route unless the user query or controlled scenario requires it. For example,
complete catalog enumeration may be one large page or several complete smaller pages.
No forced tool calls or access to labels to improve scores.

## Capture and review

Save an immutable run manifest (commit, dirty diff hash, cases/source hashes,
settings, provider/model and timestamp), sanitized request/response history, tool
observations, source snapshots, final AgentResult, provider HTTP statuses and wall
time. Store artifacts in a new `artifacts/agent_eval_<run_id>/` directory. Never
record credentials or private reasoning text; usage and reasoning character counts
are sufficient. Preserve failed responses and do not overwrite earlier runs. The
runner uses exclusive output directories/files, records partial model/tool traces
on timeout/cancellation, and stops the suite on transport/runner failure or its
global new-call limit. `skipped_cases` identifies cases not attempted. HTTP errors
are captured with bounded bodies and allowlisted fields; arbitrary error messages
and authorization headers are excluded. An HTTP attempt without usage has unknown
cost. `controlled_fault_exercised` distinguishes actually injected faults from a
controlled case where the model never called the targeted tool.

The runner creates `review.json` for attempted cases, with report hashes and null
quality scores. `review_template.json` remains a standalone full-set template.
Bind every case review to its
report SHA-256 and name the reviewer (`human` or `assistant`; do not conflate them).
The template is deliberately unscored. Review the actual cited observations, not
just the reference answer or the model's self-assessment. Record concrete claims
and supporting/missing evidence in `claim_reviews`, including citation placement.

Each applicable dimension uses **0 / 1 / 2**, with a written rationale:

| Dimension | 0 | 1 | 2 |
| --- | --- | --- | --- |
| task_completion | Wrong answer, no usable result, or unsafe invention | Materially incomplete result, or safe termination by a limit | Requested task completed, including justified refusal for absent evidence |
| claim_support | A central claim contradicts or lacks evidence | Mixed support or unsupported secondary details | Every material factual claim supported by recorded observations |
| citation_placement | Missing/unknown citations, or only a trailing marker after an unrelated closing question | Some claims have clearly attributable citations, others do not | Citations clearly identify evidence for each factual claim/list item |
| failure_handling | Invalid execution, fabricated success, hidden errors, budget violation | Safe stop/limit without recovery when recovery was expected | Appropriate recovery or refusal, errors retained, budgets respected |

Use null with an explicit reason for non-applicable dimensions: claim support and
citations for a pure greeting or a factual-claim-free refusal; failure handling for
a natural case where no failure occurred. Null is not zero and not a pass. An
unexpected error in a natural case makes failure handling applicable. Keep
`structural_validation_passed` separate: `answered` with one trailing [T1] can pass
the existing validator and still score 0 for placement. Source ID existence does
not establish entailment. Check arithmetic against the recorded expression/result;
check catalog numbers against the observed page coverage, not known fixture counts
alone. Do not demand fixed document UUID ordering; inspect the returned page.

Report per-case scores and each dimension's applicable-case denominator, plus
answered/refusal/failed/limit counts. Report natural and controlled groups separately.
No single blended pass percentage: it would hide the distinction between useful
answers, safe failures and synthetic recovery. Report model/tool counts, rejected
proposals, repair attempts, token usage, known provider cost, missing-usage counts
and end-to-end latency. Replayed usage must not be included in new paid totals;
missing provider cost is unknown, not zero. Mark p50/p95 as descriptive only on this
small heterogeneous set, and separate replay timings from fresh runs.

Freeze labels before running. If criteria, prompts, corpus or adapters change, save
a new version/configuration and keep both reports. One success on a previously
failing example does not establish improved general reliability.

## Commands

Run from the repository root. Validation is the default and does not load LLM
credentials or connect to external services:

```bash
UV_CACHE_DIR=/tmp/retrieval-proj-uv-cache uv run --no-sync python scripts/evaluate_agent.py
```

Offline infrastructure check (all cases, real local BM25, `FakeToolLLM`). The fake
model's command-based routing is not expected to solve natural-language tasks;
failed cases and unexercised faults are retained, not replaced by scripted successes:

```bash
UV_CACHE_DIR=/tmp/retrieval-proj-uv-cache uv run --no-sync python scripts/evaluate_agent.py \
  --mode mock --output-dir artifacts/agent_eval_mock_NEW_ID
```

After separate approval of paid execution, select explicit cases and a global new
model-call ceiling. This example is a command template, not authorization to run:

```bash
UV_CACHE_DIR=/tmp/retrieval-proj-uv-cache uv run --no-sync python scripts/evaluate_agent.py \
  --mode live --allow-paid --cases calculate_ratio --max-live-calls 3 \
  --output-dir artifacts/agent_eval_live_NEW_ID
```

Live mode reads existing `RAG_LLM_BASE_URL`, `RAG_LLM_MODEL`, `RAG_LLM_API_KEY` and
request timeout; configured reasoning must be false. Model/token/tool/deadline
budgets otherwise come from the frozen case policy, not the interactive API settings.
The live ceiling excludes replayed prefix turns but includes failed new attempts;
it is not a monetary ceiling. The manifest records versions, code fingerprints,
model, provider origin without credentials, effective policy and dataset hashes.
Raw reasoning is excluded and configured API-key occurrences are redacted from
saved reports. No automatic HTTP retries, no model fallback and no external tracing.

## First live subset (2026-09-19)

Local artifacts: `artifacts/agent_eval_live_20260919_v1/`, with frozen reports,
`assistant_review.json` bound to their hashes, and readable `review.md`.
Qwen `qwen/qwen3.6-35b-a3b`, reasoning disabled, unchanged cases/prompts:

| Case | Observed result |
| --- | --- |
| calculate_ratio (natural) | 75%, supported by calculate, adjacent [T1] |
| search_paged (natural) | KV-cache explanation supported by C1, citations by each numbered point, no invented paper title |
| uncited_catalog_answer (controlled) | One real correction after two scripted turns; three observed catalog items, [T1] on each item, original error retained |

All returned `answered`. Assistant review assigns 2/2 for task completion, claim
support and citation placement (natural n=2, controlled n=1); failure handling is
N/A for natural cases (n=0) and 2/2 for the controlled case (n=1). This is not an
independent human review, a full-set success rate, or proof of reliable repair.
In particular, the controlled prefix does not test model-selected catalog routing.

Five new provider HTTP calls (all 200) out of a ceiling of eight; 6124 input / 507
output tokens, summed provider `cost` 0.180785, no missing usage. The runner does
not establish currency. Two replayed turns are excluded from new paid usage.
The other eleven cases were not selected; no follow-up paid attempts were made.

## Remaining cases (separate authorization, 2026-09-19)

`artifacts/agent_eval_live_20260919_v2/` contains the other eleven cases and
hash-bound assistant judgments in `assistant_review.json`, readable `review.md`,
and `combined_review.json` aggregating the two disjoint runs. Code/script, case and
corpus hashes match. No prompt/label changes or automatic retries during testing.

The new run used 35 calls of a 36-call ceiling, all HTTP 200: 62252 input / 2594
output tokens, provider cost sum 1.543350 (currency not established). One scripted
malformed-argument prefix turn is excluded from paid usage.

Main findings:

- Malformed JSON recovery succeeded: a new valid catalog call executed once, the
  original invalid_arguments observation remained. Citation placement was partial:
  T2 accompanied has_more, while the listed items had source URLs but no T markers.
- Complete catalog enumeration used one limit=20 page with has_more=false; one
  citation repair then correctly marked every item and the total count.
- `absent_private_fact` repeated five searches yielding the same LoRA passage;
  `empty_retrieval` made four empty searches and a catalog read. Both hit the model
  budget instead of producing the expected insufficient-evidence refusal.
- `catalog_unavailable` made two failed catalog calls then three searches. Its final
  draft explained the refusal and appended INSUFFICIENT_EVIDENCE; exact sentinel
  matching did not recognize it. Citation repair had no remaining model budget.
- `catalog_page` attached T1 only to pagination, leaving list attribution incomplete.
  `search_compare` added an overly categorical original-precision claim where the
  source only establishes that LoRA itself does not quantize the base model.

Across both runs: natural cases 9 answered / 1 limit_reached; controlled cases
2 answered / 2 limit_reached. These are statuses, not semantic success percentages.
40 new HTTP calls total, 68376 input / 3101 output tokens, provider cost sum 1.724135;
three replayed turns excluded. Assistant-review mean scores (0–2) and applicable
denominators are recorded separately per group in combined_review.json. No
independent human review or held-out reliability estimate is claimed.

Next priority: stop repeated retrieval without new evidence and preserve capacity
for a valid refusal; also address brittle refusal formatting. Keep this baseline
unchanged before testing such modifications.

## Stage 10 offline tool-use review (2026-09-21)

Added [tool review policy v1](tool_review_policy.md) and `scripts/review_agent.py`.
The rubric is retrospective and assistant-reviewed; cases and original reports
remain unchanged. Local `artifacts/agent_review_20260921/final_summary.json` binds
both original quality reviews and new per-proposal labels by SHA-256 and records
the evaluator source hash. No new provider calls were made.

Natural live proposals: selection 15/15, arguments 15/15, unnecessary 4/15.
Controlled live proposals: selection 9/12, arguments 9/12, unnecessary 4/12.
Replay proposals: selection 2/2, arguments 1/2, unnecessary 0/2; excluded from live.
Task-completion full score remains natural 9/10 and controlled 2/4 under the prior
rubric; this does not imply full claim support or citation placement. Model steps
average 2.5 / 4.5 and observations 1.4 / 3.25 (natural / controlled).
Natural latency mean 3.310 s, nearest-rank p50 3.174 s, p95 5.110 s (n=10).
Controlled latency is split into fresh mean 5.281 s (n=2) and containing replay
mean 2.584 s (n=2). Descriptive only; not a throughput or production SLO test.
The calls and missing-evidence failures remain unchanged; evaluation does not
itself fix the agent. Independent human review is still pending.
