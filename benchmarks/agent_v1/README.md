# Agent development cases v1

14 fixed cases: **10 natural tasks and 4 controlled fault scenarios**. This is an
assistant-authored development set, pending human review. Some cases derive from
previous smoke failures; it is not a held-out benchmark. No runs or quality scores
have been produced for this set. The case file and review template are ready;
a runner with corpus preflight, fault injection and report capture is not implemented.
Do not send the controlled cases to `/agent` and claim the faults were exercised.

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
Use an isolated corpus containing exactly these ten active revisions. The user's
current database includes other documents and is **not** this fixture. Provision
an isolated store or validate exact membership; do not delete or replace user data.
The entire corpus must be available to the agent's BM25 retrieval and catalog.

The initial policy is six model calls, eight tool observations, 48,000 prompt bytes,
12,000 observation bytes, 4,096 output tokens per call and 180 seconds per case.
Temperature=0, reasoning disabled, no reranker or automatic HTTP retry. Pin the
actual provider/model and effective settings in the run manifest. This does not
promise deterministic provider responses or a monetary ceiling. Confirm the chosen
case subset and spending constraints separately before any paid run.

Natural cases use real model decisions and read-only tools. Controlled cases apply
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
are sufficient. Preserve failed responses and do not overwrite earlier runs.

Copy `review_template.json` into the run directory. Bind every case review to its
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
