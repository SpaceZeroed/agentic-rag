# Tool-use review v1

This policy was added **retrospectively on 2026-09-21**, after the saved September
19 live runs were inspected. It supplements the original frozen case criteria;
it is not a preregistered metric or a held-out evaluation. Do not edit cases or
old reports to improve these scores. Future policy changes require a new version.

The offline reviewer creates labels; the program validates provenance and computes
aggregates. It does not infer semantic correctness. Human review is still pending.
No LLM judge, network request, credentials or database is needed.

## Unit and denominators

Each **proposed tool call**, including a proposal blocked by the agent budget, has
three separate boolean judgments and written reasons:

- `selection`: is this tool's capability suitable for the requested operation?
  Repeated use of the correct tool can pass selection and fail necessity.
- `arguments_correct`: are the arguments syntactically valid and semantically
  suitable for the requested operation? Correct JSON alone is insufficient.
  A wrong tool cannot have task-correct arguments under this policy. Equivalent
  arithmetic expressions and search paraphrases are allowed. A harmless default
  need not be specified explicitly. Check observed input dependencies: `N * 1.25`
  must use N from the preceding catalog, not a known fixture constant alone.
- `unnecessary`: should this proposal have been omitted given the information
  available before it? This is not the same as a failed call or an empty result.
  A bounded retry after an availability error can be reasonable. The agent does
  not know the hidden fault adapter guarantees permanent failure.

Report sum(true) / number of proposals **separately by live/mock/replayed origin**.
For unnecessary, lower is better. Zero denominator means null, not 100% or 0%.
Arguments are scored over all proposals, not only correctly selected tools.

Also review `selection_complete` per case: all needed capabilities were proposed,
no unsuitable capabilities were proposed, or no tool was correctly chosen for a
greeting. This catches omitted tools, which proposal precision alone misses. It
does not require successful execution or task completion. For controlled cases
this is an end-to-end trace judgment including fixtures, not autonomous model
selection accuracy. All cases, including zero-proposal cases, enter this denominator.

The current empty-search trajectory may reasonably translate/broaden a query,
inspect the catalog for a title, then search using that title. The final close
paraphrase after these attempts is judged unnecessary. For absent personal cost,
the initial search and one reformulation are allowed; further closely related
queries after the same general LoRA passage are judged unnecessary. These are
explicit assistant judgments, not a universal retry threshold or production policy.
Search cannot substitute for the catalog's current chunk-count metadata; the three
fallback searches in `catalog_unavailable` fail selection and arguments and are
unnecessary. Read the per-proposal reasons alongside these aggregate scores.

## Other dimensions

Reuse existing hash-bound 0/1/2 quality reviews for task completion, claim support,
citation placement and failure handling. Report mean, full-score count and each
dimension's applicable denominator. N/A is excluded, not treated as zero or success.
An `answered` status is not a semantic pass; dimensions must remain separate.

Model steps count every recorded model invocation, including fixture turns and
failed attempts. Tool observations include rejections and cache returns; they are
not backend execution counts. Proposals can exceed observations. Report per-case
observation errors and repair attempts. An aborted run without a final result has
unknown observation count, not zero. Skipped cases remain explicitly listed.

Latency is recorded end-to-end case wall time. Separate cases containing replay
from fresh cases and natural from controlled cases. p50/p95 use nearest rank
(`ceil(p*n)-1` index); on tiny heterogeneous groups they are descriptive only.
They do not measure production SLOs, MCP, PostgreSQL or API latency.

## Provenance and commands

Templates leave all judgments null. Fill each boolean and rationale, reviewer
name, and reviewer type before aggregation. The original quality review is a
separate input. Hash checks bind both reviews to each report and manifest; proposal
identity uses turn and position, so duplicate call IDs cannot silently drop calls.
Review coverage must exactly match attempted cases. All output uses exclusive
creation and will refuse to overwrite files.

```bash
UV_CACHE_DIR=/tmp/retrieval-proj-uv-cache uv run --no-sync python scripts/review_agent.py template \
  --run-dir artifacts/agent_eval_live_20260919_v1 \
  --output artifacts/tool_review_TEMPLATE_NEW.json

UV_CACHE_DIR=/tmp/retrieval-proj-uv-cache uv run --no-sync python scripts/review_agent.py summarize \
  --run-dir artifacts/agent_eval_live_20260919_v1 artifacts/agent_eval_live_20260919_v2 \
  --tool-review artifacts/agent_review_20260921/tool_review_v1.json artifacts/agent_review_20260921/tool_review_v2.json \
  --quality-review artifacts/agent_eval_live_20260919_v1/assistant_review.json artifacts/agent_eval_live_20260919_v2/assistant_review.json \
  --output artifacts/agent_review_summary_NEW.json
```

To combine disjoint subsets, configuration, code/dependency hashes, corpus and case
hashes must match. README prose and overall dirty-diff hashes may differ; executable
source and machine-readable benchmark files may not. Duplicate case IDs are rejected,
not silently pooled. Raw runs and completed reviews above are local ignored artifacts;
a fresh checkout can generate templates for its own runs and run synthetic unit tests,
but does not contain the historical live reports.
