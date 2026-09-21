# External inference API benchmark

Stage 12 scope: client-observed streaming performance. Self-hosted vLLM and GPU
experiments are deferred. Uses the existing configured model and credentials;
does not modify `.env`. No new dependencies. Run from repository root:

```bash
UV_CACHE_DIR=/tmp/retrieval-proj-uv-cache uv run --no-sync python scripts/benchmark_inference.py --output-dir artifacts/inference_api_run
```

This command makes 30 paid requests: 24 measured plus 6 excluded warmups. Each
request has max_tokens=512 and a 90-second total deadline. No retries. The output
directory must not exist. Report is checkpointed after each cell. Only metadata
is saved, never credentials or generated text. Prompts are synthetic service
records generated in the script, with 8 or 128 records; lengths in tokens come
from provider usage, not estimates based on characters. Record count is not a
token count. Requested model and returned model are both recorded.

Six cells: two context sizes × concurrency 1/2/4, four requests per cell.
Deterministically shuffled order (seed 12), one shared connection pool and a
closed-loop worker pool. Request latency begins when a worker starts its call;
it excludes local waiting for a worker and includes connection pool/network/
provider waiting. Throughput denominator is the entire measured cell duration.
Cold initialization is not isolated. Identical prompts and warmups can exercise
provider prefix/response caches, which this client cannot control. Repetitions
are not independent production traffic. Four observations per cell cannot
support a stable p95; raw samples and medians are provided instead.

`first_text_seconds` is time to first nonempty visible content delta; hidden
reasoning, role-only events and HTTP headers do not end this timer. Chunk gaps
are client delivery intervals, not token intervals. Exact TTFT/TPOT is unavailable:
SSE chunks may contain multiple tokens and reported completion usage may include
reasoning. No GPU memory, batching configuration or server-only decode speed is
inferred. Output lengths can vary even with temperature=0.

Success requires the existing client to validate terminal metadata and DONE;
truncation is incomplete, even if usage is present. Timeouts cancel the stream.
Transport and protocol errors share the production client's safe error category;
HTTP response bodies are not logged. Successful latency excludes errors, so
always inspect status counts and raw error durations alongside it. Usage remains
null when unavailable. Known successful completion tokens per second is a partial
sum when coverage is below 1; it excludes errors and warmups, so it is not total
billable usage or server capacity. Their reported usage is retained in raw samples.

Reports include source hashes, Git HEAD (which may precede uncommitted benchmark
code), model request configuration and client platform. Provider hardware, model
revision, caching, internal queueing and batching are unknown. These measurements
characterize this client/provider path at that time, not a production SLA and not
RAG answer quality. Raw reports belong in ignored artifacts/.
