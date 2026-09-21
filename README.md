# Agentic RAG for ML/LLM documentation

A Python project for answering technical questions using retrieved evidence and
validated citations. It combines document ingestion, hybrid retrieval, optional
CPU reranking, a bounded tool-using agent, MCP, evaluation and request tracing.

**Status:** stages 0–13 implemented; Stage 14 adds CI and documentation polish.
Self-hosted vLLM/GPU inference is deferred. The deployed baseline uses BM25 and
an external compatible LLM API; the default smoke-test mode explicitly uses a
fake model. Local verification is documented below; a hosted GitHub Actions run
is not yet verified. This is a learning project with measured development results,
not an independently validated production service.

```mermaid
flowchart LR
    Client --> API[FastAPI: documents / query / agent]
    API --> Agent[Bounded LangGraph agent]
    API --> Retrieval[BM25 / dense / hybrid + optional reranker]
    Agent --> Retrieval
    Agent --> Tools[Catalog + MCP calculator]
    Retrieval --> PG[(PostgreSQL documents and revisions)]
    Retrieval --> Qdrant[(Qdrant vectors)]
    API --> LLM[External LLM API]
    Agent --> LLM
    API --> Tracing[JSON logs / metrics / optional Langfuse]
```

## Run the complete local stack

Requires Docker Compose v2. From a fresh checkout:

```bash
docker compose -f compose.deploy.yaml up -d --build --wait --wait-timeout 120
curl --fail http://localhost:8002/ready
```

Open http://localhost:8002/docs to ingest a document and query it. This creates
an isolated `retrieval-deploy` project with its own PostgreSQL/Qdrant volumes.
The default response uses a labelled fake generator; it does not contact an LLM.
A one-shot migration must succeed before the non-root API starts.

To enable the external model, copy `.env.example` to `.env` and configure
`RAG_LLM_BASE_URL`, `RAG_LLM_MODEL`, `RAG_LLM_API_KEY` and, if supported,
`RAG_LLM_REASONING_ENABLED`. Never commit `.env`. Then run:

```bash
RAG_DEPLOY_LLM_PROVIDER=compatible docker compose -f compose.deploy.yaml up -d --wait
```

This lean image supports BM25 without downloading embedding/reranking models.
Dense/hybrid retrieval is available in the Python application with optional
model dependencies; it is not enabled in this image. The LLM is always separate.
See [deployment guide](deployment/README.md) for configuration, probes, upgrades,
persistence, recovery commands and limitations. Stop without deleting data:

```bash
docker compose -f compose.deploy.yaml stop
```

## Python development

The tested baseline is Python 3.12 with uv 0.12.10. The package declares Python
3.12+, but CI currently exercises only 3.12. Install locked dependencies:

```bash
uv sync --locked
uv run --no-sync agentic-rag --help
uv run --no-sync agentic-rag
```

The bare CLI is a startup check that logs `application_ready` and exits. It does
not start a web server. Most tests use fake/mocked models and need no API key.
For optional local embedding/reranking work, use `uv sync --locked --extra embeddings`.
After installing extras, use `uv run --no-sync` so ordinary runs do not prune them.
Model downloads, GPU access and paid API evaluations are not prerequisites for CI.

The original `compose.yaml` is a development database stack with localhost ports
55432/6333. It is separate from `compose.deploy.yaml` and its data volumes.

## API and execution contracts

- `POST /documents`: validated Markdown/TXT ingestion with canonical revisions.
- `POST /query`: retrieval and cited answers; optional SSE streaming with provisional
  deltas and a validated terminal `result`, or an `error` event.
- `POST /agent`: bounded search, calculator and catalog tool execution, or direct answers.
- `GET /live`: the process responds. `GET /ready` (also `/health`): required backend
  databases are reachable. External LLM availability is not inferred from this probe.
- `GET /metrics`: process-local metrics when observability is enabled. Trace IDs
  belong in logs/spans, not metric labels. Metrics reset on process restart.

Agent call/prompt/output limits, tool policy checks and bounded parsing constrain
execution; they do not establish semantic correctness of all answers. MCP
capability discovery is filtered by local policy. Streaming HTTP 200 alone is
not proof of a successfully completed answer.

## Reproducible evidence

| Area | Observed result / scope | Evidence |
| --- | --- | --- |
| Retrieval | Authored development set: 10 notes, 20 chunks, 24 EN/RU questions; not held-out quality | [Dataset and commands](benchmarks/dense_v1/README.md), [dense/BM25/hybrid report](benchmarks/dense_v1/results/comparison_cpu.json) |
| Reranking | CPU experiments on the same small set; inspect quality and latency together | [Isolated report](benchmarks/dense_v1/results/reranking_cpu_isolated.json) |
| RAG generation | Frozen-context Qwen/Haiku comparison; scores include explicitly labelled assistant review | [Comparison methodology and results](benchmarks/rag_v1/results/model_comparison_v1/README.md) |
| Agent evaluation | Versioned scenarios; proposals, executions and replays must remain distinct | [Cases](benchmarks/agent_v1/README.md), [review policy](benchmarks/agent_v1/tool_review_policy.md) |
| API inference | 24 measured calls + 6 warmups; all succeeded. Client first-text/chunk timing, not server token timing | [Methodology](benchmarks/inference_v1/README.md), [measured table](benchmarks/inference_v1/results.md) |
| Deployment | Startup, migrations, restart persistence, database outage/recovery and two live LLM requests | [Verification](deployment/results.md) |

Small experiments do not establish production p95, maximum throughput, a safe
concurrency limit or general answer quality. Raw local artifacts and working
notes under `docs/` are not distributed; the linked tracked reports state the
available evidence and its limits. No synthetic latency is presented as a live
measurement. The [CI guide](deployment/ci.md) distinguishes local verification
from an actual hosted workflow run.

## Checks

```bash
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync mypy src tests scripts
uv run --no-sync pytest
uv build --no-sources
```

Real database checks (each PostgreSQL test gets an isolated schema; Qdrant tests
use separate collections):

```bash
uv run --no-sync pytest tests/integration \
  --postgres-url=postgresql+psycopg://rag:rag_local@127.0.0.1:55432/rag \
  --qdrant-url=http://127.0.0.1:6333
```

Without database URLs the corresponding tests explicitly skip. The two real CPU
model tests also skip unless given downloaded caches through `--model-cache` and
`--reranker-cache`. These skips are not successful model integration tests.

## Optional Langfuse

```bash
python3 scripts/init_langfuse.py
docker compose --env-file .env.langfuse -f compose.langfuse.yaml up -d
```

This separate stack is available at http://localhost:3000. Login is
`local@example.com`; the generated password is `LF_LOGIN_PASSWORD` in the ignored
`.env.langfuse`. The initializer does not overwrite existing credentials.
Tracing uses explicit spans; prompts, source text, tool arguments and generated
answers are omitted from those spans. Export failures do not fail user requests.
The deployment baseline enables local telemetry but leaves Langfuse export off.
See `scripts/check_langfuse.py` for verification against the local server; it uses
fixture models and does not make paid LLM calls.
