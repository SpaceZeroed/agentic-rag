# Agentic RAG for technical documentation

An evidence-first RAG system for technical documents, built as a Python modular
monolith. It ingests versioned Markdown/TXT sources, retrieves canonical chunks,
generates cited answers, exposes a bounded tool-using agent, and records enough
provenance to evaluate retrieval, context construction, generation, and operations
separately.

This is a portfolio and learning project with real integration tests and measured
development results. It is not presented as a production-ready service or as an
independently validated model benchmark.

## What is implemented

| Area | Implementation |
| --- | --- |
| Ingestion | UTF-8 Markdown/TXT parsing, deterministic chunking, revisions, checksums, stable IDs |
| Storage | PostgreSQL as the source of truth; explicit Alembic migrations and retained history |
| Retrieval | Dense E5, BM25, reciprocal-rank fusion, filters, exact search and optional cross-encoder reranking |
| RAG | Prompt budgeting, canonical context, strict citation validation, abstention and compatible LLM API |
| API | FastAPI document/query endpoints, bounded concurrency, cancellation, readiness and SSE streaming |
| Agent | Bounded LangGraph loop with search, catalog and calculator tools; MCP calculator integration |
| Evaluation | Versioned datasets, retrieval/context metrics, saved answers, review templates and agent trace review |
| Operations | Docker Compose, JSON logs, process metrics, OpenTelemetry and optional local Langfuse |

The default local and deployment configurations use BM25 with a clearly labelled
fake generator. External inference is enabled only by explicit configuration; tests
do not require an API key, GPU, or model download.

## Architecture

```mermaid
flowchart LR
    Files["Markdown / TXT"] --> Ingest["parse → normalize → chunk"]
    Ingest --> PG[(PostgreSQL<br/>text, revisions, provenance)]
    Ingest --> Embed[Embedding provider]
    Embed --> Qdrant[(Qdrant<br/>derived vectors)]

    Client --> API[FastAPI / CLI]
    API --> Search["BM25 / dense / hybrid"]
    Search --> PG
    Search --> Qdrant
    Search --> Rank[Optional reranker]
    Rank --> Context[Budgeted canonical context]
    Context --> LLM[Compatible LLM API]
    LLM --> Validate[Answer + citation validation]
    Validate --> Client

    API --> Agent[Bounded agent]
    Agent --> Search
    Agent --> Tools[Catalog + calculator + MCP]
    API -.-> Observe[Logs / metrics / traces]
    API -.-> Eval[Datasets / reports / reviews]
```

PostgreSQL owns document text and revision history. Qdrant is a rebuildable index,
never the canonical store. Citations bind an answer to a document, revision, chunk,
and source coordinates. A valid citation proves identity, not semantic support;
faithfulness still requires review.

## Quick start: local BM25 RAG

Requirements: Python 3.12+, [uv](https://docs.astral.sh/uv/) and Docker Compose v2.

```bash
uv sync --locked
cp .env.example .env
docker compose up -d --wait postgres qdrant
uv run --no-sync agentic-rag db-upgrade
uv run --no-sync agentic-rag ingest examples/ingestion_demo.md \
  --max-chars 300 --overlap 50
uv run --no-sync agentic-rag ask \
  'Какие свойства загрузки документов проверяет пример?' \
  --mode bm25 --llm fake
```

The fake generator is a deterministic workflow fixture, not a quality baseline. To
use an OpenAI-compatible provider, set `RAG_LLM_BASE_URL`, `RAG_LLM_MODEL`, and
`RAG_LLM_API_KEY` in the ignored `.env`, then select it explicitly:

```bash
uv run --no-sync agentic-rag ask \
  'Какие свойства загрузки документов проверяет пример?' \
  --mode bm25 --llm compatible
```

For dense/hybrid retrieval or reranking, install the optional CPU model stack:

```bash
uv sync --locked --extra embeddings
uv run --no-sync agentic-rag index
uv run --no-sync agentic-rag search 'hybrid retrieval' \
  --mode hybrid --candidate-k 20 --k 5 --rerank --rerank-k 20
```

Model weights are downloaded on first use unless local-only mode is enabled. The
development stack exposes PostgreSQL on `127.0.0.1:55432` and Qdrant on
`127.0.0.1:6333`; its named volumes survive `docker compose stop`.

## Run the HTTP service

For local development with BM25 and the fake generator:

```bash
RAG_API_LLM_PROVIDER=fake uv run --no-sync uvicorn \
  agentic_rag.api.app:create_app --factory --host 127.0.0.1 --port 8001
```

Swagger is available at <http://127.0.0.1:8001/docs>. Important endpoints:

- `POST /documents` — ingest versioned Markdown/TXT content;
- `POST /query` — retrieve and return a cited answer, optionally over SSE;
- `POST /agent` — run the bounded tool-using agent;
- `GET /live` — process liveness;
- `GET /ready` and `/health` — required storage readiness;
- `GET /metrics` — process-local metrics when observability is enabled.

HTTP 200 at the beginning of an SSE response is not proof of success. Consumers
must wait for a validated terminal `result`; an `error` event invalidates all
provisional deltas.

## Isolated Compose deployment

The deployment stack builds a non-root API image, applies packaged migrations,
and creates isolated PostgreSQL/Qdrant volumes:

```bash
docker compose -f compose.deploy.yaml up -d --build --wait --wait-timeout 120
curl --fail http://localhost:8002/ready
```

It starts in BM25/fake mode and does not contact an external LLM. To opt into the
provider configured in `.env`:

```bash
RAG_DEPLOY_LLM_PROVIDER=compatible \
  docker compose -f compose.deploy.yaml up -d --wait
```

See the [deployment guide](deployment/README.md) for persistence, recovery,
configuration and security boundaries. The deployment image intentionally omits
the embedding/reranking runtime; dense and hybrid experiments run from the Python
environment.

## Measured results

Results are reported at their actual scope; development datasets and assistant
reviews are not described as independent quality evidence.

| Experiment | Result | Scope and evidence |
| --- | --- | --- |
| External lexical retrieval | SciFact test: Recall@10 **0.791**, MRR@10 **0.628**, nDCG@10 **0.662** | 5,183 documents and 300 queries, whole-document BM25; [method](benchmarks/scifact/README.md) |
| Local RAG retrieval candidate | Recall@5 **0.720**, MRR@5 **0.565**, nDCG@5 **0.595** | 30 frozen docs, 50 answerable questions; authored labels pending exhaustive review; [dataset and breakdown](benchmarks/local_docs_v2/README.md) |
| Real generation, identical contexts | GLM 5.3 Flash: median **1.926 s**, **2.468 ₽**, 1 application error; Qwen 3.6: **2.621 s**, **3.631 ₽**, 2 errors | 60 sequential Rus-GPT calls per model, no retries; operational comparison only; [method and results](benchmarks/local_docs_v2/model_comparison.md) |
| Strict refusal contract | Qwen **10/10**, GLM **8/10** on intended-unanswerable cases | GLM's two failures were semantic refusals with extra text after the exact sentinel; semantic quality review remains pending |
| Streaming inference | 24 measured calls plus 6 warmups, all successful | Client-observed latency/throughput, not server token timing; [method and table](benchmarks/inference_v1/README.md) |
| Deployment recovery | Migrations, restart persistence and PostgreSQL outage/recovery verified | Single-host Compose, not HA or backup restoration; [verification](deployment/results.md) |

The new evaluation work supports three conclusions:

1. Retrieval remains the dominant unresolved quality bottleneck. The local candidate
   is weaker on its five English and five multi-source cases, and its qrels contain
   known duplicate-evidence ambiguities.
2. A faster or newer generator is not automatically a better RAG model. GLM was
   faster and cheaper in one paired run, while Qwen followed the exact abstention
   contract more reliably. No semantic winner is claimed before review.
3. Output validation is part of model quality. Both models reproduced a literal
   malformed `` `[C` `` example from source documentation and triggered the citation
   validator, revealing an application-contract edge case rather than a transport
   failure.

Earlier dense/hybrid/reranking, RAG, agent and deployment evidence is indexed in the
[results journal](docs/results.md) and under [`benchmarks/`](benchmarks/).

## Evaluation workflows

Validate and run the larger local candidate without databases or an API key:

```bash
uv run --no-sync python scripts/evaluate_rag_offline.py \
  benchmarks/local_docs_v2/dataset.json \
  --output artifacts/local_docs_v2_bm25.json
```

Run the DB-backed evaluator with a fake or compatible generator:

```bash
uv run --no-sync agentic-rag evaluate-rag \
  benchmarks/local_docs_v2/dataset.json \
  --mode bm25 --llm fake \
  --output artifacts/local_docs_v2_smoke.json
```

`review-rag` exports a report-bound annotation template and scores only complete
reviews. Reference answers are not sent to the generation provider. Reports retain
model/configuration provenance, per-case errors, retrieved chunks, context, token
usage and latency; credentials and transport response bodies are excluded.

## Verification

Fast local checks:

```bash
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync mypy src tests scripts
uv run --no-sync pytest -m 'not postgres'
uv build --no-sources
```

Integration tests against the development services:

```bash
docker compose up -d --wait postgres qdrant
uv run --no-sync pytest tests/integration \
  --postgres-url=postgresql+psycopg://rag:rag_local@127.0.0.1:55432/rag \
  --qdrant-url=http://127.0.0.1:6333
```

Each PostgreSQL integration test uses an isolated schema; Qdrant tests use separate
collections. Real CPU model tests additionally require downloaded model caches.
Skipped service/model tests are not reported as successful integrations.

## Documentation map

- [Project brief](docs/project_brief.md) and [architecture](docs/architecture.md)
- [Ingestion](docs/ingestion.md), [retrieval](docs/dense_retrieval.md),
  [hybrid search](docs/sparse_hybrid.md) and [reranking](docs/reranking.md)
- [Basic RAG](docs/basic_rag.md) and [RAG evaluation](docs/rag_evaluation.md)
- [HTTP API](docs/api.md), [agent](docs/agent.md), [MCP](docs/mcp.md) and
  [agent evaluation](docs/agent_evaluation.md)
- [Observability](docs/observability.md), [roadmap](docs/roadmap.md),
  [results](docs/results.md) and [architecture decisions](docs/adr/)

The project deliberately avoids automatic fallback from a failed real model to the
fake generator, automatic migrations during import/startup, and claims of general
answer quality from small assistant-authored datasets.
