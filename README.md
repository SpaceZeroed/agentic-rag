# Agentic RAG for ML/LLM documentation

A Python engineering and learning project for an assistant that answers questions
about ML papers and technical documentation using traceable evidence. The target
system combines retrieval, cited generation, a single tool-using agent, evaluation,
and local model inference.

**Current status: Stage 3 — dense, BM25 and hybrid retrieval.** Markdown/TXT ingestion retains
canonical revisions in PostgreSQL. A pinned multilingual E5 model runs on CPU;
Qdrant searches current indexed chunks with metadata filters. A fixed authored
development set records actual retrieval metrics. Generation and HTTP endpoints
remain future milestones.

See the Russian guides for [ingestion](docs/ingestion.md),
[dense retrieval](docs/dense_retrieval.md) and [BM25/hybrid](docs/sparse_hybrid.md), including consistency, evaluation and
reproduction commands. The first dense run measured Recall@5 = 0.9792 and
MRR@5 = 0.8438 on **10 short authored notes / 20 chunks / 24 EN+RU questions**.
This is a development smoke set with pending human label review, not held-out
quality evidence; [full results and limitations](docs/results.md).

## Run locally

Prerequisites: Python 3.12 and [uv](https://docs.astral.sh/uv/getting-started/installation/).
Python metadata permits 3.12+; the development baseline in `.python-version` is 3.12.
Only the interpreter listed in [results](docs/results.md) has been verified so far.

From the project root:

```bash
uv sync --locked
uv run --locked agentic-rag
```

The CLI emits one JSON line to **stderr**, with `message="application_ready"`,
and exits with status 0. It is a startup check, not a running web server.
An equivalent command is `uv run --locked python -m agentic_rag`.

Optionally create `.env` by copying `.env.example` and adjust the values. No
configuration file, GPU, account, API key, network service, or Docker daemon is
needed for startup, text preview, or autonomous tests after dependencies are installed.

```bash
RAG_ENVIRONMENT=test RAG_LOG_LEVEL=DEBUG uv run --locked agentic-rag
```

The inline environment syntax above is for Bash and similar shells.

## Ingest a document

Preview requires no database:

```bash
uv run --locked agentic-rag preview examples/ingestion_demo.md --max-chars 300 --overlap 50
```

Start local PostgreSQL, migrate explicitly, and ingest:

```bash
docker compose up -d --wait postgres
export RAG_DATABASE_URL='postgresql+psycopg://rag:rag_local@127.0.0.1:55432/rag'
uv run --locked agentic-rag db-upgrade
uv run --locked agentic-rag ingest examples/ingestion_demo.md --max-chars 300 --overlap 50
```

Repeat the ingestion command to get `unchanged`. Use the returned UUIDs with
`agentic-rag show DOCUMENT_UUID` or
`agentic-rag show DOCUMENT_UUID --revision REVISION_UUID`, prefixed by `uv run --locked`.
These commands emit result JSON to stdout and operational logs to stderr.
Source identity defaults to a resolved file URI. Use `--source-uri` for an explicit
portable source identity; no remote URL is fetched. Old revisions remain readable.
The sample is repository-authored demonstration text, not an evaluation dataset.

Compose exposes PostgreSQL on loopback port 55432 and persists it in a named volume.
Its sample password is for local development. `docker compose stop postgres` stops
the service while retaining data. The application URL is optional for startup and
preview, and required for `db-upgrade`, `ingest`, and `show`.

## Dense search on CPU

```bash
uv sync --locked --extra embeddings
docker compose up -d postgres qdrant
export RAG_DATABASE_URL='postgresql+psycopg://rag:rag_local@127.0.0.1:55432/rag'
uv run --locked --extra embeddings agentic-rag db-upgrade
uv run --locked --extra embeddings agentic-rag model-check
uv run --locked --extra embeddings agentic-rag evaluate benchmarks/dense_v1/dataset.json \
  --collection-prefix rag_dense_v1 --output artifacts/dense-v1.json
uv run --locked --extra embeddings agentic-rag search "How does LoRA reduce training memory?" \
  --collection-prefix rag_dense_v1 --k 3
```

First model use downloads pinned Hugging Face weights into `data/models`.
Then `RAG_MODEL_LOCAL_FILES_ONLY=true` enables offline model loading. Keep
`--extra embeddings` on uv commands while using the model; ordinary sync/run
commands may remove optional model packages. Normal tests need neither weights
nor Torch. Qdrant is bound to loopback port 6333 and retains a named volume.
Use `index` after ingesting your own documents; `evaluate` ingests/indexes its fixed
set automatically. See the [guide](docs/dense_retrieval.md) for filters and real
PostgreSQL/Qdrant/model tests.

## BM25 and hybrid search

BM25 uses current PostgreSQL chunks and requires no model or Qdrant. Hybrid uses
both dense and BM25 ranks with reciprocal rank fusion. Search defaults to dense.

```bash
uv run --locked --extra embeddings agentic-rag search "KV cache blocks" --mode bm25 --k 5
uv run --locked --extra embeddings agentic-rag search "KV cache blocks" \
  --mode hybrid --collection-prefix rag_dense_v1 --candidate-k 20 --k 5
uv run --locked --extra embeddings agentic-rag evaluate benchmarks/dense_v1/dataset.json \
  --compare --collection-prefix rag_dense_v1 --output artifacts/comparison.json
```

On the unchanged development set, hybrid MRR@5 was 0.8993 versus dense 0.8438;
Recall@5 remained 0.9792. BM25 Recall@5 was 0.7500. These are small-set observations,
not general quality guarantees. The current lexical index is rebuilt on each query;
this cost is included in reported latency. See [results](docs/results.md).

## Configuration and logs

| Environment variable | Default | Accepted values / meaning |
| --- | --- | --- |
| `RAG_ENVIRONMENT` | `local` | `local`, `test`, `production` |
| `RAG_LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `RAG_DATA_DIR` | `data` | A `pathlib.Path`; relative paths use the working directory |
| `RAG_DATABASE_URL` | unset | PostgreSQL URL using `postgresql+psycopg://`; stored as `SecretStr` |

Settings are immutable after validation. Construction does not create directories.
`.env` is optional and is read from the current working directory, not searched
upward. Values use this precedence: explicit constructor arguments, environment
variables, `.env`, defaults. Unknown `.env` keys are ignored to allow a shared
development file; this also means a misspelled key can be silently ignored. See
[Pydantic Settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/).

Invalid recognized values, such as `RAG_LOG_LEVEL=VERBOSE`, produce a JSON error
and exit code 2. Error events identify the field and validation error type without
echoing the invalid value. At `WARNING` and above, successful startup is quiet.

Logs contain a UTC timestamp, level, logger, and message. Callers can add structured
context with `extra={"fields": {"document_id": "..."}}`. Custom fields are nested
so they cannot replace the standard fields. Exceptions retain their traceback.
The formatter escapes newlines to preserve one JSON event per line. Arbitrary
objects fall back to their string representation; use JSON-native values when
their types matter. Messages and exception text are not automatically redacted:
call sites must choose what is appropriate to log.

`configure_logging()` replaces root handlers and belongs at process startup.
Importing application modules does not configure logging or load settings.

## Verify

```bash
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked mypy
uv run --locked pytest
uv build
```

By default, PostgreSQL tests skip explicitly. Run them against the local service:

```bash
uv run --locked pytest -m postgres --postgres-url='postgresql+psycopg://rag:rag_local@127.0.0.1:55432/rag'
```

Each test uses its own temporary schema and real Alembic migrations. Tests verify
round trips, repeat ingestion, retained revisions, concurrent writes, transaction
rollback, and CLI persistence across processes. Existing application tables are
not cleared. Qdrant/model tests also skip without their explicit connection/cache options.

`uv sync --locked` includes the development dependency group. `--locked` checks
that project metadata matches the lockfile and refuses to update a stale lockfile.
Use `uv add PACKAGE` or `uv add --dev PACKAGE` for deliberate dependency changes;
review and commit both `pyproject.toml` and `uv.lock`. See
[uv's locking and syncing guide](https://docs.astral.sh/uv/concepts/projects/sync/).

Tests cover configuration parsing and precedence, structured log output,
tracebacks, exit codes, logging reconfiguration, and both installed entry points
from outside the repository. The fixture clears `RAG_*` overrides and uses a
temporary working directory so your local `.env` cannot affect assertions.
There is no `sys.path` or `PYTHONPATH` patch to bypass installation.

## Docker development basics

```bash
docker build -t agentic-rag:dev .
docker run --rm agentic-rag:dev
docker run --rm agentic-rag:dev pytest
```

The development image includes quality tools and runs as a non-root user. Its
default command performs the same startup check and exits. `.dockerignore`
excludes local credentials, virtual environments, and generated data. To override
settings, pass Docker environment options, for example `-e RAG_ENVIRONMENT=test`.

Stage 0 Docker was verified on 2026-09-14: the image built, the startup command exited 0,
and all 17 tests plus Ruff and mypy passed inside a non-root container with
networking disabled. The initial WSL integration limitation no longer reproduces.
See [the verification record](docs/results.md) for commands and environment details.

This is a development image. The Python base tag is mutable; pin image digests
when we work on deployment reproducibility. Compose runs PostgreSQL and Qdrant;
the full application deployment belongs to later milestones. The image
includes the sample document, so `docker run --rm agentic-rag:dev preview
examples/ingestion_demo.md` works without host mounts. See the verification record
for Stage 1 results and the guide for database addressing inside containers.

## Repository

```text
pyproject.toml              # Build metadata, dependencies, tool configuration
uv.lock                    # Exact dependency resolution
.python-version            # Python development baseline
.env.example               # Safe local configuration example
Dockerfile / .dockerignore  # Development container
compose.yaml               # Local PostgreSQL and Qdrant with persistent volumes
examples/                  # Repository-authored demonstration document
benchmarks/                # Fixed development corpus, labels, measured report
src/agentic_rag/
    __init__.py
    __main__.py             # python -m entry point
    cli.py                  # Startup composition and exit status
    core/
        __init__.py
        config.py           # Validated settings
        logging.py          # JSON formatter and startup configuration
    ingestion/              # Parsing, explicit chunking, deterministic identities
    storage/                # SQLAlchemy adapter and packaged Alembic migrations
    embeddings/             # Provider contract and explicit CPU E5 pooling
    retrieval/              # Dense workflow and Qdrant adapter
    evaluation/             # Fixed-label loader and explicit ranking metrics
tests/
    conftest.py             # Isolated environment and working directory
    unit/                  # Configuration and logging behavior
    integration/           # Installed CLI behavior in separate processes
docs/
    architecture.md        # Requirements, diagram, boundaries, decisions, risks
    roadmap.md             # Stage deliverables and verification gates
    adr/                   # Architectural decision records
    results.md             # Actual measurements and reproducibility information
    interview_notes.md     # Concepts, tradeoffs, and self-check questions
```

Start with [architecture](docs/architecture.md), then the
[roadmap](docs/roadmap.md) and [interview notes](docs/interview_notes.md).
Stages 0–3 are implemented. The next stage, after review, is cross-encoder reranking.

To continue in a new chat, read the [conversation handoff](docs/handoff.md) and
the [original project brief](docs/project_brief.md). They preserve the working
style, completed work, verification results, limitations, and stopping point.

## Stage 4: optional CPU reranking

`search --mode hybrid --rerank --candidate-k 20 --rerank-k 20 --k 5` adds a pinned
multilingual cross-encoder after retrieval. `evaluate DATASET --rerank --output PATH`
compares the same hybrid candidates before/after reranking. Use the existing
`--extra embeddings`; models run locally on CPU. See [architecture, model choice,
commands and limits](docs/reranking.md) and [measurements](docs/results.md).
