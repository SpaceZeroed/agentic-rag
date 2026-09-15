# Agentic RAG for ML/LLM documentation

A Python engineering and learning project for an assistant that answers questions
about ML papers and technical documentation using traceable evidence. The target
system combines retrieval, cited generation, a single tool-using agent, evaluation,
and local model inference.

**Current status: Stage 0 — repository foundation.** The executable validates
configuration, emits a structured startup event, and exits. Retrieval, model calls,
databases, and HTTP endpoints belong to later milestones.

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
needed to run the CLI or tests after dependencies are installed.

```bash
RAG_ENVIRONMENT=test RAG_LOG_LEVEL=DEBUG uv run --locked agentic-rag
```

The inline environment syntax above is for Bash and similar shells.

## Configuration and logs

| Environment variable | Default | Accepted values / meaning |
| --- | --- | --- |
| `RAG_ENVIRONMENT` | `local` | `local`, `test`, `production` |
| `RAG_LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `RAG_DATA_DIR` | `data` | A `pathlib.Path`; relative paths use the working directory |

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

Docker was verified on 2026-09-14: the image built, the startup command exited 0,
and all 17 tests plus Ruff and mypy passed inside a non-root container with
networking disabled. The initial WSL integration limitation no longer reproduces.
See [the verification record](docs/results.md) for commands and environment details.

This is a development image. The Python base tag is mutable; pin image digests
when we work on deployment reproducibility. Compose with PostgreSQL and Qdrant
will arrive when those dependencies have working application paths.

## Repository

```text
pyproject.toml              # Build metadata, dependencies, tool configuration
uv.lock                    # Exact dependency resolution
.python-version            # Python development baseline
.env.example               # Safe local configuration example
Dockerfile / .dockerignore  # Development container
src/agentic_rag/
    __init__.py
    __main__.py             # python -m entry point
    cli.py                  # Startup composition and exit status
    core/
        __init__.py
        config.py           # Validated settings
        logging.py          # JSON formatter and startup configuration
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
Only Stage 0 is implemented. The next stage is explicit Markdown/TXT ingestion.

To continue in a new chat, read the [conversation handoff](docs/handoff.md) and
the [original project brief](docs/project_brief.md). They preserve the working
style, completed work, verification results, limitations, and stopping point.
