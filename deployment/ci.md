# CI and reproducibility

The GitHub Actions workflow in `.github/workflows/ci.yaml` runs on push, pull
request and manual dispatch. It does not deploy a public service, push an image,
or call paid model providers. It uses a read-only repository token and checkout
without persisted Git credentials. Actions are pinned by commit SHA; uv is pinned
to 0.12.10, Python baseline to 3.12 and Python packages to uv.lock.

Two independent jobs:

1. `checks`: locked install, Ruff lint/format, strict mypy on src/tests/scripts,
   the ordinary test suite, and wheel/sdist build without workspace source overrides.
2. `integration`: disposable PostgreSQL/Qdrant services, explicit Qdrant readiness
   wait, real database integration tests, then build/start the deployment image and
   run the fake-model restart/outage/persistence check. On failure it prints
   deployment logs. The test stack is stopped even when a step fails.

Hosted runners start without the developer's `.env`, optional model weights or
local artifacts. CPU model tests skip explicitly unless their caches are supplied;
mocked embedding tests do not validate actual model weights. Real LLM quality,
provider rate limits, GPU inference and Langfuse server ingestion are not CI gates.

A green workflow means only these checks passed for that revision. It is not a
production deployment, quality certification or benchmark improvement. A workflow
file on disk is also not a completed GitHub Actions run: the first hosted run
requires the changes to be committed and pushed. No remote run is claimed here.
Branch protection/required checks must be configured separately in GitHub.

Reproduce the Python commands from the root README. For database integration,
start `docker compose up -d` using the development compose file, then pass the
localhost URLs documented there. Do not run the deployment recovery script
against an API serving users: it intentionally stops its test databases.

Sources for workflow behavior/options:
[checkout](https://github.com/actions/checkout/tree/de0fac2e4500dabe0009e67214ff5f5447ce83dd),
[setup-uv](https://github.com/astral-sh/setup-uv/tree/94527f2e458b27549849d47d273a16bec83a01e9).
Service images and runner patch versions can change; this is not a bit-for-bit
reproducible OS image. Update action pins and dependencies deliberately and rerun
checks. The existing local deployment is separate from ephemeral CI resources.

## Local verification, 2026-09-21

A clean export of tracked commit f0d8c56 plus Stage 14 files was created in
`/tmp/retrieval-ci-stage14`, without .env, local docs, artifacts or an existing
virtual environment. `uv sync --locked` installed a new environment without
embedding extras. Ruff lint/format passed (138 Python files), strict mypy passed
(115 source files), and wheel/sdist built successfully with `uv build --no-sources`.
The full suite with real development PostgreSQL/Qdrant passed: **389 passed,
2 skipped** in 157.39 seconds. The skips were the two optional CPU model-cache
checks. Database tests used isolated schemas/collections; this was not a fresh
GitHub service-container run. No paid model calls were made for these checks.

YAML parsing, expected triggers/jobs and full-length action SHA references were
checked locally. This is not an actionlint run or execution by the Actions runner.

The Docker portion also passed from this export in a new `retrieval-ci-stage14`
project on port 8003: image build, empty-volume startup/migrations and the
fake-model restart/persistence/outage check. Evidence is retained locally in
`artifacts/ci_deployment_20260921.json`. The temporary project was stopped after
verification; existing deployment on port 8002 was not restarted or reconfigured.
