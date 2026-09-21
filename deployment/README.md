# Single-host deployment (Stage 13)

The separate `compose.deploy.yaml` project (`retrieval-deploy`) runs the API,
a one-shot migration, PostgreSQL and Qdrant. It uses new named volumes and does not
reuse or remove development or Langfuse data. LLM inference stays external.
This is a local single-host deployment, not a public Internet service or HA setup.
API binds to localhost:8002; databases have no published host ports. The bundled
PostgreSQL password is a local example. Before public deployment, supply private
credentials, authentication/TLS at a reverse proxy, and an actual backup policy.

## Start

From repository root with Docker Compose v2:

```bash
docker compose -f compose.deploy.yaml up -d --build --wait --wait-timeout 120
curl --fail http://127.0.0.1:8002/ready
```

The initial mode is BM25 + explicitly labelled fake LLM. This tests infrastructure
without using the model provider. Qdrant runs and retains data but BM25 does not
query it. The lean image deliberately omits Torch/Transformers; dense/hybrid and
reranking require an image with the embeddings extra and a model-cache volume.
They are not enabled by this deployment configuration. One Uvicorn worker keeps
existing process-local request limits and metrics semantics.

To use existing `.env` LLM credentials and endpoint (Compose reads them locally):

```bash
RAG_DEPLOY_LLM_PROVIDER=compatible docker compose -f compose.deploy.yaml up -d --wait
```

Only listed environment values reach the containers. `.env` is not baked into
images, mounted, or modified. Environment credentials are visible to Docker
administrators; do not publish `docker compose config` or container environment.
Compose service DNS is used for databases: `postgres:5432`, `qdrant:6333`.
`localhost` inside API refers to that container, not the host or another service.
The API has a read-only root filesystem, a temporary /tmp, UID 10001, no added
Linux capabilities, and no dev tools installed. Logging emits JSON to stdout;
Langfuse export is off, local spans/metrics are on. The Docker host controls log
retention and container resource limits; no capacity/SLA claim is made here.

## Startup and probes

Postgres must pass pg_isready before migration runs. `migrate` uses packaged
Alembic migrations and must exit 0 before API starts. It is intentionally an
exited container, not a long-running healthy service. Qdrant starts alongside
these services; BM25 readiness does not require it. `depends_on` is a startup
ordering mechanism, not ongoing dependency supervision.

- `/live`: ASGI process responds, independent of databases and paid LLM calls.
- `/ready` and compatibility alias `/health`: required backend database checks;
  return 503 if unavailable. BM25 reports Qdrant `not_required`.
- LLM `not_probed` means no external inference was tested by readiness.
- `/metrics`: bounded process metrics; reset when API restarts.

Docker's API healthcheck uses `/ready`. An unhealthy container is not automatically
restarted by Docker's restart policy; `unless-stopped` concerns process exit.
If readiness drops, inspect the dependency rather than restarting in a loop.

## Persistence and recovery

```bash
docker compose -f compose.deploy.yaml ps --all
docker compose -f compose.deploy.yaml logs --tail 100 migrate api postgres
docker compose -f compose.deploy.yaml restart postgres qdrant api
```

Wait for `/ready` after restart; restarting does not rerun migrations or apply
changed environment variables. Use `up -d` to recreate changed service config.
For an application/schema upgrade, stop API, back up PostgreSQL, build the image,
run migrations explicitly, then start API. Never start the upgraded API if the
migration command failed:

```bash
docker compose -f compose.deploy.yaml stop api
docker compose -f compose.deploy.yaml build
docker compose -f compose.deploy.yaml run --rm migrate
docker compose -f compose.deploy.yaml up -d --wait
```

Named volumes survive container restarts and recreation. They are not backups.
Do not run `down -v` or prune volumes when preserving data. Ordinary `stop` retains
everything; `up -d` starts it again. The development stack is a separate project.
An older API image may not support a newer schema: rollback needs a compatible
schema or a tested backup restore, not just changing the image tag.

Postgres logical backup (write to a new file, preserve the existing backup):

```bash
mkdir -p artifacts/backups
docker compose -f compose.deploy.yaml exec -T postgres pg_dump -U rag -d rag -Fc > artifacts/backups/rag.dump
```

Choose a unique filename for each backup. Check command exit status and test a
restore into a separate disposable database with `pg_restore --exit-on-error`.
Qdrant needs its own collection snapshots/backup if dense retrieval is enabled;
PostgreSQL and Qdrant do not share a transaction. Their restored revisions must
be reconciled/reindexed using the project's indexing workflow before serving.
Backup/restore and disaster recovery are not validated by the restart check.

## Reproducible restart/outage check

With the deployment running, preferably in default fake mode:

```bash
UV_CACHE_DIR=/tmp/retrieval-proj-uv-cache uv run --no-sync python scripts/check_deployment.py --output artifacts/deployment_run.json
```

This script intentionally stops/restarts **only retrieval-deploy** PostgreSQL,
Qdrant and API. It ingests a unique synthetic document, creates one probe Qdrant
collection with a point, verifies live=200/ready=503 during a PostgreSQL outage,
restarts all three services and reads the retained document/vector without
reingestion. Tiny probe data is retained for inspection. Output must be new.
Do not run against a deployment serving active users. Change --url if using
RAG_DEPLOY_PORT. A successful fake response verifies wiring, not LLM quality.
