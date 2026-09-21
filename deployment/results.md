# Stage 13 verification — 2026-09-21

- Clean creation of retrieval-deploy network and PostgreSQL/Qdrant named volumes.
- Packaged migrations exited successfully before API startup; repeated migration
  on final image recreation also exited 0.
- Default BM25/fake mode: ingested a synthetic document and a separate Qdrant
  probe point. During PostgreSQL stop, /live returned 200 and /ready returned 503.
  Restarting PostgreSQL restored readiness without an API restart.
- Subsequently restarted PostgreSQL, Qdrant and API. The saved document was
  retrieved with a citation; Qdrant retained point ID, payload and vector.
  No reingestion was needed. Probe data is retained for inspection.
- Rebuilt final API image, recreated with compatible external LLM configuration.
  Qwen qwen/qwen3.6-35b-a3b returned a grounded answer with one citation; a second
  streaming request ended in result, with no error event. X-Trace-ID present.
- Container UID 10001; no /app/.env; pytest not installed. API runs with read-only
  root and tmpfs /tmp. No existing development/Langfuse volumes were modified.
- 50 API/observability tests passed, scoped Ruff/format and mypy passed.

Final local image ID:
`sha256:2049261e001a945cb8fb0ced7097b980c36594638e9b025f55180e923563c726`.
Image tags are mutable; this ID identifies the actual verified build, not a
portable registry deployment pin. Python dependencies are uv.lock pinned.

Local ignored evidence: artifacts/deployment_20260921.json,
artifacts/deployment_live_20260921.json, artifacts/deployment_sources_20260921.json.
Live verification source retained in artifacts/check_deployment_live_20260921.py.
An initial probe used an unsupported urn source URI and received 422 before
writing data; corrected to HTTPS. Sandbox async test run hung and was interrupted;
reported tests completed outside sandbox.

API remains available at http://localhost:8002/docs in compatible LLM mode.
A plain up command without RAG_DEPLOY_LLM_PROVIDER=compatible selects default fake.
Readiness does not probe the external LLM; successful live calls are separate
verification. Qdrant is not required by BM25 and has no Docker healthcheck here.
This validates single-host startup/restart recovery, not backup restoration,
HA/failover, dense retrieval in this image, public authentication or load capacity.
