"""Exercise the isolated retrieval-deploy stack, including database outage/restarts."""

import argparse
import json
import subprocess
import time
from pathlib import Path
from uuid import uuid4

import httpx

COMPOSE = ["docker", "compose", "-f", "compose.deploy.yaml"]


def compose(*args: str, stdin: str | None = None) -> str:
    return subprocess.check_output([*COMPOSE, *args], input=stdin, text=True, timeout=120)


def ready(client: httpx.Client) -> None:
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        try:
            if client.get("/ready").status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(1)
    raise RuntimeError("Deployment did not become ready")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8002")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Output exists")
    collection = "deployment_probe_" + uuid4().hex
    with httpx.Client(base_url=args.url, timeout=20, trust_env=False) as client:
        ready(client)
        response = client.post(
            "/documents",
            json={
                "filename": "deployment.txt",
                "source_uri": "https://example.org/" + collection,
                "content": "Deployment recovery: backups are retained for 37 days.",
            },
        )
        response.raise_for_status()
        document = response.json()["document"]
        print("Document persisted; checking Qdrant", flush=True)
        setup = f"""
from qdrant_client import QdrantClient, models
client = QdrantClient(url="http://qdrant:6333")
client.create_collection(
    {collection!r},
    vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE),
)
point = models.PointStruct(id=1, vector=[1.0, 0.0], payload={{"marker": "retained"}})
client.upsert({collection!r}, points=[point], wait=True)
client.close()
"""
        compose("exec", "-T", "api", "python", "-", stdin=setup)
        try:
            compose("stop", "postgres")
            assert client.get("/live").status_code == 200
            assert client.get("/ready").status_code == 503
            print("Postgres outage: live=200, ready=503", flush=True)
        finally:
            compose("start", "postgres")
        ready(client)
        compose("restart", "postgres", "qdrant", "api")
        ready(client)
        verify = f"""
from qdrant_client import QdrantClient
client = QdrantClient(url="http://qdrant:6333")
points = client.retrieve({collection!r}, ids=[1], with_vectors=True)
assert len(points) == 1 and points[0].payload == {{"marker": "retained"}}
assert points[0].vector == [1.0, 0.0]
client.close()
"""
        compose("exec", "-T", "api", "python", "-", stdin=verify)
        response = client.post(
            "/query",
            json={
                "query": "How long are deployment backups retained?",
                "document_ids": [document["document_id"]],
            },
        )
        response.raise_for_status()
        answer = response.json()
        assert answer["status"] == "answered" and answer["citations"]
        assert "37" in answer["text"]
        report = {
            "document": document,
            "qdrant_probe_collection": collection,
            "postgres_recovered": True,
            "postgres_and_qdrant_persisted_after_restart": True,
            "outage_live_status": 200,
            "outage_ready_status": 503,
            "llm_provider": answer["llm_provider"],
            "health": client.get("/ready").json(),
            "compose_ps": compose("ps", "--all", "--format", "json"),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print("Restart/persistence/recovery checks passed: " + str(args.output), flush=True)


if __name__ == "__main__":
    main()
