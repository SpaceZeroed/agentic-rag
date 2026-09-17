import json
from pathlib import Path

import pytest
from sqlalchemy import Engine

from agentic_rag.cli import main
from agentic_rag.ingestion.models import ChunkingConfig
from agentic_rag.ingestion.service import prepare_document
from agentic_rag.storage.repository import PostgresDocumentRepository


@pytest.mark.postgres
def test_bm25_ask_fake_current_revision_and_filters(
    database: Engine,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("RAG_DATABASE_URL", database.url.render_as_string(hide_password=False))
    path = tmp_path / "paper.txt"
    path.write_text("PagedAttention uses blocks for the KV cache.", encoding="utf-8")
    original = prepare_document(
        path, ChunkingConfig(100, 0), source_uri="https://example.org/paper"
    )
    repository = PostgresDocumentRepository(database)
    repository.save(original)
    path.write_text("PagedAttention allocates KV cache in blocks on demand.", encoding="utf-8")
    current = prepare_document(path, ChunkingConfig(100, 0), source_uri=original.source_uri)
    repository.save(current)
    assert main(["ask", "PagedAttention", "--mode", "bm25", "--llm", "fake"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["llm_provider"] == "fake"
    citation = result["answer"]["citations"][0]["source"]
    assert citation["revision_id"] == str(current.revision_id)
    assert citation["text"] == current.text
    assert citation["source_uri"] == current.source_uri
    assert (
        main(
            [
                "ask",
                "PagedAttention",
                "--mode",
                "bm25",
                "--source-uri",
                "https://example.org/absent",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["answer"]["status"] == "insufficient_evidence"
    assert result["answer"]["completion"] is None


@pytest.mark.parametrize(
    "arguments",
    [
        ["--llm", "compatible"],
        ["--k", "0"],
        ["--rerank", "--rerank-k", "1"],
    ],
)
def test_ask_rejects_bad_configuration_before_database(
    arguments: list[str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["ask", "query", *arguments]) == 2
    assert capsys.readouterr().out == ""


def test_generation_errors_have_nonzero_exit_without_answer(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import argparse

    from agentic_rag import cli
    from agentic_rag.core.config import Settings
    from agentic_rag.llm.base import LLMError

    def fail(args: argparse.Namespace, settings: Settings) -> int:
        raise LLMError("LLM transport or response failure")

    monkeypatch.setattr(cli, "_execute", fail)
    assert cli.main(["ask", "private-question"]) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert "generation_failed" in output.err
    assert "private-question" not in output.err


@pytest.mark.postgres
@pytest.mark.parametrize("server_status", [200, 500])
def test_compatible_ask_with_mock_http_transport(
    database: Engine,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    server_status: int,
) -> None:
    import httpx

    from agentic_rag import cli

    monkeypatch.setenv("RAG_DATABASE_URL", database.url.render_as_string(hide_password=False))
    monkeypatch.setenv("RAG_LLM_BASE_URL", "http://inference.test/v1")
    monkeypatch.setenv("RAG_LLM_MODEL", "local-model")
    monkeypatch.setenv("RAG_LLM_API_KEY", "private-key")
    path = tmp_path / "paper.txt"
    path.write_text("PagedAttention uses blocks.", encoding="utf-8")
    PostgresDocumentRepository(database).save(prepare_document(path, ChunkingConfig(100, 0)))
    original_client = httpx.Client
    clients: list[httpx.Client] = []

    def handle(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer private-key"
        assert str(request.url) == "http://inference.test/v1/chat/completions"
        body = json.loads(request.content)
        assert body["max_tokens"] == 512
        assert body["model"] == "local-model"
        assert json.loads(body["messages"][1]["content"])["sources"][0]["id"] == "C1"
        return httpx.Response(
            server_status,
            json={
                "model": "local-model",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "Blocks [C1]"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    def client_factory(
        *,
        base_url: str,
        headers: dict[str, str],
        timeout: float,
        follow_redirects: bool,
        trust_env: bool,
    ) -> httpx.Client:
        client = original_client(
            base_url=base_url,
            headers=headers,
            timeout=timeout,
            follow_redirects=follow_redirects,
            trust_env=trust_env,
            transport=httpx.MockTransport(handle),
        )
        clients.append(client)
        return client

    monkeypatch.setattr(httpx, "Client", client_factory)
    code = cli.main(["ask", "PagedAttention", "--mode", "bm25", "--llm", "compatible"])
    output = capsys.readouterr()
    assert clients[0].is_closed
    assert "private-key" not in output.out + output.err
    if server_status == 200:
        assert code == 0
        assert json.loads(output.out)["answer"]["text"] == "Blocks [C1]"
    else:
        assert code == 1
        assert output.out == ""
        assert "generation_failed" in output.err
