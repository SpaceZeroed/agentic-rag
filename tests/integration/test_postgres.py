import json
import subprocess
import sys
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Barrier
from uuid import uuid4

import pytest
from alembic import command
from sqlalchemy import Engine, func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from sqlalchemy.schema import CreateSchema, DropSchema

from agentic_rag.ingestion.models import ChunkingConfig, IngestResult, PreparedDocument
from agentic_rag.ingestion.service import prepare_document
from agentic_rag.storage.database import create_database_engine, migration_config, upgrade_database
from agentic_rag.storage.models import ChunkRow, DocumentRow, RevisionRow
from agentic_rag.storage.repository import PostgresDocumentRepository

pytestmark = pytest.mark.postgres


@pytest.fixture
def database(request: pytest.FixtureRequest) -> Iterator[Engine]:
    url = request.config.getoption("--postgres-url")
    if not url:
        pytest.skip("Use --postgres-url to run against real PostgreSQL")
    admin = create_database_engine(str(url))
    schema = f"rag_test_{uuid4().hex}"
    with admin.begin() as connection:
        connection.execute(CreateSchema(schema))
    scoped_url = admin.url.update_query_dict(
        {"options": f"-csearch_path={schema} -cstatement_timeout=30000 -clock_timeout=10000"}
    )
    engine = create_database_engine(scoped_url.render_as_string(hide_password=False))
    try:
        upgrade_database(engine)
        yield engine
    finally:
        engine.dispose()
        with admin.begin() as connection:
            connection.execute(DropSchema(schema, cascade=True))
        admin.dispose()


def prepare(
    tmp_path: Path, text: str = "# Paper\n\nA paragraph about retrieval.\n" * 5
) -> PreparedDocument:
    path = tmp_path / "paper.md"
    path.write_text(text, encoding="utf-8")
    return prepare_document(path, ChunkingConfig(35, 7), source_uri="https://example.org/paper")


def counts(engine: Engine) -> tuple[int, int, int]:
    with Session(engine) as session:
        return (
            session.scalar(select(func.count()).select_from(DocumentRow)) or 0,
            session.scalar(select(func.count()).select_from(RevisionRow)) or 0,
            session.scalar(select(func.count()).select_from(ChunkRow)) or 0,
        )


def test_migrations_are_repeatable_and_match_declared_schema(database: Engine) -> None:
    upgrade_database(database)
    with database.begin() as connection:
        command.check(migration_config(connection))
    assert counts(database) == (0, 0, 0)


def test_roundtrip_and_duplicate_ingestion(database: Engine, tmp_path: Path) -> None:
    repository = PostgresDocumentRepository(database)
    document = prepare(tmp_path)
    assert repository.save(document).status == "created"
    assert repository.save(document).status == "unchanged"
    assert repository.get(document.document_id) == document
    assert counts(database) == (1, 1, len(document.chunks))
    assert repository.get(uuid4()) is None
    assert repository.get(document.document_id, revision_id=uuid4()) is None


def test_updates_retain_history_and_old_versions_can_be_reactivated(
    database: Engine, tmp_path: Path
) -> None:
    repository = PostgresDocumentRepository(database)
    original = prepare(tmp_path)
    repository.save(original)
    changed = prepare(tmp_path, "# Changed\n\nDifferent content.\n" * 4)
    assert repository.save(changed).status == "updated"
    assert repository.get(original.document_id) == changed
    assert repository.get(original.document_id, revision_id=original.revision_id) == original
    assert repository.save(original).status == "reactivated"
    assert repository.get(original.document_id) == original
    assert counts(database) == (1, 2, len(original.chunks) + len(changed.chunks))


def test_rechunking_is_a_new_revision(database: Engine, tmp_path: Path) -> None:
    repository = PostgresDocumentRepository(database)
    original = prepare(tmp_path)
    repository.save(original)
    rechunked = prepare_document(
        tmp_path / "paper.md", ChunkingConfig(60, 10), source_uri=original.source_uri
    )
    assert repository.save(rechunked).status == "updated"
    assert repository.get(original.document_id) == rechunked
    assert repository.get(original.document_id, revision_id=original.revision_id) == original


@pytest.mark.parametrize("has_previous", [False, True])
def test_failed_chunk_write_rolls_back_the_entire_revision(
    database: Engine, tmp_path: Path, has_previous: bool
) -> None:
    repository = PostgresDocumentRepository(database)
    original = prepare(tmp_path)
    if has_previous:
        repository.save(original)
    before = counts(database)
    changed = prepare(tmp_path, "A new version with multiple chunks. " * 8)
    invalid_chunk = replace(changed.chunks[-1], end_char=changed.chunks[-1].end_char + 1)
    broken = replace(changed, chunks=(*changed.chunks[:-1], invalid_chunk))
    with pytest.raises(IntegrityError):
        repository.save(broken)
    assert counts(database) == before
    assert repository.get(original.document_id) == (original if has_previous else None)
    assert repository.get(original.document_id, revision_id=changed.revision_id) is None


def test_concurrent_duplicates_create_one_complete_revision(
    database: Engine, tmp_path: Path
) -> None:
    document = prepare(tmp_path)
    barrier = Barrier(2)

    def save() -> IngestResult:
        barrier.wait(timeout=10)
        return PostgresDocumentRepository(database).save(document)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(save) for _ in range(2)]
        results = [future.result(timeout=15) for future in futures]
    assert sorted(result.status for result in results) == ["created", "unchanged"]
    assert counts(database) == (1, 1, len(document.chunks))
    assert PostgresDocumentRepository(database).get(document.document_id) == document


def test_concurrent_revisions_keep_one_current_and_both_histories(
    database: Engine, tmp_path: Path
) -> None:
    versions = [prepare(tmp_path), prepare(tmp_path, "Different revision. " * 8)]
    barrier = Barrier(2)

    def save(document: PreparedDocument) -> IngestResult:
        barrier.wait(timeout=10)
        return PostgresDocumentRepository(database).save(document)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(save, document) for document in versions]
        results = [future.result(timeout=15) for future in futures]
    assert sorted(result.status for result in results) == ["created", "updated"]
    repository = PostgresDocumentRepository(database)
    assert repository.get(versions[0].document_id) in versions
    for document in versions:
        assert repository.get(document.document_id, revision_id=document.revision_id) == document
    with Session(database) as session:
        assert (
            session.scalar(
                select(func.count()).select_from(RevisionRow).where(RevisionRow.is_current)
            )
            == 1
        )


def test_cli_persists_across_processes_and_can_read_a_retained_revision(
    database: Engine, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAG_DATABASE_URL", database.url.render_as_string(hide_password=False))
    path = tmp_path / "document.txt"
    path.write_text("First persisted version.", encoding="utf-8")

    def cli(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "agentic_rag", *args],
            text=True,
            capture_output=True,
            check=False,
            timeout=15,
        )

    assert cli("db-upgrade").returncode == 0
    first = cli("ingest", str(path))
    assert first.returncode == 0, first.stderr
    result = json.loads(first.stdout)
    duplicate = cli("ingest", str(path))
    assert json.loads(duplicate.stdout)["status"] == "unchanged"
    path.write_text("Second persisted version.", encoding="utf-8")
    updated = cli("ingest", str(path))
    assert json.loads(updated.stdout)["status"] == "updated"
    current = cli("show", result["document_id"])
    assert json.loads(current.stdout)["text"] == "Second persisted version."
    old = cli("show", result["document_id"], "--revision", result["revision_id"])
    assert json.loads(old.stdout)["text"] == "First persisted version."
    assert cli("show", str(uuid4())).returncode == 1


def test_cli_reports_unknown_migration_without_a_traceback(
    database: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAG_DATABASE_URL", database.url.render_as_string(hide_password=False))
    with database.begin() as connection:
        connection.execute(text("UPDATE alembic_version SET version_num = 'unknown_revision'"))
    result = subprocess.run(
        [sys.executable, "-m", "agentic_rag", "db-upgrade"],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    assert result.returncode == 1
    assert "database_failed" in result.stderr
    assert "Traceback" not in result.stderr
    assert "rag_local" not in result.stderr
