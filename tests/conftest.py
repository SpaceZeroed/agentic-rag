"""Keep tests independent of the developer's environment and local .env file."""

import os
from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import Engine
from sqlalchemy.schema import CreateSchema, DropSchema

from agentic_rag.storage.database import create_database_engine, upgrade_database


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--qdrant-url", help="Real Qdrant URL; tests create unique collections")
    parser.addoption("--model-cache", help="Existing E5 cache for optional offline CPU model test")
    parser.addoption(
        "--postgres-url",
        help="PostgreSQL test connection; each test creates and drops its own isolated schema",
    )


@pytest.fixture(autouse=True)
def isolated_configuration(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for key in os.environ:
        if key.upper().startswith("RAG_"):
            monkeypatch.delenv(key)
    monkeypatch.chdir(tmp_path)


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
