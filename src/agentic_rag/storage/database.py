"""Database connections and explicit, package-local schema migrations."""

from alembic import command
from alembic.config import Config
from sqlalchemy import Connection, Engine, create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError


def create_database_engine(url: str) -> Engine:
    try:
        parsed = make_url(url)
    except ArgumentError as exc:
        raise ValueError("Invalid database URL") from exc
    if parsed.drivername != "postgresql+psycopg":
        raise ValueError("Database URL must use postgresql+psycopg")
    options = parsed.query.get("options", "-c statement_timeout=30000 -c lock_timeout=10000")
    if not isinstance(options, str):
        raise ValueError("Database connection options must be a single value")
    return create_engine(
        parsed,
        pool_pre_ping=True,
        hide_parameters=True,
        isolation_level="READ COMMITTED",
        connect_args={
            "connect_timeout": 5,
            "options": options,
        },
    )


def migration_config(connection: Connection) -> Config:
    config = Config()
    config.set_main_option("script_location", "agentic_rag.storage:migrations")
    config.attributes["connection"] = connection
    return config


def upgrade_database(engine: Engine) -> None:
    """Run migrations explicitly; ordinary imports and ingestion never run DDL."""
    with engine.begin() as connection:
        command.upgrade(migration_config(connection), "head")
