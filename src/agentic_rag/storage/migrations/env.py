"""Alembic environment; connection ownership stays with the caller."""

from alembic import context
from sqlalchemy import Connection

from agentic_rag.storage.models import Base

connection = context.config.attributes.get("connection")
if not isinstance(connection, Connection):
    raise RuntimeError("Run migrations through agentic-rag db-upgrade")
context.configure(connection=connection, target_metadata=Base.metadata, compare_type=True)
with context.begin_transaction():
    context.run_migrations()
