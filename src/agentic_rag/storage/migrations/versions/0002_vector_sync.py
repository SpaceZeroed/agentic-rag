"""Track recoverable embedding/index writes separately from ingestion."""

import sqlalchemy as sa
from alembic import op

revision = "0002_vector_sync"
down_revision = "0001_documents"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "vector_sync_states",
        sa.Column("collection", sa.String(128), primary_key=True),
        sa.Column(
            "revision_id",
            sa.Uuid(),
            sa.ForeignKey("document_revisions.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("state", sa.String(16), nullable=False),
        sa.Column("attempt", sa.Uuid(), nullable=False),
        sa.Column("chunk_count", sa.Integer(), nullable=False),
        sa.Column("error_type", sa.String(128)),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.CheckConstraint("state IN ('pending', 'ready', 'failed')", name="ck_vector_sync_state"),
        sa.CheckConstraint("chunk_count > 0", name="ck_vector_sync_count"),
    )


def downgrade() -> None:
    op.drop_table("vector_sync_states")
