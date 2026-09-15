"""Create canonical documents, retained revisions and chunks."""

import sqlalchemy as sa
from alembic import op

revision = "0001_documents"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "documents",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("source_uri", sa.Text(), nullable=False, unique=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )
    op.create_table(
        "document_revisions",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "document_id",
            sa.Uuid(),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("is_current", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("media_type", sa.String(64), nullable=False),
        sa.Column("raw_sha256", sa.String(64), nullable=False),
        sa.Column("content_sha256", sa.String(64), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("parser_version", sa.String(64), nullable=False),
        sa.Column("chunker_version", sa.String(64), nullable=False),
        sa.Column("max_chars", sa.Integer(), nullable=False),
        sa.Column("overlap", sa.Integer(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.CheckConstraint("max_chars > 0", name="ck_revision_max_chars"),
        sa.CheckConstraint("overlap >= 0 AND overlap < max_chars", name="ck_revision_overlap"),
    )
    op.create_index("ix_revision_document", "document_revisions", ["document_id"])
    op.create_index(
        "uq_revision_current",
        "document_revisions",
        ["document_id"],
        unique=True,
        postgresql_where=sa.text("is_current"),
    )
    op.create_table(
        "chunks",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "revision_id",
            sa.Uuid(),
            sa.ForeignKey("document_revisions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("start_char", sa.Integer(), nullable=False),
        sa.Column("end_char", sa.Integer(), nullable=False),
        sa.Column("start_line", sa.Integer(), nullable=False),
        sa.Column("end_line", sa.Integer(), nullable=False),
        sa.UniqueConstraint("revision_id", "position", name="uq_chunk_position"),
        sa.CheckConstraint("position >= 0", name="ck_chunk_position"),
        sa.CheckConstraint("start_char >= 0 AND end_char > start_char", name="ck_chunk_offsets"),
        sa.CheckConstraint("char_length(text) = end_char - start_char", name="ck_chunk_length"),
        sa.CheckConstraint("start_line >= 1 AND end_line >= start_line", name="ck_chunk_lines"),
    )


def downgrade() -> None:
    op.drop_table("chunks")
    op.drop_table("document_revisions")
    op.drop_table("documents")
