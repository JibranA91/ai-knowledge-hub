"""Add wiki_files table for special files (index.md, log.md, AGENTS.md, graph cache).

Revision ID: 002
Revises: 001
Create Date: 2025-01-01
"""
from alembic import op
import sqlalchemy as sa

revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "wiki_files",
        sa.Column("key",        sa.Text(), nullable=False),
        sa.Column("content",    sa.Text(), nullable=False, server_default=""),
        sa.Column("updated_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("NOW()")),
        sa.PrimaryKeyConstraint("key"),
    )


def downgrade() -> None:
    op.drop_table("wiki_files")
