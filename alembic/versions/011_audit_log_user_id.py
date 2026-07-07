"""Add user_id to audit_log for per-entry attribution.

Revision ID: 011
Revises: 010
Create Date: 2026-05-08
"""
from typing import Sequence, Union

from alembic import op

revision: str = "011"
down_revision: Union[str, None] = "010"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE audit_log
            ADD COLUMN IF NOT EXISTS user_id UUID REFERENCES users(id) ON DELETE SET NULL
    """)
    op.execute("CREATE INDEX IF NOT EXISTS ix_audit_log_user_id ON audit_log (user_id)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_audit_log_user_id")
    op.execute("ALTER TABLE audit_log DROP COLUMN IF EXISTS user_id")
