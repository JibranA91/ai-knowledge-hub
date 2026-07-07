"""Phase 7b: Supervisor role — extend role CHECK, add supervisor_user_id to orgs.

Revision ID: 007
Revises: 006
Create Date: 2026-05-07
"""
from typing import Sequence, Union

from alembic import op

revision: str = "007"
down_revision: Union[str, None] = "006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Allow 'supervisor' as a valid role value
    op.execute("ALTER TABLE users DROP CONSTRAINT IF EXISTS users_role_check")
    op.execute(
        "ALTER TABLE users ADD CONSTRAINT users_role_check "
        "CHECK (role IN ('admin', 'supervisor', 'member'))"
    )

    # Track the designated supervisor for each organization
    op.execute("""
        ALTER TABLE organizations
        ADD COLUMN IF NOT EXISTS supervisor_user_id UUID
            REFERENCES users(id) ON DELETE SET NULL
    """)


def downgrade() -> None:
    op.execute("ALTER TABLE organizations DROP COLUMN IF EXISTS supervisor_user_id")
    op.execute("ALTER TABLE users DROP CONSTRAINT IF EXISTS users_role_check")
    op.execute(
        "ALTER TABLE users ADD CONSTRAINT users_role_check "
        "CHECK (role IN ('admin', 'member'))"
    )
