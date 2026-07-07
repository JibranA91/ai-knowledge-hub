"""Admin users have no organization — org_id is now nullable.

Remove the "admin" built-in permission template (admin role bypasses all checks).
Set org_id = NULL for existing admin-role users.

Revision ID: 008
Revises: 007
Create Date: 2026-05-08
"""
from typing import Sequence, Union

from alembic import op

revision: str = "008"
down_revision: Union[str, None] = "007"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Allow admin users to have no org
    op.execute("ALTER TABLE users ALTER COLUMN org_id DROP NOT NULL")

    # Admin role already bypasses all permission checks — the "admin" template is redundant.
    # Users assigned to it get permission_template_id = NULL (ON DELETE SET NULL), which is
    # fine since admin role grants everything.
    op.execute("""
        DELETE FROM permission_templates
        WHERE is_builtin = true AND name = 'admin'
    """)

    # Existing admin users should not belong to any org
    op.execute("UPDATE users SET org_id = NULL WHERE role = 'admin'")


def downgrade() -> None:
    # Re-attach admins to the default org
    op.execute("""
        UPDATE users SET org_id = '00000000-0000-0000-0000-000000000001'::uuid
        WHERE role = 'admin' AND org_id IS NULL
    """)
    op.execute("ALTER TABLE users ALTER COLUMN org_id SET NOT NULL")
