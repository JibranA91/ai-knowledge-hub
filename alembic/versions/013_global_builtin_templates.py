"""Make built-in permission templates global (org_id nullable).

Built-ins (read_only, contributor, power_user) are shared across all
organisations. Only custom templates are tied to a specific org.

Revision ID: 013
Revises: 012
Create Date: 2026-05-08
"""
from typing import Sequence, Union

from alembic import op

revision: str = "013"
down_revision: Union[str, None] = "012"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Drop the old FK constraint + unique constraint so we can change org_id
    op.execute("""
        ALTER TABLE permission_templates
            DROP CONSTRAINT IF EXISTS permission_templates_org_id_name_key
    """)
    op.execute("""
        ALTER TABLE permission_templates
            DROP CONSTRAINT IF EXISTS permission_templates_org_id_fkey
    """)

    # 2. Make org_id nullable
    op.execute("""
        ALTER TABLE permission_templates
            ALTER COLUMN org_id DROP NOT NULL
    """)

    # 3. Re-add the FK (now nullable — NULLs are allowed)
    op.execute("""
        ALTER TABLE permission_templates
            ADD CONSTRAINT permission_templates_org_id_fkey
            FOREIGN KEY (org_id) REFERENCES organizations(id) ON DELETE CASCADE
    """)

    # 4. New unique constraint:
    #    - (name) UNIQUE WHERE org_id IS NULL  → one global set of built-ins
    #    - (org_id, name) UNIQUE WHERE org_id IS NOT NULL  → per-org custom templates
    op.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS uq_permission_templates_global_name
            ON permission_templates (name)
            WHERE org_id IS NULL
    """)
    op.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS uq_permission_templates_org_name
            ON permission_templates (org_id, name)
            WHERE org_id IS NOT NULL
    """)

    # 5. Migrate existing per-org built-ins: keep one global copy, delete duplicates.
    #    Strategy: for each built-in name, find one row and set its org_id to NULL,
    #    then delete all other rows with that name and is_builtin=true.
    for name in ('read_only', 'contributor', 'power_user'):
        # Pick the first existing built-in row for this name (any org)
        op.execute(f"""
            UPDATE permission_templates SET org_id = NULL
            WHERE id = (
                SELECT id FROM permission_templates
                WHERE name = '{name}' AND is_builtin = true
                LIMIT 1
            )
              AND is_builtin = true
        """)
        # Delete the remaining per-org duplicates
        op.execute(f"""
            DELETE FROM permission_templates
            WHERE name = '{name}' AND is_builtin = true AND org_id IS NOT NULL
        """)


def downgrade() -> None:
    # Remove the partial unique indexes
    op.execute("DROP INDEX IF EXISTS uq_permission_templates_global_name")
    op.execute("DROP INDEX IF EXISTS uq_permission_templates_org_name")

    # Restore org_id NOT NULL (requires all rows to have an org_id — not fully reversible
    # for the global built-ins, but best-effort for development)
    op.execute("""
        ALTER TABLE permission_templates
            ALTER COLUMN org_id SET NOT NULL
    """)

    # Restore original unique constraint
    op.execute("""
        ALTER TABLE permission_templates
            ADD CONSTRAINT permission_templates_org_id_name_key UNIQUE (org_id, name)
    """)
