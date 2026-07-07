"""Multi-org memberships — a user can belong to many orgs with a per-org role.

Introduces the `org_memberships` join table carrying the full per-org access
profile (role, permission template, workspace, suspension, limit overrides).
One `users` row per identity (email stays globally unique); many memberships.

The existing `users.*` org columns (org_id, role, permission_template_id,
workspace_id, is_suspended, max_tokens_per_day, max_chat_messages_per_day) are
kept and backfilled into memberships so memberships become the source of truth
while leaving a safe rollback path. A later migration can drop the dead columns.

Admin users (role='admin', org_id IS NULL) stay global — they get no membership.

Revision ID: 023
Revises: 022
Create Date: 2026-06-02
"""
from typing import Sequence, Union

from alembic import op


revision: str = "023"
down_revision: Union[str, None] = "022"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS org_memberships (
            id                        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id                   UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            org_id                    UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
            role                      TEXT NOT NULL DEFAULT 'member'
                                          CHECK (role IN ('supervisor', 'member')),
            permission_template_id    UUID REFERENCES permission_templates(id) ON DELETE SET NULL,
            workspace_id              UUID REFERENCES workspaces(id) ON DELETE SET NULL,
            is_suspended              BOOL NOT NULL DEFAULT false,
            max_tokens_per_day        INTEGER,
            max_chat_messages_per_day INTEGER,
            created_at                TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE (user_id, org_id)
        )
    """)

    op.execute("""
        CREATE INDEX IF NOT EXISTS org_memberships_user_idx ON org_memberships (user_id)
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS org_memberships_org_idx ON org_memberships (org_id)
    """)

    # ── Backfill one membership per existing non-admin user (idempotent) ──────
    # Admins (role='admin' / no org) are intentionally excluded — they stay global.
    op.execute("""
        INSERT INTO org_memberships
            (user_id, org_id, role, permission_template_id, workspace_id,
             is_suspended, max_tokens_per_day, max_chat_messages_per_day)
        SELECT u.id, u.org_id,
               CASE WHEN u.role = 'supervisor' THEN 'supervisor' ELSE 'member' END,
               u.permission_template_id, u.workspace_id,
               u.is_suspended, u.max_tokens_per_day, u.max_chat_messages_per_day
        FROM users u
        WHERE u.org_id IS NOT NULL
          AND u.role <> 'admin'
        ON CONFLICT (user_id, org_id) DO NOTHING
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS org_memberships_org_idx")
    op.execute("DROP INDEX IF EXISTS org_memberships_user_idx")
    op.execute("DROP TABLE IF EXISTS org_memberships")
