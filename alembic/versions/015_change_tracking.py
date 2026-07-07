"""Change tracking + revert: wiki_actions, wiki_revisions, retention setting.

Every user action that mutates wiki state opens a `wiki_actions` row. Every
write performed under that action emits a `wiki_revisions` row with the
content before and after the write so the action can be reverted.

Revision ID: 015
Revises: 014
Create Date: 2026-05-16
"""
from typing import Sequence, Union

from alembic import op


revision: str = "015"
down_revision: Union[str, None] = "014"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS wiki_actions (
            id            UUID PRIMARY KEY,
            org_id        UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
            user_id       UUID,
            action_type   TEXT NOT NULL,
            summary       TEXT NOT NULL DEFAULT '',
            status        TEXT NOT NULL DEFAULT 'running',
            started_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            finished_at   TIMESTAMPTZ,
            revert_of_id  UUID REFERENCES wiki_actions(id) ON DELETE SET NULL,
            details       JSONB NOT NULL DEFAULT '{}'
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS wiki_actions_org_started_idx ON wiki_actions (org_id, started_at DESC)")
    op.execute("CREATE INDEX IF NOT EXISTS wiki_actions_org_status_idx  ON wiki_actions (org_id, status)")

    op.execute("""
        CREATE TABLE IF NOT EXISTS wiki_revisions (
            id                BIGSERIAL PRIMARY KEY,
            action_id         UUID NOT NULL REFERENCES wiki_actions(id) ON DELETE CASCADE,
            org_id            UUID NOT NULL,
            target_kind       TEXT NOT NULL,
            target_key        TEXT NOT NULL,
            op                TEXT NOT NULL,
            content_before    TEXT,
            content_after     TEXT,
            bytes_size_before BIGINT,
            bytes_size_after  BIGINT,
            created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS wiki_revisions_action_idx ON wiki_revisions (action_id)")
    op.execute("CREATE INDEX IF NOT EXISTS wiki_revisions_org_target_idx ON wiki_revisions (org_id, target_kind, target_key)")

    op.execute("""
        ALTER TABLE organizations
        ADD COLUMN IF NOT EXISTS revision_retention_days INT NOT NULL DEFAULT 90
    """)


def downgrade() -> None:
    op.execute("ALTER TABLE organizations DROP COLUMN IF EXISTS revision_retention_days")
    op.execute("DROP TABLE IF EXISTS wiki_revisions")
    op.execute("DROP TABLE IF EXISTS wiki_actions")
