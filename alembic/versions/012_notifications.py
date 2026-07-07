"""Persistent notifications table for cross-session alerts.

Revision ID: 012
Revises: 011
Create Date: 2026-05-08
"""
from typing import Sequence, Union

from alembic import op

revision: str = "012"
down_revision: Union[str, None] = "011"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS notifications (
            id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id    UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            org_id     UUID REFERENCES organizations(id) ON DELETE CASCADE,
            type       VARCHAR(50)  NOT NULL,
            title      VARCHAR(255) NOT NULL,
            body       TEXT         NOT NULL DEFAULT '',
            link       VARCHAR(500) NOT NULL DEFAULT '',
            metadata   JSONB        NOT NULL DEFAULT '{}',
            is_read    BOOLEAN      NOT NULL DEFAULT FALSE,
            created_at TIMESTAMPTZ  NOT NULL DEFAULT NOW()
        )
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_notifications_user_unread
            ON notifications (user_id, is_read, created_at DESC)
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_notifications_user_unread")
    op.execute("DROP TABLE IF EXISTS notifications")
