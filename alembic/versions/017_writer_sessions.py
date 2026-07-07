"""Writer-mode chat sessions: adds mode, draft_content, draft_filename, user_id.

Writer drafts are stored in chat_sessions rows with mode='writer'. Drafts are
per-user (the picker lists only the current user's drafts), so we add user_id
which is nullable to preserve existing legacy chat rows that pre-date this
column.

Revision ID: 017
Revises: 016
Create Date: 2026-05-16
"""
from typing import Sequence, Union

from alembic import op


revision: str = "017"
down_revision: Union[str, None] = "016"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE chat_sessions
        ADD COLUMN IF NOT EXISTS mode VARCHAR(20) NOT NULL DEFAULT 'chat'
    """)
    op.execute("""
        ALTER TABLE chat_sessions
        ADD COLUMN IF NOT EXISTS draft_content TEXT NOT NULL DEFAULT ''
    """)
    op.execute("""
        ALTER TABLE chat_sessions
        ADD COLUMN IF NOT EXISTS draft_filename TEXT NOT NULL DEFAULT ''
    """)
    op.execute("""
        ALTER TABLE chat_sessions
        ADD COLUMN IF NOT EXISTS user_id UUID
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_chat_sessions_writer_user
        ON chat_sessions (org_id, user_id, last_active_at DESC)
        WHERE mode = 'writer'
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_chat_sessions_writer_user")
    op.execute("ALTER TABLE chat_sessions DROP COLUMN IF EXISTS user_id")
    op.execute("ALTER TABLE chat_sessions DROP COLUMN IF EXISTS draft_filename")
    op.execute("ALTER TABLE chat_sessions DROP COLUMN IF EXISTS draft_content")
    op.execute("ALTER TABLE chat_sessions DROP COLUMN IF EXISTS mode")
