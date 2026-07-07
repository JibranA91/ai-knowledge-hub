"""Add draft_ready flag to chat_sessions for writer-agent gating.

Writer-mode drafts can only be ingested once the agent has emitted a
`[DRAFT_READY]` marker — the route enforces 409 when this is false. The
column defaults to FALSE so existing writer sessions are treated as
"not yet ready"; users must re-prompt the agent for confirmation.

Revision ID: 021
Revises: 020
Create Date: 2026-05-26
"""
from typing import Sequence, Union

from alembic import op


revision: str = "021"
down_revision: Union[str, None] = "020"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE chat_sessions
        ADD COLUMN IF NOT EXISTS draft_ready BOOLEAN NOT NULL DEFAULT FALSE
    """)


def downgrade() -> None:
    op.execute("ALTER TABLE chat_sessions DROP COLUMN IF EXISTS draft_ready")
