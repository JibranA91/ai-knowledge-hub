"""Phase 4: JWT auth columns on auth_tokens + rate_limit_counters table.

Revision ID: 003
Revises: 002
Create Date: 2026-05-06
"""
from typing import Sequence, Union

from alembic import op

revision: str = "003"
down_revision: Union[str, None] = "002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # auth_tokens: add token_type and username for refresh-token tracking
    op.execute("""
        ALTER TABLE auth_tokens
        ADD COLUMN IF NOT EXISTS token_type TEXT NOT NULL DEFAULT 'refresh'
    """)
    op.execute("""
        ALTER TABLE auth_tokens
        ADD COLUMN IF NOT EXISTS username TEXT NOT NULL DEFAULT ''
    """)

    # rate_limit_counters: per-user per-endpoint sliding-window counters
    op.execute("""
        CREATE TABLE IF NOT EXISTS rate_limit_counters (
            id           SERIAL PRIMARY KEY,
            identifier   TEXT NOT NULL,
            endpoint     TEXT NOT NULL,
            window_start TIMESTAMPTZ NOT NULL,
            count        INT  NOT NULL DEFAULT 1,
            CONSTRAINT rate_limit_counters_unique
                UNIQUE (identifier, endpoint, window_start)
        )
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS rate_limit_idx
        ON rate_limit_counters (identifier, endpoint, window_start)
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS rate_limit_idx")
    op.execute("DROP TABLE IF EXISTS rate_limit_counters")
    op.execute("ALTER TABLE auth_tokens DROP COLUMN IF EXISTS username")
    op.execute("ALTER TABLE auth_tokens DROP COLUMN IF EXISTS token_type")
