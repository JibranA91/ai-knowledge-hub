"""Background revert jobs — track progress + persist error message on wiki_actions.

Reverting many actions (e.g. 100 ingests) can take 30s+ and exceeds typical
proxy timeouts. The revert is now spawned as a background task and the user
polls progress via `GET /api/admin/history/revert-job/{id}`. These columns
back that polling.

Revision ID: 022
Revises: 021
Create Date: 2026-05-26
"""
from typing import Sequence, Union

from alembic import op


revision: str = "022"
down_revision: Union[str, None] = "021"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE wiki_actions
        ADD COLUMN IF NOT EXISTS progress_done INTEGER NOT NULL DEFAULT 0
    """)
    op.execute("""
        ALTER TABLE wiki_actions
        ADD COLUMN IF NOT EXISTS progress_total INTEGER NOT NULL DEFAULT 0
    """)
    op.execute("""
        ALTER TABLE wiki_actions
        ADD COLUMN IF NOT EXISTS error_message TEXT
    """)


def downgrade() -> None:
    op.execute("ALTER TABLE wiki_actions DROP COLUMN IF EXISTS error_message")
    op.execute("ALTER TABLE wiki_actions DROP COLUMN IF EXISTS progress_total")
    op.execute("ALTER TABLE wiki_actions DROP COLUMN IF EXISTS progress_done")
