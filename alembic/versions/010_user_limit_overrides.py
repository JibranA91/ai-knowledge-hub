"""Per-user token and chat message limit overrides.

When set, these columns take precedence over the assigned permission template's
limits for that individual user. NULL means "use template limit".

Revision ID: 010
Revises: 009
Create Date: 2026-05-08
"""
from typing import Sequence, Union

from alembic import op

revision: str = "010"
down_revision: Union[str, None] = "009"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE users
            ADD COLUMN IF NOT EXISTS max_tokens_per_day        INTEGER,
            ADD COLUMN IF NOT EXISTS max_chat_messages_per_day INTEGER
    """)


def downgrade() -> None:
    op.execute("ALTER TABLE users DROP COLUMN IF EXISTS max_tokens_per_day")
    op.execute("ALTER TABLE users DROP COLUMN IF EXISTS max_chat_messages_per_day")
