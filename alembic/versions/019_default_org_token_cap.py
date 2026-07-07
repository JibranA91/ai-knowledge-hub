"""Seed a 10M-token/day cap on the Default Organization.

Only applies when the cap is currently NULL (unlimited), so admins who have
already configured a different value via /api/admin/org-limits are not
overridden by this one-shot.

Revision ID: 019
Revises: 018
Create Date: 2026-05-25
"""
from typing import Sequence, Union

from alembic import op


revision: str = "019"
down_revision: Union[str, None] = "018"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_DEFAULT_ORG_ID = "00000000-0000-0000-0000-000000000001"
_DEFAULT_TOKENS_PER_DAY = 10_000_000


def upgrade() -> None:
    op.execute(f"""
        UPDATE organizations
        SET max_tokens_per_day_org = {_DEFAULT_TOKENS_PER_DAY}
        WHERE id = '{_DEFAULT_ORG_ID}'
          AND max_tokens_per_day_org IS NULL
    """)


def downgrade() -> None:
    op.execute(f"""
        UPDATE organizations
        SET max_tokens_per_day_org = NULL
        WHERE id = '{_DEFAULT_ORG_ID}'
          AND max_tokens_per_day_org = {_DEFAULT_TOKENS_PER_DAY}
    """)
