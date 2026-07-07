"""Change `revision_retention_days` default from 90 → 21.

Also updates any existing org rows still on the prior 90-day default so the
change is visible immediately. Orgs that explicitly chose a different value
(anything not equal to 90) are left untouched.

Revision ID: 016
Revises: 015
Create Date: 2026-05-16
"""
from typing import Sequence, Union

from alembic import op


revision: str = "016"
down_revision: Union[str, None] = "015"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE organizations
        ALTER COLUMN revision_retention_days SET DEFAULT 21
    """)
    op.execute("""
        UPDATE organizations
        SET revision_retention_days = 21
        WHERE revision_retention_days = 90
    """)


def downgrade() -> None:
    op.execute("""
        ALTER TABLE organizations
        ALTER COLUMN revision_retention_days SET DEFAULT 90
    """)
    op.execute("""
        UPDATE organizations
        SET revision_retention_days = 90
        WHERE revision_retention_days = 21
    """)
