"""Permission flag for document-writer tool access.

Adds `can_use_writer` to permission_templates and grants it to the built-in
`power_user` template so existing power-user accounts keep their access.

Revision ID: 018
Revises: 017
Create Date: 2026-05-22
"""
from typing import Sequence, Union

from alembic import op


revision: str = "018"
down_revision: Union[str, None] = "017"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE permission_templates
        ADD COLUMN IF NOT EXISTS can_use_writer BOOLEAN NOT NULL DEFAULT FALSE
    """)
    op.execute("""
        UPDATE permission_templates
        SET can_use_writer = TRUE
        WHERE is_builtin = TRUE AND name = 'power_user'
    """)


def downgrade() -> None:
    op.execute("ALTER TABLE permission_templates DROP COLUMN IF EXISTS can_use_writer")
