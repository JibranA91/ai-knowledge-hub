"""Separate permission for ingesting writer-agent drafts.

Splits the "save & ingest writer draft" capability out of `can_upload`, so a
template can grant writer-mode ingest without granting the regular upload
button (and vice-versa).

For back-compat the built-ins mirror their `can_upload` value, so existing
power_user/contributor accounts keep their writer-ingest access.

Revision ID: 020
Revises: 019
Create Date: 2026-05-26
"""
from typing import Sequence, Union

from alembic import op


revision: str = "020"
down_revision: Union[str, None] = "019"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE permission_templates
        ADD COLUMN IF NOT EXISTS can_upload_writer_draft BOOLEAN NOT NULL DEFAULT FALSE
    """)
    # Mirror can_upload on built-ins so existing power_user/contributor users
    # keep the same ability to ingest writer drafts.
    op.execute("""
        UPDATE permission_templates
        SET can_upload_writer_draft = can_upload
        WHERE is_builtin = TRUE
    """)


def downgrade() -> None:
    op.execute("ALTER TABLE permission_templates DROP COLUMN IF EXISTS can_upload_writer_draft")
