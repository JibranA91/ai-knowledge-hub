"""Recalibration fingerprints: per-page content hash + last-reviewed timestamp.

Lets master recalibration run idempotently. A page already reviewed at its
current content hash is skipped, and staleness is measured from the last review
(not just the original ingest date) — so repeated runs over a healthy wiki do
no work. Internal bookkeeping only; deliberately excluded from change-tracking.

Revision ID: 026
Revises: 025
Create Date: 2026-07-02
"""
from typing import Sequence, Union

from alembic import op


revision: str = "026"
down_revision: Union[str, None] = "025"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS wiki_recalibration (
            org_id      UUID        NOT NULL,
            path        TEXT        NOT NULL,
            content_sha TEXT        NOT NULL,
            reviewed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (org_id, path)
        )
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS wiki_recalibration")
