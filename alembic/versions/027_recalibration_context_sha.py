"""Recalibration fingerprints: add context_sha for relational-signal idempotency.

content_sha alone can't tell that a page's orphan / broken_links / duplicate
flag changed because a *neighbour* changed (the page's own text is untouched).
context_sha hashes those relational inputs — inbound link sources, outbound
target existence, and same-title group — so recalibration re-reviews a page when
its neighbourhood changes, while staying idempotent when nothing relevant did.

Revision ID: 027
Revises: 026
Create Date: 2026-07-02
"""
from typing import Sequence, Union

from alembic import op


revision: str = "027"
down_revision: Union[str, None] = "026"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE wiki_recalibration
        ADD COLUMN IF NOT EXISTS context_sha TEXT NOT NULL DEFAULT ''
    """)


def downgrade() -> None:
    op.execute("ALTER TABLE wiki_recalibration DROP COLUMN IF EXISTS context_sha")
