"""Durable ingest queue: persist user_notes + queued_at on ingest_jobs.

Makes the per-org write queue DB-backed: the ingest_jobs table itself is the
queue. `queued_at` gives durable FIFO ordering (survives restarts, consistent
across replicas) and `user_notes` persists the approve-time reviewer notes so a
job recovered after a restart still writes with them.

Revision ID: 025
Revises: 024
Create Date: 2026-06-28
"""
from typing import Sequence, Union

from alembic import op


revision: str = "025"
down_revision: Union[str, None] = "024"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE ingest_jobs
        ADD COLUMN IF NOT EXISTS user_notes TEXT NOT NULL DEFAULT ''
    """)
    op.execute("""
        ALTER TABLE ingest_jobs
        ADD COLUMN IF NOT EXISTS queued_at TIMESTAMPTZ
    """)
    # Drives the FIFO claim + queue-position query.
    op.execute("""
        CREATE INDEX IF NOT EXISTS ingest_jobs_org_status_queued_idx
        ON ingest_jobs (org_id, status, queued_at)
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ingest_jobs_org_status_queued_idx")
    op.execute("ALTER TABLE ingest_jobs DROP COLUMN IF EXISTS queued_at")
    op.execute("ALTER TABLE ingest_jobs DROP COLUMN IF EXISTS user_notes")
