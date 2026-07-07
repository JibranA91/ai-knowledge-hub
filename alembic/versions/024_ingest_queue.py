"""Per-org ingest queue: cancel_requested flag on ingest_jobs.

Cooperative, replica-safe cancellation. Any replica can set
cancel_requested=true; the per-org write worker checks it before starting a
queued job and the write loop checks it between page writes, so a job can be
cancelled regardless of which replica happens to be running it.

Revision ID: 024
Revises: 023
Create Date: 2026-06-19
"""
from typing import Sequence, Union

from alembic import op


revision: str = "024"
down_revision: Union[str, None] = "023"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE ingest_jobs
        ADD COLUMN IF NOT EXISTS cancel_requested BOOLEAN NOT NULL DEFAULT false
    """)


def downgrade() -> None:
    op.execute("ALTER TABLE ingest_jobs DROP COLUMN IF EXISTS cancel_requested")
