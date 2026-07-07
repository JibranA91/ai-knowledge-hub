"""Recalibration job state — one in-memory job per org_id.

PostgreSQL row tracks running/done status so the lock middleware
works correctly across multiple replicas.
"""
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone

import sqlalchemy as sa

from app.logger import get_logger

log = get_logger(__name__)


@dataclass
class RecalibrateJob:
    status: str = "idle"        # idle | running | done | done_with_errors | error
    stage: str = ""
    progress: int = 0
    details: str = ""
    fact_instructions: str = ""
    pages_improved: list[str] = field(default_factory=list)
    pages_deleted: list[str] = field(default_factory=list)
    pages_renamed: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    started_at: str = ""
    finished_at: str = ""
    task: asyncio.Task | None = field(default=None, repr=False)
    _db_id: int | None = field(default=None, repr=False)


# Per-org in-memory jobs: {org_id → RecalibrateJob}
_jobs: dict[str, RecalibrateJob] = {}


def _current_org_id() -> str:
    try:
        from app.context import get_org_id
        return get_org_id()
    except RuntimeError:
        return ""


def get(org_id: str | None = None) -> RecalibrateJob:
    key = org_id or _current_org_id()
    if key not in _jobs:
        _jobs[key] = RecalibrateJob()
    return _jobs[key]


def reset(org_id: str | None = None) -> None:
    key = org_id or _current_org_id()
    old = _jobs.get(key)
    if old and old.task and not old.task.done():
        old.task.cancel()
    _jobs[key] = RecalibrateJob()


async def persist_start(org_id: str | None = None) -> None:
    """Insert a running row in recalibrate_jobs so other replicas see the lock."""
    key = org_id or _current_org_id()
    from app.db import get_db
    async with get_db() as db:
        result = await db.execute(sa.text("""
            INSERT INTO recalibrate_jobs (org_id, status, started_at)
            VALUES (CAST(:org_id AS UUID), 'running', NOW())
            RETURNING id
        """), {"org_id": key})
        row = result.fetchone()
        if row:
            _jobs[key]._db_id = row.id


async def persist_finish(status: str, org_id: str | None = None) -> None:
    """Update the DB row when the job completes."""
    import json
    key = org_id or _current_org_id()
    job = _jobs.get(key)
    if job is None or job._db_id is None:
        return
    from app.db import get_db
    async with get_db() as db:
        await db.execute(sa.text("""
            UPDATE recalibrate_jobs SET
                status          = :status,
                stage           = :stage,
                progress        = :progress,
                pages_improved  = CAST(:pages_improved AS JSONB),
                pages_deleted   = CAST(:pages_deleted  AS JSONB),
                pages_renamed   = CAST(:pages_renamed  AS JSONB),
                errors          = CAST(:errors         AS JSONB),
                finished_at     = NOW()
            WHERE id = :id
        """), {
            "id": job._db_id,
            "status": status,
            "stage": job.stage,
            "progress": job.progress,
            "pages_improved": json.dumps(job.pages_improved),
            "pages_deleted": json.dumps(job.pages_deleted),
            "pages_renamed": json.dumps(job.pages_renamed),
            "errors": json.dumps(job.errors),
        })


async def is_running_in_db(org_id: str | None = None) -> bool:
    """Check whether any replica has a running recalibration for this org."""
    key = org_id or _current_org_id()
    from app.db import get_db
    async with get_db() as db:
        result = await db.execute(
            sa.text("""
                SELECT COUNT(*) FROM recalibrate_jobs
                WHERE org_id = CAST(:org_id AS UUID) AND status = 'running'
            """),
            {"org_id": key},
        )
        return (result.scalar() or 0) > 0


async def mark_stale_as_error() -> int:
    """On startup, mark any recalibration jobs left 'running' by a prior instance as error.

    Without this sweep a crashed/restarted server would leave the recalibrate
    lock engaged for affected orgs, blocking all write operations indefinitely.

    Returns the number of rows updated.
    """
    from app.db import get_db
    async with get_db() as db:
        result = await db.execute(sa.text("""
            UPDATE recalibrate_jobs
               SET status      = 'error',
                   errors      = errors || '["Interrupted by server restart"]'::jsonb,
                   finished_at = NOW()
             WHERE status = 'running'
        """))
        count = result.rowcount or 0
    if count:
        log.warning("startup | marked %d stale recalibration job(s) as error", count)
    return count
