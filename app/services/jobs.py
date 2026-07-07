"""Async DB-backed ingest job store — scoped to the current org via context var.

The asyncio.Task reference is kept in-process only (_tasks dict) since it
is not serialisable. All other job state is persisted to PostgreSQL.
"""
import asyncio
import json
from dataclasses import dataclass, field

import sqlalchemy as sa

from app.context import get_org_id, get_user_id
from app.db import get_db
from app.logger import get_logger

log = get_logger(__name__)

# In-process task references keyed by (org_id, filename) — not persisted
_tasks: dict[tuple[str, str], asyncio.Task] = {}


@dataclass
class IngestJob:
    filename: str
    org_id: str = ""
    user_id: str = ""
    status: str = "queued"
    message: str = ""
    pages_created: list[str] = field(default_factory=list)
    pages_updated: list[str] = field(default_factory=list)
    plan: list[dict] = field(default_factory=list)
    conflicts: list[dict] = field(default_factory=list)
    log_entry: str = ""
    doc_text: str = ""
    plan_chat_history: list[dict] = field(default_factory=list)
    cancel_requested: bool = False


def get_task(filename: str) -> asyncio.Task | None:
    try:
        org_id = get_org_id()
    except RuntimeError:
        org_id = ""
    return _tasks.get((org_id, filename))


def set_task(filename: str, task: asyncio.Task | None) -> None:
    try:
        org_id = get_org_id()
    except RuntimeError:
        org_id = ""
    key = (org_id, filename)
    if task is None:
        _tasks.pop(key, None)
    else:
        _tasks[key] = task


_INSERT = sa.text("""
    INSERT INTO ingest_jobs (org_id, user_id, filename, status)
    VALUES (CAST(:org_id AS UUID), CAST(:user_id AS UUID), :filename, 'queued')
    ON CONFLICT (org_id, filename) DO UPDATE SET
        status            = 'queued',
        user_id           = CAST(:user_id AS UUID),
        message           = '',
        plan              = '[]',
        conflicts         = '[]',
        log_entry         = '',
        doc_text          = '',
        plan_chat_history = '[]',
        pages_created     = '[]',
        pages_updated     = '[]',
        cancel_requested  = false,
        updated_at        = NOW()
""")

_SELECT = sa.text("""
    SELECT filename, org_id, user_id, status, message, plan, conflicts,
           log_entry, doc_text, plan_chat_history, pages_created, pages_updated,
           cancel_requested
    FROM ingest_jobs WHERE org_id = CAST(:org_id AS UUID) AND filename = :filename
""")

_UPDATE = sa.text("""
    UPDATE ingest_jobs SET
        status            = :status,
        message           = :message,
        plan              = CAST(:plan AS JSONB),
        conflicts         = CAST(:conflicts AS JSONB),
        log_entry         = :log_entry,
        doc_text          = :doc_text,
        plan_chat_history = CAST(:plan_chat_history AS JSONB),
        pages_created     = CAST(:pages_created AS JSONB),
        pages_updated     = CAST(:pages_updated AS JSONB),
        cancel_requested  = :cancel_requested,
        updated_at        = NOW()
    WHERE org_id = CAST(:org_id AS UUID) AND filename = :filename
""")


def _row_to_job(row) -> IngestJob:
    return IngestJob(
        filename=row.filename,
        org_id=str(row.org_id) if row.org_id else "",
        user_id=str(row.user_id) if row.user_id else "",
        status=row.status or "queued",
        message=row.message or "",
        plan=row.plan or [],
        conflicts=row.conflicts or [],
        log_entry=row.log_entry or "",
        doc_text=row.doc_text or "",
        plan_chat_history=row.plan_chat_history or [],
        pages_created=row.pages_created or [],
        pages_updated=row.pages_updated or [],
        cancel_requested=bool(getattr(row, "cancel_requested", False)),
    )


async def enqueue(filename: str) -> IngestJob:
    org_id = get_org_id()
    try:
        user_id = get_user_id()
    except RuntimeError:
        user_id = "00000000-0000-0000-0000-000000000000"
    async with get_db() as db:
        await db.execute(_INSERT, {"org_id": org_id, "user_id": user_id, "filename": filename})
    return IngestJob(filename=filename, org_id=org_id, user_id=user_id)


async def get(filename: str) -> IngestJob | None:
    org_id = get_org_id()
    async with get_db() as db:
        result = await db.execute(_SELECT, {"org_id": org_id, "filename": filename})
        row = result.fetchone()
    return _row_to_job(row) if row else None


async def save(job: IngestJob) -> None:
    org_id = job.org_id or get_org_id()
    async with get_db() as db:
        await db.execute(_UPDATE, {
            "org_id": org_id,
            "filename": job.filename,
            "status": job.status,
            "message": job.message,
            "plan": json.dumps(job.plan),
            "conflicts": json.dumps(job.conflicts),
            "log_entry": job.log_entry,
            "doc_text": job.doc_text,
            "plan_chat_history": json.dumps(job.plan_chat_history),
            "pages_created": json.dumps(job.pages_created),
            "pages_updated": json.dumps(job.pages_updated),
            "cancel_requested": job.cancel_requested,
        })


async def mark_stale_as_error() -> int:
    """On startup, mark any jobs left in 'processing' by a prior instance as error.

    Returns the number of rows updated.
    """
    async with get_db() as db:
        result = await db.execute(sa.text("""
            UPDATE ingest_jobs
               SET status  = 'error',
                   message = 'Interrupted by server restart'
             WHERE status IN ('processing', 'writing')
        """))
        count = result.rowcount or 0
    if count:
        log.warning("startup | marked %d stale processing job(s) as error", count)
    return count


async def cancel(filename: str) -> bool:
    org_id = get_org_id()
    job = await get(filename)
    if not job:
        return False
    task = _tasks.get((org_id, filename))
    if task and not task.done():
        task.cancel()
    _tasks.pop((org_id, filename), None)
    job.status = "cancelled"
    await save(job)
    return True


# Statuses where nothing is actively writing on a replica yet, so a cancel can
# flip straight to 'cancelled'. For 'writing', the write loop aborts itself
# between pages after observing cancel_requested (see ingest_agent).
_CANCEL_FLIP_STATUSES = {"queued", "processing", "pending_review", "queued_write"}


async def request_cancel(filename: str) -> bool:
    """Cooperative, replica-safe cancel.

    Persists `cancel_requested=true` so whichever replica is running the job
    aborts at its next checkpoint (the queue worker before it starts a job, the
    write loop between page writes). Also cancels the local asyncio task as a
    fast path, and flips non-writing jobs straight to 'cancelled' so the UI
    updates immediately.
    """
    org_id = get_org_id()
    job = await get(filename)
    if not job:
        return False
    job.cancel_requested = True
    task = _tasks.get((org_id, filename))
    if task and not task.done():
        task.cancel()
    _tasks.pop((org_id, filename), None)
    if job.status in _CANCEL_FLIP_STATUSES:
        job.status = "cancelled"
        job.message = "Ingest cancelled by user"
    await save(job)
    return True


async def is_cancel_requested(filename: str) -> bool:
    """Re-read the cancel flag from the DB so any replica's write loop sees it."""
    org_id = get_org_id()
    async with get_db() as db:
        result = await db.execute(sa.text("""
            SELECT cancel_requested FROM ingest_jobs
            WHERE org_id = CAST(:org_id AS UUID) AND filename = :filename
        """), {"org_id": org_id, "filename": filename})
        row = result.fetchone()
    return bool(row and row.cancel_requested)
