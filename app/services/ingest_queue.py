"""Per-org ingest write queue — DB-backed, durable, replica-safe.

The queue *is* the ``ingest_jobs`` table: an approved job sits at
``status='queued_write'`` with a ``queued_at`` timestamp. A per-org *drain*
claims jobs FIFO under a Postgres session-level advisory lock (so only one
replica writes for an org at a time), runs each write, and marks it done/error.

Why DB-backed (vs the old in-memory asyncio.Queue):
  * **Durable** — jobs and their order survive a restart (the rows persist;
    ``queued_at`` preserves FIFO). No in-memory re-enqueue needed.
  * **Replica-safe** — `pg_try_advisory_lock` guarantees a single writer per org
    across the whole cluster; a background poller lets any replica pick up work
    submitted on a since-dead replica.
  * **Visible** — queue position is a DB query, so the admin Jobs tab shows the
    true cross-replica position, not one replica's local view.
  * **Self-healing** — a job orphaned mid-write by a dead replica is reclaimed:
    whoever next acquires the org's advisory lock sees a stale ``writing`` row
    (no live writer can hold the lock) and marks it errored.

Submit fast-paths a drain on the receiving replica; the poller is the failover.
"""
import asyncio

import sqlalchemy as sa

from app.context import UserContext, current_user
from app.db import get_conn, get_db
from app.logger import get_logger

log = get_logger(__name__)

# classid for pg_advisory_lock(classid, objid) — distinct from the dead
# path-keyed helper (which used 1) so the two lock spaces never collide.
_ADVISORY_LOCK_CLASS = 42

# Background failover/recovery scan cadence. The common case (a job submitted on
# this replica) drains instantly via submit()'s kick; the poller only matters
# for picking up work left by another (possibly dead) replica.
_POLL_SECS = 15

# Max time shutdown() waits for in-flight drains to finish their current job
# before giving up. Anything still writing is left to the DB reclaim
# (_RECLAIM_STALE) on next boot, so we never block a deploy unbounded.
_SHUTDOWN_GRACE_SECS = 10

# Orgs currently being drained on THIS replica (avoids duplicate in-process drains).
_draining: set[str] = set()
_tasks: set[asyncio.Task] = set()
_poller: asyncio.Task | None = None
_stopped = False


_SUBMIT = sa.text("""
    UPDATE ingest_jobs
       SET status = 'queued_write', user_notes = :notes,
           queued_at = NOW(), cancel_requested = false, updated_at = NOW()
     WHERE org_id = CAST(:org AS UUID) AND filename = :filename
       AND status <> 'writing'
""")

_CLAIM_NEXT = sa.text("""
    UPDATE ingest_jobs SET status = 'writing', updated_at = NOW()
     WHERE id = (
        SELECT id FROM ingest_jobs
         WHERE org_id = CAST(:org AS UUID)
           AND status = 'queued_write'
           AND cancel_requested = false
         ORDER BY queued_at NULLS FIRST, id
         LIMIT 1
     )
    RETURNING filename, user_notes, user_id::text AS user_id
""")

_SWEEP_CANCELLED = sa.text("""
    UPDATE ingest_jobs
       SET status = 'cancelled', message = 'Ingest cancelled by user', updated_at = NOW()
     WHERE org_id = CAST(:org AS UUID)
       AND status = 'queued_write' AND cancel_requested = true
""")

_RECLAIM_STALE = sa.text("""
    UPDATE ingest_jobs
       SET status = 'error', message = 'Interrupted by server restart', updated_at = NOW()
     WHERE org_id = CAST(:org AS UUID) AND status = 'writing'
""")

# Defensive: a write phase that raised (a bug — _execute_ingest normally handles
# its own errors) leaves the claimed row stuck at 'writing'. Mark it errored so
# the queue isn't stalled and the row isn't orphaned.
_MARK_FAILED = sa.text("""
    UPDATE ingest_jobs
       SET status = 'error', message = 'Ingest failed', updated_at = NOW()
     WHERE org_id = CAST(:org AS UUID) AND filename = :filename AND status = 'writing'
""")

_PENDING_ORGS = sa.text("""
    SELECT DISTINCT org_id::text AS org_id FROM ingest_jobs
     WHERE status IN ('queued_write', 'writing')
""")

_POSITION = sa.text("""
    SELECT (
        SELECT COUNT(*) FROM ingest_jobs b
         WHERE b.org_id = a.org_id AND b.status = 'queued_write'
           AND (b.queued_at < a.queued_at
                OR (b.queued_at = a.queued_at AND b.id <= a.id))
    ) AS pos
    FROM ingest_jobs a
    WHERE a.org_id = CAST(:org AS UUID) AND a.filename = :filename
      AND a.status = 'queued_write'
""")


async def submit(filename: str, user_notes: str, org_id: str, user_id: str) -> None:
    """Mark an approved job queued for writing, then kick its org's drain."""
    async with get_db() as db:
        result = await db.execute(_SUBMIT, {"org": org_id, "filename": filename, "notes": user_notes or ""})
    # _SUBMIT refuses to touch a row already 'writing' (a drain is mid-ingest);
    # re-queuing it would let it be claimed and ingested twice. No eligible row
    # means it's in-flight or gone — nothing to kick.
    if not result.rowcount:
        log.warning("ingest_queue | submit skipped (already writing or missing) | org=%s | file=%s", org_id, filename)
        return
    log.info("ingest_queue | queued | org=%s | file=%s", org_id, filename)
    _kick(org_id)


def _kick(org_id: str) -> None:
    """Spawn a drain for an org unless one is already running on this replica."""
    if _stopped or org_id in _draining:
        return
    # Claim synchronously, BEFORE create_task: _drain_org adds itself only once
    # it starts running, so without claiming here a second _kick (e.g. submit
    # racing the poller) would pass the guard and spawn a duplicate drain.
    _draining.add(org_id)
    t = asyncio.create_task(_drain_org(org_id))
    _tasks.add(t)
    t.add_done_callback(_tasks.discard)


async def _drain_org(org_id: str) -> None:
    # Production always enters via _kick, which has already claimed `_draining`.
    # Direct callers (tests) skip that; the advisory lock below is the real
    # cross-process guard. We only own the `_draining` discard in `finally`.
    _draining.add(org_id)
    try:
        # The advisory lock is SESSION-level — bound to ONE physical connection,
        # held until explicitly unlocked or the connection closes. We therefore
        # hold a single dedicated connection (get_conn()) for the whole drain:
        # unlike get_db()'s Session, which returns its connection to the pool on
        # every commit() (which would STRAND the lock on a pooled connection —
        # the later unlock would run on a *different* connection and silently
        # no-op, wedging the org's queue forever). get_conn() keeps the
        # connection checked out across commits until the block exits, so
        # acquire, claim, and release all happen on the same connection.
        async with get_conn() as conn:
            got = (await conn.execute(
                sa.text("SELECT pg_try_advisory_lock(:cls, hashtext(:org))"),
                {"cls": _ADVISORY_LOCK_CLASS, "org": org_id},
            )).scalar()
            # End the acquire transaction. The lock persists on THIS connection;
            # we just must not keep an open transaction (and its row locks) while
            # awaiting the long write phase below.
            await conn.commit()
            if not got:
                return  # another replica/process holds the org's write slot
            try:
                # We hold the lock, so any 'writing' row has no live writer →
                # it was orphaned by a crash; mark it errored (partial writes
                # stay recoverable via History/revert). Then drop cancelled.
                await conn.execute(_RECLAIM_STALE, {"org": org_id})
                await conn.execute(_SWEEP_CANCELLED, {"org": org_id})
                await conn.commit()
                while not _stopped:
                    row = (await conn.execute(_CLAIM_NEXT, {"org": org_id})).fetchone()
                    # Commit the claim NOW — before the write phase. Otherwise the
                    # claim's UPDATE (status='writing') stays uncommitted, holding
                    # a row lock; _execute_ingest then updates the SAME row on a
                    # different connection and blocks on that lock → self-deadlock
                    # (Postgres can't see it; the drain is awaiting, not lock-waiting).
                    # Committing also makes 'writing' visible to the Jobs tab.
                    await conn.commit()
                    if row is None:
                        break
                    # Isolate per-job failures so one bad write can't stall the
                    # rest of the org's queue (real _execute_ingest handles its
                    # own errors; this guards against unexpected raises).
                    try:
                        await _run_job(org_id, row.filename, row.user_notes or "", row.user_id or "")
                    except Exception:
                        log.exception("ingest_queue | job failed | org=%s | file=%s", org_id, row.filename)
                        await conn.execute(_MARK_FAILED, {"org": org_id, "filename": row.filename})
                        await conn.commit()
            finally:
                await conn.execute(
                    sa.text("SELECT pg_advisory_unlock(:cls, hashtext(:org))"),
                    {"cls": _ADVISORY_LOCK_CLASS, "org": org_id},
                )
                await conn.commit()
    except Exception:
        log.exception("ingest_queue | drain error | org=%s", org_id)
    finally:
        _draining.discard(org_id)


async def _run_job(org_id: str, filename: str, user_notes: str, user_id: str) -> None:
    # Restore the submitting user's context so wiki writes (and the tracked
    # action) are attributed correctly inside this background task.
    current_user.set(UserContext(user_id=user_id, org_id=org_id, email="", role="member"))
    from app.routes.operations import _execute_ingest
    await _execute_ingest(filename, user_notes)


async def queue_position(filename: str, org_id: str) -> int | None:
    """1-based FIFO position of a still-queued job (cross-replica), else None."""
    async with get_db() as db:
        row = (await db.execute(_POSITION, {"org": org_id, "filename": filename})).fetchone()
    return int(row.pos) if row and row.pos else None


async def _initial_scan() -> None:
    """Kick drains for every org with pending/orphaned jobs (boot + each poll)."""
    try:
        async with get_db() as db:
            rows = (await db.execute(_PENDING_ORGS)).fetchall()
    except Exception:
        log.exception("ingest_queue | pending-orgs scan failed")
        return
    for r in rows:
        _kick(r.org_id)


async def _poll_loop() -> None:
    while not _stopped:
        try:
            await asyncio.sleep(_POLL_SECS)
            await _initial_scan()
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("ingest_queue | poller error")


def start() -> None:
    """Begin processing on startup: launch the failover poller + an immediate scan."""
    global _poller, _stopped
    _stopped = False
    if _poller is None:
        _poller = asyncio.create_task(_poll_loop())
    _tasks.add(asyncio.create_task(_initial_scan()))


async def shutdown() -> None:
    """Stop the poller and let in-flight drains finish their current job.

    Bounded by _SHUTDOWN_GRACE_SECS: a drain only re-checks _stopped between
    jobs, so a long in-flight ingest could otherwise block shutdown for minutes
    (risking a SIGKILL mid-write on deploy). Anything still running past the
    grace is left detached — its row stays 'writing' and is reclaimed by
    _RECLAIM_STALE on the next boot.
    """
    global _poller, _stopped
    _stopped = True
    if _poller is not None:
        _poller.cancel()
        try:
            await _poller
        except (asyncio.CancelledError, Exception):
            pass
        _poller = None
    tasks = [t for t in _tasks if not t.done()]
    if tasks:
        done, pending = await asyncio.wait(tasks, timeout=_SHUTDOWN_GRACE_SECS)
        for t in done:
            if not t.cancelled() and t.exception() is not None:
                pass  # exceptions are already logged inside _drain_org
        if pending:
            log.warning("ingest_queue | shutdown grace (%ss) exceeded; %d drain(s) left to DB reclaim",
                        _SHUTDOWN_GRACE_SECS, len(pending))
    _tasks.clear()
    _draining.clear()
