"""Integration tests for the durable, DB-backed per-org ingest write queue.

The queue is the ingest_jobs table (status='queued_write' + queued_at). Tests
drive the drain directly (deterministic) with a probe replacing the write phase,
plus an end-to-end submit→drain path. Requires PostgreSQL.
"""
import asyncio

import pytest
import pytest_asyncio
import sqlalchemy as sa
from unittest.mock import patch

from app.db import get_db
from app.services import ingest_queue
from app.services.jobs import enqueue, get, save


@pytest_asyncio.fixture(autouse=True)
async def _reset_queue():
    ingest_queue._stopped = False
    ingest_queue._draining.clear()
    yield
    await ingest_queue.shutdown()
    ingest_queue._stopped = False
    ingest_queue._draining.clear()


async def _queue(filename: str, org_id: str, notes: str = "") -> None:
    """Put a job at status='queued_write' with a queue time — no auto-kick."""
    await enqueue(filename)
    async with get_db() as db:
        await db.execute(sa.text("""
            UPDATE ingest_jobs SET status='queued_write', queued_at=NOW(), user_notes=:n,
                   updated_at=NOW()
            WHERE org_id = CAST(:o AS UUID) AND filename = :f
        """), {"o": org_id, "f": filename, "n": notes})


# ── FIFO drain + serialization ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_drain_processes_jobs_fifo(client, user_ctx, default_user):
    org_id = default_user["org_id"]
    await _queue("a.txt", org_id)
    await _queue("b.txt", org_id)

    order, active, max_concurrent = [], {"n": 0}, {"v": 0}

    async def fake_exec(filename, notes=""):
        active["n"] += 1
        max_concurrent["v"] = max(max_concurrent["v"], active["n"])
        order.append(filename)
        await asyncio.sleep(0.03)
        active["n"] -= 1

    with patch("app.routes.operations._execute_ingest", fake_exec):
        await ingest_queue._drain_org(org_id)

    assert order == ["a.txt", "b.txt"]   # FIFO
    assert max_concurrent["v"] == 1       # one write at a time


@pytest.mark.asyncio
async def test_serialised_write_has_no_lost_update(client, user_ctx, default_user):
    from app.services.wiki_db import get_wiki_page_content, upsert_wiki_page
    org_id = default_user["org_id"]
    await _queue("first.txt", org_id)
    await _queue("second.txt", org_id)

    async def fake_exec(filename, notes=""):
        existing = await get_wiki_page_content("concepts/shared.md") or ""
        await asyncio.sleep(0.02)
        await upsert_wiki_page("concepts/shared.md", existing + f"\n{filename}")

    with patch("app.routes.operations._execute_ingest", fake_exec):
        await ingest_queue._drain_org(org_id)

    assert await get_wiki_page_content("concepts/shared.md") == "\nfirst.txt\nsecond.txt"


@pytest.mark.asyncio
async def test_claim_committed_before_write_phase(client, user_ctx, default_user):
    """Regression: the claim must be committed (status='writing' visible on a
    fresh connection) BEFORE the write phase. Otherwise the drain holds the
    claim's row lock in an open transaction while awaiting _execute_ingest, whose
    own update of the same row blocks on that lock → self-deadlock (the bug that
    wedged the advisory lock in production)."""
    org_id = default_user["org_id"]
    await _queue("commit.txt", org_id)
    seen = {}

    async def fake_exec(filename, notes=""):
        from app.services.jobs import get as job_get
        j = await job_get(filename)          # fresh connection
        seen["status"] = j.status

    with patch("app.routes.operations._execute_ingest", fake_exec):
        await ingest_queue._drain_org(org_id)

    assert seen["status"] == "writing"


@pytest.mark.asyncio
async def test_drain_no_deadlock_when_write_updates_job_row(client, user_ctx, default_user):
    """The real write path updates the job row via job_store.save; that must not
    deadlock against the drain's claim. Bounded by wait_for so a regression fails
    fast instead of hanging."""
    org_id = default_user["org_id"]
    await _queue("save.txt", org_id)

    async def fake_exec(filename, notes=""):
        from app.services.jobs import get as job_get, save as job_save
        j = await job_get(filename)
        j.status = "done"
        await job_save(j)                    # updates the same ingest_jobs row

    with patch("app.routes.operations._execute_ingest", fake_exec):
        await asyncio.wait_for(ingest_queue._drain_org(org_id), timeout=10)

    assert (await get("save.txt")).status == "done"


@pytest.mark.asyncio
async def test_drain_releases_advisory_lock(client, user_ctx, default_user):
    """Regression: the SESSION-level advisory lock must be released when the drain
    ends. It is bound to one physical connection; if the drain ran on a Session
    (which returns its connection to the pool on every commit), the lock would
    strand on a pooled connection and the final unlock would run on a different
    one — silently no-op — wedging the org's queue forever. After a clean drain a
    fresh connection must be able to acquire the org's lock."""
    org_id = default_user["org_id"]
    await _queue("lock.txt", org_id)

    async def fake_exec(filename, notes=""):
        # Touch the DB from the write phase the way the real path does, so the
        # pool hands out connections and any stranding would surface.
        from app.services.jobs import get as job_get
        await job_get(filename)

    with patch("app.routes.operations._execute_ingest", fake_exec):
        await ingest_queue._drain_org(org_id)

    async with get_db() as db:
        free = (await db.execute(
            sa.text("SELECT pg_try_advisory_lock(:c, hashtext(:o))"),
            {"c": ingest_queue._ADVISORY_LOCK_CLASS, "o": org_id},
        )).scalar()
        await db.execute(
            sa.text("SELECT pg_advisory_unlock(:c, hashtext(:o))"),
            {"c": ingest_queue._ADVISORY_LOCK_CLASS, "o": org_id},
        )
    assert free is True   # lock was free → drain released it on the right connection


@pytest.mark.asyncio
async def test_drain_passes_persisted_notes(client, user_ctx, default_user):
    org_id = default_user["org_id"]
    await _queue("noted.txt", org_id, notes="MANDATORY: keep section X")
    seen = {}

    async def fake_exec(filename, notes=""):
        seen["notes"] = notes

    with patch("app.routes.operations._execute_ingest", fake_exec):
        await ingest_queue._drain_org(org_id)
    assert seen["notes"] == "MANDATORY: keep section X"


# ── self-healing + cancel ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_drain_reclaims_orphaned_writing(client, user_ctx, default_user):
    org_id = default_user["org_id"]
    await enqueue("orphan.txt")
    job = await get("orphan.txt"); job.status = "writing"; await save(job)

    ran = {"v": False}

    async def fake_exec(filename, notes=""):
        ran["v"] = True

    with patch("app.routes.operations._execute_ingest", fake_exec):
        await ingest_queue._drain_org(org_id)

    assert ran["v"] is False                       # orphan isn't re-executed
    assert (await get("orphan.txt")).status == "error"


@pytest.mark.asyncio
async def test_drain_skips_cancelled(client, user_ctx, default_user):
    org_id = default_user["org_id"]
    await _queue("cancelme.txt", org_id)
    job = await get("cancelme.txt"); job.cancel_requested = True; await save(job)

    ran = {"v": False}

    async def fake_exec(filename, notes=""):
        ran["v"] = True

    with patch("app.routes.operations._execute_ingest", fake_exec):
        await ingest_queue._drain_org(org_id)

    assert ran["v"] is False
    assert (await get("cancelme.txt")).status == "cancelled"


@pytest.mark.asyncio
async def test_drain_survives_failed_job(client, user_ctx, default_user):
    org_id = default_user["org_id"]
    await _queue("boom.txt", org_id)
    await _queue("next.txt", org_id)
    ran = []

    async def fake_exec(filename, notes=""):
        ran.append(filename)
        if filename == "boom.txt":
            raise RuntimeError("kaboom")

    with patch("app.routes.operations._execute_ingest", fake_exec):
        await ingest_queue._drain_org(org_id)

    assert ran == ["boom.txt", "next.txt"]   # one failure doesn't stall the queue


# ── queue position (cross-replica, from DB) ────────────────────────────────

@pytest.mark.asyncio
async def test_queue_position_is_fifo(client, user_ctx, default_user):
    org_id = default_user["org_id"]
    await _queue("p1.txt", org_id)
    await _queue("p2.txt", org_id)
    await _queue("p3.txt", org_id)

    assert await ingest_queue.queue_position("p1.txt", org_id) == 1
    assert await ingest_queue.queue_position("p2.txt", org_id) == 2
    assert await ingest_queue.queue_position("p3.txt", org_id) == 3
    assert await ingest_queue.queue_position("missing.txt", org_id) is None


# ── end-to-end submit + restart recovery ───────────────────────────────────

@pytest.mark.asyncio
async def test_submit_triggers_drain(client, user_ctx, default_user):
    org_id, user_id = default_user["org_id"], default_user["id"]
    await enqueue("e2e.txt")
    ran = {"v": False}

    async def fake_exec(filename, notes=""):
        ran["v"] = True

    with patch("app.routes.operations._execute_ingest", fake_exec):
        await ingest_queue.submit("e2e.txt", "", org_id, user_id)
        await asyncio.gather(*list(ingest_queue._tasks))

    assert ran["v"] is True


@pytest.mark.asyncio
async def test_initial_scan_recovers_pending(client, user_ctx, default_user):
    """A job left 'queued_write' by a prior process is drained on scan (restart)."""
    org_id = default_user["org_id"]
    await _queue("recover.txt", org_id)
    seen = []

    async def fake_exec(filename, notes=""):
        seen.append(filename)

    with patch("app.routes.operations._execute_ingest", fake_exec):
        await ingest_queue._initial_scan()
        await asyncio.gather(*list(ingest_queue._tasks))

    assert seen == ["recover.txt"]


# ── DELETE /api/ops/ingest cancels a queued_write job ─────────────────────

@pytest.mark.asyncio
async def test_cancel_queued_write_via_api(client, auth_headers, user_ctx):
    await enqueue("cqw.txt")
    job = await get("cqw.txt"); job.status = "queued_write"; await save(job)

    resp = await client.delete("/api/ops/ingest/cqw.txt", headers=auth_headers)
    assert resp.status_code == 200

    job = await get("cqw.txt")
    assert job.status == "cancelled"
    assert job.cancel_requested is True


# ── B4/B7/B6 regressions ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_submit_does_not_resurrect_a_writing_job(client, user_ctx, default_user):
    """B4: re-submitting a file whose job is mid-write must NOT flip it back to
    queued_write — that would let the same file be claimed and ingested twice."""
    org_id, user_id = default_user["org_id"], default_user["id"]
    await enqueue("dup.txt")
    job = await get("dup.txt"); job.status = "writing"; await save(job)

    await ingest_queue.submit("dup.txt", "", org_id, user_id)

    assert (await get("dup.txt")).status == "writing"   # untouched


@pytest.mark.asyncio
async def test_kick_claims_draining_synchronously(client, user_ctx, default_user):
    """B7: _kick must add to _draining synchronously (before create_task), so a
    racing second kick can't spawn a duplicate drain."""
    org_id = default_user["org_id"]
    assert org_id not in ingest_queue._draining
    ingest_queue._kick(org_id)
    assert org_id in ingest_queue._draining            # claimed before the task runs
    before = len(ingest_queue._tasks)
    ingest_queue._kick(org_id)                          # second kick → no-op
    assert len(ingest_queue._tasks) == before
    await asyncio.gather(*list(ingest_queue._tasks))    # drain finishes + discards


@pytest.mark.asyncio
async def test_shutdown_is_bounded_by_grace(monkeypatch):
    """B6: a long in-flight drain must not block shutdown unbounded."""
    import time
    monkeypatch.setattr(ingest_queue, "_SHUTDOWN_GRACE_SECS", 0.2)

    async def _hang():
        await asyncio.sleep(30)

    t = asyncio.create_task(_hang())
    ingest_queue._tasks.add(t)
    start = time.perf_counter()
    await ingest_queue.shutdown()
    elapsed = time.perf_counter() - start
    t.cancel()
    try:
        await t
    except asyncio.CancelledError:
        pass
    assert elapsed < 5    # gave up at the grace, didn't wait 30s
