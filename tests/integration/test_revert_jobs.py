"""Integration tests for the background revert-job flow (`/api/admin/history/.../revert`).

The route returns 202 immediately and the actual revert runs in an asyncio
task. Tests drive the API via httpx and poll for completion, mirroring how
the admin UI consumes the endpoint.
"""
import asyncio
import pytest
import sqlalchemy as sa


_DEFAULT_ORG_ID = "00000000-0000-0000-0000-000000000001"


async def _wait_for_job(client, headers, revert_action_id, timeout=10.0):
    """Poll the job-status endpoint until status != 'running'. Returns final state."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        r = await client.get(
            f"/api/admin/history/revert-job/{revert_action_id}",
            headers=headers,
        )
        assert r.status_code == 200, r.text
        job = r.json()
        if job["status"] != "running":
            return job
        await asyncio.sleep(0.1)
    raise AssertionError(f"Revert job {revert_action_id} did not finish in {timeout}s")


async def _create_page_action(client, headers, path: str, content: str) -> str:
    """PUT a wiki page and return the wiki_actions.id that the write produced."""
    r = await client.put(f"/api/wiki/{path}", json={"content": content}, headers=headers)
    assert r.status_code in (200, 201), r.text

    # Fetch the most-recent action for this org (admin context) so we can target it.
    hist = await client.get("/api/admin/history?limit=5", headers=headers)
    assert hist.status_code == 200
    entries = hist.json()["entries"]
    assert entries, "expected at least one action after PUT /api/wiki"
    return entries[0]["id"]


# ── Happy path ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_revert_post_returns_202_with_job_id(client, auth_headers, user_ctx):
    action_id = await _create_page_action(client, auth_headers, "tests/revert-1.md", "# v1\n")

    r = await client.post(f"/api/admin/history/{action_id}/revert", headers=auth_headers)
    assert r.status_code == 202, r.text
    body = r.json()
    assert "revert_action_id" in body
    assert body["status"] == "running"
    # progress_total >= 1 (at least the target action itself is in scope)
    assert body["progress_total"] >= 1


@pytest.mark.asyncio
async def test_revert_job_polled_to_done_restores_page(
    client, auth_headers, user_ctx, test_engine,
):
    """End-to-end: create a page, kick off revert, poll to done, page is gone."""
    action_id = await _create_page_action(client, auth_headers, "tests/revert-2.md", "# v1\n")

    # Page exists right now
    g = await client.get("/api/wiki/tests/revert-2.md", headers=auth_headers)
    assert g.status_code == 200

    r = await client.post(f"/api/admin/history/{action_id}/revert", headers=auth_headers)
    assert r.status_code == 202
    revert_id = r.json()["revert_action_id"]

    job = await _wait_for_job(client, auth_headers, revert_id)
    assert job["status"] == "done", job
    assert job["progress_done"] == job["progress_total"]
    assert job["error_message"] is None

    # Original page should now be gone (revert of "create" = delete)
    g2 = await client.get("/api/wiki/tests/revert-2.md", headers=auth_headers)
    assert g2.status_code == 404


@pytest.mark.asyncio
async def test_revert_of_revert_redoes_the_change(
    client, auth_headers, user_ctx, test_engine,
):
    """Reverting a revert is a redo: create a page, revert it (page gone),
    then revert the revert and the page comes back."""
    action_id = await _create_page_action(client, auth_headers, "tests/revert-redo.md", "# v1\n")

    # Revert the create → page gone.
    r1 = await client.post(f"/api/admin/history/{action_id}/revert", headers=auth_headers)
    assert r1.status_code == 202
    revert_id = r1.json()["revert_action_id"]
    await _wait_for_job(client, auth_headers, revert_id)
    g = await client.get("/api/wiki/tests/revert-redo.md", headers=auth_headers)
    assert g.status_code == 404

    # Revert the revert → page restored (redo). Previously this returned 409.
    r2 = await client.post(f"/api/admin/history/{revert_id}/revert", headers=auth_headers)
    assert r2.status_code == 202, r2.text
    job = await _wait_for_job(client, auth_headers, r2.json()["revert_action_id"])
    assert job["status"] == "done", job

    g2 = await client.get("/api/wiki/tests/revert-redo.md", headers=auth_headers)
    assert g2.status_code == 200
    assert "# v1" in g2.text

    # The original create action's stale 'reverted' status is reconciled back
    # to 'done' — its effect is live again, so History shouldn't still call it
    # reverted.
    hist = await client.get("/api/admin/history?limit=50", headers=auth_headers)
    entry = next(e for e in hist.json()["entries"] if e["id"] == action_id)
    assert entry["status"] == "done", entry


@pytest.mark.asyncio
async def test_revert_job_status_404_for_unknown_id(client, auth_headers):
    r = await client.get(
        "/api/admin/history/revert-job/00000000-0000-0000-0000-000000000000",
        headers=auth_headers,
    )
    assert r.status_code == 404


# ── Validation errors return synchronously ───────────────────────────────────

@pytest.mark.asyncio
async def test_revert_unknown_action_returns_409(client, auth_headers):
    r = await client.post(
        "/api/admin/history/00000000-0000-0000-0000-000000000000/revert",
        headers=auth_headers,
    )
    assert r.status_code == 409
    assert "not found" in r.json()["detail"].lower()


@pytest.mark.asyncio
async def test_revert_already_reverted_returns_409(
    client, auth_headers, user_ctx,
):
    action_id = await _create_page_action(client, auth_headers, "tests/revert-3.md", "# v1\n")
    r1 = await client.post(f"/api/admin/history/{action_id}/revert", headers=auth_headers)
    assert r1.status_code == 202
    await _wait_for_job(client, auth_headers, r1.json()["revert_action_id"])

    # Second attempt on the same target should bounce with 409.
    r2 = await client.post(f"/api/admin/history/{action_id}/revert", headers=auth_headers)
    assert r2.status_code == 409


@pytest.mark.asyncio
async def test_concurrent_revert_blocked_by_write_lock(
    client, auth_headers, user_ctx, test_engine,
):
    """While a revert is in flight for an org, every write (including a new
    revert) is blocked by the write-lock middleware with 503 — a stronger
    guarantee than the route-level 409, since the middleware also blocks
    unrelated writes (uploads, page edits) for the duration of the revert."""
    # Need a real target action so we don't bounce on "not found".
    a = await _create_page_action(client, auth_headers, "tests/revert-4a.md", "# A\n")

    # Insert a fake 'running' revert directly so the middleware sees the org
    # as currently reverting. This is faster and more deterministic than
    # racing a real worker.
    import uuid
    fake_id = uuid.uuid4().hex
    async with test_engine.begin() as conn:
        await conn.execute(sa.text("""
            INSERT INTO wiki_actions
                (id, org_id, action_type, summary, status, progress_done, progress_total)
            VALUES
                (CAST(:id AS UUID), CAST(:org AS UUID), 'revert', 'fake',
                 'running', 0, 99)
        """), {"id": fake_id, "org": _DEFAULT_ORG_ID})

    try:
        r = await client.post(f"/api/admin/history/{a}/revert", headers=auth_headers)
        assert r.status_code == 503
        assert "revert" in r.json()["detail"].lower()
    finally:
        async with test_engine.begin() as conn:
            await conn.execute(sa.text("DELETE FROM wiki_actions WHERE id = CAST(:id AS UUID)"),
                                {"id": fake_id})


@pytest.mark.asyncio
async def test_start_revert_job_rejects_when_one_already_running(
    client, auth_headers, user_ctx, test_engine,
):
    """B5: the service-level revert claim (now under an advisory xact lock) is
    atomic — with a revert already 'running' for the org, start_revert_job
    raises RevertJobAlreadyRunning rather than starting a second overlapping
    revert. (Direct call bypasses the middleware to exercise the service guard.)"""
    from app.services.wiki_state import start_revert_job, RevertJobAlreadyRunning
    import uuid

    a = await _create_page_action(client, auth_headers, "tests/revert-b5.md", "# A\n")
    fake_id = uuid.uuid4().hex
    async with test_engine.begin() as conn:
        await conn.execute(sa.text("""
            INSERT INTO wiki_actions
                (id, org_id, action_type, summary, status, progress_done, progress_total)
            VALUES (CAST(:id AS UUID), CAST(:org AS UUID), 'revert', 'fake', 'running', 0, 1)
        """), {"id": fake_id, "org": _DEFAULT_ORG_ID})
    try:
        with pytest.raises(RevertJobAlreadyRunning):
            await start_revert_job(a)
    finally:
        async with test_engine.begin() as conn:
            await conn.execute(sa.text("DELETE FROM wiki_actions WHERE id = CAST(:id AS UUID)"),
                               {"id": fake_id})


@pytest.mark.asyncio
async def test_page_revision_recorded_in_caller_transaction(user_ctx, default_user):
    """B8: record_revision(db=...) joins the caller's transaction, so a write and
    its revision commit — or roll back — together. A page change can no longer
    durably persist while losing its revision (which would make it un-revertible)."""
    from app.db import get_db
    from app.services.wiki_state import begin_action, record_revision
    key = "tests/atomic-b8.md"
    async with begin_action("manual_edit", summary="b8"):
        async with get_db() as db:
            await record_revision(db=db, target_kind="page", target_key=key,
                                  op="create", content_after="hi")
            await db.rollback()   # caller aborts → the revision must vanish with it
    async with get_db() as db:
        n = (await db.execute(
            sa.text("SELECT COUNT(*) FROM wiki_revisions WHERE target_key = :k"), {"k": key}
        )).scalar()
    assert n == 0


# ── Permission ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_revert_member_forbidden(client, auth_headers, user_ctx):
    """Only admin/supervisor can revert."""
    action_id = await _create_page_action(client, auth_headers, "tests/revert-5.md", "# v1\n")

    from app.services.orgs import create_user
    from app.services.auth import create_access_token
    user_id = await create_user(_DEFAULT_ORG_ID, "rev_member@test.com", "password123",
                                  role="member")
    token = create_access_token(
        email="rev_member@test.com", user_id=user_id,
        org_id=_DEFAULT_ORG_ID, role="member",
    )
    r = await client.post(
        f"/api/admin/history/{action_id}/revert",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 403


# ── Startup recovery ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_recover_stale_revert_jobs_marks_running_as_error(test_engine, user_ctx):
    """Stranded 'running' reverts (process crashed mid-flight) should be flipped
    to 'error' on startup so polling clients don't wait forever."""
    import uuid
    stale_id = uuid.uuid4().hex
    async with test_engine.begin() as conn:
        await conn.execute(sa.text("""
            INSERT INTO wiki_actions
                (id, org_id, action_type, summary, status, progress_done, progress_total)
            VALUES
                (CAST(:id AS UUID), CAST(:org AS UUID), 'revert', 'stale',
                 'running', 3, 10)
        """), {"id": stale_id, "org": _DEFAULT_ORG_ID})

    from app.services.wiki_state import recover_stale_revert_jobs
    recovered = await recover_stale_revert_jobs()
    assert recovered >= 1  # at least our stale row (other tests may leave more)

    async with test_engine.begin() as conn:
        row = (await conn.execute(sa.text("""
            SELECT status, error_message FROM wiki_actions WHERE id = CAST(:id AS UUID)
        """), {"id": stale_id})).fetchone()
    assert row.status == "error"
    assert "Interrupted by server restart" in (row.error_message or "")
