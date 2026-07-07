"""Integration tests for the admin Jobs API (/api/admin/jobs).

Covers the unified active-job shape and supervisor org-scoping.
Requires PostgreSQL via testcontainers.
"""
import pytest
import pytest_asyncio


@pytest_asyncio.fixture
async def supervisor_headers(default_user):
    from app.services.orgs import create_user
    from app.services.auth import create_access_token
    uid = await create_user(default_user["org_id"], "sup-jobs@test.com",
                            "password123", role="supervisor")
    token = create_access_token(
        email="sup-jobs@test.com", user_id=uid,
        org_id=default_user["org_id"], role="supervisor",
    )
    return {"Authorization": f"Bearer {token}"}


async def _enqueue_in(org_id: str, user_id: str, filename: str, status: str) -> None:
    """Enqueue an ingest job in a specific org by setting the context var."""
    from app.context import UserContext, current_user
    from app.services.jobs import enqueue, get, save
    current_user.set(UserContext(user_id=user_id, org_id=org_id,
                                 email="x@test.com", role="member"))
    await enqueue(filename)
    job = await get(filename)
    job.status = status
    await save(job)


# ── GET /api/admin/jobs ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_jobs_shape_lists_active_ingest(client, auth_headers, default_user):
    await _enqueue_in(default_user["org_id"], default_user["id"], "shape.txt", "pending_review")

    resp = await client.get("/api/admin/jobs", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert "jobs" in data
    ingest = [j for j in data["jobs"] if j["type"] == "ingest"]
    match = next((j for j in ingest if j["target"] == "shape.txt"), None)
    assert match is not None
    # Required keys on every job entry.
    for key in ("type", "id", "target", "status", "cancellable", "started_at"):
        assert key in match
    assert match["status"] == "pending_review"
    assert match["cancellable"] is True


@pytest.mark.asyncio
async def test_jobs_excludes_terminal_ingest(client, auth_headers, default_user):
    await _enqueue_in(default_user["org_id"], default_user["id"], "done.txt", "done")

    resp = await client.get("/api/admin/jobs", headers=auth_headers)
    targets = [j["target"] for j in resp.json()["jobs"]]
    assert "done.txt" not in targets


@pytest.mark.asyncio
async def test_supervisor_sees_only_own_org_jobs(client, supervisor_headers, default_user):
    from app.services.orgs import create_org
    other_org = await create_org("Jobs Other Org")

    await _enqueue_in(default_user["org_id"], default_user["id"], "mine.txt", "pending_review")
    await _enqueue_in(other_org, default_user["id"], "theirs.txt", "pending_review")

    resp = await client.get("/api/admin/jobs", headers=supervisor_headers)
    assert resp.status_code == 200
    targets = [j["target"] for j in resp.json()["jobs"]]
    assert "mine.txt" in targets
    assert "theirs.txt" not in targets


@pytest.mark.asyncio
async def test_jobs_requires_supervisor_role(client, default_user):
    from app.services.orgs import create_user
    from app.services.auth import create_access_token
    uid = await create_user(default_user["org_id"], "member-jobs@test.com",
                            "password123", role="member")
    token = create_access_token(
        email="member-jobs@test.com", user_id=uid,
        org_id=default_user["org_id"], role="member",
    )
    resp = await client.get("/api/admin/jobs", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 403


# ── POST /api/admin/jobs/ingest/{filename}/cancel ─────────────────────────

@pytest.mark.asyncio
async def test_admin_cancel_ingest_job(client, auth_headers, default_user):
    await _enqueue_in(default_user["org_id"], default_user["id"], "adm-cancel.txt", "queued_write")

    resp = await client.post(
        "/api/admin/jobs/ingest/adm-cancel.txt/cancel", headers=auth_headers,
    )
    assert resp.status_code == 200
    assert resp.json()["cancelled"] == "adm-cancel.txt"

    # Restore context to the default org and confirm the job was cancelled.
    from app.context import UserContext, current_user
    from app.services.jobs import get
    current_user.set(UserContext(user_id=default_user["id"], org_id=default_user["org_id"],
                                 email="admin", role="admin"))
    job = await get("adm-cancel.txt")
    assert job.status == "cancelled"


@pytest.mark.asyncio
async def test_admin_cancel_unknown_ingest_404(client, auth_headers):
    resp = await client.post(
        "/api/admin/jobs/ingest/nope.txt/cancel", headers=auth_headers,
    )
    assert resp.status_code == 404
