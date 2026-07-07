"""Integration tests for the member-facing wiki activity API.

Exercises /api/activity/recent and /api/activity/page against real tracked
actions/revisions. Requires PostgreSQL.
"""
import pytest
import pytest_asyncio

from app.context import UserContext, current_user
from app.services.wiki_state import begin_action
from app.services.wiki_db import upsert_wiki_page


def _set_ctx(user_id: str, org_id: str, email: str = "admin") -> None:
    current_user.set(UserContext(user_id=user_id, org_id=org_id, email=email, role="admin"))


async def _tracked_edit(path: str, content: str, summary: str = "edit",
                        action_type: str = "manual_edit") -> None:
    async with begin_action(action_type, summary=summary):
        await upsert_wiki_page(path, content)


@pytest_asyncio.fixture
async def member_headers(default_user):
    from app.services.orgs import create_user
    from app.services.auth import create_access_token
    uid = await create_user(default_user["org_id"], "member-act@test.com",
                            "password123", role="member")
    token = create_access_token(email="member-act@test.com", user_id=uid,
                                org_id=default_user["org_id"], role="member")
    return {"Authorization": f"Bearer {token}"}


# ── GET /api/activity/recent ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_recent_lists_page_changes(client, auth_headers, default_user):
    _set_ctx(default_user["id"], default_user["org_id"])
    await _tracked_edit("concepts/alpha.md", "# Alpha\n\nv1", summary="create alpha")

    resp = await client.get("/api/activity/recent", headers=auth_headers)
    assert resp.status_code == 200
    changes = resp.json()["changes"]
    match = next((c for c in changes if "concepts/alpha.md" in c["pages"]), None)
    assert match is not None
    for key in ("id", "action_type", "user_email", "when", "page_count", "pages"):
        assert key in match
    assert match["page_count"] >= 1


@pytest.mark.asyncio
async def test_recent_newest_first(client, auth_headers, default_user):
    _set_ctx(default_user["id"], default_user["org_id"])
    await _tracked_edit("concepts/one.md", "# One", summary="first")
    await _tracked_edit("concepts/two.md", "# Two", summary="second")

    changes = (await client.get("/api/activity/recent", headers=auth_headers)).json()["changes"]
    # Most recent (two) should appear before the older (one).
    idx = {c["summary"]: i for i, c in enumerate(changes)}
    assert idx["second"] < idx["first"]


@pytest.mark.asyncio
async def test_recent_is_org_scoped(client, auth_headers, default_user):
    from app.services.orgs import create_org
    org_b = await create_org("Activity Org B")
    _set_ctx(default_user["id"], org_b)
    await _tracked_edit("concepts/in-b.md", "# B-only", summary="b edit")

    # Admin token resolves to the default org — org B's change must not leak.
    resp = await client.get("/api/activity/recent", headers=auth_headers)
    pages = [p for c in resp.json()["changes"] for p in c["pages"]]
    assert "concepts/in-b.md" not in pages


@pytest.mark.asyncio
async def test_recent_accessible_to_members(client, member_headers):
    # The whole point of this feature: ordinary members (who have can_view_wiki
    # by default) can see the activity feed, not just supervisors/admins.
    resp = await client.get("/api/activity/recent", headers=member_headers)
    assert resp.status_code == 200
    assert "changes" in resp.json()


@pytest.mark.asyncio
async def test_activity_denied_for_suspended_member(client, default_user):
    from app.services.orgs import create_user, update_membership
    from app.services.auth import create_access_token
    uid = await create_user(default_user["org_id"], "suspended-act@test.com",
                            "password123", role="member")
    await update_membership(uid, default_user["org_id"],
                            {"role": "member", "is_suspended": True})
    token = create_access_token(email="suspended-act@test.com", user_id=uid,
                                org_id=default_user["org_id"], role="member")
    headers = {"Authorization": f"Bearer {token}"}
    resp = await client.get("/api/activity/recent", headers=headers)
    assert resp.status_code == 403


# ── GET /api/activity/page ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_page_activity_counts_and_contributors(client, auth_headers, default_user):
    _set_ctx(default_user["id"], default_user["org_id"])
    await _tracked_edit("concepts/beta.md", "# Beta v1", summary="create")
    await _tracked_edit("concepts/beta.md", "# Beta v2", summary="update")

    resp = await client.get("/api/activity/page?path=concepts/beta.md", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["change_count"] == 2
    assert data["last_edited_by"] is not None
    assert data["last_edited_at"] is not None
    assert any(c["count"] == 2 for c in data["contributors"])


@pytest.mark.asyncio
async def test_page_activity_untracked_page_returns_zero(client, auth_headers):
    resp = await client.get("/api/activity/page?path=concepts/never.md", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["change_count"] == 0
    assert data["last_edited_by"] is None
    assert data["contributors"] == []


@pytest.mark.asyncio
async def test_page_activity_accessible_to_members(client, member_headers):
    resp = await client.get("/api/activity/page?path=concepts/x.md", headers=member_headers)
    assert resp.status_code == 200
    assert "change_count" in resp.json()


# ── GET /api/activity/page/history ────────────────────────────────────────

@pytest.mark.asyncio
async def test_page_history_timeline_newest_first(client, auth_headers, default_user):
    _set_ctx(default_user["id"], default_user["org_id"])
    await _tracked_edit("concepts/gamma.md", "# Gamma v1", summary="create")
    await _tracked_edit("concepts/gamma.md", "# Gamma v2", summary="update",
                        action_type="recalibrate")

    resp = await client.get("/api/activity/page/history?path=concepts/gamma.md",
                            headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["change_count"] == 2
    changes = data["changes"]
    assert len(changes) == 2
    for key in ("user_email", "action_type", "op", "when"):
        assert key in changes[0]
    # Newest first: the recalibrate (latest) precedes the original create.
    assert changes[0]["action_type"] == "recalibrate"
    assert changes[-1]["op"] == "create"


@pytest.mark.asyncio
async def test_page_history_untracked_returns_empty(client, auth_headers):
    resp = await client.get("/api/activity/page/history?path=concepts/none.md",
                            headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["change_count"] == 0
    assert data["changes"] == []


@pytest.mark.asyncio
async def test_page_history_accessible_to_members(client, member_headers):
    resp = await client.get("/api/activity/page/history?path=concepts/x.md",
                            headers=member_headers)
    assert resp.status_code == 200
    assert "changes" in resp.json()


# ── GET /api/activity/revision/{revision_id} ──────────────────────────────

async def _latest_revision_id(client, headers, path: str) -> int:
    hist = (await client.get(f"/api/activity/page/history?path={path}",
                             headers=headers)).json()["changes"]
    return hist[0]["id"]


@pytest.mark.asyncio
async def test_revision_content_returns_before_after(client, auth_headers, default_user):
    _set_ctx(default_user["id"], default_user["org_id"])
    await _tracked_edit("concepts/delta.md", "# Delta\n\nfirst body", summary="create")
    await _tracked_edit("concepts/delta.md", "# Delta\n\nsecond body", summary="update")

    rev_id = await _latest_revision_id(client, auth_headers, "concepts/delta.md")
    resp = await client.get(f"/api/activity/revision/{rev_id}", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["op"] == "update"
    assert "first body" in (data["content_before"] or "")
    assert "second body" in (data["content_after"] or "")


@pytest.mark.asyncio
async def test_revision_not_found_returns_404(client, auth_headers):
    resp = await client.get("/api/activity/revision/999999999", headers=auth_headers)
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_revision_content_is_org_scoped(client, auth_headers, default_user):
    # A revision created in org B must not be readable with the default-org token.
    from app.services.orgs import create_org, create_user
    from app.services.auth import create_access_token
    org_b = await create_org("Revision Org B")
    uid_b = await create_user(org_b, "rev-b@test.com", "password123", role="admin")
    token_b = create_access_token(email="rev-b@test.com", user_id=uid_b,
                                  org_id=org_b, role="admin")
    headers_b = {"Authorization": f"Bearer {token_b}"}

    _set_ctx(uid_b, org_b, email="rev-b@test.com")
    await _tracked_edit("concepts/b-only.md", "# B-only", summary="b create")
    rev_id = await _latest_revision_id(client, headers_b, "concepts/b-only.md")

    resp = await client.get(f"/api/activity/revision/{rev_id}", headers=auth_headers)
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_revision_content_accessible_to_members(client, auth_headers, member_headers,
                                                      default_user):
    _set_ctx(default_user["id"], default_user["org_id"])
    await _tracked_edit("concepts/epsilon.md", "# Epsilon", summary="create")
    rev_id = await _latest_revision_id(client, auth_headers, "concepts/epsilon.md")

    resp = await client.get(f"/api/activity/revision/{rev_id}", headers=member_headers)
    assert resp.status_code == 200
    assert "content_after" in resp.json()
