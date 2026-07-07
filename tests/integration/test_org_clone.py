"""Integration tests for Clone Org (admin-only full wiki copy).

Requires PostgreSQL. Verifies that a clone copies wiki pages + schema into a new
org, regenerates links, and does NOT carry over members.
"""
import pytest
import pytest_asyncio
import sqlalchemy as sa

from app.context import UserContext, current_user
from app.db import get_db
from app.services import orgs as orgs_svc
from app.services.wiki_db import (
    upsert_wiki_page, set_wiki_file, get_wiki_page_content, get_wiki_file,
    list_wiki_paths, get_wiki_links,
)


def _ctx(org_id: str, user_id: str = "00000000-0000-0000-0000-000000000000") -> None:
    current_user.set(UserContext(user_id=user_id, org_id=org_id, email="sys", role="admin"))


async def _seed_source(org_id: str, user_id: str) -> None:
    _ctx(org_id, user_id)
    await set_wiki_file("schema/AGENTS.md", "# Custom Schema\n\n## Page Types\nconcept\n")
    await upsert_wiki_page("concepts/alpha.md", "# Alpha\n\nLinks to [[beta]].")
    await upsert_wiki_page("concepts/beta.md", "# Beta\n\nStandalone.")


@pytest_asyncio.fixture
async def supervisor_headers(default_user):
    from app.services.orgs import create_user
    from app.services.auth import create_access_token
    uid = await create_user(default_user["org_id"], "sup-clone@test.com",
                            "password123", role="supervisor")
    token = create_access_token(email="sup-clone@test.com", user_id=uid,
                               org_id=default_user["org_id"], role="supervisor")
    return {"Authorization": f"Bearer {token}"}


# ── service: clone_org ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_clone_copies_pages_and_schema(default_user):
    src = default_user["org_id"]
    await _seed_source(src, default_user["id"])

    new_org = await orgs_svc.clone_org(src, "Cloned Wiki Org")
    assert new_org and new_org != src

    _ctx(new_org, default_user["id"])
    assert "Alpha" in (await get_wiki_page_content("concepts/alpha.md") or "")
    assert "Beta" in (await get_wiki_page_content("concepts/beta.md") or "")
    assert "Custom Schema" in (await get_wiki_file("schema/AGENTS.md") or "")

    paths = await list_wiki_paths()
    assert "concepts/alpha.md" in paths and "concepts/beta.md" in paths


@pytest.mark.asyncio
async def test_clone_regenerates_links(default_user):
    src = default_user["org_id"]
    await _seed_source(src, default_user["id"])
    new_org = await orgs_svc.clone_org(src, "Linked Clone")
    assert new_org

    _ctx(new_org, default_user["id"])
    links = await get_wiki_links("concepts/alpha.md")
    assert any("beta" in t for t in links.get("outgoing", []))


@pytest.mark.asyncio
async def test_clone_does_not_copy_members(default_user):
    src = default_user["org_id"]
    # Give the source a member so "no members copied" is a real assertion.
    await orgs_svc.create_user(src, "member-clone@test.com", "password123", role="member")
    await _seed_source(src, default_user["id"])

    new_org = await orgs_svc.clone_org(src, "Memberless Clone")
    assert new_org

    async with get_db() as db:
        count = (await db.execute(
            sa.text("SELECT COUNT(*) FROM org_memberships WHERE org_id = CAST(:o AS UUID)"),
            {"o": new_org},
        )).scalar()
    assert count == 0


@pytest.mark.asyncio
async def test_clone_rejects_duplicate_name(default_user):
    src = default_user["org_id"]
    await _seed_source(src, default_user["id"])
    first = await orgs_svc.clone_org(src, "Dupe Name Org")
    assert first
    second = await orgs_svc.clone_org(src, "Dupe Name Org")
    assert second is None


@pytest.mark.asyncio
async def test_clone_missing_source_returns_none(default_user):
    missing = "00000000-0000-0000-0000-0000000000ff"
    assert await orgs_svc.clone_org(missing, "From Nothing") is None


# ── route: POST /api/admin/organizations/{id}/clone ───────────────────────

@pytest.mark.asyncio
async def test_clone_endpoint_admin_ok(client, auth_headers, default_user, user_ctx):
    await upsert_wiki_page("concepts/api.md", "# Api page")  # uses user_ctx (default org)

    resp = await client.post(
        f"/api/admin/organizations/{default_user['org_id']}/clone",
        json={"new_name": "Clone Via API"}, headers=auth_headers,
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["cloned_from"] == default_user["org_id"]
    assert body["id"] != default_user["org_id"]


@pytest.mark.asyncio
async def test_clone_endpoint_forbidden_for_supervisor(client, supervisor_headers, default_user):
    resp = await client.post(
        f"/api/admin/organizations/{default_user['org_id']}/clone",
        json={"new_name": "Nope"}, headers=supervisor_headers,
    )
    assert resp.status_code == 403
