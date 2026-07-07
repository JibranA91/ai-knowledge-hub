"""Integration tests for multi-org memberships.

Covers the membership management API (GET/POST/PUT/DELETE
/api/admin/users/{id}/memberships[/{org_id}]), role/scoping rules, the
per-request active-org resolution done in auth_middleware, and per-org
suspension isolation.

Requires PostgreSQL via testcontainers (Docker).
"""
import pytest
import pytest_asyncio

_DEFAULT_ORG_ID = "00000000-0000-0000-0000-000000000001"


def _bearer(user_id, email="u@test.com", org_id="", role="member"):
    from app.services.auth import create_access_token
    tok = create_access_token(email=email, user_id=user_id, org_id=org_id, role=role)
    return {"Authorization": f"Bearer {tok}"}


@pytest_asyncio.fixture
async def org_b(client, auth_headers):
    r = await client.post("/api/admin/organizations",
                          json={"name": "Org B Memberships"}, headers=auth_headers)
    assert r.status_code == 201
    return r.json()["id"]


@pytest_asyncio.fixture
async def member_user(default_user):
    """A member in the default org (created via create_user → seeds a membership)."""
    from app.services.orgs import create_user
    uid = await create_user(_DEFAULT_ORG_ID, "mem@test.com", "password123", role="member")
    return {"id": uid, "email": "mem@test.com"}


@pytest_asyncio.fixture
async def supervisor_headers(default_user):
    """Auth headers for a supervisor of the default org."""
    from app.services.orgs import create_user
    uid = await create_user(_DEFAULT_ORG_ID, "sup@test.com", "password123", role="supervisor")
    return _bearer(uid, "sup@test.com", org_id=_DEFAULT_ORG_ID, role="supervisor")


# ── Membership CRUD ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_list_user_memberships(client, auth_headers, member_user, org_b):
    await client.post(f"/api/admin/users/{member_user['id']}/memberships",
                      json={"org_id": org_b, "role": "supervisor"}, headers=auth_headers)
    r = await client.get(f"/api/admin/users/{member_user['id']}/memberships", headers=auth_headers)
    assert r.status_code == 200
    by_org = {m["org_id"]: m["role"] for m in r.json()}
    assert by_org[_DEFAULT_ORG_ID] == "member"
    assert by_org[org_b] == "supervisor"


@pytest.mark.asyncio
async def test_add_membership_duplicate_returns_409(client, auth_headers, member_user):
    r = await client.post(f"/api/admin/users/{member_user['id']}/memberships",
                          json={"org_id": _DEFAULT_ORG_ID, "role": "member"}, headers=auth_headers)
    assert r.status_code == 409


@pytest.mark.asyncio
async def test_cannot_add_membership_to_admin(client, auth_headers):
    from app.services.orgs import create_user
    admin_id = await create_user(None, "admin2@test.com", "password123", role="admin")
    r = await client.post(f"/api/admin/users/{admin_id}/memberships",
                          json={"org_id": _DEFAULT_ORG_ID, "role": "member"}, headers=auth_headers)
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_add_membership_role_admin_rejected(client, auth_headers, member_user, org_b):
    r = await client.post(f"/api/admin/users/{member_user['id']}/memberships",
                          json={"org_id": org_b, "role": "admin"}, headers=auth_headers)
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_update_membership_role_and_suspension(client, auth_headers, member_user):
    r = await client.put(
        f"/api/admin/users/{member_user['id']}/memberships/{_DEFAULT_ORG_ID}",
        json={"role": "supervisor", "is_suspended": True}, headers=auth_headers,
    )
    assert r.status_code == 200
    ms = (await client.get(f"/api/admin/users/{member_user['id']}/memberships",
                           headers=auth_headers)).json()
    m = next(m for m in ms if m["org_id"] == _DEFAULT_ORG_ID)
    assert m["role"] == "supervisor"
    assert m["is_suspended"] is True


@pytest.mark.asyncio
async def test_remove_membership_keeps_other_orgs(client, auth_headers, member_user, org_b):
    await client.post(f"/api/admin/users/{member_user['id']}/memberships",
                      json={"org_id": org_b, "role": "member"}, headers=auth_headers)
    r = await client.delete(
        f"/api/admin/users/{member_user['id']}/memberships/{_DEFAULT_ORG_ID}",
        headers=auth_headers,
    )
    assert r.status_code == 200
    ms = (await client.get(f"/api/admin/users/{member_user['id']}/memberships",
                           headers=auth_headers)).json()
    assert [m["org_id"] for m in ms] == [org_b]


@pytest.mark.asyncio
async def test_remove_membership_not_a_member_returns_404(client, auth_headers, member_user, org_b):
    r = await client.delete(
        f"/api/admin/users/{member_user['id']}/memberships/{org_b}", headers=auth_headers,
    )
    assert r.status_code == 404


# ── Scoping: admin vs supervisor vs member ──────────────────────────────────────

@pytest.mark.asyncio
async def test_supervisor_can_only_manage_own_org(client, supervisor_headers, org_b):
    from app.services.orgs import create_user
    floating = await create_user(None, "float1@test.com", "password123", role="member")

    # Own org → allowed.
    r_own = await client.post(f"/api/admin/users/{floating}/memberships",
                              json={"org_id": _DEFAULT_ORG_ID, "role": "member"},
                              headers=supervisor_headers)
    assert r_own.status_code == 201

    # A different org → forbidden.
    r_other = await client.post(f"/api/admin/users/{floating}/memberships",
                                json={"org_id": org_b, "role": "member"},
                                headers=supervisor_headers)
    assert r_other.status_code == 403


@pytest.mark.asyncio
async def test_member_cannot_manage_memberships(client, member_user, org_b):
    hdr = _bearer(member_user["id"], member_user["email"])
    r = await client.post(f"/api/admin/users/{member_user['id']}/memberships",
                          json={"org_id": org_b, "role": "member"}, headers=hdr)
    assert r.status_code == 403


# ── Active-org resolution (auth_middleware) ─────────────────────────────────────

@pytest.mark.asyncio
async def test_active_org_resolution_per_header(client, auth_headers, member_user, org_b):
    """A user who is a member in one org and supervisor in another gets the role
    of whichever org the X-Org-Context header selects."""
    await client.post(f"/api/admin/users/{member_user['id']}/memberships",
                      json={"org_id": org_b, "role": "supervisor"}, headers=auth_headers)
    hdr = _bearer(member_user["id"], member_user["email"])

    # Multiple memberships + no header → no active org, but the highest role
    # across orgs is preserved (supervisor here) so the dashboard still loads.
    me = (await client.get("/api/admin/me", headers=hdr)).json()
    assert me["org_id"] == ""
    assert me["role"] == "supervisor"

    # Header selects the org and its role.
    me_a = (await client.get("/api/admin/me",
                             headers={**hdr, "X-Org-Context": _DEFAULT_ORG_ID})).json()
    assert me_a["org_id"] == _DEFAULT_ORG_ID and me_a["role"] == "member"

    me_b = (await client.get("/api/admin/me",
                             headers={**hdr, "X-Org-Context": org_b})).json()
    assert me_b["org_id"] == org_b and me_b["role"] == "supervisor"


@pytest.mark.asyncio
async def test_multi_org_supervisor_reaches_dashboard(client, auth_headers, org_b):
    """A supervisor of two orgs must reach the admin dashboard with no org header,
    and /admin/me must list the orgs they supervise for the org switcher."""
    from app.services.orgs import create_user
    uid = await create_user(_DEFAULT_ORG_ID, "multisup@test.com", "password123",
                            role="supervisor")
    await client.post(f"/api/admin/users/{uid}/memberships",
                      json={"org_id": org_b, "role": "supervisor"}, headers=auth_headers)
    hdr = _bearer(uid, "multisup@test.com")

    # No header: supervisor role (dashboard loads), no org selected yet.
    me = (await client.get("/api/admin/me", headers=hdr)).json()
    assert me["role"] == "supervisor"
    assert me["org_id"] == ""
    supervised = {m["org_id"] for m in me["memberships"] if m["role"] == "supervisor"}
    assert supervised == {_DEFAULT_ORG_ID, org_b}

    # Selecting an org scopes a supervisor-only endpoint to that org.
    r = await client.get("/api/admin/templates", headers={**hdr, "X-Org-Context": org_b})
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_header_for_unjoined_org_grants_no_access(client, member_user, org_b):
    """Pointing X-Org-Context at an org the user doesn't belong to yields no context."""
    hdr = _bearer(member_user["id"], member_user["email"])
    me = (await client.get("/api/admin/me", headers={**hdr, "X-Org-Context": org_b})).json()
    assert me["org_id"] == ""


@pytest.mark.asyncio
async def test_suspension_is_per_org(client, auth_headers, member_user, org_b):
    """Suspending a user in one org must not affect their access in another."""
    await client.post(f"/api/admin/users/{member_user['id']}/memberships",
                      json={"org_id": org_b, "role": "member"}, headers=auth_headers)
    await client.put(f"/api/admin/users/{member_user['id']}/memberships/{org_b}",
                     json={"role": "member", "is_suspended": True}, headers=auth_headers)

    hdr = _bearer(member_user["id"], member_user["email"])
    p_default = (await client.get("/api/ops/permissions",
                                  headers={**hdr, "X-Org-Context": _DEFAULT_ORG_ID})).json()
    p_b = (await client.get("/api/ops/permissions",
                            headers={**hdr, "X-Org-Context": org_b})).json()
    assert p_default["is_suspended"] is False
    assert p_b["is_suspended"] is True
