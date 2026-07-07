"""Integration tests for /api/auth routes.

Requires PostgreSQL via testcontainers.
Login now returns {access_token, refresh_token, token_type} — JWT Phase 4.
"""
import pytest


@pytest.mark.asyncio
async def test_get_config_no_auth(client):
    resp = await client.get("/api/auth/config")
    assert resp.status_code == 200
    data = resp.json()
    assert "company_name" in data
    assert "chat_stream" in data
    assert "max_upload_size_mb" in data


@pytest.mark.asyncio
async def test_login_valid_credentials(client):
    resp = await client.post(
        "/api/auth/login",
        json={"username": "admin", "password": "admin"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["token_type"] == "bearer"
    # JWT access tokens are longer than the old 64-char hex tokens
    assert len(data["access_token"]) > 64
    # Refresh tokens are still 64-char hex
    assert len(data["refresh_token"]) == 64


@pytest.mark.asyncio
async def test_admin_login_returns_empty_memberships(client):
    """Admins are global — login returns no memberships and no active org."""
    resp = await client.post("/api/auth/login", json={"username": "admin", "password": "admin"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["role"] == "admin"
    assert data["memberships"] == []
    assert data["active_org_id"] is None


@pytest.mark.asyncio
async def test_member_login_returns_memberships(client, default_user):
    """A member's login response lists their org memberships; a single membership
    is auto-selected as the active org."""
    from app.services.orgs import create_user
    _DEFAULT_ORG_ID = "00000000-0000-0000-0000-000000000001"
    await create_user(_DEFAULT_ORG_ID, "loginmem@test.com", "password123", role="member")

    resp = await client.post("/api/auth/login",
                             json={"username": "loginmem@test.com", "password": "password123"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["role"] == "member"
    assert len(data["memberships"]) == 1
    assert data["memberships"][0]["org_id"] == _DEFAULT_ORG_ID
    assert data["active_org_id"] == _DEFAULT_ORG_ID


@pytest.mark.asyncio
async def test_multi_org_member_login_has_no_auto_active_org(client, auth_headers, default_user):
    """A member of several orgs gets all memberships but no auto-selected active
    org (the client must choose)."""
    from app.services.orgs import create_user
    _DEFAULT_ORG_ID = "00000000-0000-0000-0000-000000000001"
    uid = await create_user(_DEFAULT_ORG_ID, "multi2@test.com", "password123", role="member")
    org_b = (await client.post("/api/admin/organizations", json={"name": "LoginOrgB"},
                               headers=auth_headers)).json()["id"]
    await client.post(f"/api/admin/users/{uid}/memberships",
                      json={"org_id": org_b, "role": "member"}, headers=auth_headers)

    data = (await client.post("/api/auth/login",
                              json={"username": "multi2@test.com", "password": "password123"})).json()
    assert len(data["memberships"]) == 2
    assert data["active_org_id"] is None


@pytest.mark.asyncio
async def test_login_wrong_password(client):
    resp = await client.post(
        "/api/auth/login",
        json={"username": "admin", "password": "wrong"},
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_login_wrong_username(client):
    resp = await client.post(
        "/api/auth/login",
        json={"username": "hacker", "password": "admin"},
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_logout_invalidates_refresh_token(client, auth_headers):
    # Login to get tokens
    r = await client.post(
        "/api/auth/login", json={"username": "admin", "password": "admin"}
    )
    refresh_token = r.json()["refresh_token"]

    # Confirm the refresh token works before logout
    r2 = await client.post(
        "/api/auth/refresh", json={"refresh_token": refresh_token}
    )
    assert r2.status_code == 200

    # Logout (revokes the refresh token)
    await client.post("/api/auth/logout", json={"refresh_token": refresh_token})

    # Refresh token should now be invalid
    r3 = await client.post(
        "/api/auth/refresh", json={"refresh_token": refresh_token}
    )
    assert r3.status_code == 401


@pytest.mark.asyncio
async def test_api_requires_auth(client):
    resp = await client.get("/api/wiki")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_api_accepts_valid_token(client, auth_headers):
    resp = await client.get("/api/wiki", headers=auth_headers)
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_api_rejects_garbage_token(client):
    resp = await client.get(
        "/api/wiki", headers={"Authorization": "Bearer not-a-real-token"}
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_two_logins_produce_different_access_tokens(client):
    r1 = await client.post(
        "/api/auth/login", json={"username": "admin", "password": "admin"}
    )
    r2 = await client.post(
        "/api/auth/login", json={"username": "admin", "password": "admin"}
    )
    assert r1.json()["access_token"] != r2.json()["access_token"]


@pytest.mark.asyncio
async def test_refresh_returns_new_access_token(client):
    r = await client.post(
        "/api/auth/login", json={"username": "admin", "password": "admin"}
    )
    original_access = r.json()["access_token"]
    refresh_token = r.json()["refresh_token"]

    r2 = await client.post(
        "/api/auth/refresh", json={"refresh_token": refresh_token}
    )
    assert r2.status_code == 200
    data = r2.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    # New access token is different from original (different exp claim)
    assert data["access_token"] != original_access


@pytest.mark.asyncio
async def test_refresh_preserves_session_org(client, default_user):
    """B11: the refreshed access token keeps the refresh token's (session) org,
    not the user's home org — so a multi-org member doesn't silently switch org
    on refresh."""
    from app.services import auth as auth_service
    from app.services.orgs import create_org

    org_b = await create_org("Refresh Session Org B")
    rt = await auth_service.create_refresh_token(
        email=default_user["email"], user_id=default_user["id"], org_id=org_b,
    )
    r = await client.post("/api/auth/refresh", json={"refresh_token": rt})
    assert r.status_code == 200
    ctx = auth_service.validate_access_token(r.json()["access_token"])
    assert ctx.org_id == org_b


@pytest.mark.asyncio
async def test_refresh_with_bad_token_returns_401(client):
    resp = await client.post(
        "/api/auth/refresh", json={"refresh_token": "not-a-real-refresh-token"}
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_health_endpoint(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["db"] == "ok"
