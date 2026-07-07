"""Integration tests for /api/orgs — org management endpoints."""
import pytest


# ── GET /api/orgs/me ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_my_org(client, auth_headers, default_user):
    resp = await client.get("/api/orgs/me", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == default_user["org_id"]
    assert "name" in data


# ── GET /api/orgs/me/users ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_list_org_users_admin(client, auth_headers, default_user):
    # Admin users have no org (org_id=None since migration 008), so the list is empty
    resp = await client.get("/api/orgs/me/users", headers=auth_headers)
    assert resp.status_code == 200
    users = resp.json()
    assert isinstance(users, list)


# ── POST /api/orgs/me/users ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_create_user_in_org(client, auth_headers):
    payload = {"email": "newbie@example.com", "password": "secret123", "role": "member"}
    resp = await client.post("/api/orgs/me/users", json=payload, headers=auth_headers)
    assert resp.status_code == 201
    data = resp.json()
    assert data["email"] == "newbie@example.com"
    assert data["role"] == "member"


@pytest.mark.asyncio
async def test_create_user_duplicate_email(client, auth_headers):
    payload = {"email": "dup@example.com", "password": "pw", "role": "member"}
    r1 = await client.post("/api/orgs/me/users", json=payload, headers=auth_headers)
    assert r1.status_code == 201
    r2 = await client.post("/api/orgs/me/users", json=payload, headers=auth_headers)
    assert r2.status_code == 409


@pytest.mark.asyncio
async def test_create_user_invalid_role(client, auth_headers):
    payload = {"email": "x@x.com", "password": "pw", "role": "superuser"}
    resp = await client.post("/api/orgs/me/users", json=payload, headers=auth_headers)
    assert resp.status_code == 400


# ── DELETE /api/orgs/me/users/{id} ────────────────────────────────────────

@pytest.mark.asyncio
async def test_delete_user(client, auth_headers):
    # Create a user first
    create_resp = await client.post(
        "/api/orgs/me/users",
        json={"email": "todelete@x.com", "password": "pw", "role": "member"},
        headers=auth_headers,
    )
    assert create_resp.status_code == 201
    user_id = create_resp.json()["id"]

    del_resp = await client.delete(f"/api/orgs/me/users/{user_id}", headers=auth_headers)
    assert del_resp.status_code == 200


@pytest.mark.asyncio
async def test_delete_nonexistent_user(client, auth_headers):
    resp = await client.delete(
        "/api/orgs/me/users/00000000-0000-0000-0000-000000000099",
        headers=auth_headers,
    )
    assert resp.status_code == 404


# ── GET /api/orgs/me/usage ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_usage_empty(client, auth_headers):
    resp = await client.get("/api/orgs/me/usage", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert "usage" in data
    assert isinstance(data["usage"], list)


# ── POST /api/orgs (system admin — create new org) ─────────────────────────

@pytest.mark.asyncio
async def test_create_org(client, auth_headers):
    payload = {
        "name": "Acme Corp",
        "slug": "acme",
        "admin_email": "acme-admin@acme.com",
        "admin_password": "acme-pass",
    }
    resp = await client.post("/api/orgs", json=payload, headers=auth_headers)
    assert resp.status_code == 201
    data = resp.json()
    assert "org_id" in data
    assert "admin_user_id" in data


# ── Login flow (POST /api/auth/login) ─────────────────────────────────────

@pytest.mark.asyncio
async def test_login_returns_org_id(client, default_user):
    resp = await client.post(
        "/api/auth/login",
        json={"username": "admin", "password": "admin"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert "org_id" in data
    # Admin users have no org_id (org_id=None since migration 008)
    assert data["org_id"] is None


@pytest.mark.asyncio
async def test_login_wrong_password(client, default_user):
    resp = await client.post(
        "/api/auth/login",
        json={"username": "admin", "password": "wrong"},
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_org_not_accessible_without_token(client, default_user):
    resp = await client.get("/api/orgs/me")
    assert resp.status_code == 401
