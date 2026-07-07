"""Integration tests for admin dashboard endpoints.

Requires testcontainers (Docker). All tests skip if unavailable.
"""
import pytest
import pytest_asyncio
import sqlalchemy as sa

pytestmark = pytest.mark.skipif(
    False, reason="integration tests"
)

_DEFAULT_ORG_ID = "00000000-0000-0000-0000-000000000001"

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def member_user(test_engine, default_user):
    """Create a non-admin member user and return their credentials."""
    from app.services.orgs import create_user
    user_id = await create_user(_DEFAULT_ORG_ID, "member@test.com", "password123", role="member")
    assert user_id is not None
    return {"id": user_id, "email": "member@test.com", "role": "member", "org_id": _DEFAULT_ORG_ID}


@pytest.fixture
def member_auth_headers(member_user):
    from app.services.auth import create_access_token
    token = create_access_token(
        email=member_user["email"],
        user_id=member_user["id"],
        org_id=member_user["org_id"],
        role="member",
    )
    return {"Authorization": f"Bearer {token}"}


# ── Permission Templates CRUD ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_list_templates_admin(client, auth_headers):
    r = await client.get("/api/admin/templates", headers=auth_headers)
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    # Built-in templates seeded by migration are present
    names = {t["name"] for t in data}
    assert "read_only" in names
    assert "contributor" in names
    assert "power_user" in names


@pytest.mark.asyncio
async def test_list_templates_member_forbidden(client, member_auth_headers):
    r = await client.get("/api/admin/templates", headers=member_auth_headers)
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_create_and_get_template(client, auth_headers):
    body = {
        "name": "reviewer",
        "description": "Can view and query but not upload",
        "can_view_wiki": True,
        "can_query": True,
        "can_chat": False,
        "can_upload": False,
        "max_queries_per_day": 30,
    }
    r = await client.post("/api/admin/templates", json=body, headers=auth_headers)
    assert r.status_code == 201
    tpl_id = r.json()["id"]

    r2 = await client.get(f"/api/admin/templates/{tpl_id}", headers=auth_headers)
    assert r2.status_code == 200
    tpl = r2.json()
    assert tpl["name"] == "reviewer"
    assert tpl["can_view_wiki"] is True
    assert tpl["can_upload"] is False
    assert tpl["max_queries_per_day"] == 30


@pytest.mark.asyncio
async def test_create_template_name_conflict(client, auth_headers):
    body = {"name": "conflict_tpl", "description": "first"}
    r1 = await client.post("/api/admin/templates", json=body, headers=auth_headers)
    assert r1.status_code == 201
    r2 = await client.post("/api/admin/templates", json=body, headers=auth_headers)
    assert r2.status_code == 409


@pytest.mark.asyncio
async def test_update_custom_template(client, auth_headers):
    create_r = await client.post("/api/admin/templates",
                                  json={"name": "editable", "description": "orig"},
                                  headers=auth_headers)
    assert create_r.status_code == 201
    tpl_id = create_r.json()["id"]

    upd = await client.put(f"/api/admin/templates/{tpl_id}",
                            json={"name": "editable", "description": "updated", "can_upload": True},
                            headers=auth_headers)
    assert upd.status_code == 200

    r = await client.get(f"/api/admin/templates/{tpl_id}", headers=auth_headers)
    assert r.json()["description"] == "updated"
    assert r.json()["can_upload"] is True


@pytest.mark.asyncio
async def test_update_builtin_template_forbidden(client, auth_headers):
    templates = (await client.get("/api/admin/templates", headers=auth_headers)).json()
    builtin = next(t for t in templates if t["is_builtin"])
    r = await client.put(f"/api/admin/templates/{builtin['id']}",
                          json={"name": builtin["name"], "description": "hacked"},
                          headers=auth_headers)
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_delete_custom_template(client, auth_headers):
    create_r = await client.post("/api/admin/templates",
                                  json={"name": "to_delete", "description": ""},
                                  headers=auth_headers)
    tpl_id = create_r.json()["id"]
    del_r = await client.delete(f"/api/admin/templates/{tpl_id}", headers=auth_headers)
    assert del_r.status_code == 200
    get_r = await client.get(f"/api/admin/templates/{tpl_id}", headers=auth_headers)
    assert get_r.status_code == 404


@pytest.mark.asyncio
async def test_delete_builtin_template_forbidden(client, auth_headers):
    templates = (await client.get("/api/admin/templates", headers=auth_headers)).json()
    builtin = next(t for t in templates if t["is_builtin"])
    r = await client.delete(f"/api/admin/templates/{builtin['id']}", headers=auth_headers)
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_clone_template(client, auth_headers):
    templates = (await client.get("/api/admin/templates", headers=auth_headers)).json()
    src = templates[0]
    # Admin users have no org_id; must supply a target org when cloning global built-ins
    r = await client.post(f"/api/admin/templates/{src['id']}/clone",
                           json={"name": "cloned_tpl", "org_id": _DEFAULT_ORG_ID},
                           headers=auth_headers)
    assert r.status_code == 201
    cloned_id = r.json()["id"]

    cloned = (await client.get(f"/api/admin/templates/{cloned_id}", headers=auth_headers)).json()
    assert cloned["name"] == "cloned_tpl"
    assert cloned["is_builtin"] is False


# ── Template user-count endpoint ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_template_user_count_no_users(client, auth_headers):
    """A freshly created template has zero assigned users."""
    create_r = await client.post("/api/admin/templates",
                                  json={"name": "count_empty_tpl"},
                                  headers=auth_headers)
    assert create_r.status_code == 201
    tpl_id = create_r.json()["id"]

    r = await client.get(f"/api/admin/templates/{tpl_id}/user-count", headers=auth_headers)
    assert r.status_code == 200
    assert r.json()["count"] == 0


@pytest.mark.asyncio
async def test_template_user_count_with_assigned_user(client, auth_headers, member_user):
    """Count reflects users actually assigned to the template."""
    templates = (await client.get("/api/admin/templates", headers=auth_headers)).json()
    tpl = next(t for t in templates if t["name"] == "contributor")

    await client.put(f"/api/admin/users/{member_user['id']}",
                     json={"permission_template_id": tpl["id"], "role": "member"},
                     headers=auth_headers)

    r = await client.get(f"/api/admin/templates/{tpl['id']}/user-count", headers=auth_headers)
    assert r.status_code == 200
    assert r.json()["count"] >= 1


@pytest.mark.asyncio
async def test_template_user_count_member_forbidden(client, member_auth_headers):
    r = await client.get("/api/admin/templates/some-id/user-count", headers=member_auth_headers)
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_delete_template_nulls_user_assignment(client, auth_headers, member_user):
    """Deleting a template sets affected users to NULL → read_only permissions."""
    from app.services.auth import create_access_token
    member_hdrs = {"Authorization": f"Bearer {create_access_token(
        email=member_user['email'],
        user_id=member_user['id'],
        org_id=member_user['org_id'],
        role='member',
    )}"}

    # Create a custom template with can_query=True and assign the member to it
    create_r = await client.post("/api/admin/templates",
                                  json={"name": "to_delete_with_user", "can_query": True,
                                        "org_id": _DEFAULT_ORG_ID},
                                  headers=auth_headers)
    assert create_r.status_code == 201
    tpl_id = create_r.json()["id"]

    await client.put(f"/api/admin/users/{member_user['id']}",
                     json={"permission_template_id": tpl_id, "role": "member"},
                     headers=auth_headers)

    # User has can_query via the template
    perms_before = (await client.get("/api/ops/permissions", headers=member_hdrs)).json()
    assert perms_before["can_query"] is True

    # Confirm user-count endpoint reflects the assignment
    count_r = await client.get(f"/api/admin/templates/{tpl_id}/user-count", headers=auth_headers)
    assert count_r.json()["count"] == 1

    # Delete the template — DB ON DELETE SET NULL fires
    del_r = await client.delete(f"/api/admin/templates/{tpl_id}", headers=auth_headers)
    assert del_r.status_code == 200

    # User now falls back to read_only defaults (NULL template_id)
    perms_after = (await client.get("/api/ops/permissions", headers=member_hdrs)).json()
    assert perms_after["can_query"] is False


# ── User management ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_list_users_admin(client, auth_headers, member_user):
    r = await client.get("/api/admin/users", headers=auth_headers)
    assert r.status_code == 200
    users = r.json()
    emails = {u["email"] for u in users}
    assert "member@test.com" in emails


@pytest.mark.asyncio
async def test_list_users_member_forbidden(client, member_auth_headers):
    r = await client.get("/api/admin/users", headers=member_auth_headers)
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_update_user_assign_template(client, auth_headers, member_user):
    templates = (await client.get("/api/admin/templates", headers=auth_headers)).json()
    tpl = next(t for t in templates if t["name"] == "contributor")

    r = await client.put(f"/api/admin/users/{member_user['id']}",
                          json={"permission_template_id": tpl["id"], "role": "member"},
                          headers=auth_headers)
    assert r.status_code == 200

    users = (await client.get("/api/admin/users", headers=auth_headers)).json()
    updated = next(u for u in users if u["id"] == member_user["id"])
    assert updated["permission_template_id"] == tpl["id"]
    assert updated["template_name"] == "contributor"


@pytest.mark.asyncio
async def test_update_user_suspend(client, auth_headers, member_user):
    r = await client.put(f"/api/admin/users/{member_user['id']}",
                          json={"is_suspended": True, "role": "member"},
                          headers=auth_headers)
    assert r.status_code == 200
    users = (await client.get("/api/admin/users", headers=auth_headers)).json()
    u = next(u for u in users if u["id"] == member_user["id"])
    assert u["is_suspended"] is True


@pytest.mark.asyncio
async def test_suspended_user_cannot_access_api(client, member_user, auth_headers):
    """After suspension, member's own requests should get 403."""
    # First assign a template that grants can_query
    templates = (await client.get("/api/admin/templates", headers=auth_headers)).json()
    tpl = next(t for t in templates if t["name"] == "contributor")

    await client.put(f"/api/admin/users/{member_user['id']}",
                      json={"permission_template_id": tpl["id"], "is_suspended": False, "role": "member"},
                      headers=auth_headers)

    # Suspend
    await client.put(f"/api/admin/users/{member_user['id']}",
                      json={"permission_template_id": tpl["id"], "is_suspended": True, "role": "member"},
                      headers=auth_headers)

    from app.services.auth import create_access_token
    member_token = create_access_token(
        email=member_user["email"], user_id=member_user["id"],
        org_id=member_user["org_id"], role="member",
    )
    member_headers = {"Authorization": f"Bearer {member_token}"}
    # With suspension, any permission-gated endpoint should return 403
    r = await client.post("/api/ops/query",
                           json={"question": "test"},
                           headers=member_headers)
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_reset_password(client, auth_headers, member_user):
    r = await client.post(f"/api/admin/users/{member_user['id']}/reset-password",
                           json={"new_password": "newpassword123"},
                           headers=auth_headers)
    assert r.status_code == 200
    # Verify new password works (login endpoint uses "username" field)
    login_r = await client.post("/api/auth/login",
                                 json={"username": "member@test.com", "password": "newpassword123"})
    assert login_r.status_code == 200


@pytest.mark.asyncio
async def test_reset_password_too_short(client, auth_headers, member_user):
    r = await client.post(f"/api/admin/users/{member_user['id']}/reset-password",
                           json={"new_password": "short"},
                           headers=auth_headers)
    assert r.status_code == 400


# ── Supervisor assignment on user create ─────────────────────────────────────

@pytest.mark.asyncio
async def test_create_user_with_role_supervisor_sets_org_supervisor(client, auth_headers):
    """POST /api/admin/users with role=supervisor + org_id must set the org's
    supervisor pointer so the new user's email appears in the Organizations tab."""
    r = await client.post(
        "/api/admin/users",
        json={
            "email": "sup1@test.com", "password": "password123",
            "role": "supervisor", "org_id": _DEFAULT_ORG_ID,
        },
        headers=auth_headers,
    )
    assert r.status_code == 201
    new_user_id = r.json()["id"]

    orgs = (await client.get("/api/admin/organizations", headers=auth_headers)).json()
    default_org = next(o for o in orgs if o["id"] == _DEFAULT_ORG_ID)
    assert default_org["supervisor_email"] == "sup1@test.com"
    assert default_org["supervisor_user_id"] == new_user_id


@pytest.mark.asyncio
async def test_create_user_role_supervisor_overwrites_existing_pointer(client, auth_headers):
    """Matches PUT /organizations/{id}/supervisor semantics: the latest assignment wins."""
    r1 = await client.post(
        "/api/admin/users",
        json={"email": "sup_a@test.com", "password": "password123",
              "role": "supervisor", "org_id": _DEFAULT_ORG_ID},
        headers=auth_headers,
    )
    assert r1.status_code == 201

    r2 = await client.post(
        "/api/admin/users",
        json={"email": "sup_b@test.com", "password": "password123",
              "role": "supervisor", "org_id": _DEFAULT_ORG_ID},
        headers=auth_headers,
    )
    assert r2.status_code == 201

    orgs = (await client.get("/api/admin/organizations", headers=auth_headers)).json()
    default_org = next(o for o in orgs if o["id"] == _DEFAULT_ORG_ID)
    assert default_org["supervisor_email"] == "sup_b@test.com"


@pytest.mark.asyncio
async def test_create_user_role_member_does_not_touch_supervisor_pointer(client, auth_headers):
    """A member create must not alter the org's supervisor pointer."""
    # Seed an existing supervisor first
    await client.post(
        "/api/admin/users",
        json={"email": "existing_sup@test.com", "password": "password123",
              "role": "supervisor", "org_id": _DEFAULT_ORG_ID},
        headers=auth_headers,
    )

    r = await client.post(
        "/api/admin/users",
        json={"email": "newmember@test.com", "password": "password123",
              "role": "member", "org_id": _DEFAULT_ORG_ID},
        headers=auth_headers,
    )
    assert r.status_code == 201

    orgs = (await client.get("/api/admin/organizations", headers=auth_headers)).json()
    default_org = next(o for o in orgs if o["id"] == _DEFAULT_ORG_ID)
    assert default_org["supervisor_email"] == "existing_sup@test.com"


# ── Floating identities (Add User without an org) + identities endpoint ────────

@pytest.mark.asyncio
async def test_create_floating_user_has_no_org(client, auth_headers):
    """Add User with no org_id creates a credential-only identity with no membership."""
    r = await client.post(
        "/api/admin/users",
        json={"email": "floating@test.com", "password": "password123", "role": "member"},
        headers=auth_headers,
    )
    assert r.status_code == 201
    assert r.json()["org_id"] is None

    identities = (await client.get("/api/admin/identities", headers=auth_headers)).json()
    me = next(i for i in identities if i["email"] == "floating@test.com")
    assert me["is_admin"] is False
    assert me["org_count"] == 0


@pytest.mark.asyncio
async def test_create_admin_user_no_org(client, auth_headers):
    r = await client.post(
        "/api/admin/users",
        json={"email": "root2@test.com", "password": "password123", "role": "admin"},
        headers=auth_headers,
    )
    assert r.status_code == 201
    assert r.json()["org_id"] is None

    identities = (await client.get("/api/admin/identities", headers=auth_headers)).json()
    me = next(i for i in identities if i["email"] == "root2@test.com")
    assert me["is_admin"] is True
    assert me["org_count"] == 0


@pytest.mark.asyncio
async def test_list_identities_member_forbidden(client, member_auth_headers):
    r = await client.get("/api/admin/identities", headers=member_auth_headers)
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_floating_user_gains_access_via_membership(client, auth_headers):
    """Create a floating identity, then grant org access from the membership endpoint."""
    created = (await client.post(
        "/api/admin/users",
        json={"email": "later@test.com", "password": "password123", "role": "member"},
        headers=auth_headers,
    )).json()
    uid = created["id"]

    # Not a member of any org yet → absent from the default org's member rows.
    before = (await client.get("/api/admin/users", headers=auth_headers)).json()
    assert not any(u["id"] == uid and u["org_id"] == _DEFAULT_ORG_ID for u in before)

    # Grant membership in the default org.
    r = await client.post(
        f"/api/admin/users/{uid}/memberships",
        json={"org_id": _DEFAULT_ORG_ID, "role": "member"},
        headers=auth_headers,
    )
    assert r.status_code == 201

    after = (await client.get("/api/admin/users", headers=auth_headers)).json()
    assert any(u["id"] == uid and u["org_id"] == _DEFAULT_ORG_ID for u in after)

    identities = (await client.get("/api/admin/identities", headers=auth_headers)).json()
    me = next(i for i in identities if i["email"] == "later@test.com")
    assert me["org_count"] == 1


# ── Permission enforcement on routes ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_member_without_template_cannot_upload(client, member_user):
    """Member with no template defaults to read_only — upload denied."""
    from app.services.auth import create_access_token
    token = create_access_token(
        email=member_user["email"], user_id=member_user["id"],
        org_id=member_user["org_id"], role="member",
    )
    headers = {"Authorization": f"Bearer {token}"}
    # No file content needed — 403 should fire before file parsing
    import io
    files = {"file": ("test.txt", io.BytesIO(b"content"), "text/plain")}
    r = await client.post("/api/documents/upload", headers=headers, files=files)
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_contributor_member_can_query(client, auth_headers, member_user):
    """Assign contributor template → can_query should pass (LLM mocked so query may fail with 500, not 403)."""
    templates = (await client.get("/api/admin/templates", headers=auth_headers)).json()
    tpl = next(t for t in templates if t["name"] == "contributor")
    await client.put(f"/api/admin/users/{member_user['id']}",
                      json={"permission_template_id": tpl["id"], "role": "member"},
                      headers=auth_headers)

    from app.services.auth import create_access_token
    member_token = create_access_token(
        email=member_user["email"], user_id=member_user["id"],
        org_id=member_user["org_id"], role="member",
    )
    headers = {"Authorization": f"Bearer {member_token}"}
    r = await client.post("/api/ops/query", json={"question": "test"}, headers=headers)
    # 403 would indicate permission denied; anything else means the permission check passed
    assert r.status_code != 403


@pytest.mark.asyncio
async def test_writer_endpoint_member_without_template_denied(client, member_user):
    """A member with no template (read_only fallback) cannot reach writer endpoints."""
    from app.services.auth import create_access_token
    token = create_access_token(
        email=member_user["email"], user_id=member_user["id"],
        org_id=member_user["org_id"], role="member",
    )
    r = await client.get(
        "/api/ops/writer/sessions",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 403
    assert "writer" in r.json()["detail"].lower()


@pytest.mark.asyncio
async def test_writer_endpoint_member_with_read_only_denied(client, auth_headers, member_user):
    """read_only template has can_use_writer=false → writer endpoints return 403."""
    templates = (await client.get("/api/admin/templates", headers=auth_headers)).json()
    tpl = next(t for t in templates if t["name"] == "read_only")
    await client.put(
        f"/api/admin/users/{member_user['id']}",
        json={"permission_template_id": tpl["id"], "role": "member"},
        headers=auth_headers,
    )

    from app.services.auth import create_access_token
    token = create_access_token(
        email=member_user["email"], user_id=member_user["id"],
        org_id=member_user["org_id"], role="member",
    )
    r = await client.get(
        "/api/ops/writer/sessions",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_writer_endpoint_member_with_power_user_allowed(client, auth_headers, member_user):
    """power_user template has can_use_writer=true → writer endpoints pass the gate."""
    templates = (await client.get("/api/admin/templates", headers=auth_headers)).json()
    tpl = next(t for t in templates if t["name"] == "power_user")
    await client.put(
        f"/api/admin/users/{member_user['id']}",
        json={"permission_template_id": tpl["id"], "role": "member"},
        headers=auth_headers,
    )

    from app.services.auth import create_access_token
    token = create_access_token(
        email=member_user["email"], user_id=member_user["id"],
        org_id=member_user["org_id"], role="member",
    )
    r = await client.get(
        "/api/ops/writer/sessions",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_writer_endpoint_member_with_custom_can_use_writer_allowed(client, auth_headers, member_user):
    """A custom template that only sets can_use_writer=true should grant access."""
    create_r = await client.post(
        "/api/admin/templates",
        json={
            "name": "writer_only",
            "description": "Writer access only",
            "org_id": _DEFAULT_ORG_ID,
            "can_use_writer": True,
        },
        headers=auth_headers,
    )
    assert create_r.status_code == 201
    tpl_id = create_r.json()["id"]

    await client.put(
        f"/api/admin/users/{member_user['id']}",
        json={"permission_template_id": tpl_id, "role": "member"},
        headers=auth_headers,
    )

    from app.services.auth import create_access_token
    token = create_access_token(
        email=member_user["email"], user_id=member_user["id"],
        org_id=member_user["org_id"], role="member",
    )
    r = await client.get(
        "/api/ops/writer/sessions",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200


# ── can_upload_writer_draft enforcement ──────────────────────────────────────

async def _seed_writer_draft(test_engine, org_id, user_id, content="# Hello\n\nBody",
                              draft_ready: bool = True):
    """Insert a writer-mode chat_sessions row directly, return its session_id.

    Defaults to draft_ready=True so existing tests (which exercise unrelated
    permission gates) aren't affected by the readiness gate added in
    migration 021. Tests that want to exercise the gate itself pass
    draft_ready=False explicitly.
    """
    import uuid
    session_id = str(uuid.uuid4())
    async with test_engine.begin() as conn:
        await conn.execute(sa.text("""
            INSERT INTO chat_sessions (
                org_id, session_id, messages, summary, mode,
                draft_content, draft_filename, draft_ready,
                user_id, last_active_at
            ) VALUES (
                CAST(:org_id AS UUID), :session_id, '[]'::jsonb, '', 'writer',
                :content, '', :draft_ready,
                CAST(:uid AS UUID), NOW()
            )
        """), {"org_id": org_id, "session_id": session_id,
               "content": content, "draft_ready": draft_ready, "uid": user_id})
    return session_id


@pytest.mark.asyncio
async def test_writer_ingest_denied_when_can_upload_writer_draft_false(
    client, auth_headers, member_user, test_engine,
):
    """can_upload=true + can_upload_writer_draft=false → /api/documents/upload OK
       but /api/ops/writer/{sid}/ingest returns 403 with a clear message."""
    tpl_r = await client.post(
        "/api/admin/templates",
        json={"name": "upload_no_writer", "can_upload": True,
              "can_upload_writer_draft": False, "can_use_writer": True},
        headers=auth_headers,
    )
    assert tpl_r.status_code == 201
    tpl_id = tpl_r.json()["id"]

    await client.put(
        f"/api/admin/users/{member_user['id']}",
        json={"permission_template_id": tpl_id, "role": "member"},
        headers=auth_headers,
    )

    sid = await _seed_writer_draft(test_engine, member_user["org_id"], member_user["id"])

    from app.services.auth import create_access_token
    token = create_access_token(
        email=member_user["email"], user_id=member_user["id"],
        org_id=member_user["org_id"], role="member",
    )
    r = await client.post(
        f"/api/ops/writer/{sid}/ingest",
        json={"filename": "draft1.md"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 403
    assert "can_upload_writer_draft" in r.json()["detail"]


@pytest.mark.asyncio
async def test_writer_ingest_allowed_when_only_can_upload_writer_draft_true(
    client, auth_headers, member_user, test_engine,
):
    """The headline use case: can_upload=false, can_upload_writer_draft=true.
       Writer ingest must pass the gate; direct upload must be denied."""
    tpl_r = await client.post(
        "/api/admin/templates",
        json={"name": "writer_only", "can_upload": False,
              "can_upload_writer_draft": True, "can_use_writer": True},
        headers=auth_headers,
    )
    assert tpl_r.status_code == 201
    tpl_id = tpl_r.json()["id"]

    await client.put(
        f"/api/admin/users/{member_user['id']}",
        json={"permission_template_id": tpl_id, "role": "member"},
        headers=auth_headers,
    )

    sid = await _seed_writer_draft(test_engine, member_user["org_id"], member_user["id"])

    from app.services.auth import create_access_token
    token = create_access_token(
        email=member_user["email"], user_id=member_user["id"],
        org_id=member_user["org_id"], role="member",
    )
    hdrs = {"Authorization": f"Bearer {token}"}

    # Direct upload is blocked
    import io
    direct = await client.post(
        "/api/documents/upload",
        headers=hdrs,
        files={"file": ("foo.txt", io.BytesIO(b"x"), "text/plain")},
    )
    assert direct.status_code == 403

    # Writer ingest passes the permission gate (200 means the gate let it through)
    ingest = await client.post(
        f"/api/ops/writer/{sid}/ingest",
        json={"filename": "writer-only.md"},
        headers=hdrs,
    )
    assert ingest.status_code != 403, f"expected gate to pass, got {ingest.status_code}: {ingest.text}"


@pytest.mark.asyncio
async def test_writer_ingest_denied_when_both_false(
    client, auth_headers, member_user, test_engine,
):
    tpl_r = await client.post(
        "/api/admin/templates",
        json={"name": "neither_upload", "can_upload": False,
              "can_upload_writer_draft": False, "can_use_writer": True},
        headers=auth_headers,
    )
    assert tpl_r.status_code == 201
    tpl_id = tpl_r.json()["id"]

    await client.put(
        f"/api/admin/users/{member_user['id']}",
        json={"permission_template_id": tpl_id, "role": "member"},
        headers=auth_headers,
    )

    sid = await _seed_writer_draft(test_engine, member_user["org_id"], member_user["id"])

    from app.services.auth import create_access_token
    token = create_access_token(
        email=member_user["email"], user_id=member_user["id"],
        org_id=member_user["org_id"], role="member",
    )
    r = await client.post(
        f"/api/ops/writer/{sid}/ingest",
        json={"filename": "blocked.md"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_writer_ingest_respects_per_user_upload_quota(
    client, auth_headers, member_user, test_engine,
):
    """writer_ingest now goes through check_upload_quota — once the daily
    upload cap is exhausted (via any ingest_jobs row), writer-ingest 429s."""
    tpl_r = await client.post(
        "/api/admin/templates",
        json={"name": "tight_writer", "can_use_writer": True,
              "can_upload_writer_draft": True, "max_uploads_per_day": 1},
        headers=auth_headers,
    )
    assert tpl_r.status_code == 201
    tpl_id = tpl_r.json()["id"]

    await client.put(
        f"/api/admin/users/{member_user['id']}",
        json={"permission_template_id": tpl_id, "role": "member"},
        headers=auth_headers,
    )

    # Burn the daily quota via a stand-in ingest_jobs row.
    async with test_engine.begin() as conn:
        await conn.execute(sa.text("""
            INSERT INTO ingest_jobs (org_id, user_id, filename, status)
            VALUES (CAST(:org AS UUID), CAST(:uid AS UUID), 'seed.txt', 'done')
        """), {"org": member_user["org_id"], "uid": member_user["id"]})

    sid = await _seed_writer_draft(test_engine, member_user["org_id"], member_user["id"])

    from app.services.auth import create_access_token
    token = create_access_token(
        email=member_user["email"], user_id=member_user["id"],
        org_id=member_user["org_id"], role="member",
    )
    r = await client.post(
        f"/api/ops/writer/{sid}/ingest",
        json={"filename": "quota.md"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 429
    assert "upload" in r.json()["detail"].lower()


# ── draft_ready gate ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_writer_ingest_blocked_when_draft_not_ready(
    client, auth_headers, member_user, test_engine,
):
    """Even with every permission, an un-ready draft must 409 — hard gate."""
    tpl_r = await client.post(
        "/api/admin/templates",
        json={"name": "ready_test_full", "can_upload_writer_draft": True,
              "can_use_writer": True},
        headers=auth_headers,
    )
    assert tpl_r.status_code == 201
    await client.put(
        f"/api/admin/users/{member_user['id']}",
        json={"permission_template_id": tpl_r.json()["id"], "role": "member"},
        headers=auth_headers,
    )

    sid = await _seed_writer_draft(
        test_engine, member_user["org_id"], member_user["id"], draft_ready=False,
    )

    from app.services.auth import create_access_token
    token = create_access_token(
        email=member_user["email"], user_id=member_user["id"],
        org_id=member_user["org_id"], role="member",
    )
    r = await client.post(
        f"/api/ops/writer/{sid}/ingest",
        json={"filename": "not-ready.md"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert "not marked" in detail.lower() or "not ready" in detail.lower()
    assert "Traceback" not in detail


@pytest.mark.asyncio
async def test_writer_ingest_allowed_when_draft_ready(
    client, auth_headers, member_user, test_engine,
):
    tpl_r = await client.post(
        "/api/admin/templates",
        json={"name": "ready_test_pass", "can_upload_writer_draft": True,
              "can_use_writer": True},
        headers=auth_headers,
    )
    assert tpl_r.status_code == 201
    await client.put(
        f"/api/admin/users/{member_user['id']}",
        json={"permission_template_id": tpl_r.json()["id"], "role": "member"},
        headers=auth_headers,
    )

    sid = await _seed_writer_draft(
        test_engine, member_user["org_id"], member_user["id"], draft_ready=True,
    )

    from app.services.auth import create_access_token
    token = create_access_token(
        email=member_user["email"], user_id=member_user["id"],
        org_id=member_user["org_id"], role="member",
    )
    r = await client.post(
        f"/api/ops/writer/{sid}/ingest",
        json={"filename": "ready.md"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code != 409, f"expected gate to pass, got 409: {r.text}"
    assert r.status_code != 403


@pytest.mark.asyncio
async def test_writer_ingest_gate_applies_to_admin_too(
    client, auth_headers, default_user, test_engine,
):
    """No role escape hatch: admins are subject to the same content gate."""
    sid = await _seed_writer_draft(
        test_engine, default_user["org_id"], default_user["id"], draft_ready=False,
    )
    r = await client.post(
        f"/api/ops/writer/{sid}/ingest",
        json={"filename": "admin-not-ready.md"},
        headers=auth_headers,
    )
    assert r.status_code == 409


@pytest.mark.asyncio
async def test_writer_ingest_permission_gate_fires_before_readiness_gate(
    client, auth_headers, member_user, test_engine,
):
    """A member without can_upload_writer_draft must see 403, NOT 409 —
    permission failures must be reported even if the draft is unready."""
    tpl_r = await client.post(
        "/api/admin/templates",
        json={"name": "no_upload_writer", "can_upload_writer_draft": False,
              "can_use_writer": True},
        headers=auth_headers,
    )
    assert tpl_r.status_code == 201
    await client.put(
        f"/api/admin/users/{member_user['id']}",
        json={"permission_template_id": tpl_r.json()["id"], "role": "member"},
        headers=auth_headers,
    )

    sid = await _seed_writer_draft(
        test_engine, member_user["org_id"], member_user["id"], draft_ready=False,
    )

    from app.services.auth import create_access_token
    token = create_access_token(
        email=member_user["email"], user_id=member_user["id"],
        org_id=member_user["org_id"], role="member",
    )
    r = await client.post(
        f"/api/ops/writer/{sid}/ingest",
        json={"filename": "still-403.md"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_get_draft_returns_draft_ready_flag(
    client, auth_headers, default_user, test_engine,
):
    """The /draft endpoint must surface the readiness flag for the UI to gate on."""
    sid = await _seed_writer_draft(
        test_engine, default_user["org_id"], default_user["id"], draft_ready=True,
    )
    r = await client.get(f"/api/ops/writer/{sid}/draft", headers=auth_headers)
    assert r.status_code == 200
    body = r.json()
    assert body["draft_ready"] is True

    sid2 = await _seed_writer_draft(
        test_engine, default_user["org_id"], default_user["id"], draft_ready=False,
    )
    r2 = await client.get(f"/api/ops/writer/{sid2}/draft", headers=auth_headers)
    assert r2.json()["draft_ready"] is False


@pytest.mark.asyncio
async def test_template_create_persists_can_upload_writer_draft(client, auth_headers):
    r = await client.post(
        "/api/admin/templates",
        json={"name": "uwd_round_trip", "can_upload_writer_draft": True},
        headers=auth_headers,
    )
    assert r.status_code == 201
    tpl_id = r.json()["id"]

    r2 = await client.get(f"/api/admin/templates/{tpl_id}", headers=auth_headers)
    assert r2.status_code == 200
    body = r2.json()
    assert body["can_upload_writer_draft"] is True
    # Independent of can_upload (we didn't set it; defaults to False)
    assert body["can_upload"] is False


@pytest.mark.asyncio
async def test_template_create_persists_can_use_writer(client, auth_headers):
    """The new field must round-trip through POST then GET on /api/admin/templates."""
    r = await client.post(
        "/api/admin/templates",
        json={"name": "writer_round_trip", "can_use_writer": True},
        headers=auth_headers,
    )
    assert r.status_code == 201
    tpl_id = r.json()["id"]

    r2 = await client.get(f"/api/admin/templates/{tpl_id}", headers=auth_headers)
    assert r2.status_code == 200
    assert r2.json()["can_use_writer"] is True


@pytest.mark.asyncio
async def test_read_only_member_cannot_edit_wiki(client, auth_headers, member_user):
    templates = (await client.get("/api/admin/templates", headers=auth_headers)).json()
    tpl = next(t for t in templates if t["name"] == "read_only")
    await client.put(f"/api/admin/users/{member_user['id']}",
                      json={"permission_template_id": tpl["id"], "role": "member"},
                      headers=auth_headers)

    from app.services.auth import create_access_token
    member_token = create_access_token(
        email=member_user["email"], user_id=member_user["id"],
        org_id=member_user["org_id"], role="member",
    )
    headers = {"Authorization": f"Bearer {member_token}"}
    r = await client.put("/api/wiki/some/page.md",
                          json={"content": "hacked"},
                          headers=headers)
    assert r.status_code == 403


# ── Workspaces ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_create_and_list_workspace(client, auth_headers):
    r = await client.post("/api/admin/workspaces",
                           json={"name": "test-workspace"},
                           headers=auth_headers)
    assert r.status_code == 201
    ws_id = r.json()["id"]

    r2 = await client.get("/api/admin/workspaces", headers=auth_headers)
    assert r2.status_code == 200
    ws_list = r2.json()
    ids = {w["id"] for w in ws_list}
    assert ws_id in ids


@pytest.mark.asyncio
async def test_workspace_name_conflict(client, auth_headers):
    await client.post("/api/admin/workspaces", json={"name": "dup-ws"}, headers=auth_headers)
    r2 = await client.post("/api/admin/workspaces", json={"name": "dup-ws"}, headers=auth_headers)
    assert r2.status_code == 409


@pytest.mark.asyncio
async def test_delete_workspace(client, auth_headers):
    r = await client.post("/api/admin/workspaces",
                           json={"name": "to-delete-ws"},
                           headers=auth_headers)
    ws_id = r.json()["id"]
    del_r = await client.delete(f"/api/admin/workspaces/{ws_id}", headers=auth_headers)
    assert del_r.status_code == 200
    ws_list = (await client.get("/api/admin/workspaces", headers=auth_headers)).json()
    assert ws_id not in {w["id"] for w in ws_list}


# ── Org limits ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_default_org_starts_with_10m_token_cap(client, auth_headers):
    """The Default Organization is seeded with max_tokens_per_day_org=10,000,000
    (see migration 019 and orgs.DEFAULT_ORG_MAX_TOKENS_PER_DAY)."""
    from app.services.orgs import DEFAULT_ORG_MAX_TOKENS_PER_DAY
    assert DEFAULT_ORG_MAX_TOKENS_PER_DAY == 10_000_000

    r = await client.get("/api/admin/org-limits", headers=auth_headers)
    assert r.status_code == 200
    data = r.json()
    default = (
        next(o for o in data if o["id"] == _DEFAULT_ORG_ID)
        if isinstance(data, list) else data
    )
    assert default["max_tokens_per_day_org"] == DEFAULT_ORG_MAX_TOKENS_PER_DAY


@pytest.mark.asyncio
async def test_get_and_update_org_limits(client, auth_headers):
    r = await client.get("/api/admin/org-limits", headers=auth_headers)
    assert r.status_code == 200
    # Admin receives a list of all orgs; supervisor receives a single dict
    data = r.json()
    if isinstance(data, list):
        assert len(data) > 0
        assert "max_uploads_per_day_org" in data[0]
        target_org_id = data[0]["id"]
    else:
        assert "max_uploads_per_day_org" in data
        target_org_id = data["id"]

    upd = await client.put("/api/admin/org-limits",
                            json={"org_id": target_org_id,
                                  "max_uploads_per_day_org": 100, "max_tokens_per_day_org": 500000,
                                  "max_members": 50},
                            headers=auth_headers)
    assert upd.status_code == 200

    r2 = await client.get("/api/admin/org-limits", headers=auth_headers)
    data2 = r2.json()
    if isinstance(data2, list):
        updated = next(o for o in data2 if o["id"] == target_org_id)
    else:
        updated = data2
    assert updated["max_uploads_per_day_org"] == 100
    assert updated["max_tokens_per_day_org"] == 500000
    assert updated["max_members"] == 50


@pytest.mark.asyncio
async def test_org_limits_member_forbidden(client, member_auth_headers):
    r = await client.get("/api/admin/org-limits", headers=member_auth_headers)
    assert r.status_code == 403


# ── Audit log ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_audit_log_returns_paginated(client, auth_headers):
    r = await client.get("/api/admin/audit-log?page=1&limit=10", headers=auth_headers)
    assert r.status_code == 200
    data = r.json()
    assert "entries" in data
    assert "total" in data
    assert "page" in data
    assert isinstance(data["entries"], list)


@pytest.mark.asyncio
async def test_audit_log_member_forbidden(client, member_auth_headers):
    r = await client.get("/api/admin/audit-log", headers=member_auth_headers)
    assert r.status_code == 403


# ── Upload quota enforcement ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_upload_quota_enforced(client, auth_headers, member_user, test_engine):
    """Assign contributor template with max 1 upload/day, then verify second upload is blocked."""
    # Create a strict template
    tpl_r = await client.post("/api/admin/templates",
                               json={"name": "strict_uploader", "can_upload": True,
                                     "max_uploads_per_day": 1},
                               headers=auth_headers)
    assert tpl_r.status_code == 201
    tpl_id = tpl_r.json()["id"]

    await client.put(f"/api/admin/users/{member_user['id']}",
                      json={"permission_template_id": tpl_id, "role": "member"},
                      headers=auth_headers)

    # Manually insert a fake job from today so quota is used up
    async with test_engine.begin() as conn:
        await conn.execute(sa.text("""
            INSERT INTO ingest_jobs (org_id, user_id, filename, status)
            VALUES (CAST(:org AS UUID), CAST(:uid AS UUID), 'previous.txt', 'done')
        """), {"org": _DEFAULT_ORG_ID, "uid": member_user["id"]})

    from app.services.auth import create_access_token
    token = create_access_token(
        email=member_user["email"], user_id=member_user["id"],
        org_id=member_user["org_id"], role="member",
    )
    headers = {"Authorization": f"Bearer {token}"}
    import io
    files = {"file": ("new.txt", io.BytesIO(b"content"), "text/plain")}
    r = await client.post("/api/documents/upload", headers=headers, files=files)
    assert r.status_code == 429
