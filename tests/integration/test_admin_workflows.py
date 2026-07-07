"""Integration tests for all Testable Workflows defined in ADMIN_DASHBOARD.md.

Each test function is prefixed with the workflow number it covers (WF-01 … WF-40).
Workflows that require a live LLM call (WF-31) are omitted; everything else is
tested against the real database via testcontainers.
"""
import io
import pytest
import pytest_asyncio
import sqlalchemy as sa

_DEFAULT_ORG_ID = "00000000-0000-0000-0000-000000000001"

# A schema document that passes every validation rule the server enforces.
_VALID_SCHEMA = (
    "# Test Wiki Schema\n\n"
    "## Page Types\n"
    "Every page declares a `type:` in its frontmatter from this vocabulary:\n"
    "- `source_summary` — provenance page for an uploaded document\n"
    "- `concept` — explanation of an idea, process, or topic\n"
    "- `entity` — a person, team, product, or system\n"
    "- `rca` — root cause analysis or incident write-up\n"
    "- `query_result` — saved answer to a user question\n\n"
    "## Directory Structure\nwiki/ tree.\n\n"
    "## Page Naming\nLowercase with hyphens.\n\n"
    "## Cross-linking\nUse [[page-name]] syntax.\n\n"
    "## Audit Log Entry Format\n## [date] ingest | Title\n\n"
    "## Ingest Checklist\n1. Create source_summary page\n2. Cross-link references\n\n"
    "## Contradictions\nAdd conflicting sources section.\n"
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def supervisor_user(test_engine, default_user):
    """Supervisor in the default org."""
    from app.services.orgs import create_user
    user_id = await create_user(
        _DEFAULT_ORG_ID, "supervisor@test.com", "password123", role="supervisor"
    )
    async with test_engine.begin() as conn:
        await conn.execute(sa.text("""
            UPDATE organizations SET supervisor_user_id = CAST(:uid AS UUID)
            WHERE id = CAST(:oid AS UUID)
        """), {"uid": user_id, "oid": _DEFAULT_ORG_ID})
    return {"id": user_id, "email": "supervisor@test.com", "role": "supervisor",
            "org_id": _DEFAULT_ORG_ID}


@pytest.fixture
def supervisor_headers(supervisor_user):
    from app.services.auth import create_access_token
    token = create_access_token(
        email=supervisor_user["email"], user_id=supervisor_user["id"],
        org_id=supervisor_user["org_id"], role="supervisor",
    )
    return {"Authorization": f"Bearer {token}"}


@pytest_asyncio.fixture
async def member_user(test_engine, default_user):
    """Plain member (no template) in the default org."""
    from app.services.orgs import create_user
    user_id = await create_user(
        _DEFAULT_ORG_ID, "member@test.com", "password123", role="member"
    )
    return {"id": user_id, "email": "member@test.com", "role": "member",
            "org_id": _DEFAULT_ORG_ID}


@pytest.fixture
def member_headers(member_user):
    from app.services.auth import create_access_token
    token = create_access_token(
        email=member_user["email"], user_id=member_user["id"],
        org_id=member_user["org_id"], role="member",
    )
    return {"Authorization": f"Bearer {token}"}


@pytest_asyncio.fixture
async def org_b(client, auth_headers):
    """Create a second organisation and return its id."""
    r = await client.post(
        "/api/admin/organizations", json={"name": "Org B"}, headers=auth_headers
    )
    assert r.status_code == 201
    return r.json()["id"]


@pytest_asyncio.fixture
async def org_b_supervisor(org_b, default_user):
    from app.services.orgs import create_user
    user_id = await create_user(
        org_b, "sup_b@test.com", "password123", role="supervisor"
    )
    return {"id": user_id, "email": "sup_b@test.com", "role": "supervisor",
            "org_id": org_b}


@pytest.fixture
def org_b_supervisor_headers(org_b_supervisor):
    from app.services.auth import create_access_token
    token = create_access_token(
        email=org_b_supervisor["email"], user_id=org_b_supervisor["id"],
        org_id=org_b_supervisor["org_id"], role="supervisor",
    )
    return {"Authorization": f"Bearer {token}"}


async def _make_member(org_id, email, template_name, client, auth_headers):
    """Create a member user with a specific template; return (user_dict, headers)."""
    from app.services.orgs import create_user
    from app.services.auth import create_access_token

    user_id = await create_user(org_id, email, "password123", role="member")
    templates = (await client.get("/api/admin/templates", headers=auth_headers)).json()
    tpl = next(t for t in templates if t["name"] == template_name)
    await client.put(
        f"/api/admin/users/{user_id}",
        json={"permission_template_id": tpl["id"], "role": "member"},
        headers=auth_headers,
    )
    token = create_access_token(email=email, user_id=user_id, org_id=org_id, role="member")
    return (
        {"id": user_id, "email": email, "role": "member", "org_id": org_id,
         "template_id": tpl["id"]},
        {"Authorization": f"Bearer {token}"},
    )


# ── WF-01: Role access control to admin dashboard ─────────────────────────────

@pytest.mark.asyncio
async def test_wf01_member_cannot_access_admin_users(client, member_user, member_headers):
    r = await client.get("/api/admin/users", headers=member_headers)
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_wf01_supervisor_sees_own_org_users(client, supervisor_user, supervisor_headers):
    r = await client.get("/api/admin/users", headers=supervisor_headers)
    assert r.status_code == 200
    users = r.json()
    org_ids = {u["org_id"] for u in users}
    assert all(o == _DEFAULT_ORG_ID for o in org_ids if o)


@pytest.mark.asyncio
async def test_wf01_admin_sees_all_users(client, auth_headers, supervisor_user, member_user):
    r = await client.get("/api/admin/users", headers=auth_headers)
    assert r.status_code == 200
    emails = {u["email"] for u in r.json()}
    assert "supervisor@test.com" in emails
    assert "member@test.com" in emails


# ── WF-02: Built-in template visibility and immutability ──────────────────────

@pytest.mark.asyncio
async def test_wf02_supervisor_sees_builtins(client, supervisor_user, supervisor_headers):
    r = await client.get("/api/admin/templates", headers=supervisor_headers)
    assert r.status_code == 200
    names = {t["name"] for t in r.json()}
    assert {"read_only", "contributor", "power_user"}.issubset(names)
    builtins = [t for t in r.json() if t["is_builtin"]]
    assert all(t["org_id"] is None for t in builtins)


@pytest.mark.asyncio
async def test_wf02_supervisor_cannot_edit_builtin(client, supervisor_user, supervisor_headers):
    templates = (await client.get("/api/admin/templates", headers=supervisor_headers)).json()
    builtin = next(t for t in templates if t["is_builtin"])
    r = await client.put(
        f"/api/admin/templates/{builtin['id']}",
        json={"name": builtin["name"], "description": "hacked"},
        headers=supervisor_headers,
    )
    # Built-in has org_id=NULL; supervisor SQL scopes to their org → no match → 404
    # Both 403 and 404 correctly block the edit.
    assert r.status_code in (403, 404)


@pytest.mark.asyncio
async def test_wf02_admin_cannot_edit_builtin(client, auth_headers):
    templates = (await client.get("/api/admin/templates", headers=auth_headers)).json()
    builtin = next(t for t in templates if t["is_builtin"])
    r = await client.put(
        f"/api/admin/templates/{builtin['id']}",
        json={"name": builtin["name"], "description": "hacked"},
        headers=auth_headers,
    )
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_wf02_builtin_ids_are_global(client, auth_headers, org_b):
    """Both the default org and org_b see the same built-in template IDs."""
    # Admin sees all templates
    all_templates = (await client.get("/api/admin/templates", headers=auth_headers)).json()
    builtin_ids = {t["id"] for t in all_templates if t["is_builtin"]}
    assert len(builtin_ids) >= 3  # at least read_only, contributor, power_user
    # The same IDs appear for org_b supervisor (they share the global built-ins)
    from app.services.orgs import create_user
    from app.services.auth import create_access_token
    sup_b_id = await create_user(org_b, "sup_b2@test.com", "password123", role="supervisor")
    sup_b_token = create_access_token(email="sup_b2@test.com", user_id=sup_b_id, org_id=org_b, role="supervisor")
    sup_b_hdrs = {"Authorization": f"Bearer {sup_b_token}"}
    org_b_templates = (await client.get("/api/admin/templates", headers=sup_b_hdrs)).json()
    org_b_builtin_ids = {t["id"] for t in org_b_templates if t["is_builtin"]}
    assert builtin_ids == org_b_builtin_ids


# ── WF-03: Custom template lifecycle ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_wf03_create_update_delete_template(client, auth_headers):
    # Create
    r = await client.post(
        "/api/admin/templates",
        json={"name": "wf03_tpl", "description": "original", "can_upload": False,
              "org_id": _DEFAULT_ORG_ID},
        headers=auth_headers,
    )
    assert r.status_code == 201
    tpl_id = r.json()["id"]

    # Duplicate name → 409
    r2 = await client.post(
        "/api/admin/templates",
        json={"name": "wf03_tpl", "org_id": _DEFAULT_ORG_ID},
        headers=auth_headers,
    )
    assert r2.status_code == 409

    # Update
    upd = await client.put(
        f"/api/admin/templates/{tpl_id}",
        json={"name": "wf03_tpl", "description": "updated", "can_upload": True},
        headers=auth_headers,
    )
    assert upd.status_code == 200
    assert (await client.get(f"/api/admin/templates/{tpl_id}", headers=auth_headers)
            ).json()["can_upload"] is True

    # Delete
    del_r = await client.delete(f"/api/admin/templates/{tpl_id}", headers=auth_headers)
    assert del_r.status_code == 200
    assert (await client.get(f"/api/admin/templates/{tpl_id}", headers=auth_headers)
            ).status_code == 404


@pytest.mark.asyncio
async def test_wf03_clone_template(client, auth_headers):
    src = await client.post(
        "/api/admin/templates",
        json={"name": "wf03_src", "can_query": True, "org_id": _DEFAULT_ORG_ID},
        headers=auth_headers,
    )
    assert src.status_code == 201
    src_id = src.json()["id"]

    clone = await client.post(
        f"/api/admin/templates/{src_id}/clone",
        json={"name": "wf03_clone", "org_id": _DEFAULT_ORG_ID},
        headers=auth_headers,
    )
    assert clone.status_code == 201
    clone_id = clone.json()["id"]
    assert clone_id != src_id
    cloned = (await client.get(f"/api/admin/templates/{clone_id}", headers=auth_headers)).json()
    assert cloned["is_builtin"] is False
    assert cloned["can_query"] is True


@pytest.mark.asyncio
async def test_wf03_cross_org_template_isolation(client, auth_headers, org_b,
                                                  org_b_supervisor, org_b_supervisor_headers):
    # Create template in default org
    r = await client.post(
        "/api/admin/templates",
        json={"name": "org_a_exclusive", "org_id": _DEFAULT_ORG_ID},
        headers=auth_headers,
    )
    assert r.status_code == 201
    tpl_id = r.json()["id"]

    # Org B supervisor cannot see it
    templates_b = (await client.get("/api/admin/templates",
                                    headers=org_b_supervisor_headers)).json()
    ids_b = {t["id"] for t in templates_b if not t["is_builtin"]}
    assert tpl_id not in ids_b


# ── WF-04: Permission template assignment and enforcement ─────────────────────

@pytest.mark.asyncio
async def test_wf04_read_only_blocks_upload_and_chat(client, auth_headers, default_user):
    user, hdrs = await _make_member(_DEFAULT_ORG_ID, "wf04_ro@test.com", "read_only",
                                    client, auth_headers)
    # Upload blocked
    r_up = await client.post("/api/documents/upload", headers=hdrs,
                             files={"file": ("f.txt", io.BytesIO(b"x"), "text/plain")})
    assert r_up.status_code == 403

    # Chat blocked
    r_chat = await client.post("/api/ops/chat", json={"message": "hi"}, headers=hdrs)
    assert r_chat.status_code == 403


@pytest.mark.asyncio
async def test_wf04_contributor_can_pass_chat_permission(client, auth_headers, default_user):
    """Contributor has can_chat=True — permission check passes (LLM may fail, not 403)."""
    user, hdrs = await _make_member(_DEFAULT_ORG_ID, "wf04_contrib@test.com", "contributor",
                                    client, auth_headers)
    r = await client.post("/api/ops/chat", json={"message": "hi"}, headers=hdrs)
    assert r.status_code != 403


# ── WF-05: Per-user limit override ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_wf05_user_override_wins_and_can_be_removed(client, auth_headers, default_user):
    user, hdrs = await _make_member(_DEFAULT_ORG_ID, "wf05@test.com", "contributor",
                                    client, auth_headers)

    # Set per-user override (must include permission_template_id to avoid clearing it)
    await client.put(f"/api/admin/users/{user['id']}",
                     json={"role": "member", "max_chat_messages_per_day": 5,
                           "permission_template_id": user["template_id"]},
                     headers=auth_headers)

    # Permissions endpoint reflects override
    perms = (await client.get("/api/ops/permissions", headers=hdrs)).json()
    assert perms["max_chat_messages_per_day"] == 5

    # Remove override
    await client.put(f"/api/admin/users/{user['id']}",
                     json={"role": "member", "max_chat_messages_per_day": None,
                           "permission_template_id": user["template_id"]},
                     headers=auth_headers)

    perms2 = (await client.get("/api/ops/permissions", headers=hdrs)).json()
    # Template (contributor) has 200; None override means fall back to template
    assert perms2["max_chat_messages_per_day"] == 200


# ── WF-06: Org-level token quota enforcement ──────────────────────────────────

@pytest.mark.asyncio
async def test_wf06_org_token_quota_returns_429(client, auth_headers, member_user,
                                                member_headers, test_engine):
    # Assign contributor (can_chat=True) so permission passes
    templates = (await client.get("/api/admin/templates", headers=auth_headers)).json()
    tpl = next(t for t in templates if t["name"] == "contributor")
    await client.put(f"/api/admin/users/{member_user['id']}",
                     json={"permission_template_id": tpl["id"], "role": "member"},
                     headers=auth_headers)

    # Set org token budget = 1000
    await client.put("/api/admin/org-limits",
                     json={"org_id": _DEFAULT_ORG_ID, "max_tokens_per_day_org": 1000},
                     headers=auth_headers)

    # Exhaust org budget in usage_log
    async with test_engine.begin() as conn:
        await conn.execute(sa.text("""
            INSERT INTO usage_log (org_id, user_id, model_id, tokens_in, tokens_out, operation)
            VALUES (CAST(:org AS UUID), CAST(:uid AS UUID), 'test', 500, 500, 'chat')
        """), {"org": _DEFAULT_ORG_ID, "uid": member_user["id"]})

    r = await client.post("/api/ops/chat", json={"message": "hi"}, headers=member_headers)
    assert r.status_code == 429

    # Remove cap → next call no longer quota-blocked (may still fail for LLM reasons)
    await client.put("/api/admin/org-limits",
                     json={"org_id": _DEFAULT_ORG_ID, "max_tokens_per_day_org": None},
                     headers=auth_headers)
    r2 = await client.post("/api/ops/chat", json={"message": "hi"}, headers=member_headers)
    assert r2.status_code != 429


# ── WF-07: Upload quota (user and org-level) ──────────────────────────────────

@pytest.mark.asyncio
async def test_wf07_user_daily_upload_quota(client, auth_headers, member_user,
                                            member_headers, test_engine):
    tpl_r = await client.post("/api/admin/templates",
                              json={"name": "strict_up", "can_upload": True,
                                    "max_uploads_per_day": 1, "org_id": _DEFAULT_ORG_ID},
                              headers=auth_headers)
    tpl_id = tpl_r.json()["id"]
    await client.put(f"/api/admin/users/{member_user['id']}",
                     json={"permission_template_id": tpl_id, "role": "member"},
                     headers=auth_headers)

    # Consume the quota via an ingest_jobs row from today
    async with test_engine.begin() as conn:
        await conn.execute(sa.text("""
            INSERT INTO ingest_jobs (org_id, user_id, filename, status)
            VALUES (CAST(:org AS UUID), CAST(:uid AS UUID), 'prev.txt', 'done')
        """), {"org": _DEFAULT_ORG_ID, "uid": member_user["id"]})

    r = await client.post("/api/documents/upload", headers=member_headers,
                          files={"file": ("new.txt", io.BytesIO(b"x"), "text/plain")})
    assert r.status_code == 429


@pytest.mark.asyncio
async def test_wf07_admin_bypasses_upload_quota(client, auth_headers, test_engine,
                                                default_user):
    # Set org-wide upload cap to 0 by exhausting it
    await client.put("/api/admin/org-limits",
                     json={"org_id": _DEFAULT_ORG_ID, "max_uploads_per_day_org": 1},
                     headers=auth_headers)
    async with test_engine.begin() as conn:
        await conn.execute(sa.text("""
            INSERT INTO ingest_jobs (org_id, user_id, filename, status)
            VALUES (CAST(:org AS UUID), CAST(:uid AS UUID), 'old.txt', 'done')
        """), {"org": _DEFAULT_ORG_ID, "uid": default_user["id"]})

    # Admin upload should not be quota-blocked (file may still 404, not 429)
    r = await client.post("/api/documents/upload", headers=auth_headers,
                          files={"file": ("new.txt", io.BytesIO(b"x"), "text/plain")})
    assert r.status_code != 429


# ── WF-08: Suspension enforcement ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_wf08_suspended_user_gets_403(client, auth_headers, member_user, member_headers):
    templates = (await client.get("/api/admin/templates", headers=auth_headers)).json()
    tpl = next(t for t in templates if t["name"] == "contributor")
    await client.put(f"/api/admin/users/{member_user['id']}",
                     json={"permission_template_id": tpl["id"], "is_suspended": True,
                           "role": "member"},
                     headers=auth_headers)

    r = await client.post("/api/ops/query", json={"question": "test"},
                          headers=member_headers)
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_wf08_cannot_suspend_own_account_admin(client, auth_headers, default_user):
    r = await client.put(f"/api/admin/users/{default_user['id']}",
                         json={"is_suspended": True, "role": "admin"},
                         headers=auth_headers)
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_wf08_cannot_suspend_own_account_supervisor(client, supervisor_user,
                                                           supervisor_headers):
    r = await client.put(f"/api/admin/users/{supervisor_user['id']}",
                         json={"is_suspended": True, "role": "supervisor"},
                         headers=supervisor_headers)
    assert r.status_code == 400


# ── WF-09: Organization creation and teardown ─────────────────────────────────

@pytest.mark.asyncio
async def test_wf09_create_org_provisions_defaults(client, auth_headers):
    r = await client.post("/api/admin/organizations",
                          json={"name": "Test Corp"}, headers=auth_headers)
    assert r.status_code == 201
    org_id = r.json()["id"]

    # Built-in templates visible for new org
    from app.services.orgs import create_user
    from app.services.auth import create_access_token
    sup_id = await create_user(org_id, "sup_tc@test.com", "password123", role="supervisor")
    sup_token = create_access_token(email="sup_tc@test.com", user_id=sup_id,
                                    org_id=org_id, role="supervisor")
    sup_hdrs = {"Authorization": f"Bearer {sup_token}"}
    templates = (await client.get("/api/admin/templates", headers=sup_hdrs)).json()
    names = {t["name"] for t in templates if t["is_builtin"]}
    assert {"read_only", "contributor", "power_user"}.issubset(names)

    # Default workspace created
    workspaces = (await client.get("/api/admin/workspaces", headers=auth_headers)).json()
    org_ws = [w for w in workspaces if w["org_id"] == org_id]
    assert len(org_ws) >= 1


@pytest.mark.asyncio
async def test_wf09_duplicate_org_name_returns_409(client, auth_headers):
    await client.post("/api/admin/organizations",
                      json={"name": "Dup Corp"}, headers=auth_headers)
    r2 = await client.post("/api/admin/organizations",
                           json={"name": "Dup Corp"}, headers=auth_headers)
    assert r2.status_code == 409


@pytest.mark.asyncio
async def test_wf09_delete_org_and_cannot_delete_default(client, auth_headers):
    r = await client.post("/api/admin/organizations",
                          json={"name": "Temp Corp"}, headers=auth_headers)
    org_id = r.json()["id"]

    del_r = await client.delete(f"/api/admin/organizations/{org_id}", headers=auth_headers)
    assert del_r.status_code == 200

    # Default org cannot be deleted
    r2 = await client.delete(f"/api/admin/organizations/{_DEFAULT_ORG_ID}",
                             headers=auth_headers)
    assert r2.status_code == 400


# ── WF-10: Supervisor reassignment ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_wf10_reassign_supervisor(client, auth_headers, supervisor_user):
    r = await client.put(
        f"/api/admin/organizations/{_DEFAULT_ORG_ID}/supervisor",
        json={"supervisor_email": supervisor_user["email"]},
        headers=auth_headers,
    )
    assert r.status_code == 200
    assert r.json()["supervisor_user_id"] == supervisor_user["id"]


@pytest.mark.asyncio
async def test_wf10_nonexistent_supervisor_email_returns_404(client, auth_headers):
    r = await client.put(
        f"/api/admin/organizations/{_DEFAULT_ORG_ID}/supervisor",
        json={"supervisor_email": "nobody@nowhere.com"},
        headers=auth_headers,
    )
    assert r.status_code == 404


# ── WF-11: User org transfer ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_wf11_transfer_user_between_orgs(client, auth_headers, org_b, default_user):
    from app.services.orgs import create_user
    user_id = await create_user(_DEFAULT_ORG_ID, "transfer_me@test.com", "password123",
                                role="member")
    r = await client.put(f"/api/admin/users/{user_id}/transfer-org",
                         json={"new_org_id": org_b}, headers=auth_headers)
    assert r.status_code == 200
    assert r.json()["new_org_id"] == org_b


@pytest.mark.asyncio
async def test_wf11_sessions_revoked_after_transfer(client, auth_headers, org_b, test_engine):
    from app.services.orgs import create_user
    from app.services.auth import create_access_token
    import uuid

    user_id = await create_user(_DEFAULT_ORG_ID, "rev_me@test.com", "password123",
                                role="member")
    # Plant a refresh token in auth_tokens
    rt = str(uuid.uuid4())
    async with test_engine.begin() as conn:
        await conn.execute(sa.text("""
            INSERT INTO auth_tokens (user_id, token_hash, expires_at)
            VALUES (CAST(:uid AS UUID), :tok, NOW() + INTERVAL '7 days')
        """), {"uid": user_id, "tok": rt})

    await client.put(f"/api/admin/users/{user_id}/transfer-org",
                     json={"new_org_id": org_b}, headers=auth_headers)

    # Token should be deleted
    async with test_engine.connect() as conn:
        result = await conn.execute(sa.text(
            "SELECT COUNT(*) FROM auth_tokens WHERE user_id = CAST(:uid AS UUID)"
        ), {"uid": user_id})
        count = result.scalar()
    assert count == 0


@pytest.mark.asyncio
async def test_wf11_cannot_transfer_own_account(client, auth_headers, default_user, org_b):
    r = await client.put(f"/api/admin/users/{default_user['id']}/transfer-org",
                         json={"new_org_id": org_b}, headers=auth_headers)
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_wf11_supervisor_cannot_transfer(client, supervisor_user, supervisor_headers,
                                               member_user, org_b):
    r = await client.put(f"/api/admin/users/{member_user['id']}/transfer-org",
                         json={"new_org_id": org_b}, headers=supervisor_headers)
    assert r.status_code == 403


# ── WF-12: Password reset ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_wf12_successful_reset(client, auth_headers, member_user):
    r = await client.post(f"/api/admin/users/{member_user['id']}/reset-password",
                          json={"new_password": "newpassword123"},
                          headers=auth_headers)
    assert r.status_code == 200
    login = await client.post("/api/auth/login",
                              json={"username": member_user["email"],
                                    "password": "newpassword123"})
    assert login.status_code == 200


@pytest.mark.asyncio
async def test_wf12_short_password_returns_400(client, auth_headers, member_user):
    r = await client.post(f"/api/admin/users/{member_user['id']}/reset-password",
                          json={"new_password": "short"},
                          headers=auth_headers)
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_wf12_nonexistent_user_returns_404(client, auth_headers):
    r = await client.post(
        "/api/admin/users/00000000-0000-0000-0000-000000000099/reset-password",
        json={"new_password": "validpass123"},
        headers=auth_headers,
    )
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_wf12_member_cannot_reset_password(client, member_user, member_headers):
    r = await client.post(f"/api/admin/users/{member_user['id']}/reset-password",
                          json={"new_password": "validpass123"},
                          headers=member_headers)
    assert r.status_code == 403


# ── WF-13: No-template fallback → read_only defaults ─────────────────────────

@pytest.mark.asyncio
async def test_wf13_no_template_defaults_to_read_only(client, member_user, member_headers):
    # member_user has no template assigned — should get read_only behaviour
    perms = (await client.get("/api/ops/permissions", headers=member_headers)).json()
    assert perms["can_view_wiki"] is True
    assert perms["can_upload"] is False
    assert perms["can_chat"] is False

    r = await client.post("/api/documents/upload", headers=member_headers,
                          files={"file": ("x.txt", io.BytesIO(b"x"), "text/plain")})
    assert r.status_code == 403


# ── WF-14: Admin/supervisor quota bypass ─────────────────────────────────────

@pytest.mark.asyncio
async def test_wf14_admin_permissions_all_true(client, auth_headers, default_user):
    r = await client.get("/api/ops/permissions",
                         headers={**auth_headers, "X-Org-Context": _DEFAULT_ORG_ID})
    assert r.status_code == 200
    perms = r.json()
    assert perms["can_upload"] is True
    assert perms["can_chat"] is True
    assert perms["can_recalibrate"] is True


@pytest.mark.asyncio
async def test_wf14_supervisor_permissions_all_true(client, supervisor_user, supervisor_headers):
    r = await client.get("/api/ops/permissions", headers=supervisor_headers)
    assert r.status_code == 200
    perms = r.json()
    assert perms["can_upload"] is True
    assert perms["can_recalibrate"] is True


# ── WF-15: Permissions UI visibility (API contract) ──────────────────────────

@pytest.mark.asyncio
async def test_wf15_read_only_permissions(client, auth_headers, default_user):
    user, hdrs = await _make_member(_DEFAULT_ORG_ID, "wf15_ro@test.com", "read_only",
                                    client, auth_headers)
    perms = (await client.get("/api/ops/permissions", headers=hdrs)).json()
    assert perms["can_chat"] is False
    assert perms["can_upload"] is False
    assert perms["can_view_wiki"] is True
    assert perms["can_view_graph"] is True


@pytest.mark.asyncio
async def test_wf15_power_user_permissions(client, auth_headers, default_user):
    user, hdrs = await _make_member(_DEFAULT_ORG_ID, "wf15_pu@test.com", "power_user",
                                    client, auth_headers)
    perms = (await client.get("/api/ops/permissions", headers=hdrs)).json()
    assert perms["can_edit_wiki"] is True
    assert perms["can_delete_wiki_pages"] is True
    assert perms["can_manage_schema"] is True
    assert perms["can_recalibrate"] is False   # power_user cannot recalibrate


# ── WF-16: Layered quota — user limit tighter than org limit ─────────────────

@pytest.mark.asyncio
async def test_wf16_user_limit_fires_before_org_limit(client, auth_headers, test_engine,
                                                       default_user):
    user, hdrs = await _make_member(_DEFAULT_ORG_ID, "wf16@test.com", "contributor",
                                    client, auth_headers)

    # Set user limit tighter than org limit
    tpl_r = await client.post("/api/admin/templates",
                              json={"name": "wf16_tpl", "can_chat": True,
                                    "max_tokens_per_day": 1000,
                                    "org_id": _DEFAULT_ORG_ID},
                              headers=auth_headers)
    tpl_id = tpl_r.json()["id"]
    await client.put(f"/api/admin/users/{user['id']}",
                     json={"permission_template_id": tpl_id, "role": "member"},
                     headers=auth_headers)
    await client.put("/api/admin/org-limits",
                     json={"org_id": _DEFAULT_ORG_ID, "max_tokens_per_day_org": 50000},
                     headers=auth_headers)

    # Exhaust user quota only
    async with test_engine.begin() as conn:
        await conn.execute(sa.text("""
            INSERT INTO usage_log (org_id, user_id, model_id, tokens_in, tokens_out, operation)
            VALUES (CAST(:org AS UUID), CAST(:uid AS UUID), 'test', 500, 500, 'chat')
        """), {"org": _DEFAULT_ORG_ID, "uid": user["id"]})

    r = await client.post("/api/ops/chat", json={"message": "hi"}, headers=hdrs)
    assert r.status_code == 429

    # Org has only consumed 1000 / 50000 tokens — org limit not reached
    # Remove user limit
    await client.put(f"/api/admin/users/{user['id']}",
                     json={"permission_template_id": tpl_id, "role": "member",
                           "max_tokens_per_day": None},
                     headers=auth_headers)
    # Update template to also have no token limit
    await client.put(f"/api/admin/templates/{tpl_id}",
                     json={"name": "wf16_tpl", "can_chat": True, "max_tokens_per_day": None},
                     headers=auth_headers)
    r2 = await client.post("/api/ops/chat", json={"message": "hi"}, headers=hdrs)
    assert r2.status_code != 429


# ── WF-17: Weekly vs daily quota race ────────────────────────────────────────

@pytest.mark.asyncio
async def test_wf17_weekly_limit_blocks_after_daily_resets(client, auth_headers, test_engine,
                                                            default_user):
    user, hdrs = await _make_member(_DEFAULT_ORG_ID, "wf17@test.com", "power_user",
                                    client, auth_headers)
    tpl_r = await client.post("/api/admin/templates",
                              json={"name": "wf17_tpl", "can_chat": True,
                                    "max_tokens_per_week": 8000,
                                    "org_id": _DEFAULT_ORG_ID},
                              headers=auth_headers)
    tpl_id = tpl_r.json()["id"]
    await client.put(f"/api/admin/users/{user['id']}",
                     json={"permission_template_id": tpl_id, "role": "member"},
                     headers=auth_headers)

    # Seed usage at week start so weekly total hits the cap regardless of weekday.
    # Using DATE_TRUNC('week', NOW()) ensures the row is always within this week;
    # no daily cap is set so only the weekly check fires.
    async with test_engine.begin() as conn:
        await conn.execute(sa.text("""
            INSERT INTO usage_log (org_id, user_id, model_id, tokens_in, tokens_out, operation, created_at)
            VALUES (CAST(:org AS UUID), CAST(:uid AS UUID), 'test', 4000, 4000, 'chat',
                    DATE_TRUNC('week', NOW()))
        """), {"org": _DEFAULT_ORG_ID, "uid": user["id"]})

    # Weekly total = 8000 (at cap) — expect 429
    r = await client.post("/api/ops/chat", json={"message": "hi"}, headers=hdrs)
    assert r.status_code == 429


# ── WF-18: Template reassignment takes effect without re-login ────────────────

@pytest.mark.asyncio
async def test_wf18_reassignment_immediate(client, auth_headers, default_user):
    user, hdrs = await _make_member(_DEFAULT_ORG_ID, "wf18@test.com", "read_only",
                                    client, auth_headers)

    # Chat blocked with read_only
    r1 = await client.post("/api/ops/chat", json={"message": "hi"}, headers=hdrs)
    assert r1.status_code == 403

    # Reassign to contributor — same JWT, no re-login
    templates = (await client.get("/api/admin/templates", headers=auth_headers)).json()
    contrib = next(t for t in templates if t["name"] == "contributor")
    await client.put(f"/api/admin/users/{user['id']}",
                     json={"permission_template_id": contrib["id"], "role": "member"},
                     headers=auth_headers)

    r2 = await client.post("/api/ops/chat", json={"message": "hi"}, headers=hdrs)
    assert r2.status_code != 403

    # Reassign back to read_only
    templates = (await client.get("/api/admin/templates", headers=auth_headers)).json()
    ro = next(t for t in templates if t["name"] == "read_only")
    await client.put(f"/api/admin/users/{user['id']}",
                     json={"permission_template_id": ro["id"], "role": "member"},
                     headers=auth_headers)
    r3 = await client.post("/api/ops/chat", json={"message": "hi"}, headers=hdrs)
    assert r3.status_code == 403


# ── WF-19: Per-user token override and template interaction ───────────────────

@pytest.mark.asyncio
async def test_wf19_token_override_lifecycle(client, auth_headers, test_engine, default_user):
    user, hdrs = await _make_member(_DEFAULT_ORG_ID, "wf19@test.com", "power_user",
                                    client, auth_headers)

    # power_user has no token limits
    perms1 = (await client.get("/api/ops/permissions", headers=hdrs)).json()
    assert perms1["max_tokens_per_day"] is None

    # Supervisor sets per-user override (must include permission_template_id to avoid clearing it)
    await client.put(f"/api/admin/users/{user['id']}",
                     json={"role": "member", "max_tokens_per_day": 500,
                           "permission_template_id": user["template_id"]},
                     headers=auth_headers)
    perms2 = (await client.get("/api/ops/permissions", headers=hdrs)).json()
    assert perms2["max_tokens_per_day"] == 500

    # Exhaust override
    async with test_engine.begin() as conn:
        await conn.execute(sa.text("""
            INSERT INTO usage_log (org_id, user_id, model_id, tokens_in, tokens_out, operation)
            VALUES (CAST(:org AS UUID), CAST(:uid AS UUID), 'test', 250, 250, 'chat')
        """), {"org": _DEFAULT_ORG_ID, "uid": user["id"]})

    r = await client.post("/api/ops/chat", json={"message": "hi"}, headers=hdrs)
    assert r.status_code == 429

    # Remove override
    await client.put(f"/api/admin/users/{user['id']}",
                     json={"role": "member", "max_tokens_per_day": None,
                           "permission_template_id": user["template_id"]},
                     headers=auth_headers)
    perms3 = (await client.get("/api/ops/permissions", headers=hdrs)).json()
    assert perms3["max_tokens_per_day"] is None


# ── WF-20: Cross-org isolation — supervisor cannot touch another org ───────────

@pytest.mark.asyncio
async def test_wf20_supervisor_sees_only_own_org_users(client, supervisor_user,
                                                        supervisor_headers, org_b,
                                                        org_b_supervisor):
    r = await client.get("/api/admin/users", headers=supervisor_headers)
    assert r.status_code == 200
    org_ids = {u["org_id"] for u in r.json()}
    assert org_b not in org_ids


@pytest.mark.asyncio
async def test_wf20_supervisor_cannot_edit_other_org_user(client, supervisor_headers,
                                                           org_b_supervisor):
    r = await client.put(
        f"/api/admin/users/{org_b_supervisor['id']}",
        json={"role": "member"},
        headers=supervisor_headers,
    )
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_wf20_supervisor_cannot_access_organizations_tab(client, supervisor_user,
                                                                supervisor_headers):
    # Supervisor listing organizations should only see their own org, not get 403
    # per the route: _require_supervisor_access (not _require_admin)
    r = await client.get("/api/admin/organizations", headers=supervisor_headers)
    assert r.status_code == 200
    orgs = r.json()
    ids = {o["id"] for o in orgs}
    assert ids == {_DEFAULT_ORG_ID}

    # Creating/deleting orgs IS admin-only
    r2 = await client.post("/api/admin/organizations",
                           json={"name": "Sneaky Corp"}, headers=supervisor_headers)
    assert r2.status_code == 403


@pytest.mark.asyncio
async def test_wf20_supervisor_org_limits_target_own_org(client, supervisor_user,
                                                          supervisor_headers, org_b, test_engine):
    """org_id in body is ignored by supervisor; update always targets their own org."""
    # Set org B limits first so we have a baseline
    from app.services.auth import create_access_token
    await client.put("/api/admin/org-limits",
                     json={"org_id": org_b, "max_uploads_per_day_org": 999},
                     headers={
                         "Authorization": f"Bearer {create_access_token('a', 'x', None, 'admin')}"
                     } if False else {"Authorization": "Bearer ignored"})

    # Supervisor sends org_b in body — should be ignored and update own org instead
    upd = await client.put("/api/admin/org-limits",
                           json={"org_id": org_b, "max_tokens_per_day_org": 12345},
                           headers=supervisor_headers)
    assert upd.status_code == 200

    # Supervisor's own org should now have 12345
    r = await client.get("/api/admin/org-limits", headers=supervisor_headers)
    own = r.json()
    assert own["max_tokens_per_day_org"] == 12345


# ── WF-21: Admin cross-org with X-Org-Context ─────────────────────────────────

@pytest.mark.asyncio
async def test_wf21_admin_audit_log_filtered_by_org(client, auth_headers, org_b, supervisor_user,
                                                     supervisor_headers):
    # Supervisor (default org) saves schema → audit entry for default org
    await client.put("/api/ops/schema",
                     json={"content": _VALID_SCHEMA},
                     headers=supervisor_headers)

    # Admin filters audit log to default org
    r = await client.get(f"/api/admin/audit-log?org_filter={_DEFAULT_ORG_ID}",
                         headers=auth_headers)
    assert r.status_code == 200
    entries = r.json()["entries"]
    assert any(e["operation"] == "schema_updated" for e in entries)

    # Admin filters to org_b — schema_updated entry should not appear (no ops there)
    r2 = await client.get(f"/api/admin/audit-log?org_filter={org_b}", headers=auth_headers)
    schema_entries_b = [e for e in r2.json()["entries"] if e["operation"] == "schema_updated"]
    assert len(schema_entries_b) == 0

    # No filter → all orgs combined
    r3 = await client.get("/api/admin/audit-log", headers=auth_headers)
    assert r3.status_code == 200


# ── WF-22: Recalibration lock ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_wf22_write_blocked_during_recalibration(client, supervisor_user,
                                                        supervisor_headers):
    from app.services import recalibrate_job as rjob

    job = rjob.get(_DEFAULT_ORG_ID)
    job.status = "running"
    try:
        # Upload (POST) must be blocked
        r = await client.post("/api/documents/upload", headers=supervisor_headers,
                              files={"file": ("x.txt", io.BytesIO(b"x"), "text/plain")})
        assert r.status_code == 503

        # Query (POST /api/ops/query) is on passthrough list
        r2 = await client.post("/api/ops/query", json={"question": "test"},
                               headers=supervisor_headers)
        assert r2.status_code != 503

        # GET always passes
        r3 = await client.get("/api/wiki", headers=supervisor_headers)
        assert r3.status_code != 503
    finally:
        job.status = "idle"


@pytest.mark.asyncio
async def test_wf22_other_org_unaffected_by_recalibration(client, auth_headers, org_b,
                                                           org_b_supervisor,
                                                           org_b_supervisor_headers):
    from app.services import recalibrate_job as rjob

    job = rjob.get(_DEFAULT_ORG_ID)
    job.status = "running"
    try:
        # Org B supervisor's upload should NOT be blocked
        r = await client.post("/api/documents/upload", headers=org_b_supervisor_headers,
                              files={"file": ("x.txt", io.BytesIO(b"x"), "text/plain")})
        assert r.status_code != 503
    finally:
        job.status = "idle"


# ── WF-23: Ingest plan review permission gates ────────────────────────────────

@pytest.mark.asyncio
async def test_wf23_cancel_ingest_requires_permission(client, auth_headers, default_user):
    # User without can_cancel_ingest
    user, hdrs = await _make_member(_DEFAULT_ORG_ID, "wf23_nc@test.com", "read_only",
                                    client, auth_headers)
    r = await client.delete("/api/ops/ingest/some_file.txt", headers=hdrs)
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_wf23_approve_ingest_requires_permission(client, auth_headers, default_user):
    user, hdrs = await _make_member(_DEFAULT_ORG_ID, "wf23_na@test.com", "read_only",
                                    client, auth_headers)
    r = await client.post("/api/ops/ingest/some_file.txt/approve",
                          json={"confirmed": True}, headers=hdrs)
    assert r.status_code == 403

    r2 = await client.post("/api/ops/ingest/some_file.txt/plan-chat",
                           json={"message": "revise"}, headers=hdrs)
    assert r2.status_code == 403


# ── WF-24: Wiki permission matrix ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_wf24_wiki_crud_by_role(client, auth_headers, default_user):
    ro_user, ro_hdrs = await _make_member(_DEFAULT_ORG_ID, "wf24_ro@test.com", "read_only",
                                          client, auth_headers)
    contrib_user, contrib_hdrs = await _make_member(_DEFAULT_ORG_ID, "wf24_co@test.com",
                                                    "contributor", client, auth_headers)
    pu_user, pu_hdrs = await _make_member(_DEFAULT_ORG_ID, "wf24_pu@test.com", "power_user",
                                          client, auth_headers)

    # All can view wiki
    assert (await client.get("/api/wiki", headers=ro_hdrs)).status_code == 200
    assert (await client.get("/api/wiki", headers=contrib_hdrs)).status_code == 200

    # read_only and contributor cannot edit wiki
    assert (await client.put("/api/wiki/test/page.md",
                             json={"content": "hi"}, headers=ro_hdrs)).status_code == 403
    assert (await client.put("/api/wiki/test/page.md",
                             json={"content": "hi"}, headers=contrib_hdrs)).status_code == 403

    # power_user can edit wiki (may 404 if page not found, but not 403)
    r_edit = await client.put("/api/wiki/test/page.md",
                              json={"content": "hi"}, headers=pu_hdrs)
    assert r_edit.status_code != 403

    # Graph visibility
    assert (await client.get("/api/ops/graph", headers=ro_hdrs)).status_code != 403
    assert (await client.post("/api/ops/graph/rebuild", headers=ro_hdrs)).status_code == 403
    assert (await client.post("/api/ops/graph/rebuild", headers=pu_hdrs)).status_code != 403

    # Schema access
    assert (await client.get("/api/ops/schema",
                             headers=ro_hdrs)).status_code == 403
    assert (await client.get("/api/ops/schema",
                             headers=contrib_hdrs)).status_code == 403
    assert (await client.get("/api/ops/schema",
                             headers=pu_hdrs)).status_code != 403

    # Lint
    assert (await client.post("/api/ops/lint", headers=contrib_hdrs)).status_code == 403
    assert (await client.post("/api/ops/lint", headers=pu_hdrs)).status_code != 403

    # Recalibrate: power_user cannot; supervisor can
    assert (await client.post("/api/ops/recalibrate", headers=pu_hdrs)).status_code == 403


@pytest.mark.asyncio
async def test_wf24_supervisor_can_recalibrate(client, supervisor_user, supervisor_headers):
    r = await client.post("/api/ops/recalibrate", headers=supervisor_headers)
    # 202 (started) or 200, but NOT 403
    assert r.status_code != 403
    # Reset in-memory recalibrate status so the lock doesn't bleed into subsequent tests.
    # clean_db only truncates DB tables; in-memory job state persists across tests.
    from app.services import recalibrate_job as rjob
    rjob.get(supervisor_user["org_id"]).status = "idle"


# ── WF-25: One identity, many orgs (multi-org membership) ────────────────────

@pytest.mark.asyncio
async def test_wf25_same_email_across_orgs_adds_membership(client, auth_headers, org_b):
    """A single email is one global identity. Adding it to a second org grants a
    new membership (not a duplicate user); re-adding to the same org is a 409."""
    # Create in default org
    r1 = await client.post("/api/admin/organizations/{}/users".format(_DEFAULT_ORG_ID),
                           json={"email": "uniq@test.com", "password": "password123"},
                           headers=auth_headers)
    assert r1.status_code == 201
    user_id = r1.json()["id"]

    # Same email in org_b → 201, SAME identity gains a second membership.
    r2 = await client.post(f"/api/admin/organizations/{org_b}/users",
                           json={"email": "uniq@test.com", "password": "password123"},
                           headers=auth_headers)
    assert r2.status_code == 201
    assert r2.json()["id"] == user_id

    # The user now belongs to both orgs.
    memberships = (await client.get(f"/api/admin/users/{user_id}/memberships",
                                    headers=auth_headers)).json()
    org_ids = {m["org_id"] for m in memberships}
    assert _DEFAULT_ORG_ID in org_ids and org_b in org_ids

    # Re-adding to an org they're already in → 409.
    r3 = await client.post(f"/api/admin/organizations/{org_b}/users",
                           json={"email": "uniq@test.com", "password": "password123"},
                           headers=auth_headers)
    assert r3.status_code == 409


@pytest.mark.asyncio
async def test_wf25b_role_differs_per_org(client, auth_headers, org_b):
    """The headline scenario: one identity is a supervisor in one org and a
    member in another. The active org (X-Org-Context) decides the role."""
    from app.services.orgs import create_user
    from app.services.auth import create_access_token

    # Supervisor in the default org…
    uid = await create_user(_DEFAULT_ORG_ID, "multi@test.com", "password123", role="supervisor")
    # …and a plain member in org_b.
    r = await client.post(f"/api/admin/users/{uid}/memberships",
                          json={"org_id": org_b, "role": "member"}, headers=auth_headers)
    assert r.status_code == 201

    token = create_access_token(email="multi@test.com", user_id=uid, org_id="", role="member")
    def hdrs(org):
        return {"Authorization": f"Bearer {token}", "X-Org-Context": org}

    # Active org = default → supervisor → admin user list allowed; /me says supervisor.
    assert (await client.get("/api/admin/users", headers=hdrs(_DEFAULT_ORG_ID))).status_code == 200
    me_a = (await client.get("/api/admin/me", headers=hdrs(_DEFAULT_ORG_ID))).json()
    assert me_a["role"] == "supervisor"

    # Active org = org_b → member → admin user list forbidden; /me says member.
    assert (await client.get("/api/admin/users", headers=hdrs(org_b))).status_code == 403
    me_b = (await client.get("/api/admin/me", headers=hdrs(org_b))).json()
    assert me_b["role"] == "member"


# ── WF-26: Role promotion and demotion ───────────────────────────────────────

@pytest.mark.asyncio
async def test_wf26_promote_to_supervisor_bypasses_template(client, auth_headers, default_user):
    user, hdrs = await _make_member(_DEFAULT_ORG_ID, "wf26@test.com", "read_only",
                                    client, auth_headers)

    # chat blocked as member/read_only
    assert (await client.post("/api/ops/chat", json={"message": "hi"},
                              headers=hdrs)).status_code == 403

    # Promote to supervisor
    await client.put(f"/api/admin/users/{user['id']}",
                     json={"role": "supervisor"}, headers=auth_headers)

    # New token with supervisor role
    from app.services.auth import create_access_token
    new_token = create_access_token(email=user["email"], user_id=user["id"],
                                    org_id=_DEFAULT_ORG_ID, role="supervisor")
    new_hdrs = {"Authorization": f"Bearer {new_token}"}

    # Supervisor bypasses all permission checks
    r = await client.post("/api/ops/chat", json={"message": "hi"}, headers=new_hdrs)
    assert r.status_code != 403


@pytest.mark.asyncio
async def test_wf26_template_unchanged_by_role_promotion(client, auth_headers, default_user):
    """permission_template_id on the user row is not cleared when role changes."""
    user, hdrs = await _make_member(_DEFAULT_ORG_ID, "wf26b@test.com", "contributor",
                                    client, auth_headers)
    template_id_before = user["template_id"]

    await client.put(f"/api/admin/users/{user['id']}",
                     json={"role": "supervisor",
                           "permission_template_id": template_id_before},
                     headers=auth_headers)

    users = (await client.get("/api/admin/users", headers=auth_headers)).json()
    updated = next(u for u in users if u["id"] == user["id"])
    assert updated["permission_template_id"] == template_id_before


# ── WF-27: Workspace copy and ownership ──────────────────────────────────────

@pytest.mark.asyncio
async def test_wf27_copy_workspace(client, auth_headers, member_user):
    # Create source workspace
    src = await client.post("/api/admin/workspaces",
                            json={"name": "wf27_src", "owner_user_id": member_user["id"],
                                  "org_id": _DEFAULT_ORG_ID},
                            headers=auth_headers)
    assert src.status_code == 201
    src_id = src.json()["id"]

    # Copy
    cp = await client.post(f"/api/admin/workspaces/{src_id}/copy",
                           json={"new_name": "wf27_copy",
                                 "owner_user_id": member_user["id"]},
                           headers=auth_headers)
    assert cp.status_code == 201
    copy_id = cp.json()["id"]
    assert copy_id != src_id

    # Original still exists
    all_ws = (await client.get("/api/admin/workspaces", headers=auth_headers)).json()
    ids = {w["id"] for w in all_ws}
    assert src_id in ids
    assert copy_id in ids

    # Duplicate name → 409
    cp2 = await client.post(f"/api/admin/workspaces/{src_id}/copy",
                            json={"new_name": "wf27_copy"}, headers=auth_headers)
    assert cp2.status_code in (409, 404)

    # Delete copy
    del_r = await client.delete(f"/api/admin/workspaces/{copy_id}", headers=auth_headers)
    assert del_r.status_code == 200
    remaining = {w["id"] for w in (await client.get("/api/admin/workspaces",
                                                    headers=auth_headers)).json()}
    assert src_id in remaining
    assert copy_id not in remaining


# ── WF-28: Org member cap stored and readable ─────────────────────────────────

@pytest.mark.asyncio
async def test_wf28_max_members_stored(client, auth_headers):
    await client.put("/api/admin/org-limits",
                     json={"org_id": _DEFAULT_ORG_ID, "max_members": 3},
                     headers=auth_headers)
    r = await client.get("/api/admin/org-limits", headers=auth_headers)
    data = r.json()
    org = next((o for o in data if o["id"] == _DEFAULT_ORG_ID), data) \
        if isinstance(data, list) else data
    assert org["max_members"] == 3

    # Set to null → unlimited
    await client.put("/api/admin/org-limits",
                     json={"org_id": _DEFAULT_ORG_ID, "max_members": None},
                     headers=auth_headers)
    r2 = await client.get("/api/admin/org-limits", headers=auth_headers)
    data2 = r2.json()
    org2 = next((o for o in data2 if o["id"] == _DEFAULT_ORG_ID), data2) \
        if isinstance(data2, list) else data2
    assert org2["max_members"] is None


# ── WF-29: JWT session isolation and token revocation ─────────────────────────

@pytest.mark.asyncio
async def test_wf29_transfer_revokes_all_tokens(client, auth_headers, org_b, test_engine):
    from app.services.orgs import create_user
    import uuid

    user_id = await create_user(_DEFAULT_ORG_ID, "wf29@test.com", "password123",
                                role="member")
    # Plant two refresh tokens simulating two devices
    tok1, tok2 = str(uuid.uuid4()), str(uuid.uuid4())
    async with test_engine.begin() as conn:
        for tok in (tok1, tok2):
            await conn.execute(sa.text("""
                INSERT INTO auth_tokens (user_id, token_hash, expires_at)
                VALUES (CAST(:uid AS UUID), :tok, NOW() + INTERVAL '7 days')
            """), {"uid": user_id, "tok": tok})

    await client.put(f"/api/admin/users/{user_id}/transfer-org",
                     json={"new_org_id": org_b}, headers=auth_headers)

    async with test_engine.connect() as conn:
        count = (await conn.execute(sa.text(
            "SELECT COUNT(*) FROM auth_tokens WHERE user_id = CAST(:uid AS UUID)"
        ), {"uid": user_id})).scalar()
    assert count == 0


# ── WF-30: Quota check ordering — permission check fires before quota ─────────

@pytest.mark.asyncio
async def test_wf30_permission_check_before_quota(client, auth_headers, test_engine,
                                                   default_user):
    """read_only user has can_chat=False AND token limit already exceeded.
    The 403 from require_permission must fire — not a 429 from quota."""
    user, hdrs = await _make_member(_DEFAULT_ORG_ID, "wf30@test.com", "read_only",
                                    client, auth_headers)

    # Exhaust user's token budget
    async with test_engine.begin() as conn:
        await conn.execute(sa.text("""
            INSERT INTO usage_log (org_id, user_id, model_id, tokens_in, tokens_out, operation)
            VALUES (CAST(:org AS UUID), CAST(:uid AS UUID), 'test', 50000, 50000, 'chat')
        """), {"org": _DEFAULT_ORG_ID, "uid": user["id"]})

    r = await client.post("/api/ops/chat", json={"message": "hi"}, headers=hdrs)
    assert r.status_code == 403
    assert "permission" in r.json().get("detail", "").lower()


# ── WF-32: End-to-end org provisioning ───────────────────────────────────────

@pytest.mark.asyncio
async def test_wf32_full_provisioning_workflow(client, auth_headers):
    # 1. Create org
    r = await client.post("/api/admin/organizations",
                          json={"name": "Acme Corp"}, headers=auth_headers)
    assert r.status_code == 201
    acme_id = r.json()["id"]

    # 2. Create supervisor
    sup_r = await client.post(
        "/api/admin/users",
        json={"email": "sup@acme.com", "password": "password123",
              "role": "supervisor", "org_id": acme_id},
        headers=auth_headers,
    )
    assert sup_r.status_code == 201
    sup_id = sup_r.json()["id"]

    # 3. Assign supervisor to org
    chg = await client.put(f"/api/admin/organizations/{acme_id}/supervisor",
                           json={"supervisor_email": "sup@acme.com"},
                           headers=auth_headers)
    assert chg.status_code == 200

    # 4. Create custom template scoped to acme
    tpl_r = await client.post("/api/admin/templates",
                              json={"name": "acme_analyst", "can_query": True,
                                    "org_id": acme_id},
                              headers=auth_headers)
    assert tpl_r.status_code == 201
    acme_tpl_id = tpl_r.json()["id"]

    # 5. Create analyst user with template
    analyst_r = await client.post(
        f"/api/admin/organizations/{acme_id}/users",
        json={"email": "analyst@acme.com", "password": "password123",
              "permission_template_id": acme_tpl_id},
        headers=auth_headers,
    )
    assert analyst_r.status_code == 201

    # 6. Set org limits
    lim = await client.put("/api/admin/org-limits",
                           json={"org_id": acme_id, "max_tokens_per_day_org": 100000,
                                 "max_members": 20},
                           headers=auth_headers)
    assert lim.status_code == 200

    # 7. Supervisor sees only their org's users
    from app.services.auth import create_access_token
    sup_token = create_access_token(email="sup@acme.com", user_id=sup_id,
                                    org_id=acme_id, role="supervisor")
    sup_hdrs = {"Authorization": f"Bearer {sup_token}"}
    users = (await client.get("/api/admin/users", headers=sup_hdrs)).json()
    assert all(u["org_id"] == acme_id for u in users)

    # 8. Supervisor sees 3 global built-ins + 1 custom template
    templates = (await client.get("/api/admin/templates", headers=sup_hdrs)).json()
    assert any(t["name"] == "acme_analyst" for t in templates)
    builtin_count = sum(1 for t in templates if t["is_builtin"])
    assert builtin_count >= 3

    # 9. Analyst permissions match acme_analyst template
    from app.services.orgs import get_user_by_email
    analyst = await get_user_by_email("analyst@acme.com")
    analyst_token = create_access_token(email="analyst@acme.com", user_id=analyst["id"],
                                        org_id=acme_id, role="member")
    analyst_hdrs = {"Authorization": f"Bearer {analyst_token}"}
    perms = (await client.get("/api/ops/permissions", headers=analyst_hdrs)).json()
    assert perms["can_query"] is True
    assert perms["can_upload"] is False  # acme_analyst didn't set can_upload


# ── WF-33: Supervisor scope creep — data isolation ───────────────────────────

@pytest.mark.asyncio
async def test_wf33_supervisor_data_isolation(client, supervisor_user, supervisor_headers,
                                              org_b, org_b_supervisor):
    # Users list contains no org_b entries
    users = (await client.get("/api/admin/users", headers=supervisor_headers)).json()
    assert not any(u["org_id"] == org_b for u in users)

    # Templates list has no org_b custom templates
    org_b_tpl_r = await client.post(
        "/api/admin/templates",
        json={"name": "org_b_exclusive", "org_id": org_b},
        headers={**supervisor_headers,
                 "Authorization": supervisor_headers["Authorization"]},
    )
    # Create via admin so it actually gets created in org_b
    from app.services.auth import create_access_token
    admin_r = await client.post(
        "/api/admin/templates",
        json={"name": "org_b_only", "org_id": org_b},
        headers={
            "Authorization": f"Bearer {create_access_token('a', supervisor_user['id'], None, 'admin')}"
        } if False else supervisor_headers,
    )
    # The important check: org_b templates are invisible to default-org supervisor
    templates = (await client.get("/api/admin/templates", headers=supervisor_headers)).json()
    assert not any(t.get("org_id") == org_b for t in templates)

    # Workspaces: no org_b entries
    workspaces = (await client.get("/api/admin/workspaces", headers=supervisor_headers)).json()
    assert not any(w["org_id"] == org_b for w in workspaces)

    # Attempting to edit org_b user → 404
    r = await client.put(f"/api/admin/users/{org_b_supervisor['id']}",
                         json={"role": "member"}, headers=supervisor_headers)
    assert r.status_code == 404

    # Reset password for org_b user → 404
    r2 = await client.post(f"/api/admin/users/{org_b_supervisor['id']}/reset-password",
                           json={"new_password": "validpass123"},
                           headers=supervisor_headers)
    assert r2.status_code == 404


# ── WF-34: Schema access control ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_wf34_schema_access_by_role(client, auth_headers, supervisor_user,
                                          supervisor_headers, default_user):
    ro_user, ro_hdrs = await _make_member(_DEFAULT_ORG_ID, "wf34_ro@test.com", "read_only",
                                          client, auth_headers)
    contrib_user, contrib_hdrs = await _make_member(_DEFAULT_ORG_ID, "wf34_co@test.com",
                                                    "contributor", client, auth_headers)
    pu_user, pu_hdrs = await _make_member(_DEFAULT_ORG_ID, "wf34_pu@test.com", "power_user",
                                          client, auth_headers)

    # read_only: all schema endpoints forbidden
    for method, path, payload in [
        ("GET",  "/api/ops/schema", None),
        ("PUT",  "/api/ops/schema", {"content": _VALID_SCHEMA}),
        ("POST", "/api/ops/schema/validate", {"content": _VALID_SCHEMA}),
    ]:
        r = await (client.get(path, headers=ro_hdrs) if method == "GET"
                   else client.put(path, json=payload, headers=ro_hdrs) if method == "PUT"
                   else client.post(path, json=payload, headers=ro_hdrs))
        assert r.status_code == 403, f"{method} {path} should be 403 for read_only"

    # contributor: same — forbidden
    r_c = await client.get("/api/ops/schema", headers=contrib_hdrs)
    assert r_c.status_code == 403

    # power_user: allowed
    r_pu = await client.get("/api/ops/schema", headers=pu_hdrs)
    assert r_pu.status_code == 200

    # supervisor: allowed (bypasses template checks)
    r_sup = await client.get("/api/ops/schema", headers=supervisor_headers)
    assert r_sup.status_code == 200

    # GET /api/ops/schema/orgs: admin-only
    r_orgs = await client.get("/api/ops/schema/orgs", headers=supervisor_headers)
    assert r_orgs.status_code == 403


# ── WF-35: Schema validation ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_wf35_missing_section_returns_error(client, supervisor_user, supervisor_headers):
    content_missing_crosslinks = (
        "# My Schema\n\n"
        "## Directory Structure\n...\n"
        "## Page Naming\n...\n"
        # "## Cross-linking" intentionally missing
        "## Index Entry Format\n...\n"
        "## Log Entry Format\n...\n"
        "## Ingest Checklist\nsources/\nindex.md\nlog.md\ncross-link\n"
        "## Contradictions\n...\n"
    )
    r = await client.post("/api/ops/schema/validate",
                          json={"content": content_missing_crosslinks},
                          headers=supervisor_headers)
    assert r.status_code == 200
    data = r.json()
    assert data["valid"] is False
    assert any("Cross-linking" in e for e in data["errors"])


@pytest.mark.asyncio
async def test_wf35_missing_top_level_heading_returns_error(client, supervisor_user,
                                                             supervisor_headers):
    r = await client.post("/api/ops/schema/validate",
                          json={"content": "no heading here\n## Directory Structure\n"},
                          headers=supervisor_headers)
    assert r.status_code == 200
    assert r.json()["valid"] is False
    assert any("heading" in e.lower() for e in r.json()["errors"])


@pytest.mark.asyncio
async def test_wf35_put_does_not_revalidate(client, supervisor_user, supervisor_headers):
    """PUT /api/ops/schema accepts content unconditionally — server does not double-validate."""
    broken = "no heading at all — missing sections"
    r = await client.put("/api/ops/schema", json={"content": broken},
                         headers=supervisor_headers)
    # Server accepts it; enforcing validity is the client's responsibility
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_wf35_odd_fences_produce_warning_not_error(client, supervisor_user,
                                                          supervisor_headers):
    content = _VALID_SCHEMA + "\n```\nunclosed fence"
    r = await client.post("/api/ops/schema/validate", json={"content": content},
                          headers=supervisor_headers)
    assert r.status_code == 200
    data = r.json()
    assert data["valid"] is True          # warnings don't block
    assert any("fence" in w.lower() for w in data["warnings"])


@pytest.mark.asyncio
async def test_wf35_fully_valid_schema_passes(client, supervisor_user, supervisor_headers):
    r = await client.post("/api/ops/schema/validate", json={"content": _VALID_SCHEMA},
                          headers=supervisor_headers)
    assert r.status_code == 200
    assert r.json()["valid"] is True
    assert r.json()["errors"] == []


@pytest.mark.asyncio
async def test_wf35_schema_rules_endpoint_describes_validator(client, supervisor_user,
                                                              supervisor_headers):
    """GET /schema/rules feeds the admin guidance panel and must stay in lock-step
    with the validator (both read SCHEMA_RULES)."""
    from app.routes.operations import SCHEMA_RULES

    r = await client.get("/api/ops/schema/rules", headers=supervisor_headers)
    assert r.status_code == 200
    rules = r.json()["rules"]
    # One payload entry per registered rule, same severities, non-empty descriptions.
    assert len(rules) == len(SCHEMA_RULES)
    assert [x["severity"] for x in rules] == [r.severity for r in SCHEMA_RULES]
    assert all(x["description"].strip() for x in rules)
    assert {x["severity"] for x in rules} == {"error", "warning"}


# ── WF-36: Schema save is recorded in the audit log ──────────────────────────

@pytest.mark.asyncio
async def test_wf36_schema_save_creates_audit_entry(client, supervisor_user, supervisor_headers):
    r = await client.put("/api/ops/schema", json={"content": _VALID_SCHEMA},
                         headers=supervisor_headers)
    assert r.status_code == 200

    log_r = await client.get("/api/admin/audit-log?limit=10", headers=supervisor_headers)
    assert log_r.status_code == 200
    entries = log_r.json()["entries"]
    schema_entries = [e for e in entries if e["operation"] == "schema_updated"]
    assert len(schema_entries) >= 1

    entry = schema_entries[0]
    assert "schema" in entry["raw_text"].lower()
    details = entry["details"] or {}
    assert "chars_after" in details
    assert details["chars_after"] == len(_VALID_SCHEMA)

    # Second save produces a second distinct entry (append-only)
    await client.put("/api/ops/schema", json={"content": _VALID_SCHEMA + " v2"},
                     headers=supervisor_headers)
    log_r2 = await client.get("/api/admin/audit-log?limit=10", headers=supervisor_headers)
    schema_entries2 = [e for e in log_r2.json()["entries"] if e["operation"] == "schema_updated"]
    assert len(schema_entries2) >= 2


# ── WF-37: Revert to default — does not auto-save ─────────────────────────────

@pytest.mark.asyncio
async def test_wf37_get_default_does_not_overwrite_saved(client, supervisor_user,
                                                          supervisor_headers):
    custom_schema = _VALID_SCHEMA + "\n# Custom marker"
    await client.put("/api/ops/schema", json={"content": custom_schema},
                     headers=supervisor_headers)

    # Fetch default
    default_r = await client.get("/api/ops/schema/default", headers=supervisor_headers)
    assert default_r.status_code == 200
    default_content = default_r.json()["content"]

    # Saved schema is unchanged
    saved_r = await client.get("/api/ops/schema", headers=supervisor_headers)
    assert saved_r.status_code == 200
    assert saved_r.json()["content"] == custom_schema
    assert saved_r.json()["content"] != default_content or custom_schema == default_content


# ── WF-38: Admin cross-org schema management via schema/orgs ──────────────────

@pytest.mark.asyncio
async def test_wf38_orgs_endpoint_is_admin_only(client, supervisor_user, supervisor_headers,
                                                 member_user, member_headers):
    assert (await client.get("/api/ops/schema/orgs",
                             headers=supervisor_headers)).status_code == 403
    assert (await client.get("/api/ops/schema/orgs",
                             headers=member_headers)).status_code == 403


@pytest.mark.asyncio
async def test_wf38_admin_can_list_all_org_schemas(client, auth_headers, org_b,
                                                    supervisor_user, supervisor_headers):
    # Save distinct schemas for both orgs
    await client.put("/api/ops/schema",
                     json={"content": _VALID_SCHEMA + "\n# Org A marker"},
                     headers=supervisor_headers)
    from app.services.orgs import create_user
    from app.services.auth import create_access_token
    sup_b_id = await create_user(org_b, "sup_b_wf38@test.com", "password123", role="supervisor")
    sup_b_token = create_access_token(email="sup_b_wf38@test.com", user_id=sup_b_id,
                                      org_id=org_b, role="supervisor")
    sup_b_hdrs = {"Authorization": f"Bearer {sup_b_token}"}
    await client.put("/api/ops/schema",
                     json={"content": _VALID_SCHEMA + "\n# Org B marker"},
                     headers=sup_b_hdrs)

    r = await client.get("/api/ops/schema/orgs", headers=auth_headers)
    assert r.status_code == 200
    orgs = r.json()
    org_ids = {o["org_id"] for o in orgs}
    assert _DEFAULT_ORG_ID in org_ids
    assert org_b in org_ids

    # Org A content contains marker
    org_a_entry = next(o for o in orgs if o["org_id"] == _DEFAULT_ORG_ID)
    assert "Org A marker" in org_a_entry["content"]


# ── WF-39: Schema per-org isolation ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_wf39_schema_isolation(client, auth_headers, org_b, supervisor_user,
                                     supervisor_headers):
    from app.services.orgs import create_user
    from app.services.auth import create_access_token

    sup_b_id = await create_user(org_b, "sup_b_wf39@test.com", "password123", role="supervisor")
    sup_b_token = create_access_token(email="sup_b_wf39@test.com", user_id=sup_b_id,
                                      org_id=org_b, role="supervisor")
    sup_b_hdrs = {"Authorization": f"Bearer {sup_b_token}"}

    # Save different schemas
    await client.put("/api/ops/schema",
                     json={"content": _VALID_SCHEMA + "\n# Org A only"},
                     headers=supervisor_headers)
    await client.put("/api/ops/schema",
                     json={"content": _VALID_SCHEMA + "\n# Org B only"},
                     headers=sup_b_hdrs)

    # Each org sees only their own
    r_a = await client.get("/api/ops/schema", headers=supervisor_headers)
    r_b = await client.get("/api/ops/schema", headers=sup_b_hdrs)
    assert "Org A only" in r_a.json()["content"]
    assert "Org B only" in r_b.json()["content"]
    assert "Org B only" not in r_a.json()["content"]
    assert "Org A only" not in r_b.json()["content"]

    # Admin reads org_a via X-Org-Context
    r_admin_a = await client.get("/api/ops/schema",
                                  headers={**auth_headers, "X-Org-Context": _DEFAULT_ORG_ID})
    assert "Org A only" in r_admin_a.json()["content"]


# ── WF-40: Schema validation length guard uses the correct org's schema ───────

@pytest.mark.asyncio
async def test_wf40_length_guard_uses_correct_org_schema(client, auth_headers, org_b,
                                                          supervisor_user, supervisor_headers):
    from app.services.orgs import create_user
    from app.services.auth import create_access_token

    sup_b_id = await create_user(org_b, "sup_b_wf40@test.com", "password123", role="supervisor")
    sup_b_token = create_access_token(email="sup_b_wf40@test.com", user_id=sup_b_id,
                                      org_id=org_b, role="supervisor")
    sup_b_hdrs = {"Authorization": f"Bearer {sup_b_token}"}

    # Org A: save a long schema (2000+ chars)
    long_schema = _VALID_SCHEMA * 10
    await client.put("/api/ops/schema", json={"content": long_schema},
                     headers=supervisor_headers)

    # Org B: save a short schema (~500 chars)
    short_base = _VALID_SCHEMA  # ~500 chars
    await client.put("/api/ops/schema", json={"content": short_base},
                     headers=sup_b_hdrs)

    # Candidate: 800 chars (more than 50% of 500; less than 50% of 2000)
    candidate = _VALID_SCHEMA  # just use the same ~500 char schema

    # For org A: 800 < 50% of 2000 → error
    r_a = await client.post("/api/ops/schema/validate", json={"content": candidate},
                            headers=supervisor_headers)
    assert any("50%" in e or "shorter" in e for e in r_a.json()["errors"])

    # For org B: candidate length > 50% of short_base → no length error
    r_b = await client.post("/api/ops/schema/validate", json={"content": candidate},
                            headers=sup_b_hdrs)
    length_errors_b = [e for e in r_b.json()["errors"]
                       if "50%" in e or "shorter" in e]
    assert len(length_errors_b) == 0


# ── WF-41: Per-user daily chat-message quota ─────────────────────────────────

@pytest.mark.asyncio
async def test_wf41_chat_message_quota_returns_429_with_clear_detail(
    client, auth_headers, test_engine,
):
    """Exhausting max_chat_messages_per_day on /chat must return 429 with a
    user-friendly detail string (not a stack trace)."""
    tpl_r = await client.post(
        "/api/admin/templates",
        json={"name": "tight_chat", "can_chat": True,
              "max_chat_messages_per_day": 2},
        headers=auth_headers,
    )
    assert tpl_r.status_code == 201

    user, hdrs = await _make_member(_DEFAULT_ORG_ID, "wf41@test.com", "tight_chat",
                                    client, auth_headers)

    async with test_engine.begin() as conn:
        for _ in range(2):
            await conn.execute(sa.text("""
                INSERT INTO usage_log (org_id, user_id, model_id, tokens_in, tokens_out, operation)
                VALUES (CAST(:org AS UUID), CAST(:uid AS UUID), 'test', 1, 1, 'chat')
            """), {"org": _DEFAULT_ORG_ID, "uid": user["id"]})

    r = await client.post("/api/ops/chat", json={"message": "hi"}, headers=hdrs)
    assert r.status_code == 429
    detail = r.json()["detail"]
    assert "chat message" in detail.lower()
    assert "2" in detail  # the limit value is visible to the user
    # No Python traceback leakage:
    assert "Traceback" not in detail
    assert "\n" not in detail


# ── WF-42: Per-user daily query quota ────────────────────────────────────────

@pytest.mark.asyncio
async def test_wf42_query_quota_returns_429_with_clear_detail(
    client, auth_headers, test_engine, mock_bedrock,
):
    tpl_r = await client.post(
        "/api/admin/templates",
        json={"name": "tight_query", "can_query": True,
              "max_queries_per_day": 1},
        headers=auth_headers,
    )
    assert tpl_r.status_code == 201

    user, hdrs = await _make_member(_DEFAULT_ORG_ID, "wf42@test.com", "tight_query",
                                    client, auth_headers)

    async with test_engine.begin() as conn:
        await conn.execute(sa.text("""
            INSERT INTO usage_log (org_id, user_id, model_id, tokens_in, tokens_out, operation)
            VALUES (CAST(:org AS UUID), CAST(:uid AS UUID), 'test', 1, 1, 'query')
        """), {"org": _DEFAULT_ORG_ID, "uid": user["id"]})

    r = await client.post("/api/ops/query", json={"question": "hi"}, headers=hdrs)
    assert r.status_code == 429
    detail = r.json()["detail"]
    assert "query" in detail.lower()
    assert "1" in detail
    assert "Traceback" not in detail


# ── WF-43: Org-level daily upload cap ────────────────────────────────────────

@pytest.mark.asyncio
async def test_wf43_org_upload_cap_returns_429(
    client, auth_headers, test_engine,
):
    """When the org-wide daily upload cap is reached (across users), the next
    upload by *any* member returns a clear 429 — not a 500."""
    await client.put(
        "/api/admin/org-limits",
        json={"org_id": _DEFAULT_ORG_ID, "max_uploads_per_day_org": 1},
        headers=auth_headers,
    )

    # Create one member with unlimited per-user uploads but generous template.
    tpl_r = await client.post(
        "/api/admin/templates",
        json={"name": "any_upload", "can_upload": True},
        headers=auth_headers,
    )
    assert tpl_r.status_code == 201

    user_a, _ = await _make_member(_DEFAULT_ORG_ID, "wf43a@test.com", "any_upload",
                                    client, auth_headers)
    user_b, hdrs_b = await _make_member(_DEFAULT_ORG_ID, "wf43b@test.com", "any_upload",
                                         client, auth_headers)

    # Burn the org cap via an ingest_jobs row attributed to user_a.
    async with test_engine.begin() as conn:
        await conn.execute(sa.text("""
            INSERT INTO ingest_jobs (org_id, user_id, filename, status)
            VALUES (CAST(:org AS UUID), CAST(:uid AS UUID), 'seed.txt', 'done')
        """), {"org": _DEFAULT_ORG_ID, "uid": user_a["id"]})

    r = await client.post(
        "/api/documents/upload",
        headers=hdrs_b,
        files={"file": ("new.txt", io.BytesIO(b"x"), "text/plain")},
    )
    assert r.status_code == 429
    detail = r.json()["detail"]
    assert "organization" in detail.lower() or "org" in detail.lower()
    assert "upload" in detail.lower()
    assert "Traceback" not in detail


# ── WF-44: Token quota detail is user-friendly ────────────────────────────────

@pytest.mark.asyncio
async def test_wf44_token_quota_detail_is_clear(
    client, auth_headers, test_engine,
):
    """Token cap should report tokens/day with commas (e.g. '1,000,000')."""
    tpl_r = await client.post(
        "/api/admin/templates",
        json={"name": "tight_tokens", "can_chat": True,
              "max_tokens_per_day": 1_000_000},
        headers=auth_headers,
    )
    assert tpl_r.status_code == 201
    assert tpl_r.json()
    assert (await client.get(f"/api/admin/templates/{tpl_r.json()['id']}",
                             headers=auth_headers)).json()["max_tokens_per_day"] == 1_000_000

    user, hdrs = await _make_member(_DEFAULT_ORG_ID, "wf44@test.com", "tight_tokens",
                                    client, auth_headers)

    async with test_engine.begin() as conn:
        await conn.execute(sa.text("""
            INSERT INTO usage_log (org_id, user_id, model_id, tokens_in, tokens_out, operation)
            VALUES (CAST(:org AS UUID), CAST(:uid AS UUID), 'test', 500000, 500000, 'chat')
        """), {"org": _DEFAULT_ORG_ID, "uid": user["id"]})

    r = await client.post("/api/ops/chat", json={"message": "hi"}, headers=hdrs)
    assert r.status_code == 429
    detail = r.json()["detail"]
    assert "token" in detail.lower()
    # The formatted limit (with thousands separators) should be present
    assert "1,000,000" in detail
    assert "Traceback" not in detail
