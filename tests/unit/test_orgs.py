"""Unit tests for app.services.orgs — password hashing, verify_user, create flows."""
import pytest
from unittest.mock import patch, AsyncMock

from tests.conftest import MockResult, MockRow, make_db_mock


# ── Password helpers ───────────────────────────────────────────────────────

def test_hash_and_verify_password():
    from app.services.orgs import _hash_password, _verify_password
    h = _hash_password("secret123")
    assert _verify_password("secret123", h)
    assert not _verify_password("wrong", h)


def test_sha256_legacy_verify():
    """Legacy sha256: hashes (pre-bcrypt) must still verify correctly."""
    import hashlib
    from app.services.orgs import _verify_password
    legacy = "sha256:" + hashlib.sha256("mypassword".encode()).hexdigest()
    assert _verify_password("mypassword", legacy)
    assert not _verify_password("bad", legacy)


def test_hash_produces_bcrypt():
    from app.services.orgs import _hash_password
    h = _hash_password("anypassword")
    assert h.startswith("$2b$") or h.startswith("$2a$")


# ── verify_user ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_verify_user_valid():
    from app.services.orgs import _hash_password

    row = MockRow(
        id="user-uuid",
        org_id="org-uuid",
        email="alice@example.com",
        password_hash=_hash_password("pass"),
        role="admin",
        org_name=None,
    )
    mock_db, session, _ = make_db_mock(MockResult(row=row))
    session.execute = AsyncMock(return_value=MockResult(row=row, rowcount=1))

    with patch("app.services.orgs.get_db", mock_db):
        from app.services.orgs import verify_user
        result = await verify_user("alice@example.com", "pass")
    assert result is not None
    assert result["email"] == "alice@example.com"
    assert result["role"] == "admin"


@pytest.mark.asyncio
async def test_verify_user_wrong_password():
    from app.services.orgs import _hash_password

    row = MockRow(
        id="user-uuid",
        org_id="org-uuid",
        email="bob@example.com",
        password_hash=_hash_password("correct"),
        role="member",
    )
    mock_db, session, _ = make_db_mock(MockResult(row=row))

    with patch("app.services.orgs.get_db", mock_db):
        from app.services.orgs import verify_user
        result = await verify_user("bob@example.com", "wrong")
    assert result is None


@pytest.mark.asyncio
async def test_verify_user_not_found():
    mock_db, _, _ = make_db_mock(MockResult(row=None))
    with patch("app.services.orgs.get_db", mock_db):
        from app.services.orgs import verify_user
        result = await verify_user("nobody@example.com", "pass")
    assert result is None


# ── create_user ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_create_user_success():
    new_row = MockRow(id="new-user-uuid")
    mock_db, session, _ = make_db_mock(MockResult(row=new_row))

    with patch("app.services.orgs.get_db", mock_db):
        from app.services.orgs import create_user
        uid = await create_user("org-uuid", "carol@example.com", "pw", "member")
    assert uid is not None


@pytest.mark.asyncio
async def test_create_user_email_conflict():
    """ON CONFLICT DO NOTHING returns no row — should return None."""
    mock_db, _, _ = make_db_mock(MockResult(row=None))
    with patch("app.services.orgs.get_db", mock_db):
        from app.services.orgs import create_user
        uid = await create_user("org-uuid", "duplicate@example.com", "pw")
    assert uid is None


# ── create_org ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_create_org_returns_id():
    row = MockRow(id="new-org-uuid")
    mock_db, _, _ = make_db_mock(MockResult(row=row))
    with patch("app.services.orgs.get_db", mock_db):
        from app.services.orgs import create_org
        org_id = await create_org("Acme Corp", "acme")
    assert org_id is not None


@pytest.mark.asyncio
async def test_create_org_seeds_agents_md():
    """New org should have the bundled AGENTS.md written to wiki_files."""
    row = MockRow(id="new-org-uuid")
    mock_db, _, _ = make_db_mock(MockResult(row=row))
    fake_content = "# Wiki Agent Schema\n\nDefault schema content for testing."
    mock_set_wiki = AsyncMock()

    with (
        patch("app.services.orgs.get_db", mock_db),
        patch("app.services.permissions.seed_builtin_templates", new_callable=AsyncMock),
        patch("app.services.wiki_db.get_wiki_file", new_callable=AsyncMock, return_value=None),
        patch("app.services.wiki_db.set_wiki_file", mock_set_wiki),
        patch("pathlib.Path.exists", return_value=True),
        patch("pathlib.Path.read_text", return_value=fake_content),
    ):
        from app.services.orgs import create_org
        await create_org("New Corp", "new-corp")

    mock_set_wiki.assert_called_once_with("schema/AGENTS.md", fake_content)


@pytest.mark.asyncio
async def test_create_org_does_not_overwrite_existing_agents_md():
    """If schema/AGENTS.md already exists for the org, it must not be overwritten."""
    row = MockRow(id="existing-org-uuid")
    mock_db, _, _ = make_db_mock(MockResult(row=row))
    mock_set_wiki = AsyncMock()

    with (
        patch("app.services.orgs.get_db", mock_db),
        patch("app.services.permissions.seed_builtin_templates", new_callable=AsyncMock),
        patch("app.services.wiki_db.get_wiki_file", new_callable=AsyncMock, return_value="# Existing schema"),
        patch("app.services.wiki_db.set_wiki_file", mock_set_wiki),
    ):
        from app.services.orgs import create_org
        await create_org("Existing Corp", "existing-corp")

    mock_set_wiki.assert_not_called()


@pytest.mark.asyncio
async def test_create_org_skips_agents_md_when_bundled_file_missing():
    """If the bundled AGENTS.md file doesn't exist on disk, no write should occur."""
    row = MockRow(id="new-org-uuid-2")
    mock_db, _, _ = make_db_mock(MockResult(row=row))
    mock_set_wiki = AsyncMock()

    with (
        patch("app.services.orgs.get_db", mock_db),
        patch("app.services.permissions.seed_builtin_templates", new_callable=AsyncMock),
        patch("app.services.wiki_db.get_wiki_file", new_callable=AsyncMock, return_value=None),
        patch("app.services.wiki_db.set_wiki_file", mock_set_wiki),
        patch("pathlib.Path.exists", return_value=False),
    ):
        from app.services.orgs import create_org
        await create_org("No Schema Corp", "no-schema")

    mock_set_wiki.assert_not_called()


# ── list_users ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_list_users_empty():
    mock_db, _, _ = make_db_mock(MockResult(rows=[]))
    with patch("app.services.orgs.get_db", mock_db):
        from app.services.orgs import list_users
        users = await list_users("org-uuid")
    assert users == []


@pytest.mark.asyncio
async def test_list_users_returns_rows():
    import datetime
    rows = [
        MockRow(id="u1", email="a@x.com", role="admin",
                created_at=datetime.datetime(2025, 1, 1), last_login_at=None),
        MockRow(id="u2", email="b@x.com", role="member",
                created_at=datetime.datetime(2025, 2, 1), last_login_at=None),
    ]
    mock_db, _, _ = make_db_mock(MockResult(rows=rows))
    with patch("app.services.orgs.get_db", mock_db):
        from app.services.orgs import list_users
        users = await list_users("org-uuid")
    assert len(users) == 2
    assert users[0]["email"] == "a@x.com"
    assert users[1]["role"] == "member"


# ── Memberships ──────────────────────────────────────────────────────────────

def _membership_row(**overrides):
    defaults = dict(
        id="m1", user_id="u1", org_id="org-a", role="member",
        permission_template_id=None, workspace_id=None,
        is_suspended=False, max_tokens_per_day=None, max_chat_messages_per_day=None,
    )
    defaults.update(overrides)
    return MockRow(**defaults)


@pytest.mark.asyncio
async def test_add_membership_created():
    mock_db, _, _ = make_db_mock(MockResult(row=MockRow(id="m-new")))
    with patch("app.services.orgs.get_db", mock_db):
        from app.services.orgs import add_membership
        created = await add_membership("u1", "org-a", "supervisor")
    assert created is True


@pytest.mark.asyncio
async def test_add_membership_already_member_returns_false():
    """ON CONFLICT DO NOTHING → no row → already a member."""
    mock_db, _, _ = make_db_mock(MockResult(row=None))
    with patch("app.services.orgs.get_db", mock_db):
        from app.services.orgs import add_membership
        created = await add_membership("u1", "org-a", "member")
    assert created is False


@pytest.mark.asyncio
async def test_update_membership_rowcount():
    mock_db, _, _ = make_db_mock(MockResult(rowcount=1))
    with patch("app.services.orgs.get_db", mock_db):
        from app.services.orgs import update_membership
        ok = await update_membership("u1", "org-a", {"role": "member"})
    assert ok is True


@pytest.mark.asyncio
async def test_remove_membership_not_found():
    mock_db, _, _ = make_db_mock(MockResult(rowcount=0))
    with patch("app.services.orgs.get_db", mock_db):
        from app.services.orgs import remove_membership
        ok = await remove_membership("u1", "org-a")
    assert ok is False


@pytest.mark.asyncio
async def test_get_membership_maps_row():
    row = _membership_row(role="supervisor", permission_template_id="tpl-9",
                          is_suspended=True, max_tokens_per_day=5000)
    mock_db, _, _ = make_db_mock(MockResult(row=row))
    with patch("app.services.orgs.get_db", mock_db):
        from app.services.orgs import get_membership
        m = await get_membership("u1", "org-a")
    assert m["role"] == "supervisor"
    assert m["permission_template_id"] == "tpl-9"
    assert m["is_suspended"] is True
    assert m["max_tokens_per_day"] == 5000


@pytest.mark.asyncio
async def test_get_membership_none_when_missing():
    mock_db, _, _ = make_db_mock(MockResult(row=None))
    with patch("app.services.orgs.get_db", mock_db):
        from app.services.orgs import get_membership
        assert await get_membership("u1", "org-a") is None


@pytest.mark.asyncio
async def test_list_memberships_for_user_maps_rows():
    rows = [
        MockRow(org_id="org-a", org_name="Org A", role="member",
                permission_template_id=None, workspace_id=None, is_suspended=False),
        MockRow(org_id="org-b", org_name="Org B", role="supervisor",
                permission_template_id="t1", workspace_id=None, is_suspended=True),
    ]
    mock_db, _, _ = make_db_mock(MockResult(rows=rows))
    with patch("app.services.orgs.get_db", mock_db):
        from app.services.orgs import list_memberships_for_user
        out = await list_memberships_for_user("u1")
    assert [m["org_id"] for m in out] == ["org-a", "org-b"]
    assert out[1]["role"] == "supervisor"
    assert out[1]["org_name"] == "Org B"


@pytest.mark.asyncio
async def test_list_members_maps_rows():
    import datetime
    rows = [
        MockRow(id="u1", email="a@x.com", role="member",
                created_at=datetime.datetime(2025, 1, 1), last_login_at=None),
        MockRow(id="u2", email="b@x.com", role="supervisor",
                created_at=datetime.datetime(2025, 2, 1), last_login_at=None),
    ]
    mock_db, _, _ = make_db_mock(MockResult(rows=rows))
    with patch("app.services.orgs.get_db", mock_db):
        from app.services.orgs import list_members
        members = await list_members("org-a")
    assert len(members) == 2
    assert members[0]["email"] == "a@x.com"
    assert members[1]["role"] == "supervisor"


# ── resolve_active_membership ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_resolve_active_membership_admin_uses_header_org():
    """Admins are global: role='admin' and the requested (header) org is used."""
    admin_row = MockRow(id="admin-1", org_id=None, email="admin@x.com", role="admin")
    mock_db, _, _ = make_db_mock(MockResult(row=admin_row))
    with patch("app.services.orgs.get_db", mock_db):
        from app.services.orgs import resolve_active_membership
        active = await resolve_active_membership("admin-1", "org-z")
    assert active == {"org_id": "org-z", "role": "admin"}


@pytest.mark.asyncio
async def test_resolve_active_membership_member_requested_org():
    """A member acting on an org they belong to gets that membership's role."""
    user_row = MockRow(id="u1", org_id="org-a", email="u@x.com", role="member")
    membership_row = _membership_row(role="supervisor", org_id="org-a")

    from contextlib import asynccontextmanager
    call = 0

    @asynccontextmanager
    async def multi():
        nonlocal call
        from unittest.mock import AsyncMock as _AM
        session = _AM()
        call += 1
        session.execute.return_value = MockResult(row=user_row if call == 1 else membership_row)
        yield session

    with patch("app.services.orgs.get_db", multi):
        from app.services.orgs import resolve_active_membership
        active = await resolve_active_membership("u1", "org-a")
    assert active == {"org_id": "org-a", "role": "supervisor"}


@pytest.mark.asyncio
async def test_resolve_active_membership_requested_org_not_a_member():
    """Header points at an org the member doesn't belong to → no org context."""
    user_row = MockRow(id="u1", org_id="org-a", email="u@x.com", role="member")

    from contextlib import asynccontextmanager
    call = 0

    @asynccontextmanager
    async def multi():
        nonlocal call
        from unittest.mock import AsyncMock as _AM
        session = _AM()
        call += 1
        session.execute.return_value = MockResult(row=user_row if call == 1 else None)
        yield session

    with patch("app.services.orgs.get_db", multi):
        from app.services.orgs import resolve_active_membership
        active = await resolve_active_membership("u1", "org-other")
    assert active == {"org_id": "", "role": "member"}


@pytest.mark.asyncio
async def test_resolve_active_membership_single_membership_no_header():
    """No header + exactly one membership → auto-select it."""
    user_row = MockRow(id="u1", org_id="org-a", email="u@x.com", role="member")
    list_row = MockRow(org_id="org-a", role="member", permission_template_id=None,
                       workspace_id=None, is_suspended=False, org_name="Org A")

    from contextlib import asynccontextmanager
    call = 0

    @asynccontextmanager
    async def multi():
        nonlocal call
        from unittest.mock import AsyncMock as _AM
        session = _AM()
        call += 1
        if call == 1:
            session.execute.return_value = MockResult(row=user_row)
        else:
            session.execute.return_value = MockResult(rows=[list_row])
        yield session

    with patch("app.services.orgs.get_db", multi):
        from app.services.orgs import resolve_active_membership
        active = await resolve_active_membership("u1", "")
    assert active == {"org_id": "org-a", "role": "member"}


@pytest.mark.asyncio
async def _resolve_no_header_for_memberships(membership_rows):
    """Helper: run resolve_active_membership("u1", "") with the given memberships.

    First DB call returns the user row (role member), the second returns the
    membership list — matching resolve_active_membership's two queries.
    """
    user_row = MockRow(id="u1", org_id="org-a", email="u@x.com", role="member")

    from contextlib import asynccontextmanager
    call = 0

    @asynccontextmanager
    async def multi():
        nonlocal call
        from unittest.mock import AsyncMock as _AM
        session = _AM()
        call += 1
        if call == 1:
            session.execute.return_value = MockResult(row=user_row)
        else:
            session.execute.return_value = MockResult(rows=membership_rows)
        yield session

    with patch("app.services.orgs.get_db", multi):
        from app.services.orgs import resolve_active_membership
        return await resolve_active_membership("u1", "")


async def test_resolve_active_membership_multi_membership_no_header():
    """No header + multiple memberships → no org selected (client must choose),
    but the highest role across orgs is preserved so a supervisor of multiple
    orgs still reaches the admin dashboard."""
    rows = [
        MockRow(org_id="org-a", role="member", permission_template_id=None,
                workspace_id=None, is_suspended=False, org_name="Org A"),
        MockRow(org_id="org-b", role="supervisor", permission_template_id=None,
                workspace_id=None, is_suspended=False, org_name="Org B"),
    ]
    active = await _resolve_no_header_for_memberships(rows)
    assert active == {"org_id": "", "role": "supervisor"}


async def test_resolve_active_membership_multi_membership_all_member_no_header():
    """No header + multiple memberships that are all `member` → stays member."""
    rows = [
        MockRow(org_id="org-a", role="member", permission_template_id=None,
                workspace_id=None, is_suspended=False, org_name="Org A"),
        MockRow(org_id="org-b", role="member", permission_template_id=None,
                workspace_id=None, is_suspended=False, org_name="Org B"),
    ]
    active = await _resolve_no_header_for_memberships(rows)
    assert active == {"org_id": "", "role": "member"}
