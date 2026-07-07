"""Unit tests for app/services/permissions.py.

All DB calls are mocked via make_db_mock / MockResult.
"""
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from tests.conftest import MockResult, MockRow, make_db_mock


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_request(role: str = "member", user_id: str = "uid-1", org_id: str = "org-1"):
    req = MagicMock()
    req.state.role = role
    req.state.user_id = user_id
    req.state.org_id = org_id
    return req


@pytest.mark.asyncio
async def test_request_user_permissions_memoizes_per_request():
    """R5: stacked dependencies in one request reuse a single permissions query."""
    import types
    from app.services import permissions
    req = types.SimpleNamespace(state=types.SimpleNamespace())
    calls = {"n": 0}

    async def _fake(uid, oid):
        calls["n"] += 1
        return {"can_chat": True}

    with patch.object(permissions, "get_user_permissions", _fake):
        p1 = await permissions._request_user_permissions(req, "u", "o")
        p2 = await permissions._request_user_permissions(req, "u", "o")

    assert p1 is p2
    assert calls["n"] == 1   # queried once, served from the per-request cache after


def _template_row(**overrides):
    defaults = dict(
        is_suspended=False,
        permission_template_id="tpl-1",
        can_upload=True,
        can_upload_writer_draft=True,
        can_delete_files=False,
        can_download_files=True,
        max_upload_size_mb=None,
        max_uploads_per_day=10,
        max_uploads_per_week=50,
        can_view_wiki=True,
        can_edit_wiki=False,
        can_delete_wiki_pages=False,
        can_query=True,
        can_chat=True,
        can_use_writer=False,
        max_queries_per_day=20,
        max_chat_messages_per_day=50,
        user_max_chat_messages_per_day=None,
        max_tokens_per_day=100000,
        user_max_tokens_per_day=None,
        max_tokens_per_week=None,
        can_recalibrate=False,
        can_run_lint=False,
        can_manage_schema=False,
        can_view_audit_log=False,
        can_view_graph=True,
        can_rebuild_graph=False,
        can_manage_workspace=False,
        can_approve_ingest=True,
        can_cancel_ingest=True,
    )
    defaults.update(overrides)
    return MockRow(**defaults)


# ── get_user_permissions ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_user_permissions_with_template():
    row = _template_row()
    mock_db, _session, _ = make_db_mock(MockResult(row=row))
    with patch("app.services.permissions.get_db", mock_db):
        from app.services.permissions import get_user_permissions
        perms = await get_user_permissions("uid-1", "org-1")
    assert perms["can_upload"] is True
    assert perms["can_delete_files"] is False
    assert perms["max_uploads_per_day"] == 10
    assert perms["can_recalibrate"] is False


@pytest.mark.asyncio
async def test_get_user_permissions_no_template_returns_read_only():
    row = MockRow(is_suspended=False, permission_template_id=None)
    mock_db, _session, _ = make_db_mock(MockResult(row=row))
    with patch("app.services.permissions.get_db", mock_db):
        from app.services.permissions import get_user_permissions
        perms = await get_user_permissions("uid-1", "org-1")
    assert perms["can_upload"] is False
    assert perms["can_view_wiki"] is True
    assert perms["can_query"] is False


@pytest.mark.asyncio
async def test_get_user_permissions_no_user_returns_read_only():
    mock_db, _session, _ = make_db_mock(MockResult(row=None))
    with patch("app.services.permissions.get_db", mock_db):
        from app.services.permissions import get_user_permissions
        perms = await get_user_permissions("unknown", "org-1")
    assert perms["can_upload"] is False


@pytest.mark.asyncio
async def test_get_user_permissions_db_error_returns_read_only():
    async def _bad_get_db():
        raise Exception("DB down")

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _err():
        raise Exception("DB down")
        yield  # noqa: unreachable

    with patch("app.services.permissions.get_db", _err):
        from app.services.permissions import get_user_permissions
        perms = await get_user_permissions("uid-1", "org-1")
    assert perms["can_upload"] is False  # falls back to read_only


@pytest.mark.asyncio
async def test_get_user_permissions_can_use_writer_propagates_true():
    row = _template_row(can_use_writer=True)
    mock_db, _session, _ = make_db_mock(MockResult(row=row))
    with patch("app.services.permissions.get_db", mock_db):
        from app.services.permissions import get_user_permissions
        perms = await get_user_permissions("uid-1", "org-1")
    assert perms["can_use_writer"] is True


@pytest.mark.asyncio
async def test_get_user_permissions_can_use_writer_propagates_false():
    row = _template_row(can_use_writer=False)
    mock_db, _session, _ = make_db_mock(MockResult(row=row))
    with patch("app.services.permissions.get_db", mock_db):
        from app.services.permissions import get_user_permissions
        perms = await get_user_permissions("uid-1", "org-1")
    assert perms["can_use_writer"] is False


@pytest.mark.asyncio
async def test_get_user_permissions_no_template_can_use_writer_false():
    """Read-only fallback (no template assigned) must deny writer access."""
    row = MockRow(is_suspended=False, permission_template_id=None)
    mock_db, _session, _ = make_db_mock(MockResult(row=row))
    with patch("app.services.permissions.get_db", mock_db):
        from app.services.permissions import get_user_permissions
        perms = await get_user_permissions("uid-1", "org-1")
    assert perms["can_use_writer"] is False


def test_read_only_defaults_can_use_writer_false():
    from app.services.permissions import _READ_ONLY_DEFAULTS
    assert _READ_ONLY_DEFAULTS["can_use_writer"] is False


def test_admin_full_can_use_writer_true():
    from app.services.permissions import _ADMIN_FULL
    assert _ADMIN_FULL["can_use_writer"] is True


def test_template_params_extracts_can_use_writer():
    """create/update template helpers must round-trip the new flag."""
    from app.services.permissions import _template_params
    params = _template_params({"name": "x", "can_use_writer": True})
    assert params["can_use_writer"] is True
    params = _template_params({"name": "x"})  # missing → default False
    assert params["can_use_writer"] is False


# ── can_upload_writer_draft propagation ───────────────────────────────────────

@pytest.mark.asyncio
async def test_get_user_permissions_can_upload_writer_draft_propagates_true():
    row = _template_row(can_upload_writer_draft=True)
    mock_db, _session, _ = make_db_mock(MockResult(row=row))
    with patch("app.services.permissions.get_db", mock_db):
        from app.services.permissions import get_user_permissions
        perms = await get_user_permissions("uid-1", "org-1")
    assert perms["can_upload_writer_draft"] is True


@pytest.mark.asyncio
async def test_get_user_permissions_can_upload_writer_draft_independent_of_can_upload():
    """The two flags are decoupled — one must not bleed into the other."""
    row = _template_row(can_upload=False, can_upload_writer_draft=True)
    mock_db, _session, _ = make_db_mock(MockResult(row=row))
    with patch("app.services.permissions.get_db", mock_db):
        from app.services.permissions import get_user_permissions
        perms = await get_user_permissions("uid-1", "org-1")
    assert perms["can_upload"] is False
    assert perms["can_upload_writer_draft"] is True


def test_read_only_defaults_can_upload_writer_draft_false():
    from app.services.permissions import _READ_ONLY_DEFAULTS
    assert _READ_ONLY_DEFAULTS["can_upload_writer_draft"] is False


def test_admin_full_can_upload_writer_draft_true():
    from app.services.permissions import _ADMIN_FULL
    assert _ADMIN_FULL["can_upload_writer_draft"] is True


def test_template_params_extracts_can_upload_writer_draft():
    from app.services.permissions import _template_params
    params = _template_params({"name": "x", "can_upload_writer_draft": True})
    assert params["can_upload_writer_draft"] is True
    params = _template_params({"name": "x"})  # missing → default False
    assert params["can_upload_writer_draft"] is False


@pytest.mark.asyncio
async def test_get_user_permissions_suspended():
    row = _template_row(is_suspended=True)
    mock_db, _session, _ = make_db_mock(MockResult(row=row))
    with patch("app.services.permissions.get_db", mock_db):
        from app.services.permissions import get_user_permissions
        perms = await get_user_permissions("uid-1", "org-1")
    assert perms["is_suspended"] is True


# ── require_permission ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_require_permission_admin_always_passes():
    from app.services.permissions import require_permission
    dep = require_permission("can_recalibrate")
    req = _make_request(role="admin")
    await dep(req)  # should not raise


@pytest.mark.asyncio
async def test_require_permission_member_with_flag_passes():
    row = _template_row(can_query=True)
    mock_db, _session, _ = make_db_mock(MockResult(row=row))
    with patch("app.services.permissions.get_db", mock_db):
        from app.services.permissions import require_permission
        dep = require_permission("can_query")
        await dep(_make_request(role="member"))  # should not raise


@pytest.mark.asyncio
async def test_require_permission_member_without_flag_raises_403():
    row = _template_row(can_recalibrate=False)
    mock_db, _session, _ = make_db_mock(MockResult(row=row))
    with patch("app.services.permissions.get_db", mock_db):
        from app.services.permissions import require_permission
        from fastapi import HTTPException
        dep = require_permission("can_recalibrate")
        with pytest.raises(HTTPException) as exc_info:
            await dep(_make_request(role="member"))
        assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_require_permission_suspended_raises_403():
    row = _template_row(is_suspended=True, can_upload=True)
    mock_db, _session, _ = make_db_mock(MockResult(row=row))
    with patch("app.services.permissions.get_db", mock_db):
        from app.services.permissions import require_permission
        from fastapi import HTTPException
        dep = require_permission("can_upload")
        with pytest.raises(HTTPException) as exc_info:
            await dep(_make_request(role="member"))
        assert exc_info.value.status_code == 403
        assert "suspended" in exc_info.value.detail.lower()


# ── check_upload_quota ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_check_upload_quota_admin_skips():
    from app.services.permissions import check_upload_quota
    dep = check_upload_quota()
    await dep(_make_request(role="admin"))  # must not call DB


@pytest.mark.asyncio
async def test_check_upload_quota_under_daily_limit():
    row = _template_row(max_uploads_per_day=10, max_uploads_per_week=None)
    perms_db, _, _ = make_db_mock(MockResult(row=row))
    count_db, cs, _ = make_db_mock(MockResult(scalar_val=5))

    call_count = 0

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def multi_mock():
        nonlocal call_count
        session = AsyncMock()
        call_count += 1
        if call_count == 1:
            session.execute.return_value = MockResult(row=row)
        else:
            session.execute.return_value = MockResult(scalar_val=5)
        yield session

    with patch("app.services.permissions.get_db", multi_mock):
        from app.services.permissions import check_upload_quota
        dep = check_upload_quota()
        await dep(_make_request(role="member"))  # under limit — should not raise


@pytest.mark.asyncio
async def test_check_upload_quota_exceeds_daily_limit():
    from contextlib import asynccontextmanager
    call_count = 0
    row = _template_row(max_uploads_per_day=5, max_uploads_per_week=None)

    @asynccontextmanager
    async def multi_mock():
        nonlocal call_count
        session = AsyncMock()
        call_count += 1
        if call_count == 1:
            session.execute.return_value = MockResult(row=row)
        else:
            session.execute.return_value = MockResult(scalar_val=5)  # exactly at limit
        yield session

    with patch("app.services.permissions.get_db", multi_mock):
        from app.services.permissions import check_upload_quota
        from fastapi import HTTPException
        dep = check_upload_quota()
        with pytest.raises(HTTPException) as exc_info:
            await dep(_make_request(role="member"))
        assert exc_info.value.status_code == 429


@pytest.mark.asyncio
async def test_check_upload_quota_no_limit_skips_count_query():
    """When template has no limits set, no COUNT queries should be executed."""
    row = _template_row(max_uploads_per_day=None, max_uploads_per_week=None)
    query_count = 0

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def counting_mock():
        nonlocal query_count
        session = AsyncMock()
        if query_count == 0:
            session.execute.return_value = MockResult(row=row)
        else:
            query_count += 1
        yield session
        query_count += 1

    with patch("app.services.permissions.get_db", counting_mock):
        from app.services.permissions import check_upload_quota
        dep = check_upload_quota()
        await dep(_make_request(role="member"))  # should not raise


# ── check_query_quota ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_check_query_quota_admin_skips():
    from app.services.permissions import check_query_quota
    dep = check_query_quota()
    await dep(_make_request(role="admin"))


@pytest.mark.asyncio
async def test_check_query_quota_exceeds_raises_429():
    from contextlib import asynccontextmanager
    call_count = 0
    row = _template_row(max_queries_per_day=5)

    @asynccontextmanager
    async def multi():
        nonlocal call_count
        session = AsyncMock()
        call_count += 1
        if call_count == 1:
            session.execute.return_value = MockResult(row=row)
        else:
            session.execute.return_value = MockResult(scalar_val=5)
        yield session

    with patch("app.services.permissions.get_db", multi):
        from app.services.permissions import check_query_quota
        from fastapi import HTTPException
        dep = check_query_quota()
        with pytest.raises(HTTPException) as exc_info:
            await dep(_make_request(role="member"))
        assert exc_info.value.status_code == 429


# ── check_chat_quota ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_check_chat_quota_under_limit():
    from contextlib import asynccontextmanager
    call_count = 0
    row = _template_row(max_chat_messages_per_day=100)

    @asynccontextmanager
    async def multi():
        nonlocal call_count
        session = AsyncMock()
        call_count += 1
        if call_count == 1:
            session.execute.return_value = MockResult(row=row)
        else:
            session.execute.return_value = MockResult(scalar_val=50)
        yield session

    with patch("app.services.permissions.get_db", multi):
        from app.services.permissions import check_chat_quota
        dep = check_chat_quota()
        await dep(_make_request(role="member"))  # should not raise


# ── check_token_quota ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_check_token_quota_admin_skips():
    from app.services.permissions import check_token_quota
    dep = check_token_quota()
    await dep(_make_request(role="admin"))


@pytest.mark.asyncio
async def test_check_token_quota_exceeds_daily():
    from contextlib import asynccontextmanager
    call_count = 0
    row = _template_row(max_tokens_per_day=50000, max_tokens_per_week=None)

    @asynccontextmanager
    async def multi():
        nonlocal call_count
        session = AsyncMock()
        call_count += 1
        if call_count == 1:
            session.execute.return_value = MockResult(row=row)
        else:
            session.execute.return_value = MockResult(scalar_val=50000)
        yield session

    with patch("app.services.permissions.get_db", multi):
        from app.services.permissions import check_token_quota
        from fastapi import HTTPException
        dep = check_token_quota()
        with pytest.raises(HTTPException) as exc_info:
            await dep(_make_request(role="member"))
        assert exc_info.value.status_code == 429
        assert "token" in exc_info.value.detail.lower()


# ── check_chat_quota — exceeds ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_check_chat_quota_exceeds_raises_429():
    from contextlib import asynccontextmanager
    call_count = 0
    row = _template_row(max_chat_messages_per_day=10)

    @asynccontextmanager
    async def multi():
        nonlocal call_count
        session = AsyncMock()
        call_count += 1
        if call_count == 1:
            session.execute.return_value = MockResult(row=row)
        else:
            session.execute.return_value = MockResult(scalar_val=10)  # at cap
        yield session

    with patch("app.services.permissions.get_db", multi):
        from app.services.permissions import check_chat_quota
        from fastapi import HTTPException
        dep = check_chat_quota()
        with pytest.raises(HTTPException) as exc_info:
            await dep(_make_request(role="member"))
        assert exc_info.value.status_code == 429
        assert "chat message" in exc_info.value.detail.lower()


# ── check_token_quota — weekly + org cap ──────────────────────────────────────

@pytest.mark.asyncio
async def test_check_token_quota_exceeds_weekly():
    """Weekly cap fires when daily is unlimited but weekly is at limit."""
    from contextlib import asynccontextmanager
    call_count = 0
    row = _template_row(max_tokens_per_day=None, max_tokens_per_week=5_000_000)

    @asynccontextmanager
    async def multi():
        nonlocal call_count
        session = AsyncMock()
        call_count += 1
        if call_count == 1:
            session.execute.return_value = MockResult(row=row)
        else:
            session.execute.return_value = MockResult(scalar_val=5_000_000)
        yield session

    with patch("app.services.permissions.get_db", multi):
        from app.services.permissions import check_token_quota
        from fastapi import HTTPException
        dep = check_token_quota()
        with pytest.raises(HTTPException) as exc_info:
            await dep(_make_request(role="member"))
        assert exc_info.value.status_code == 429
        assert "weekly" in exc_info.value.detail.lower()
        assert "token" in exc_info.value.detail.lower()


@pytest.mark.asyncio
async def test_check_token_quota_org_cap_exceeded_with_no_user_limit():
    """With no per-user token caps set, an exhausted org cap must still 429."""
    from contextlib import asynccontextmanager
    call_count = 0
    user_row = _template_row(max_tokens_per_day=None, max_tokens_per_week=None)
    org_row = MockRow(max_tokens_per_day_org=2_000_000, max_uploads_per_day_org=None, max_members=None)

    @asynccontextmanager
    async def multi():
        nonlocal call_count
        session = AsyncMock()
        call_count += 1
        if call_count == 1:
            session.execute.return_value = MockResult(row=user_row)
        elif call_count == 2:
            # User-limits block: no executes happen (both caps None) — return value irrelevant.
            session.execute.return_value = MockResult()
        elif call_count == 3:
            session.execute.return_value = MockResult(row=org_row)
        else:
            session.execute.return_value = MockResult(scalar_val=2_000_000)
        yield session

    with patch("app.services.permissions.get_db", multi):
        from app.services.permissions import check_token_quota
        from fastapi import HTTPException
        dep = check_token_quota()
        with pytest.raises(HTTPException) as exc_info:
            await dep(_make_request(role="member"))
        assert exc_info.value.status_code == 429
        assert "organization" in exc_info.value.detail.lower()


# ── check_upload_quota — weekly + org cap ─────────────────────────────────────

@pytest.mark.asyncio
async def test_check_upload_quota_exceeds_weekly():
    from contextlib import asynccontextmanager
    call_count = 0
    row = _template_row(max_uploads_per_day=None, max_uploads_per_week=10)

    @asynccontextmanager
    async def multi():
        nonlocal call_count
        session = AsyncMock()
        call_count += 1
        if call_count == 1:
            session.execute.return_value = MockResult(row=row)
        else:
            session.execute.return_value = MockResult(scalar_val=10)
        yield session

    with patch("app.services.permissions.get_db", multi):
        from app.services.permissions import check_upload_quota
        from fastapi import HTTPException
        dep = check_upload_quota()
        with pytest.raises(HTTPException) as exc_info:
            await dep(_make_request(role="member"))
        assert exc_info.value.status_code == 429
        assert "weekly" in exc_info.value.detail.lower()
        assert "upload" in exc_info.value.detail.lower()


@pytest.mark.asyncio
async def test_check_upload_quota_org_cap_exceeded():
    """Call sequence:
       1) get_user_permissions
       2) user-limits block — no execute (both caps None)
       3) _GET_ORG_LIMITS
       4) _COUNT_ORG_UPLOADS_TODAY
    """
    from contextlib import asynccontextmanager
    call_count = 0
    user_row = _template_row(max_uploads_per_day=None, max_uploads_per_week=None)
    org_row = MockRow(max_uploads_per_day_org=5, max_tokens_per_day_org=None, max_members=None)

    @asynccontextmanager
    async def multi():
        nonlocal call_count
        session = AsyncMock()
        call_count += 1
        if call_count == 1:
            session.execute.return_value = MockResult(row=user_row)
        elif call_count == 3:
            session.execute.return_value = MockResult(row=org_row)
        elif call_count == 4:
            session.execute.return_value = MockResult(scalar_val=5)
        else:
            session.execute.return_value = MockResult()
        yield session

    with patch("app.services.permissions.get_db", multi):
        from app.services.permissions import check_upload_quota
        from fastapi import HTTPException
        dep = check_upload_quota()
        with pytest.raises(HTTPException) as exc_info:
            await dep(_make_request(role="member"))
        assert exc_info.value.status_code == 429
        assert "organization" in exc_info.value.detail.lower()
        assert "upload" in exc_info.value.detail.lower()


# ── Template CRUD ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_list_templates_returns_list():
    from tests.conftest import MockRow
    rows = [
        MockRow(id="t1", org_id="org1", name="read_only", description="", is_builtin=True,
                can_upload=False, can_upload_writer_draft=False,
                can_delete_files=False, can_download_files=True,
                max_upload_size_mb=None, max_uploads_per_day=None, max_uploads_per_week=None,
                can_view_wiki=True, can_edit_wiki=False, can_delete_wiki_pages=False,
                can_query=False, can_chat=False, can_use_writer=False, max_queries_per_day=None,
                max_chat_messages_per_day=None, max_tokens_per_day=None, max_tokens_per_week=None,
                can_recalibrate=False, can_run_lint=False, can_manage_schema=False,
                can_view_audit_log=False, can_view_graph=True, can_rebuild_graph=False,
                can_manage_workspace=False, can_approve_ingest=False, can_cancel_ingest=False,
                created_at=None, updated_at=None),
    ]
    mock_db, _, _ = make_db_mock(MockResult(rows=rows))
    with patch("app.services.permissions.get_db", mock_db):
        from app.services.permissions import list_templates
        result = await list_templates("org1")
    assert len(result) == 1
    assert result[0]["name"] == "read_only"
    assert result[0]["is_builtin"] is True


@pytest.mark.asyncio
async def test_create_template_returns_id():
    new_id = "new-tpl-uuid"
    row = MockRow(id=new_id)
    mock_db, _, _ = make_db_mock(MockResult(row=row))
    with patch("app.services.permissions.get_db", mock_db):
        from app.services.permissions import create_template
        result = await create_template("org1", {
            "name": "custom", "description": "test",
            "can_upload": True, "can_delete_files": False,
        })
    assert result == new_id


@pytest.mark.asyncio
async def test_create_template_name_conflict_returns_none():
    mock_db, _, _ = make_db_mock(MockResult(row=None))
    with patch("app.services.permissions.get_db", mock_db):
        from app.services.permissions import create_template
        result = await create_template("org1", {"name": "existing"})
    assert result is None


@pytest.mark.asyncio
async def test_update_template_success():
    mock_db, _, _ = make_db_mock(MockResult(rowcount=1))
    with patch("app.services.permissions.get_db", mock_db):
        from app.services.permissions import update_template
        result = await update_template("tpl-1", "org-1", {"name": "new_name"})
    assert result is True


@pytest.mark.asyncio
async def test_update_template_builtin_returns_false():
    mock_db, _, _ = make_db_mock(MockResult(rowcount=0))
    with patch("app.services.permissions.get_db", mock_db):
        from app.services.permissions import update_template
        result = await update_template("builtin-id", "org-1", {"name": "read_only"})
    assert result is False


@pytest.mark.asyncio
async def test_delete_template_success():
    mock_db, _, _ = make_db_mock(MockResult(rowcount=1))
    with patch("app.services.permissions.get_db", mock_db):
        from app.services.permissions import delete_template
        result = await delete_template("tpl-1", "org-1")
    assert result is True


@pytest.mark.asyncio
async def test_delete_template_builtin_returns_false():
    mock_db, _, _ = make_db_mock(MockResult(rowcount=0))
    with patch("app.services.permissions.get_db", mock_db):
        from app.services.permissions import delete_template
        result = await delete_template("builtin-id", "org-1")
    assert result is False


# ── count_template_users ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_count_template_users_returns_count():
    mock_db, _, _ = make_db_mock(MockResult(scalar_val=3))
    with patch("app.services.permissions.get_db", mock_db):
        from app.services.permissions import count_template_users
        result = await count_template_users("tpl-1")
    assert result == 3


@pytest.mark.asyncio
async def test_count_template_users_zero():
    mock_db, _, _ = make_db_mock(MockResult(scalar_val=0))
    with patch("app.services.permissions.get_db", mock_db):
        from app.services.permissions import count_template_users
        result = await count_template_users("tpl-1")
    assert result == 0


@pytest.mark.asyncio
async def test_count_template_users_null_scalar_returns_zero():
    """DB COUNT returns NULL for an empty set in some drivers — must default to 0."""
    mock_db, _, _ = make_db_mock(MockResult(scalar_val=None))
    with patch("app.services.permissions.get_db", mock_db):
        from app.services.permissions import count_template_users
        result = await count_template_users("tpl-1")
    assert result == 0


# ── Defaults ──────────────────────────────────────────────────────────────────

def test_admin_full_all_booleans_true():
    from app.services.permissions import _ADMIN_FULL, _READ_ONLY_DEFAULTS
    bool_keys = [k for k, v in _READ_ONLY_DEFAULTS.items()
                 if isinstance(v, bool) and k != "is_suspended"]
    for k in bool_keys:
        assert _ADMIN_FULL[k] is True, f"_ADMIN_FULL[{k!r}] should be True"


def test_read_only_all_booleans_except_view_are_false():
    from app.services.permissions import _READ_ONLY_DEFAULTS
    allowed_true = {"can_view_wiki", "can_view_graph", "can_download_files"}
    for k, v in _READ_ONLY_DEFAULTS.items():
        if isinstance(v, bool) and k != "is_suspended" and k not in allowed_true:
            assert v is False, f"_READ_ONLY_DEFAULTS[{k!r}] should be False"
