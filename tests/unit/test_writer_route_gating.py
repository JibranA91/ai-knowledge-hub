"""Unit tests for `_require_writer_role` in app.routes.operations.

After migration 018, writer access for members is governed by the
``can_use_writer`` permission flag; admins and supervisors still bypass.
"""
import pytest
from unittest.mock import MagicMock, patch

from fastapi import HTTPException

from app.routes.operations import _require_writer_role


def _req(role: str = "member", user_id: str = "uid-1", org_id: str = "org-1"):
    r = MagicMock()
    r.state.role = role
    r.state.user_id = user_id
    r.state.org_id = org_id
    return r


@pytest.mark.asyncio
async def test_admin_bypasses_without_db_lookup():
    """Admin requires no perms lookup — must not call get_user_permissions."""
    async def _should_not_be_called(*_a, **_kw):
        raise AssertionError("get_user_permissions should not be invoked for admin")
    with patch("app.services.permissions.get_user_permissions", _should_not_be_called):
        await _require_writer_role(_req(role="admin"))


@pytest.mark.asyncio
async def test_supervisor_bypasses_without_db_lookup():
    async def _should_not_be_called(*_a, **_kw):
        raise AssertionError("get_user_permissions should not be invoked for supervisor")
    with patch("app.services.permissions.get_user_permissions", _should_not_be_called):
        await _require_writer_role(_req(role="supervisor"))


@pytest.mark.asyncio
async def test_member_with_flag_passes():
    async def _perms(user_id, org_id):
        return {"is_suspended": False, "can_use_writer": True}
    with patch("app.services.permissions.get_user_permissions", _perms):
        await _require_writer_role(_req(role="member"))


@pytest.mark.asyncio
async def test_member_without_flag_raises_403():
    async def _perms(user_id, org_id):
        return {"is_suspended": False, "can_use_writer": False}
    with patch("app.services.permissions.get_user_permissions", _perms):
        with pytest.raises(HTTPException) as exc_info:
            await _require_writer_role(_req(role="member"))
        assert exc_info.value.status_code == 403
        assert "writer" in exc_info.value.detail.lower()


@pytest.mark.asyncio
async def test_member_missing_flag_key_raises_403():
    """An older perms response that lacks the key at all must still be denied."""
    async def _perms(user_id, org_id):
        return {"is_suspended": False}  # key absent
    with patch("app.services.permissions.get_user_permissions", _perms):
        with pytest.raises(HTTPException) as exc_info:
            await _require_writer_role(_req(role="member"))
        assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_suspended_member_with_flag_still_blocked():
    """Suspension overrides any granted flag."""
    async def _perms(user_id, org_id):
        return {"is_suspended": True, "can_use_writer": True}
    with patch("app.services.permissions.get_user_permissions", _perms):
        with pytest.raises(HTTPException) as exc_info:
            await _require_writer_role(_req(role="member"))
        assert exc_info.value.status_code == 403
        assert "suspended" in exc_info.value.detail.lower()
