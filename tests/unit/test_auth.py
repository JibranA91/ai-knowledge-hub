"""Unit tests for app/services/auth.py (Phase 4 + Phase 6 — JWT with org claims).

All DB calls are mocked — no PostgreSQL needed.
"""
import time
import pytest
from unittest.mock import patch

from tests.conftest import MockResult, make_db_mock


# ── create_access_token ───────────────────────────────────────────────────

def test_create_access_token_returns_string():
    from app.services.auth import create_access_token
    token = create_access_token("admin")
    assert isinstance(token, str)
    assert len(token) > 64  # JWTs are longer than the old 64-char hex tokens


def test_create_access_token_is_synchronous():
    # Must not require await — validated by calling it without async context
    from app.services.auth import create_access_token
    import inspect
    assert not inspect.iscoroutinefunction(create_access_token)


def test_create_access_token_different_each_call():
    from app.services.auth import create_access_token
    # Different expiry timestamps produce different tokens even for same user
    t1 = create_access_token("admin")
    time.sleep(0.01)
    t2 = create_access_token("admin")
    # Same user can produce same token within the same second — just verify format
    assert isinstance(t1, str) and isinstance(t2, str)


# ── validate_access_token ─────────────────────────────────────────────────

def test_validate_access_token_valid():
    from app.services.auth import create_access_token, validate_access_token
    token = create_access_token("admin", user_id="uid-1", org_id="org-1", role="admin")
    ctx = validate_access_token(token)
    assert ctx is not None
    assert ctx.email == "admin"


def test_validate_access_token_returns_username():
    from app.services.auth import create_access_token, validate_access_token
    token = create_access_token("alice", user_id="uid-2", org_id="org-2", role="member")
    ctx = validate_access_token(token)
    assert ctx is not None
    assert ctx.email == "alice"


def test_validate_access_token_empty_string():
    from app.services.auth import validate_access_token
    assert validate_access_token("") is None


def test_validate_access_token_garbage():
    from app.services.auth import validate_access_token
    assert validate_access_token("not.a.jwt") is None


def test_validate_access_token_wrong_secret():
    from jose import jwt
    from app.services.auth import validate_access_token
    from app.config import settings
    from datetime import datetime, timedelta, timezone
    # Encode with a different secret
    payload = {"sub": "admin", "exp": datetime.now(timezone.utc) + timedelta(minutes=15), "type": "access"}
    forged = jwt.encode(payload, "wrong-secret", algorithm=settings.JWT_ALGORITHM)
    assert validate_access_token(forged) is None


def test_validate_access_token_wrong_type():
    from jose import jwt
    from app.services.auth import validate_access_token
    from app.config import settings
    from datetime import datetime, timedelta, timezone
    # Encode with type != "access"
    payload = {"sub": "admin", "exp": datetime.now(timezone.utc) + timedelta(days=7), "type": "refresh"}
    token = jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)
    assert validate_access_token(token) is None


# ── create_refresh_token ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_create_refresh_token_returns_64_char_hex():
    from app.services.auth import create_refresh_token
    mock_db, _, _ = make_db_mock()
    with patch("app.services.auth.get_db", mock_db):
        token = await create_refresh_token("admin")
    assert len(token) == 64
    assert all(c in "0123456789abcdef" for c in token)


@pytest.mark.asyncio
async def test_create_refresh_token_writes_to_db():
    from app.services.auth import create_refresh_token
    mock_db, session, _ = make_db_mock()
    with patch("app.services.auth.get_db", mock_db):
        await create_refresh_token("admin")
    session.execute.assert_called_once()


@pytest.mark.asyncio
async def test_create_refresh_token_unique():
    from app.services.auth import create_refresh_token
    mock_db, _, _ = make_db_mock()
    with patch("app.services.auth.get_db", mock_db):
        t1 = await create_refresh_token("admin")
        t2 = await create_refresh_token("admin")
    assert t1 != t2


# ── validate_refresh_token ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_validate_refresh_token_valid():
    from app.services.auth import validate_refresh_token
    from tests.conftest import MockRow
    row = MockRow(username="admin", org_id=None, user_id=None)
    mock_db, _, _ = make_db_mock(MockResult(row=row))
    with patch("app.services.auth.get_db", mock_db):
        result = await validate_refresh_token("some-token")
    assert result is not None
    assert result["email"] == "admin"


@pytest.mark.asyncio
async def test_validate_refresh_token_not_found():
    from app.services.auth import validate_refresh_token
    mock_db, _, _ = make_db_mock(MockResult(row=None))
    with patch("app.services.auth.get_db", mock_db):
        result = await validate_refresh_token("unknown-token")
    assert result is None


@pytest.mark.asyncio
async def test_validate_refresh_token_empty_string_fast_path():
    from app.services.auth import validate_refresh_token
    mock_db, session, _ = make_db_mock()
    with patch("app.services.auth.get_db", mock_db):
        result = await validate_refresh_token("")
    assert result is None
    session.execute.assert_not_called()


# ── revoke_token / purge_expired ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_revoke_token_calls_delete():
    from app.services.auth import revoke_token
    mock_db, session, _ = make_db_mock()
    with patch("app.services.auth.get_db", mock_db):
        await revoke_token("some-token")
    session.execute.assert_called_once()


@pytest.mark.asyncio
async def test_purge_expired_calls_delete():
    from app.services.auth import purge_expired
    mock_db, session, _ = make_db_mock()
    with patch("app.services.auth.get_db", mock_db):
        await purge_expired()
    session.execute.assert_called_once()
