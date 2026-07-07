"""Unit tests for app/services/rate_limit.py.

All DB calls are mocked — no PostgreSQL needed.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from tests.conftest import MockResult, make_db_mock


def _make_request(user: str = "admin") -> MagicMock:
    req = MagicMock()
    req.state.user = user
    return req


# ── check_rate_limit ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_check_rate_limit_under_limit_does_not_raise():
    from app.services.rate_limit import check_rate_limit
    mock_db, _, _ = make_db_mock(MockResult(scalar_val=1))
    with patch("app.services.rate_limit.get_db", mock_db):
        await check_rate_limit("alice", "upload", limit=10)  # count=1, limit=10 → ok


@pytest.mark.asyncio
async def test_check_rate_limit_at_limit_does_not_raise():
    from app.services.rate_limit import check_rate_limit
    mock_db, _, _ = make_db_mock(MockResult(scalar_val=10))
    with patch("app.services.rate_limit.get_db", mock_db):
        await check_rate_limit("alice", "upload", limit=10)  # count=limit → ok


@pytest.mark.asyncio
async def test_check_rate_limit_over_limit_raises_429():
    from fastapi import HTTPException
    from app.services.rate_limit import check_rate_limit
    mock_db, _, _ = make_db_mock(MockResult(scalar_val=11))
    with patch("app.services.rate_limit.get_db", mock_db):
        with pytest.raises(HTTPException) as exc_info:
            await check_rate_limit("alice", "upload", limit=10)
    assert exc_info.value.status_code == 429


@pytest.mark.asyncio
async def test_check_rate_limit_writes_to_db():
    from app.services.rate_limit import check_rate_limit
    mock_db, session, _ = make_db_mock(MockResult(scalar_val=1))
    with patch("app.services.rate_limit.get_db", mock_db):
        await check_rate_limit("alice", "upload", limit=10)
    session.execute.assert_called_once()


# ── per-endpoint dependency functions ────────────────────────────────────

@pytest.mark.asyncio
async def test_rate_limit_upload_passes_correct_endpoint():
    from app.services.rate_limit import rate_limit_upload
    mock_db, session, _ = make_db_mock(MockResult(scalar_val=1))
    with patch("app.services.rate_limit.get_db", mock_db):
        await rate_limit_upload(_make_request("alice"))
    call_args = session.execute.call_args[0][1]
    assert call_args["endpoint"] == "upload"
    assert call_args["identifier"] == "alice"


@pytest.mark.asyncio
async def test_rate_limit_query_passes_correct_endpoint():
    from app.services.rate_limit import rate_limit_query
    mock_db, session, _ = make_db_mock(MockResult(scalar_val=1))
    with patch("app.services.rate_limit.get_db", mock_db):
        await rate_limit_query(_make_request("bob"))
    call_args = session.execute.call_args[0][1]
    assert call_args["endpoint"] == "query"


@pytest.mark.asyncio
async def test_rate_limit_chat_passes_correct_endpoint():
    from app.services.rate_limit import rate_limit_chat
    mock_db, session, _ = make_db_mock(MockResult(scalar_val=1))
    with patch("app.services.rate_limit.get_db", mock_db):
        await rate_limit_chat(_make_request("carol"))
    call_args = session.execute.call_args[0][1]
    assert call_args["endpoint"] == "chat"


@pytest.mark.asyncio
async def test_rate_limit_recalibrate_passes_correct_endpoint():
    from app.services.rate_limit import rate_limit_recalibrate
    mock_db, session, _ = make_db_mock(MockResult(scalar_val=1))
    with patch("app.services.rate_limit.get_db", mock_db):
        await rate_limit_recalibrate(_make_request("dave"))
    call_args = session.execute.call_args[0][1]
    assert call_args["endpoint"] == "recalibrate"


@pytest.mark.asyncio
async def test_rate_limit_uses_anonymous_when_no_user():
    from app.services.rate_limit import rate_limit_upload
    req = MagicMock()
    del req.state.user  # simulate unauthenticated request
    req.state = MagicMock(spec=[])  # no .user attribute
    mock_db, session, _ = make_db_mock(MockResult(scalar_val=1))
    with patch("app.services.rate_limit.get_db", mock_db):
        await rate_limit_upload(req)
    call_args = session.execute.call_args[0][1]
    assert call_args["identifier"] == "anonymous"
