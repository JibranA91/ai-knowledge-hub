"""Unit tests for app/services/chat_sessions.py."""
import pytest
from unittest.mock import patch

from tests.conftest import MockResult, MockRow, make_db_mock
from app.context import UserContext, current_user


def _set_ctx(org_id: str = "00000000-0000-0000-0000-000000000001") -> None:
    current_user.set(UserContext(
        user_id="00000000-0000-0000-0000-000000000099",
        org_id=org_id,
        email="test@test.com",
        role="admin",
    ))


def _session_row(session_id="sid-1", messages=None, summary="", mode="chat",
                 draft_content="", draft_filename="", user_id=None):
    return MockRow(
        session_id=session_id,
        messages=messages if messages is not None else [],
        summary=summary,
        mode=mode,
        draft_content=draft_content,
        draft_filename=draft_filename,
        user_id=user_id,
    )


# ── get_or_create ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_or_create_none_creates_new():
    _set_ctx()
    from app.services.chat_sessions import get_or_create
    mock_db, _, _ = make_db_mock()
    with patch("app.services.chat_sessions.get_db", mock_db):
        session = await get_or_create(None)
    assert session.session_id  # has a UUID
    assert session.messages == []
    assert session.summary == ""


@pytest.mark.asyncio
async def test_get_or_create_existing_id_returns_existing():
    _set_ctx()
    from app.services.chat_sessions import get_or_create
    row = _session_row(
        session_id="abc-123",
        messages=[{"role": "user", "text": "hello"}],
        summary="Prior chat",
    )
    mock_db, _, result = make_db_mock()
    result._row = row
    with patch("app.services.chat_sessions.get_db", mock_db):
        session = await get_or_create("abc-123")
    assert session.session_id == "abc-123"
    assert session.messages == [{"role": "user", "text": "hello"}]
    assert session.summary == "Prior chat"


@pytest.mark.asyncio
async def test_get_or_create_unknown_id_creates_new():
    """When session_id not found in DB, a new session is created (different UUID)."""
    _set_ctx()
    from app.services.chat_sessions import get_or_create

    call_count = 0

    async def _side_effect(*a, **kw):
        nonlocal call_count
        call_count += 1
        result = MockResult()
        result._row = None  # SELECT returns nothing
        return result

    from contextlib import asynccontextmanager
    from unittest.mock import AsyncMock

    session_mock = AsyncMock()
    session_mock.execute.side_effect = _side_effect

    @asynccontextmanager
    async def _get_db():
        yield session_mock

    with patch("app.services.chat_sessions.get_db", _get_db):
        new_session = await get_or_create("nonexistent-uuid")

    assert new_session.session_id != "nonexistent-uuid"
    assert new_session.session_id  # is a real UUID


@pytest.mark.asyncio
async def test_get_or_create_second_call_returns_same():
    """Calling get_or_create with the same existing session_id twice returns same session."""
    _set_ctx()
    from app.services.chat_sessions import get_or_create
    row = _session_row(session_id="same-sid", messages=[], summary="x")
    mock_db, _, result = make_db_mock()
    result._row = row
    with patch("app.services.chat_sessions.get_db", mock_db):
        s1 = await get_or_create("same-sid")
        s2 = await get_or_create("same-sid")
    assert s1.session_id == s2.session_id == "same-sid"


# ── save ──────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_save_calls_upsert():
    _set_ctx()
    from app.services.chat_sessions import save, ChatSession
    session = ChatSession(session_id="s1", messages=[{"role": "user", "text": "hi"}])
    mock_db, session_mock, _ = make_db_mock()
    with patch("app.services.chat_sessions.get_db", mock_db):
        await save(session)
    session_mock.execute.assert_called_once()


# ── clear ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_clear_returns_true_when_deleted():
    _set_ctx()
    from app.services.chat_sessions import clear
    mock_db, _, result = make_db_mock(rowcount=1)
    result.rowcount = 1
    with patch("app.services.chat_sessions.get_db", mock_db):
        assert await clear("existing-sid") is True


@pytest.mark.asyncio
async def test_clear_returns_false_when_not_found():
    _set_ctx()
    from app.services.chat_sessions import clear
    mock_db, _, result = make_db_mock(rowcount=0)
    result.rowcount = 0
    with patch("app.services.chat_sessions.get_db", mock_db):
        assert await clear("no-such-sid") is False


# ── constants ─────────────────────────────────────────────────────────────

def test_summarize_threshold():
    from app.services.chat_sessions import SUMMARIZE_AFTER, KEEP_RECENT
    assert SUMMARIZE_AFTER == 10
    assert KEEP_RECENT == 4
    assert KEEP_RECENT < SUMMARIZE_AFTER
