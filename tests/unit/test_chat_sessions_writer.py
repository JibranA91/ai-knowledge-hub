"""Unit tests for writer-mode helpers in chat_sessions: section patcher,
draft excerpt, pruning."""
import pytest
from unittest.mock import patch, AsyncMock
from contextlib import asynccontextmanager

from app.context import UserContext, current_user
from app.services.chat_sessions import (
    _find_section_bounds,
    _excerpt,
    ChatSession,
)
from tests.conftest import MockResult, MockRow, make_db_mock


def _set_ctx(org_id: str = "00000000-0000-0000-0000-000000000001",
             user_id: str = "00000000-0000-0000-0000-000000000099") -> None:
    current_user.set(UserContext(
        user_id=user_id, org_id=org_id, email="t@t.com", role="supervisor",
    ))


def _draft_row(draft_content="", draft_filename=""):
    return MockRow(
        session_id="sid-w",
        messages=[],
        summary="",
        mode="writer",
        draft_content=draft_content,
        draft_filename=draft_filename,
        user_id=None,
    )


# ── _excerpt ──────────────────────────────────────────────────────────────

def test_excerpt_empty():
    assert _excerpt("") == ""


def test_excerpt_collapses_whitespace():
    assert _excerpt("Hello\n\nworld   tab\there") == "Hello world tab here"


def test_excerpt_truncates_at_limit():
    long = "a" * 200
    assert _excerpt(long, max_chars=50) == "a" * 50


# ── _find_section_bounds ──────────────────────────────────────────────────

def test_find_section_bounds_single_match():
    content = "# Title\n\n## Overview\nbody A\n\n## Details\nbody B\n"
    matches, err = _find_section_bounds(content, "Overview")
    assert err is None
    assert len(matches) == 1
    start, body_end, level = matches[0]
    assert level == 2
    assert content[start:body_end] == "## Overview\nbody A\n\n"


def test_find_section_bounds_not_found():
    content = "# Title\n\n## Overview\nbody\n"
    matches, err = _find_section_bounds(content, "Nope")
    assert matches == []
    assert "not found" in err


def test_find_section_bounds_ambiguous():
    content = "# Title\n\n## Overview\nA\n\n## Other\n## Overview\nB\n"
    matches, err = _find_section_bounds(content, "Overview")
    assert len(matches) == 2
    assert "ambiguous" in err


def test_find_section_bounds_replaces_until_same_or_higher_level():
    """Sub-headings inside the target section are included in the section bounds."""
    content = (
        "# Title\n\n"
        "## Overview\n"
        "intro paragraph\n\n"
        "### Sub heading inside overview\n"
        "sub body\n\n"
        "## Next\n"
        "next body\n"
    )
    matches, err = _find_section_bounds(content, "Overview")
    assert err is None
    start, body_end, _ = matches[0]
    # Everything up to ## Next is part of Overview
    assert content[start:body_end].endswith("sub body\n\n")
    assert "Next" not in content[start:body_end]


def test_find_section_bounds_to_end_of_doc():
    content = "# Title\n\n## Final\nlast body\n"
    matches, err = _find_section_bounds(content, "Final")
    assert err is None
    start, body_end, _ = matches[0]
    assert body_end == len(content)


# ── patch_draft_section ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_patch_draft_section_happy_path():
    _set_ctx()
    from app.services.chat_sessions import patch_draft_section

    content = "# T\n\n## Overview\nold\n\n## Next\nnext body\n"
    row = _draft_row(draft_content=content)

    # First call (get_draft → SELECT) returns row; second (UPDATE) just returns OK
    session_mock = AsyncMock()
    select_result = MockResult()
    select_result._row = row
    update_result = MockResult(rowcount=1)
    session_mock.execute.side_effect = [select_result, update_result]

    @asynccontextmanager
    async def _get_db():
        yield session_mock

    with patch("app.services.chat_sessions.get_db", _get_db):
        new_content, err = await patch_draft_section("sid-w", "Overview", "## Overview\nbrand new")

    assert err is None
    assert "## Overview\nbrand new\n" in new_content
    assert "## Next\nnext body\n" in new_content
    assert "old" not in new_content


@pytest.mark.asyncio
async def test_patch_draft_section_ambiguous_returns_error():
    _set_ctx()
    from app.services.chat_sessions import patch_draft_section

    content = "# T\n\n## A\none\n\n## B\n\n## A\ntwo\n"
    row = _draft_row(draft_content=content)
    session_mock = AsyncMock()
    select_result = MockResult()
    select_result._row = row
    session_mock.execute.side_effect = [select_result]

    @asynccontextmanager
    async def _get_db():
        yield session_mock

    with patch("app.services.chat_sessions.get_db", _get_db):
        new_content, err = await patch_draft_section("sid-w", "A", "## A\nzzz")

    assert new_content is None
    assert err is not None
    assert "ambiguous" in err
    # Only the SELECT should have been called — no UPDATE on ambiguity
    assert session_mock.execute.call_count == 1


@pytest.mark.asyncio
async def test_patch_draft_section_not_found_returns_error():
    _set_ctx()
    from app.services.chat_sessions import patch_draft_section

    row = _draft_row(draft_content="# T\n\n## Other\nbody\n")
    session_mock = AsyncMock()
    select_result = MockResult()
    select_result._row = row
    session_mock.execute.side_effect = [select_result]

    @asynccontextmanager
    async def _get_db():
        yield session_mock

    with patch("app.services.chat_sessions.get_db", _get_db):
        new_content, err = await patch_draft_section("sid-w", "Missing", "## Missing\n")

    assert new_content is None
    assert "not found" in err


@pytest.mark.asyncio
async def test_patch_draft_section_empty_draft_returns_helpful_error():
    """If the draft is empty (no full DRAFT yet), section-patch must hint
    at the right next action rather than just saying 'heading not found'."""
    _set_ctx()
    from app.services.chat_sessions import patch_draft_section

    row = _draft_row(draft_content="")  # empty draft
    session_mock = AsyncMock()
    select_result = MockResult()
    select_result._row = row
    session_mock.execute.side_effect = [select_result]

    @asynccontextmanager
    async def _get_db():
        yield session_mock

    with patch("app.services.chat_sessions.get_db", _get_db):
        new_content, err = await patch_draft_section("sid-w", "Summary", "## Summary\nhi")

    assert new_content is None
    assert err is not None
    assert "empty" in err and "DRAFT_START" in err


@pytest.mark.asyncio
async def test_patch_draft_section_session_not_found():
    _set_ctx()
    from app.services.chat_sessions import patch_draft_section

    session_mock = AsyncMock()
    select_result = MockResult()
    select_result._row = None  # No session row
    session_mock.execute.return_value = select_result

    @asynccontextmanager
    async def _get_db():
        yield session_mock

    with patch("app.services.chat_sessions.get_db", _get_db):
        new_content, err = await patch_draft_section("nope", "X", "## X")

    assert new_content is None
    assert err == "session not found"


# ── prune_expired_writer_sessions ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_prune_returns_rowcount():
    _set_ctx()
    from app.services.chat_sessions import prune_expired_writer_sessions
    mock_db, _, result = make_db_mock(rowcount=5)
    result.rowcount = 5
    with patch("app.services.chat_sessions.get_db", mock_db):
        count = await prune_expired_writer_sessions(30)
    assert count == 5


@pytest.mark.asyncio
async def test_prune_zero_when_no_expired():
    _set_ctx()
    from app.services.chat_sessions import prune_expired_writer_sessions
    mock_db, _, result = make_db_mock(rowcount=0)
    result.rowcount = 0
    with patch("app.services.chat_sessions.get_db", mock_db):
        count = await prune_expired_writer_sessions(30)
    assert count == 0


# ── get_draft ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_draft_returns_content_and_filename():
    _set_ctx()
    from app.services.chat_sessions import get_draft
    row = _draft_row(draft_content="# Hello", draft_filename="hello.md")
    mock_db, _, result = make_db_mock()
    result._row = row
    with patch("app.services.chat_sessions.get_db", mock_db):
        d = await get_draft("sid-w")
    assert d == {
        "draft_content": "# Hello",
        "draft_filename": "hello.md",
        "draft_ready": False,  # _draft_row helper omits the field → defaults False
    }


@pytest.mark.asyncio
async def test_get_draft_returns_draft_ready_when_set():
    """When the underlying row has draft_ready=True, get_draft must surface it."""
    _set_ctx()
    from app.services.chat_sessions import get_draft
    row = MockRow(
        session_id="sid-w",
        messages=[],
        summary="",
        mode="writer",
        draft_content="# Hello",
        draft_filename="hello.md",
        draft_ready=True,
        user_id=None,
    )
    mock_db, _, result = make_db_mock()
    result._row = row
    with patch("app.services.chat_sessions.get_db", mock_db):
        d = await get_draft("sid-w")
    assert d["draft_ready"] is True


@pytest.mark.asyncio
async def test_get_draft_returns_none_for_unknown():
    _set_ctx()
    from app.services.chat_sessions import get_draft
    mock_db, _, result = make_db_mock()
    result._row = None
    with patch("app.services.chat_sessions.get_db", mock_db):
        d = await get_draft("nope")
    assert d is None


# ── ChatSession dataclass writer fields ───────────────────────────────────

def test_chatsession_defaults_to_chat_mode():
    s = ChatSession()
    assert s.mode == "chat"
    assert s.draft_content == ""
    assert s.draft_filename == ""
    assert s.user_id is None


def test_chatsession_writer_mode_assignable():
    s = ChatSession(mode="writer", draft_content="x", draft_filename="x.md",
                    user_id="u1")
    assert s.mode == "writer"
    assert s.draft_content == "x"
    assert s.user_id == "u1"
