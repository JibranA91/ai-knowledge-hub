"""Unit tests for app/services/jobs.py.

In-process task helpers (get_task/set_task) need no mocks.
All DB-touching functions are mocked via make_db_mock.
"""
import asyncio
import pytest
from unittest.mock import patch, MagicMock

from tests.conftest import MockResult, MockRow, make_db_mock
from app.context import UserContext, current_user


def _set_ctx(org_id: str = "00000000-0000-0000-0000-000000000001") -> None:
    current_user.set(UserContext(
        user_id="00000000-0000-0000-0000-000000000099",
        org_id=org_id,
        email="test@test.com",
        role="admin",
    ))


def _make_job_row(**overrides):
    defaults = dict(
        filename="doc.pdf",
        org_id=None,
        user_id=None,
        status="queued",
        message="",
        plan=[],
        conflicts=[],
        log_entry="",
        doc_text="",
        plan_chat_history=[],
        pages_created=[],
        pages_updated=[],
    )
    defaults.update(overrides)
    return MockRow(**defaults)


# ── get_task / set_task ───────────────────────────────────────────────────

def test_get_task_unknown():
    from app.services.jobs import get_task
    assert get_task("nonexistent.pdf") is None


def test_set_task_and_get():
    from app.services.jobs import get_task, set_task
    task = MagicMock(spec=asyncio.Task)
    set_task("my.pdf", task)
    assert get_task("my.pdf") is task
    set_task("my.pdf", None)  # cleanup


def test_set_task_none_removes():
    from app.services.jobs import get_task, set_task
    task = MagicMock(spec=asyncio.Task)
    set_task("rm.pdf", task)
    set_task("rm.pdf", None)
    assert get_task("rm.pdf") is None


# ── enqueue ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_enqueue_returns_queued_job():
    _set_ctx()
    from app.services.jobs import enqueue
    mock_db, _, _ = make_db_mock()
    with patch("app.services.jobs.get_db", mock_db):
        job = await enqueue("new.pdf")
    assert job.filename == "new.pdf"
    assert job.status == "queued"


@pytest.mark.asyncio
async def test_enqueue_writes_to_db():
    _set_ctx()
    from app.services.jobs import enqueue
    mock_db, session, _ = make_db_mock()
    with patch("app.services.jobs.get_db", mock_db):
        await enqueue("new.pdf")
    session.execute.assert_called_once()


# ── get ───────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_returns_none_for_unknown():
    _set_ctx()
    from app.services.jobs import get
    mock_db, _, result = make_db_mock()
    result._row = None
    with patch("app.services.jobs.get_db", mock_db):
        job = await get("missing.pdf")
    assert job is None


@pytest.mark.asyncio
async def test_get_returns_job():
    _set_ctx()
    from app.services.jobs import get
    row = _make_job_row(filename="existing.pdf", status="done")
    mock_db, _, result = make_db_mock()
    result._row = row
    with patch("app.services.jobs.get_db", mock_db):
        job = await get("existing.pdf")
    assert job is not None
    assert job.filename == "existing.pdf"
    assert job.status == "done"


@pytest.mark.asyncio
async def test_get_handles_null_fields():
    """DB rows with None in JSONB fields should be treated as empty collections."""
    _set_ctx()
    from app.services.jobs import get
    row = _make_job_row(plan=None, conflicts=None, pages_created=None, pages_updated=None)
    mock_db, _, result = make_db_mock()
    result._row = row
    with patch("app.services.jobs.get_db", mock_db):
        job = await get("partial.pdf")
    assert job.plan == []
    assert job.conflicts == []
    assert job.pages_created == []
    assert job.pages_updated == []


# ── save ──────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_save_updates_db():
    _set_ctx()
    from app.services.jobs import save, IngestJob
    job = IngestJob(filename="save.pdf", status="processing")
    mock_db, session, _ = make_db_mock()
    with patch("app.services.jobs.get_db", mock_db):
        await save(job)
    session.execute.assert_called_once()


# ── cancel ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_cancel_returns_false_for_unknown():
    _set_ctx()
    from app.services.jobs import cancel
    mock_db, _, result = make_db_mock()
    result._row = None
    with patch("app.services.jobs.get_db", mock_db):
        assert await cancel("missing.pdf") is False


@pytest.mark.asyncio
async def test_cancel_sets_status_cancelled():
    _set_ctx()
    from app.services.jobs import cancel, set_task
    row = _make_job_row(filename="running.pdf", status="processing")

    call_count = 0

    async def _side_effect(*a, **kw):
        nonlocal call_count
        call_count += 1
        result = MagicMock()
        result.rowcount = 1
        if call_count == 1:
            result.fetchone.return_value = row
        else:
            result.fetchone.return_value = None
        return result

    from contextlib import asynccontextmanager
    from unittest.mock import AsyncMock

    session = AsyncMock()
    session.execute.side_effect = _side_effect

    @asynccontextmanager
    async def _get_db():
        yield session

    with patch("app.services.jobs.get_db", _get_db):
        result = await cancel("running.pdf")

    assert result is True
    # The second execute call is save(), which serialises status=cancelled
    assert call_count == 2


@pytest.mark.asyncio
async def test_cancel_cancels_asyncio_task():
    _set_ctx()
    from app.services.jobs import cancel, set_task, get_task

    async def _long():
        await asyncio.sleep(999)

    task = asyncio.create_task(_long())
    set_task("bg.pdf", task)

    row = _make_job_row(filename="bg.pdf", status="processing")

    call_count = 0

    async def _side_effect(*a, **kw):
        nonlocal call_count
        call_count += 1
        result = MagicMock()
        result.rowcount = 1
        result.fetchone.return_value = row if call_count == 1 else None
        return result

    from contextlib import asynccontextmanager
    from unittest.mock import AsyncMock

    session = AsyncMock()
    session.execute.side_effect = _side_effect

    @asynccontextmanager
    async def _get_db():
        yield session

    with patch("app.services.jobs.get_db", _get_db):
        await cancel("bg.pdf")

    # Allow event loop to process the cancellation
    await asyncio.sleep(0)
    assert task.cancelled()
    assert get_task("bg.pdf") is None


# ── request_cancel / is_cancel_requested ──────────────────────────────────

@pytest.mark.asyncio
async def test_request_cancel_unknown_returns_false():
    _set_ctx()
    from app.services.jobs import request_cancel
    mock_db, _, result = make_db_mock()
    result._row = None
    with patch("app.services.jobs.get_db", mock_db):
        assert await request_cancel("missing.pdf") is False


@pytest.mark.asyncio
async def test_request_cancel_flips_queued_write_to_cancelled():
    _set_ctx()
    from app.services.jobs import request_cancel
    row = _make_job_row(filename="qw.pdf", status="queued_write")

    captured = {}
    call_count = 0

    async def _side_effect(stmt, params=None, *a, **kw):
        nonlocal call_count
        call_count += 1
        result = MagicMock()
        result.rowcount = 1
        if call_count == 1:
            result.fetchone.return_value = row  # SELECT in get()
        else:
            captured.update(params or {})       # UPDATE in save()
            result.fetchone.return_value = None
        return result

    from contextlib import asynccontextmanager
    from unittest.mock import AsyncMock
    session = AsyncMock()
    session.execute.side_effect = _side_effect

    @asynccontextmanager
    async def _get_db():
        yield session

    with patch("app.services.jobs.get_db", _get_db):
        assert await request_cancel("qw.pdf") is True

    # The flag is persisted and a non-writing job flips straight to cancelled.
    assert captured.get("cancel_requested") is True
    assert captured.get("status") == "cancelled"


@pytest.mark.asyncio
async def test_request_cancel_writing_keeps_status_sets_flag():
    """A job actively writing isn't flipped here — the write loop aborts it."""
    _set_ctx()
    from app.services.jobs import request_cancel
    row = _make_job_row(filename="wr.pdf", status="writing")

    captured = {}
    call_count = 0

    async def _side_effect(stmt, params=None, *a, **kw):
        nonlocal call_count
        call_count += 1
        result = MagicMock()
        result.rowcount = 1
        if call_count == 1:
            result.fetchone.return_value = row
        else:
            captured.update(params or {})
            result.fetchone.return_value = None
        return result

    from contextlib import asynccontextmanager
    from unittest.mock import AsyncMock
    session = AsyncMock()
    session.execute.side_effect = _side_effect

    @asynccontextmanager
    async def _get_db():
        yield session

    with patch("app.services.jobs.get_db", _get_db):
        assert await request_cancel("wr.pdf") is True

    assert captured.get("cancel_requested") is True
    assert captured.get("status") == "writing"


@pytest.mark.asyncio
async def test_is_cancel_requested_reads_db():
    _set_ctx()
    from app.services.jobs import is_cancel_requested
    mock_db, _, result = make_db_mock()
    result._row = MockRow(cancel_requested=True)
    with patch("app.services.jobs.get_db", mock_db):
        assert await is_cancel_requested("x.pdf") is True

    result._row = MockRow(cancel_requested=False)
    with patch("app.services.jobs.get_db", mock_db):
        assert await is_cancel_requested("x.pdf") is False

    result._row = None
    with patch("app.services.jobs.get_db", mock_db):
        assert await is_cancel_requested("x.pdf") is False


# ── status transition helpers ─────────────────────────────────────────────

def test_ingest_job_defaults():
    from app.services.jobs import IngestJob
    job = IngestJob(filename="x.pdf")
    assert job.status == "queued"
    assert job.plan == []
    assert job.conflicts == []
    assert job.pages_created == []
    assert job.pages_updated == []
    assert job.cancel_requested is False
    assert not hasattr(job, "index_additions")
