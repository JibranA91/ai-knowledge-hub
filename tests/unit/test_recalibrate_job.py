"""Unit tests for the recalibration job state module."""
import pytest
from unittest.mock import patch

from tests.conftest import make_db_mock


@pytest.mark.asyncio
async def test_mark_stale_as_error_returns_rowcount():
    """Startup sweep updates running rows and returns the count."""
    from app.services.recalibrate_job import mark_stale_as_error
    mock_db, _, result = make_db_mock(rowcount=2)
    result.rowcount = 2
    with patch("app.db.get_db", mock_db):
        count = await mark_stale_as_error()
    assert count == 2


@pytest.mark.asyncio
async def test_mark_stale_as_error_returns_zero_when_no_stale_rows():
    from app.services.recalibrate_job import mark_stale_as_error
    mock_db, _, result = make_db_mock(rowcount=0)
    result.rowcount = 0
    with patch("app.db.get_db", mock_db):
        count = await mark_stale_as_error()
    assert count == 0


@pytest.mark.asyncio
async def test_mark_stale_as_error_handles_none_rowcount():
    """Some drivers return None instead of 0 — we must coerce to 0."""
    from app.services.recalibrate_job import mark_stale_as_error
    mock_db, _, result = make_db_mock()
    result.rowcount = None
    with patch("app.db.get_db", mock_db):
        count = await mark_stale_as_error()
    assert count == 0
