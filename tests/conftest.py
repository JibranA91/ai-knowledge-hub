"""Shared test helpers used by both unit and integration suites."""
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock


class MockResult:
    """Mimics a SQLAlchemy CursorResult just enough for tests."""

    def __init__(self, scalar_val=None, row=None, rows=None, rowcount=0):
        self._scalar = scalar_val
        self._row = row
        self._rows = rows or []
        self.rowcount = rowcount

    def scalar(self):
        return self._scalar

    def fetchone(self):
        return self._row

    def fetchall(self):
        return self._rows


class MockRow:
    """Simple attribute bag that mimics a SQLAlchemy Row."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def make_db_mock(result: MockResult | None = None, rowcount: int = 0):
    """Return (get_db_fn, session_mock, result_mock).

    get_db_fn is a drop-in for ``app.db.get_db`` — an async context manager
    that yields *session_mock*.  Patch the service's local reference to
    ``get_db`` with *get_db_fn* in your tests.

    Example::

        mock_db, session, res = make_db_mock(MockResult(scalar_val=1))
        with patch("app.services.auth.get_db", mock_db):
            assert await validate_token("t") is True
    """
    if result is None:
        result = MockResult(rowcount=rowcount)

    session = AsyncMock()
    session.execute.return_value = result

    @asynccontextmanager
    async def _get_db():
        yield session

    return _get_db, session, result
