"""Shared helpers for change-tracking + revert tests.

Provides:
  - `RecordingDB`: a fake async DB session that records every executed
    statement + params so tests can assert on what was written.
  - `make_recording_get_db`: builds an async context manager drop-in for
    `app.db.get_db`, optionally pre-loaded with rows for SELECTs.
  - `set_user_context` / `clear_user_context`: helpers to set the auth
    ContextVar during tests so `get_org_id()` doesn't raise.
"""
from contextlib import asynccontextmanager, contextmanager
from typing import Any
from unittest.mock import AsyncMock

from app.context import UserContext, current_action, current_user


# ── Recording DB ──────────────────────────────────────────────────────────

class _Row:
    def __init__(self, **kw: Any) -> None:
        for k, v in kw.items():
            setattr(self, k, v)


class _Result:
    def __init__(self, rows: list | None = None, scalar_val: Any = None) -> None:
        self._rows = rows or []
        self._scalar_val = scalar_val

    def fetchall(self) -> list:
        return self._rows

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def scalar(self):
        return self._scalar_val


class RecordingDB:
    """Captures every executed SQL statement and its bound parameters."""

    def __init__(self, select_results: list[_Result] | None = None) -> None:
        self.calls: list[tuple[str, dict]] = []
        self._select_iter = iter(select_results or [])

    async def execute(self, statement, params: dict | None = None):
        sql = str(getattr(statement, "text", statement))
        self.calls.append((sql, params or {}))
        if _looks_like_select(sql):
            return next(self._select_iter, _Result())
        return _Result()

    def queries_matching(self, fragment: str) -> list[tuple[str, dict]]:
        return [(s, p) for s, p in self.calls if fragment in s]


def _looks_like_select(sql: str) -> bool:
    head = sql.lstrip().upper()
    return head.startswith("SELECT") or head.startswith("WITH")


def make_recording_get_db(select_results: list[_Result] | None = None):
    """Return `(get_db_fn, db_instance)` so tests can patch + inspect.

    Use::

        get_db, db = make_recording_get_db([_Result(rows=[row])])
        with patch("app.services.wiki_db.get_db", get_db), \\
             patch("app.services.wiki_state.get_db", get_db):
            await upsert_wiki_page("a.md", "content")
        assert any("INSERT INTO wiki_revisions" in s for s, _ in db.calls)
    """
    db = RecordingDB(select_results)

    @asynccontextmanager
    async def _get_db():
        yield db

    return _get_db, db


# ── User context helpers ──────────────────────────────────────────────────

@contextmanager
def set_user_context(*, org_id: str = "org-1", user_id: str = "user-1", role: str = "member"):
    """Set the auth ContextVar for the duration of the block."""
    token = current_user.set(UserContext(
        user_id=user_id,
        org_id=org_id,
        email="test@example.com",
        role=role,
    ))
    try:
        yield
    finally:
        current_user.reset(token)


@contextmanager
def set_action_context(action_id: str | None):
    """Set the current_action ContextVar for the duration of the block."""
    token = current_action.set(action_id)
    try:
        yield
    finally:
        current_action.reset(token)


__all__ = [
    "RecordingDB",
    "_Result",
    "_Row",
    "make_recording_get_db",
    "set_user_context",
    "set_action_context",
]
