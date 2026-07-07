"""Tests for retention pruning (§11.9)."""
from unittest.mock import MagicMock, patch

import pytest

from app.services.wiki_state import prune_expired_revisions
from tests.fixtures.revisions import _Result, _Row, set_user_context


from contextlib import asynccontextmanager


class _SequencedDB:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    async def execute(self, statement, params=None):
        sql = str(getattr(statement, "text", statement))
        self.calls.append((sql, params or {}))
        return self.results.pop(0) if self.results else _Result()


def _make_db(results):
    db = _SequencedDB(results)

    @asynccontextmanager
    async def _get_db():
        yield db

    return _get_db, db


@pytest.mark.asyncio
async def test_prune_no_expired_actions_is_noop():
    get_db, db = _make_db([_Result(rows=[])])
    fake_s3 = MagicMock()
    with set_user_context(), \
         patch("app.services.wiki_state.get_db", get_db), \
         patch("app.services.wiki_state.s3", fake_s3):
        result = await prune_expired_revisions()
    assert result == {"actions_pruned": 0, "archives_deleted": 0}
    fake_s3.delete.assert_not_called()


@pytest.mark.asyncio
async def test_prune_removes_action_and_archive_objects():
    """Old action with S3-raw revisions → action deleted + archives swept."""
    results = [
        _Result(rows=[_Row(id="old-1"), _Row(id="old-2")]),
        _Result(rows=[
            _Row(content_before="archive/old-1/org/raw/a.pdf", content_after=None),
            _Row(content_before=None, content_after=None),
        ]),
        _Result(),
        _Result(rows=[
            _Row(content_before="archive/old-2/org/raw/b.pdf", content_after=None),
        ]),
        _Result(),
    ]
    get_db, db = _make_db(results)

    fake_s3 = MagicMock()
    with set_user_context(), \
         patch("app.services.wiki_state.get_db", get_db), \
         patch("app.services.wiki_state.s3", fake_s3):
        result = await prune_expired_revisions()

    deleted_keys = [c.args[0] for c in fake_s3.delete.call_args_list]
    assert "archive/old-1/org/raw/a.pdf" in deleted_keys
    assert "archive/old-2/org/raw/b.pdf" in deleted_keys
    assert result == {"actions_pruned": 2, "archives_deleted": 2}

    deletes = [s for s, _ in db.calls if "DELETE FROM wiki_actions" in s]
    assert len(deletes) == 2


@pytest.mark.asyncio
async def test_prune_skips_non_archive_paths():
    """Revisions whose content_before isn't an `archive/` key must be left alone.
    (e.g. wiki_page revisions hold inline content, not S3 paths.)
    """
    results = [
        _Result(rows=[_Row(id="old-1")]),
        _Result(rows=[_Row(content_before="some page body text",
                            content_after="another page body")]),
        _Result(),
    ]
    get_db, _ = _make_db(results)
    fake_s3 = MagicMock()
    with set_user_context(), \
         patch("app.services.wiki_state.get_db", get_db), \
         patch("app.services.wiki_state.s3", fake_s3):
        result = await prune_expired_revisions()

    fake_s3.delete.assert_not_called()
    assert result["archives_deleted"] == 0
    assert result["actions_pruned"] == 1


@pytest.mark.asyncio
async def test_prune_s3_failure_does_not_block_action_delete():
    """If S3 throws on a delete, we should still proceed to delete the action."""
    results = [
        _Result(rows=[_Row(id="old-1")]),
        _Result(rows=[_Row(content_before="archive/old-1/x", content_after=None)]),
        _Result(),
    ]
    get_db, db = _make_db(results)
    fake_s3 = MagicMock()
    fake_s3.delete.side_effect = RuntimeError("S3 down")
    with set_user_context(), \
         patch("app.services.wiki_state.get_db", get_db), \
         patch("app.services.wiki_state.s3", fake_s3):
        result = await prune_expired_revisions()

    # The action still got deleted despite the S3 error.
    deletes = [s for s, _ in db.calls if "DELETE FROM wiki_actions" in s]
    assert len(deletes) == 1
    assert result["actions_pruned"] == 1
    assert result["archives_deleted"] == 0  # nothing succeeded
