"""Unit tests for wiki_db helpers added during the index/log refactor.

Covers:
  - get_compact_index  — derives a path—title listing from wiki_pages
  - semantic_search_wiki — cosine-similarity pre-fetch for planner context
  - get_rendered_log  — renders audit_log rows as markdown
"""

import pytest
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch


# ── Helpers ───────────────────────────────────────────────────────────────

class _Row:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


class _Result:
    def __init__(self, rows=None, scalar_val=None):
        self._rows = rows or []
        self._scalar_val = scalar_val

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def scalar(self):
        return self._scalar_val


def _make_db(rows_per_call):
    """Return an async context manager whose session returns rows_per_call in order."""
    call_iter = iter(rows_per_call)

    session = AsyncMock()

    async def _execute(query, params=None):
        return next(call_iter, _Result())

    session.execute.side_effect = _execute

    @asynccontextmanager
    async def _get_db():
        yield session

    return _get_db, session


# ── get_compact_index ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_compact_index_returns_path_title_lines():
    from app.services.wiki_db import get_compact_index

    rows = [
        _Row(path="concepts/alpha.md", title="Alpha"),
        _Row(path="rca/issue-1.md", title="Issue One"),
    ]
    get_db, _ = _make_db([_Result(rows=rows)])

    with patch("app.services.wiki_db.get_org_id", return_value="org-1"), \
         patch("app.services.wiki_db.get_db", get_db):
        index = await get_compact_index()

    assert "concepts/alpha.md — Alpha" in index
    assert "rca/issue-1.md — Issue One" in index


@pytest.mark.asyncio
async def test_compact_index_empty_returns_placeholder():
    from app.services.wiki_db import get_compact_index

    get_db, _ = _make_db([_Result(rows=[])])

    with patch("app.services.wiki_db.get_org_id", return_value="org-1"), \
         patch("app.services.wiki_db.get_db", get_db):
        index = await get_compact_index()

    assert index == "No pages yet."


@pytest.mark.asyncio
async def test_compact_index_single_page():
    from app.services.wiki_db import get_compact_index

    rows = [_Row(path="index/only.md", title="Only Page")]
    get_db, _ = _make_db([_Result(rows=rows)])

    with patch("app.services.wiki_db.get_org_id", return_value="org-1"), \
         patch("app.services.wiki_db.get_db", get_db):
        index = await get_compact_index()

    lines = index.strip().splitlines()
    assert len(lines) == 1
    assert lines[0] == "index/only.md — Only Page"


# ── semantic_search_wiki ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_semantic_search_returns_empty_when_column_absent():
    """When embedding column does not exist, return [] without hitting DB."""
    from app.services.wiki_db import semantic_search_wiki

    with patch("app.services.wiki_db._embedding_col_exists", new_callable=AsyncMock, return_value=False):
        results = await semantic_search_wiki([0.1] * 10, top_k=5)

    assert results == []


@pytest.mark.asyncio
async def test_semantic_search_returns_ranked_pages(monkeypatch):
    from app import model
    monkeypatch.setattr(model.settings, "MODEL_EMBEDDING", "titanembedv1")
    from app.services.wiki_db import semantic_search_wiki

    rows = [
        _Row(path="concepts/a.md", title="A", content="content a", score=0.92),
        _Row(path="concepts/b.md", title="B", content="content b", score=0.75),
    ]
    get_db, _ = _make_db([_Result(rows=rows)])

    with patch("app.services.wiki_db._embedding_col_exists", new_callable=AsyncMock, return_value=True), \
         patch("app.services.wiki_db.get_org_id", return_value="org-1"), \
         patch("app.services.wiki_db.get_db", get_db), \
         patch("app.services.embeddings.vec_to_pg", return_value="[0.1]"):
        results = await semantic_search_wiki([0.1] * 1536, top_k=5)

    assert len(results) == 2
    assert results[0]["path"] == "concepts/a.md"
    assert results[0]["score"] == pytest.approx(0.92)
    assert results[1]["path"] == "concepts/b.md"
    assert "content" in results[0]
    assert "title" in results[0]


@pytest.mark.asyncio
async def test_semantic_search_returns_empty_when_no_matches():
    from app.services.wiki_db import semantic_search_wiki

    get_db, _ = _make_db([_Result(rows=[])])

    with patch("app.services.wiki_db._embedding_col_exists", new_callable=AsyncMock, return_value=True), \
         patch("app.services.wiki_db.get_org_id", return_value="org-1"), \
         patch("app.services.wiki_db.get_db", get_db), \
         patch("app.services.embeddings.vec_to_pg", return_value="[0.1]"):
        results = await semantic_search_wiki([0.1] * 10, top_k=5)

    assert results == []


# ── get_rendered_log ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_rendered_log_joins_entries_with_blank_lines():
    from app.services.wiki_db import get_rendered_log

    rows = [
        _Row(raw_text="## [2026-05-07] ingest\nIngested file.pdf"),
        _Row(raw_text="## [2026-05-06] recalibrate\nFixed stubs"),
    ]
    get_db, _ = _make_db([_Result(rows=rows)])

    with patch("app.services.wiki_db.get_org_id", return_value="org-1"), \
         patch("app.services.wiki_db.get_db", get_db):
        content = await get_rendered_log(limit=10)

    assert "## [2026-05-07] ingest" in content
    assert "## [2026-05-06] recalibrate" in content
    assert "\n\n" in content  # entries separated by blank line


@pytest.mark.asyncio
async def test_rendered_log_empty_returns_empty_string():
    from app.services.wiki_db import get_rendered_log

    get_db, _ = _make_db([_Result(rows=[])])

    with patch("app.services.wiki_db.get_org_id", return_value="org-1"), \
         patch("app.services.wiki_db.get_db", get_db):
        content = await get_rendered_log()

    assert content == ""


@pytest.mark.asyncio
async def test_rendered_log_single_entry():
    from app.services.wiki_db import get_rendered_log

    rows = [_Row(raw_text="## [2026-05-07] ingest\nOne entry")]
    get_db, _ = _make_db([_Result(rows=rows)])

    with patch("app.services.wiki_db.get_org_id", return_value="org-1"), \
         patch("app.services.wiki_db.get_db", get_db):
        content = await get_rendered_log()

    assert content == "## [2026-05-07] ingest\nOne entry"
    assert "\n\n" not in content  # no separator for single entry
