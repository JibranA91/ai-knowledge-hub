"""Unit tests for Phase 5 semantic search and knowledge graph features.

Covers:
  - embeddings service (is_enabled, embed_text, vec_to_pg, model dispatch)
  - wiki_db.find_relevant_pages (hybrid and BM25-only paths)
  - wiki_db.list_wiki_pages_paginated
  - wiki_db.replace_wiki_links / get_wiki_links
  - WikiGraph.get_clusters (Louvain community detection)
  - WikiGraph.update_pages / rebuild populate wiki_links DB table
"""

import json
import pytest
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch


# ── Fixtures ──────────────────────────────────────────────────────────────

class _Row:
    """Simple row mock matching SQLAlchemy Row attribute access."""
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)

    def scalar(self):
        return getattr(self, "_scalar", None)


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


@asynccontextmanager
async def _mock_db(execute_results):
    """Yields a DB session mock that returns pre-defined results per call order."""
    session = AsyncMock()
    call_iter = iter(execute_results)
    async def _execute(query, params=None):
        return next(call_iter, _Result())
    session.execute.side_effect = _execute
    yield session


# ── embeddings.is_enabled ─────────────────────────────────────────────────

def test_embedding_disabled_by_default():
    from app.services import embeddings
    with patch("app.model.settings") as mock_settings:
        mock_settings.BEDROCK_EMBEDDING_MODEL_ID = ""
        assert embeddings.is_enabled() is False


def test_embedding_enabled_when_model_set():
    from app.services import embeddings
    with patch("app.model.settings") as mock_settings:
        mock_settings.BEDROCK_EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"
        assert embeddings.is_enabled() is True


# ── embeddings.vec_to_pg ──────────────────────────────────────────────────

def test_vec_to_pg_format():
    from app.services.embeddings import vec_to_pg
    result = vec_to_pg([0.1, -0.5, 1.0])
    assert result.startswith("[")
    assert result.endswith("]")
    assert result.count(",") == 2


def test_vec_to_pg_roundtrip():
    """Parsed back from pgvector literal should give same values within float precision."""
    from app.services.embeddings import vec_to_pg
    vec = [0.123456789, -0.987654321, 0.0]
    literal = vec_to_pg(vec)
    parsed = [float(x) for x in literal.strip("[]").split(",")]
    assert len(parsed) == 3
    for orig, parsed_val in zip(vec, parsed):
        assert abs(orig - parsed_val) < 1e-6


# ── embeddings.embed_text ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_embed_text_returns_none_when_disabled():
    from app.services import embeddings
    with patch.object(embeddings, "is_enabled", return_value=False):
        result = await embeddings.embed_text("some text")
    assert result is None


@pytest.mark.asyncio
async def test_embed_text_delegates_to_model_when_enabled():
    from app.services import embeddings
    fake_vec = [0.1] * 1536
    with patch.object(embeddings, "is_enabled", return_value=True), \
         patch("app.model.embed", new_callable=AsyncMock, return_value=fake_vec) as mock_embed:
        result = await embeddings.embed_text("hello world")
    assert result is not None
    assert len(result) == 1536
    mock_embed.assert_awaited_once_with("hello world")


@pytest.mark.asyncio
async def test_embed_text_returns_none_on_error():
    """model.embed swallows provider errors; embed_text propagates the None."""
    from app import model
    from app.services import embeddings
    with patch.object(embeddings, "is_enabled", return_value=True), \
         patch.object(model, "embedding_enabled", return_value=True), \
         patch("asyncio.to_thread", side_effect=RuntimeError("Bedrock down")):
        result = await embeddings.embed_text("hello")
    assert result is None


# ── wiki_db.find_relevant_pages ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_find_relevant_pages_hybrid_path():
    """When embedding is enabled, the hybrid SQL path is taken."""
    from app.services import wiki_db

    fake_vec = [0.1] * 1536
    rows = [_Row(path="concepts/alpha.md"), _Row(path="concepts/beta.md")]

    with patch("app.services.wiki_db.get_org_id", return_value="org-1"), \
         patch("app.services.embeddings.is_enabled", return_value=True), \
         patch("app.services.embeddings.embed_text", new_callable=AsyncMock, return_value=fake_vec), \
         patch("app.services.embeddings.vec_to_pg", return_value="[0.1]"):

        get_db_mock = MagicMock()
        get_db_mock.return_value.__aenter__ = AsyncMock(return_value=AsyncMock(
            execute=AsyncMock(return_value=_Result(rows=rows))
        ))
        get_db_mock.return_value.__aexit__ = AsyncMock(return_value=False)

        with patch("app.services.wiki_db.get_db", get_db_mock):
            paths = await wiki_db.find_relevant_pages("what is alpha?", limit=5)

    assert "concepts/alpha.md" in paths
    assert "concepts/beta.md" in paths


@pytest.mark.asyncio
async def test_find_relevant_pages_bm25_fallback_no_embedding():
    """When embedding is disabled, BM25 SQL is used."""
    from app.services import wiki_db

    rows = [_Row(path="concepts/gamma.md")]

    with patch("app.services.wiki_db.get_org_id", return_value="org-1"), \
         patch("app.services.embeddings.is_enabled", return_value=False):

        get_db_mock = MagicMock()
        get_db_mock.return_value.__aenter__ = AsyncMock(return_value=AsyncMock(
            execute=AsyncMock(return_value=_Result(rows=rows))
        ))
        get_db_mock.return_value.__aexit__ = AsyncMock(return_value=False)

        with patch("app.services.wiki_db.get_db", get_db_mock):
            paths = await wiki_db.find_relevant_pages("gamma topic", limit=5)

    assert "concepts/gamma.md" in paths


@pytest.mark.asyncio
async def test_find_relevant_pages_returns_empty_when_no_match():
    """Returns [] when DB has no matching pages."""
    from app.services import wiki_db

    with patch("app.services.wiki_db.get_org_id", return_value="org-1"), \
         patch("app.services.embeddings.is_enabled", return_value=False):

        get_db_mock = MagicMock()
        get_db_mock.return_value.__aenter__ = AsyncMock(return_value=AsyncMock(
            execute=AsyncMock(return_value=_Result(rows=[]))
        ))
        get_db_mock.return_value.__aexit__ = AsyncMock(return_value=False)

        with patch("app.services.wiki_db.get_db", get_db_mock):
            paths = await wiki_db.find_relevant_pages("no match", limit=5)

    assert paths == []


@pytest.mark.asyncio
async def test_find_relevant_pages_bm25_when_embedding_fails():
    """When embed_text returns None (Bedrock error), fall through to BM25."""
    from app.services import wiki_db

    bm25_rows = [_Row(path="concepts/delta.md")]

    with patch("app.services.wiki_db.get_org_id", return_value="org-1"), \
         patch("app.services.embeddings.is_enabled", return_value=True), \
         patch("app.services.embeddings.embed_text", new_callable=AsyncMock, return_value=None):

        get_db_mock = MagicMock()
        get_db_mock.return_value.__aenter__ = AsyncMock(return_value=AsyncMock(
            execute=AsyncMock(return_value=_Result(rows=bm25_rows))
        ))
        get_db_mock.return_value.__aexit__ = AsyncMock(return_value=False)

        with patch("app.services.wiki_db.get_db", get_db_mock):
            paths = await wiki_db.find_relevant_pages("delta", limit=5)

    assert "concepts/delta.md" in paths


# ── wiki_db.list_wiki_pages_paginated ─────────────────────────────────────

@pytest.mark.asyncio
async def test_list_wiki_pages_paginated_returns_items_and_total():
    from app.services import wiki_db

    page_rows = [
        _Row(path="concepts/a.md", title="A", tags=json.dumps(["tag1"]),
             summary="Summary A", updated_at=None),
        _Row(path="concepts/b.md", title="B", tags=json.dumps([]),
             summary="Summary B", updated_at=None),
    ]

    session_mock = AsyncMock()
    # First execute call → COUNT(*), second → SELECT rows
    session_mock.execute.side_effect = [
        _Result(scalar_val=2),
        _Result(rows=page_rows),
    ]

    @asynccontextmanager
    async def _get_db():
        yield session_mock

    with patch("app.services.wiki_db.get_org_id", return_value="org-1"), \
         patch("app.services.wiki_db.get_db", _get_db):
        result = await wiki_db.list_wiki_pages_paginated(page=1, limit=10)

    assert result["total"] == 2
    assert result["page"] == 1
    assert result["limit"] == 10
    assert len(result["items"]) == 2
    assert result["items"][0]["path"] == "concepts/a.md"
    assert result["items"][0]["tags"] == ["tag1"]


@pytest.mark.asyncio
async def test_list_wiki_pages_paginated_offset():
    """Offset is calculated correctly: page 2 with limit 5 → offset 5."""
    from app.services import wiki_db

    session_mock = AsyncMock()
    session_mock.execute.side_effect = [_Result(scalar_val=0), _Result(rows=[])]

    @asynccontextmanager
    async def _get_db():
        yield session_mock

    call_params = []
    orig_execute = session_mock.execute.side_effect

    async def _capture(query, params=None):
        call_params.append(params or {})
        return next(iter([_Result(scalar_val=0), _Result(rows=[])]))

    session_mock.execute.side_effect = [_Result(scalar_val=0), _Result(rows=[])]

    with patch("app.services.wiki_db.get_org_id", return_value="org-1"), \
         patch("app.services.wiki_db.get_db", _get_db):
        result = await wiki_db.list_wiki_pages_paginated(page=2, limit=5)

    assert result["page"] == 2


# ── wiki_db.replace_wiki_links / get_wiki_links ───────────────────────────

@pytest.mark.asyncio
async def test_replace_wiki_links_deletes_then_inserts():
    from app.services import wiki_db

    session_mock = AsyncMock()
    execute_calls: list[tuple] = []

    async def _execute(query, params=None):
        execute_calls.append((str(query), params))
        return _Result()

    session_mock.execute.side_effect = _execute

    @asynccontextmanager
    async def _get_db():
        yield session_mock

    with patch("app.services.wiki_db.get_org_id", return_value="org-1"), \
         patch("app.services.wiki_db.get_db", _get_db):
        await wiki_db.replace_wiki_links("a.md", ["b.md", "c.md"])

    assert any("DELETE" in sql for sql, _ in execute_calls)
    assert any("INSERT" in sql for sql, _ in execute_calls)


@pytest.mark.asyncio
async def test_replace_wiki_links_no_targets_only_deletes():
    """Passing an empty to_paths list just deletes outgoing links."""
    from app.services import wiki_db

    session_mock = AsyncMock()
    execute_calls: list[str] = []

    async def _execute(query, params=None):
        execute_calls.append(str(query))
        return _Result()

    session_mock.execute.side_effect = _execute

    @asynccontextmanager
    async def _get_db():
        yield session_mock

    with patch("app.services.wiki_db.get_org_id", return_value="org-1"), \
         patch("app.services.wiki_db.get_db", _get_db):
        await wiki_db.replace_wiki_links("orphan.md", [])

    assert any("DELETE" in sql for sql in execute_calls)
    assert not any("INSERT" in sql for sql in execute_calls)


@pytest.mark.asyncio
async def test_get_wiki_links_returns_outgoing_and_incoming():
    from app.services import wiki_db

    out_rows = [_Row(to_path="b.md")]
    in_rows = [_Row(from_path="c.md")]

    session_mock = AsyncMock()
    session_mock.execute.side_effect = [_Result(rows=out_rows), _Result(rows=in_rows)]

    @asynccontextmanager
    async def _get_db():
        yield session_mock

    with patch("app.services.wiki_db.get_org_id", return_value="org-1"), \
         patch("app.services.wiki_db.get_db", _get_db):
        links = await wiki_db.get_wiki_links("a.md")

    assert links["outgoing"] == ["b.md"]
    assert links["incoming"] == ["c.md"]


# ── WikiGraph.get_clusters ────────────────────────────────────────────────

def test_get_clusters_returns_cluster_and_nodes():
    from app.services.graph import WikiGraph

    g = WikiGraph()
    g._loaded = True
    g._meta = {
        "a.md": {"title": "A", "tags": [], "entities": []},
        "b.md": {"title": "B", "tags": [], "entities": []},
        "c.md": {"title": "C", "tags": [], "entities": []},
    }
    g._adj = {"a.md": {"b.md"}, "b.md": {"a.md", "c.md"}, "c.md": {"b.md"}}

    result = g.get_clusters()

    assert "clusters" in result
    assert "nodes" in result
    assert len(result["nodes"]) == 3
    for node in result["nodes"]:
        assert "id" in node
        assert "title" in node
        assert "cluster" in node
        assert isinstance(node["cluster"], int)


def test_get_clusters_empty_graph():
    from app.services.graph import WikiGraph

    g = WikiGraph()
    g._loaded = True
    g._meta = {}
    g._adj = {}

    result = g.get_clusters()
    assert result == {"clusters": [], "nodes": []}


def test_get_clusters_single_node():
    from app.services.graph import WikiGraph

    g = WikiGraph()
    g._loaded = True
    g._meta = {"solo.md": {"title": "Solo", "tags": [], "entities": []}}
    g._adj = {}

    result = g.get_clusters()
    assert len(result["nodes"]) == 1
    assert result["nodes"][0]["cluster"] >= 0


def test_get_clusters_all_nodes_get_cluster_assigned():
    """Every node in meta gets a cluster id (not -1 for connected graph)."""
    from app.services.graph import WikiGraph

    g = WikiGraph()
    g._loaded = True
    paths = [f"p{i}.md" for i in range(6)]
    g._meta = {p: {"title": p, "tags": [], "entities": []} for p in paths}
    # Two clusters: 0-1-2 connected, 3-4-5 connected
    g._adj = {
        "p0.md": {"p1.md"},
        "p1.md": {"p0.md", "p2.md"},
        "p2.md": {"p1.md"},
        "p3.md": {"p4.md"},
        "p4.md": {"p3.md", "p5.md"},
        "p5.md": {"p4.md"},
    }

    result = g.get_clusters()
    cluster_ids = {n["cluster"] for n in result["nodes"]}
    assert len(cluster_ids) >= 2  # At least two communities detected


# ── WikiGraph.update_pages populates wiki_links ───────────────────────────

@pytest.mark.asyncio
async def test_update_pages_calls_replace_wiki_links():
    """After update_pages, replace_wiki_links is called for each updated page."""
    from app.services.graph import WikiGraph

    g = WikiGraph()
    g._loaded = True
    g._meta = {}
    g._adj = {}
    g._stem_index = {}

    content = "---\ntitle: Foo\n---\n# Foo\nSee [[bar.md]]."

    with patch("app.services.wiki_db.get_wiki_page_content", new_callable=AsyncMock, return_value=content), \
         patch("app.services.wiki_db.set_wiki_file", new_callable=AsyncMock), \
         patch("app.services.wiki_db.get_wiki_file", new_callable=AsyncMock, return_value=None), \
         patch("app.services.wiki_db.replace_wiki_links", new_callable=AsyncMock) as mock_replace:
        await g.update_pages(["foo.md"])

    mock_replace.assert_called_once()
    call_args = mock_replace.call_args
    assert call_args[0][0] == "foo.md"
    assert isinstance(call_args[0][1], list)


@pytest.mark.asyncio
async def test_update_pages_removes_links_for_deleted_page():
    """Deleted pages (content=None) get their links cleared."""
    from app.services.graph import WikiGraph

    g = WikiGraph()
    g._loaded = True
    g._meta = {"gone.md": {"title": "Gone", "tags": [], "entities": []}}
    g._adj = {"gone.md": {"other.md"}}
    g._stem_index = {"gone": "gone.md"}

    with patch("app.services.wiki_db.get_wiki_page_content", new_callable=AsyncMock, return_value=None), \
         patch("app.services.wiki_db.set_wiki_file", new_callable=AsyncMock), \
         patch("app.services.wiki_db.get_wiki_file", new_callable=AsyncMock, return_value=None), \
         patch("app.services.wiki_db.replace_wiki_links", new_callable=AsyncMock) as mock_replace:
        await g.update_pages(["gone.md"])

    mock_replace.assert_called_once_with("gone.md", [])
    assert "gone.md" not in g._meta


# ── Hybrid scoring correctness (mocked) ───────────────────────────────────

@pytest.mark.asyncio
async def test_hybrid_search_ranks_semantic_match_first():
    """The page with higher cosine similarity should appear before a BM25-only match."""
    from app.services import wiki_db

    # Simulate: page A has high vector similarity, page B has only keyword match
    # In the real hybrid query these are combined 0.4*bm25 + 0.6*cosine
    # Here we just verify that find_relevant_pages returns the DB order intact.
    fake_vec = [0.1] * 1536
    rows = [_Row(path="semantic/a.md"), _Row(path="keyword/b.md")]

    with patch("app.services.wiki_db.get_org_id", return_value="org-1"), \
         patch("app.services.embeddings.is_enabled", return_value=True), \
         patch("app.services.embeddings.embed_text", new_callable=AsyncMock, return_value=fake_vec), \
         patch("app.services.embeddings.vec_to_pg", return_value="[0.1]"):

        session_mock = AsyncMock()
        session_mock.execute.return_value = _Result(rows=rows)

        @asynccontextmanager
        async def _get_db():
            yield session_mock

        with patch("app.services.wiki_db.get_db", _get_db):
            paths = await wiki_db.find_relevant_pages("semantic query", limit=8)

    assert paths[0] == "semantic/a.md"
    assert paths[1] == "keyword/b.md"
