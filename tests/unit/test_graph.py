"""Unit tests for app/services/graph.py.

WikiGraph has significant pure-Python logic that can be tested without any
database or file-system access.  We build graph instances directly by
populating _adj, _meta, and _stem_index.
"""
import pytest
from unittest.mock import AsyncMock, patch

from app.services.graph import WikiGraph, _norm, _resolve


# ── Pure helper functions ─────────────────────────────────────────────────

def test_norm_lowercase():
    assert _norm("Foo Bar") == "foo-bar"


def test_norm_underscores():
    assert _norm("foo_bar") == "foo-bar"


def test_norm_hyphens_collapsed():
    assert _norm("foo--bar") == "foo-bar"


def test_resolve_relative():
    assert _resolve("concepts", "target.md") == "concepts/target.md"


def test_resolve_from_root():
    assert _resolve("", "target.md") == "target.md"


def test_resolve_normalises_dotdot():
    assert _resolve("concepts/sub", "../other.md") == "concepts/other.md"


# Frontmatter parsing moved to app.utils.parse_frontmatter (shared with the
# page store); it's covered by tests/unit/test_utils_frontmatter.py.


# ── WikiGraph helpers ─────────────────────────────────────────────────────

def _make_graph(adj=None, meta=None, stem_index=None) -> WikiGraph:
    g = WikiGraph()
    g._loaded = True
    g._adj = {k: set(v) for k, v in (adj or {}).items()}
    g._meta = meta or {}
    g._stem_index = stem_index or {}
    return g


# ── _extract_meta ─────────────────────────────────────────────────────────

def test_extract_meta_from_heading():
    g = WikiGraph()
    meta = g._extract_meta("concepts/foo.md", "# My Concept\n\nSome content.")
    assert meta["title"] == "My Concept"


def test_extract_meta_from_frontmatter():
    g = WikiGraph()
    text = "---\ntitle: Override Title\ntags: [alpha]\n---\n# Ignored\nContent."
    meta = g._extract_meta("concepts/foo.md", text)
    assert meta["title"] == "Override Title"
    assert "alpha" in meta["tags"]


def test_extract_meta_fallback_stem():
    g = WikiGraph()
    meta = g._extract_meta("concepts/my-concept.md", "No heading here.")
    assert meta["title"] == "My Concept"


# ── _extract_links ────────────────────────────────────────────────────────

def test_extract_links_markdown_link():
    g = _make_graph(
        meta={"concepts/target.md": {}},
        stem_index={"target": "concepts/target.md"},
    )
    links = g._extract_links("page.md", "[See target](concepts/target.md)", {})
    assert "concepts/target.md" in links


def test_extract_links_relative_markdown():
    g = _make_graph(
        meta={"concepts/target.md": {}},
        stem_index={},
    )
    links = g._extract_links("concepts/source.md", "[link](target.md)", {})
    assert "concepts/target.md" in links


def test_extract_links_wikilink():
    g = _make_graph(
        meta={"concepts/foo.md": {}},
        stem_index={"foo": "concepts/foo.md"},
    )
    links = g._extract_links("page.md", "See [[foo]] for details.", {})
    assert "concepts/foo.md" in links


def test_extract_links_wikilink_with_alias():
    g = _make_graph(
        meta={"concepts/foo.md": {}},
        stem_index={"foo": "concepts/foo.md"},
    )
    links = g._extract_links("page.md", "See [[foo|Foo Concept]] here.", {})
    assert "concepts/foo.md" in links


def test_extract_links_frontmatter_related():
    g = _make_graph(
        meta={"concepts/related.md": {}},
        stem_index={},
    )
    fm = {"related": ["concepts/related.md"]}
    links = g._extract_links("page.md", "Content.", fm)
    assert "concepts/related.md" in links


def test_extract_links_no_external_http():
    g = _make_graph(meta={}, stem_index={})
    links = g._extract_links("page.md", "[External](https://example.com/foo.md)", {})
    assert not links


def test_extract_links_only_known_targets():
    """Links to pages not in self._meta are filtered out."""
    g = _make_graph(meta={}, stem_index={})
    links = g._extract_links("page.md", "[Missing](missing.md)", {})
    assert not links


def test_extract_links_self_wikilink_excluded():
    """A page cannot link to itself via wikilink."""
    g = _make_graph(
        meta={"concepts/foo.md": {}},
        stem_index={"foo": "concepts/foo.md"},
    )
    links = g._extract_links("concepts/foo.md", "See [[foo]].", {})
    assert "concepts/foo.md" not in links


# ── neighbors ─────────────────────────────────────────────────────────────

def test_neighbors_hops_1_direct():
    g = _make_graph(
        adj={"a.md": ["b.md"], "b.md": ["c.md"], "c.md": []},
        meta={"a.md": {}, "b.md": {}, "c.md": {}},
    )
    result = g.neighbors(["a.md"], hops=1)
    assert "b.md" in result
    assert "c.md" not in result
    assert "a.md" not in result


def test_neighbors_hops_2():
    g = _make_graph(
        adj={"a.md": ["b.md"], "b.md": ["c.md"], "c.md": []},
        meta={"a.md": {}, "b.md": {}, "c.md": {}},
    )
    result = g.neighbors(["a.md"], hops=2)
    assert "b.md" in result
    assert "c.md" in result


def test_neighbors_bidirectional():
    """A→B means B appears in neighbors of A AND A appears in neighbors of B."""
    g = _make_graph(
        adj={"a.md": ["b.md"], "b.md": []},
        meta={"a.md": {}, "b.md": {}},
    )
    result = g.neighbors(["b.md"], hops=1)
    assert "a.md" in result


def test_neighbors_empty_start():
    g = _make_graph(adj={}, meta={})
    assert g.neighbors([], hops=1) == []


def test_neighbors_isolated_node():
    g = _make_graph(
        adj={"a.md": [], "b.md": []},
        meta={"a.md": {}, "b.md": {}},
    )
    result = g.neighbors(["a.md"], hops=1)
    assert "b.md" not in result


# ── as_dict ───────────────────────────────────────────────────────────────

def test_as_dict_nodes_and_edges():
    g = _make_graph(
        adj={"a.md": ["b.md"], "b.md": []},
        meta={
            "a.md": {"title": "A", "tags": [], "entities": []},
            "b.md": {"title": "B", "tags": [], "entities": []},
        },
    )
    data = g.as_dict()
    node_ids = {n["id"] for n in data["nodes"]}
    assert "a.md" in node_ids
    assert "b.md" in node_ids
    assert len(data["edges"]) == 1
    edge = data["edges"][0]
    assert {edge["source"], edge["target"]} == {"a.md", "b.md"}


def test_as_dict_no_duplicate_edges():
    """Mutual links A→B and B→A should produce exactly one undirected edge."""
    g = _make_graph(
        adj={"a.md": ["b.md"], "b.md": ["a.md"]},
        meta={
            "a.md": {"title": "A", "tags": [], "entities": []},
            "b.md": {"title": "B", "tags": [], "entities": []},
        },
    )
    data = g.as_dict()
    assert len(data["edges"]) == 1


def test_as_dict_degree_counts_both_in_and_out():
    # a.md → b.md → c.md
    # b.md has in-degree 1 (from a) and out-degree 1 (to c) → degree 2
    g = _make_graph(
        adj={"a.md": ["b.md"], "b.md": ["c.md"], "c.md": []},
        meta={
            "a.md": {"title": "A", "tags": [], "entities": []},
            "b.md": {"title": "B", "tags": [], "entities": []},
            "c.md": {"title": "C", "tags": [], "entities": []},
        },
    )
    data = g.as_dict()
    b_node = next(n for n in data["nodes"] if n["id"] == "b.md")
    assert b_node["degree"] == 2


def test_as_dict_empty_graph():
    g = _make_graph()
    data = g.as_dict()
    assert data == {"nodes": [], "edges": []}


# ── rebuild (mocked DB) ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_rebuild_empty_wiki():
    g = WikiGraph()
    with patch("app.services.wiki_db.list_wiki_pages_with_content", new_callable=AsyncMock) as mock_list, \
         patch.object(g, "_save", new_callable=AsyncMock):
        mock_list.return_value = []
        await g.rebuild()
    assert g._meta == {}
    assert g._adj == {}


@pytest.mark.asyncio
async def test_rebuild_builds_graph_from_pages():
    g = WikiGraph()
    pages = [
        ("concepts/foo.md", "---\ntitle: Foo\n---\n# Foo\n\nSee [Bar](bar.md)."),
        ("concepts/bar.md", "---\ntitle: Bar\n---\n# Bar\n\nContent."),
    ]
    with patch("app.services.wiki_db.list_wiki_pages_with_content", new_callable=AsyncMock) as mock_list, \
         patch.object(g, "_save", new_callable=AsyncMock):
        mock_list.return_value = pages
        await g.rebuild()
    assert "concepts/foo.md" in g._meta
    assert "concepts/bar.md" in g._meta
    # foo links to bar
    assert "concepts/bar.md" in g._adj.get("concepts/foo.md", set())


@pytest.mark.asyncio
async def test_rebuild_skips_special_files():
    g = WikiGraph()
    pages = [
        ("index.md", "# Index"),
        ("log.md", "## Log"),
        ("concepts/real.md", "# Real page"),
    ]
    with patch("app.services.wiki_db.list_wiki_pages_with_content", new_callable=AsyncMock) as mock_list, \
         patch.object(g, "_save", new_callable=AsyncMock):
        mock_list.return_value = pages
        await g.rebuild()
    assert "index.md" not in g._meta
    assert "log.md" not in g._meta
    assert "concepts/real.md" in g._meta


# ── stem index ────────────────────────────────────────────────────────────

def test_stem_index_built_on_rebuild():
    g = _make_graph(
        meta={"concepts/foo-bar.md": {"title": "Foo Bar", "tags": [], "entities": []}},
    )
    g._rebuild_stem_index()
    assert "foo-bar" in g._stem_index
    assert g._stem_index["foo-bar"] == "concepts/foo-bar.md"


# ── _save / _load ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_save_persists_to_db():
    g = _make_graph(
        adj={"a.md": ["b.md"]},
        meta={"a.md": {"title": "A"}, "b.md": {"title": "B"}},
    )
    with patch("app.services.wiki_db.set_wiki_file", new_callable=AsyncMock) as mock_set:
        await g._save()
    mock_set.assert_awaited_once()
    key, content = mock_set.call_args[0]
    assert key == "wiki/.graph.json"
    data = __import__("json").loads(content)
    assert "a.md" in data["adj"]
    assert "b.md" in data["adj"]["a.md"]


@pytest.mark.asyncio
async def test_load_populates_graph():
    import json
    raw = json.dumps({
        "adj": {"a.md": ["b.md"], "b.md": []},
        "meta": {"a.md": {"title": "A"}, "b.md": {"title": "B"}},
    })
    g = WikiGraph()
    with patch("app.services.wiki_db.get_wiki_file", new_callable=AsyncMock, return_value=raw):
        await g._load()
    assert "a.md" in g._adj
    assert "b.md" in g._adj["a.md"]
    assert g._meta["a.md"]["title"] == "A"


@pytest.mark.asyncio
async def test_load_empty_when_db_returns_none():
    g = WikiGraph()
    with patch("app.services.wiki_db.get_wiki_file", new_callable=AsyncMock, return_value=None):
        await g._load()
    assert g._adj == {}
    assert g._meta == {}


@pytest.mark.asyncio
async def test_ensure_loaded_calls_load_once():
    g = WikiGraph()
    with patch("app.services.wiki_db.get_wiki_file", new_callable=AsyncMock, return_value=None):
        await g.ensure_loaded()
        await g.ensure_loaded()  # second call should be no-op
    assert g._loaded is True


# ── generate_html ─────────────────────────────────────────────────────────

def test_generate_html_empty_graph_returns_placeholder():
    g = _make_graph()
    html = g.generate_html()
    assert "No wiki pages" in html


def test_generate_html_with_nodes_returns_html():
    g = _make_graph(
        adj={"concepts/a.md": ["concepts/b.md"], "concepts/b.md": []},
        meta={
            "concepts/a.md": {"title": "A", "tags": [], "entities": []},
            "concepts/b.md": {"title": "B", "tags": [], "entities": []},
        },
    )
    html = g.generate_html()
    assert "<html" in html.lower() or "body" in html.lower()


# ── update_pages ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_update_pages_rebuilds_for_touched():
    g = WikiGraph()
    g._loaded = True

    with patch("app.services.wiki_db.get_wiki_page_content", new_callable=AsyncMock, return_value="# A\n\nContent."), \
         patch.object(g, "_save", new_callable=AsyncMock):
        await g.update_pages(["concepts/a.md"])

    assert "concepts/a.md" in g._meta


@pytest.mark.asyncio
async def test_update_pages_removes_deleted():
    g = _make_graph(
        adj={"concepts/a.md": []},
        meta={"concepts/a.md": {"title": "A"}},
    )
    with patch("app.services.wiki_db.get_wiki_page_content", new_callable=AsyncMock, return_value=None), \
         patch.object(g, "_save", new_callable=AsyncMock):
        await g.update_pages(["concepts/a.md"])

    assert "concepts/a.md" not in g._meta
