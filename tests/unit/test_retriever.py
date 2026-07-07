"""Unit tests for the retriever.py script bundled inside wiki exports.

The script is embedded as _RETRIEVER_TEMPLATE in wiki_engine.py. We compile
and exec it into a throw-away module so we can call its functions directly
without writing to disk.
"""
import types

import pytest


# ── Bootstrap the retriever module from the template string ───────────────

@pytest.fixture(scope="module")
def retriever():
    from app.services.wiki_engine import _RETRIEVER_TEMPLATE
    mod = types.ModuleType("retriever")
    exec(compile(_RETRIEVER_TEMPLATE, "retriever.py", "exec"), mod.__dict__)
    return mod


@pytest.fixture(autouse=True)
def reset_wiki_cache(retriever):
    """Clear the module-level cache between tests."""
    retriever._wiki_cache = None
    yield
    retriever._wiki_cache = None


# ── Helpers ────────────────────────────────────────────────────────────────

def _make_page(path: str, content: str, embedding=None) -> dict:
    return {"path": path, "title": path.split("/")[-1].replace(".md", ""), "content": content, "embedding": embedding}


# ── _tokenize ─────────────────────────────────────────────────────────────

def test_tokenize_lowercases_and_splits(retriever):
    tokens = retriever._tokenize("Hello World! It's 2026.")
    assert "hello" in tokens
    assert "world" in tokens
    assert "2026" in tokens


def test_tokenize_empty_string(retriever):
    assert retriever._tokenize("") == []


# ── _bm25_score ───────────────────────────────────────────────────────────

def test_bm25_score_higher_for_matching_page(retriever):
    pages = [
        _make_page("vacation.md", "Employees get 20 days of vacation leave per year."),
        _make_page("security.md", "All systems must use two-factor authentication."),
    ]
    scores = retriever._bm25_score(pages, ["vacation", "leave"])
    assert scores[0] > scores[1]


def test_bm25_score_all_zero_when_no_terms_match(retriever):
    pages = [_make_page("a.md", "unrelated content here")]
    scores = retriever._bm25_score(pages, ["xyzzy", "foobar"])
    assert scores[0] == 0.0


def test_bm25_score_returns_one_score_per_page(retriever):
    pages = [_make_page(f"page{i}.md", f"content {i}") for i in range(5)]
    scores = retriever._bm25_score(pages, ["content"])
    assert len(scores) == 5


# ── _cosine_score ─────────────────────────────────────────────────────────

def test_cosine_score_returns_zeros_for_pages_without_embeddings(retriever):
    pages = [_make_page("a.md", "content")]
    embeddings = {}
    scores = retriever._cosine_score([1.0, 0.0], pages, embeddings)
    assert scores == [0.0]


def test_cosine_score_identical_vectors_give_one(retriever):
    pytest.importorskip("numpy")
    pages = [_make_page("a.md", "content", embedding=[1.0, 0.0])]
    embeddings = {"a.md": [1.0, 0.0]}
    scores = retriever._cosine_score([1.0, 0.0], pages, embeddings)
    assert abs(scores[0] - 1.0) < 1e-6


def test_cosine_score_orthogonal_vectors_give_zero(retriever):
    pytest.importorskip("numpy")
    pages = [_make_page("a.md", "content", embedding=[0.0, 1.0])]
    embeddings = {"a.md": [0.0, 1.0]}
    scores = retriever._cosine_score([1.0, 0.0], pages, embeddings)
    assert abs(scores[0]) < 1e-6


def test_cosine_score_zero_query_vector_returns_zeros(retriever):
    pytest.importorskip("numpy")
    pages = [_make_page("a.md", "content", embedding=[1.0, 0.0])]
    embeddings = {"a.md": [1.0, 0.0]}
    scores = retriever._cosine_score([0.0, 0.0], pages, embeddings)
    assert scores == [0.0]


# ── find_relevant ─────────────────────────────────────────────────────────

def test_find_relevant_returns_matching_page_first(retriever):
    retriever._wiki_cache = {
        "pages": [
            _make_page("vacation.md", "Employees get 20 days vacation leave per year."),
            _make_page("security.md", "All systems must use two-factor authentication."),
        ],
        "graph": {},
        "embeddings": {},
    }
    results = retriever.find_relevant("how many vacation days", top_k=2)
    assert results[0]["path"] == "vacation.md"


def test_find_relevant_returns_empty_for_empty_wiki(retriever):
    retriever._wiki_cache = {"pages": [], "graph": {}, "embeddings": {}}
    results = retriever.find_relevant("anything")
    assert results == []


def test_find_relevant_respects_top_k(retriever):
    retriever._wiki_cache = {
        "pages": [_make_page(f"page{i}.md", f"content about topic {i}") for i in range(10)],
        "graph": {},
        "embeddings": {},
    }
    results = retriever.find_relevant("content topic", top_k=3)
    assert len(results) <= 3


def test_find_relevant_bm25_only_without_query_vector(retriever):
    retriever._wiki_cache = {
        "pages": [
            _make_page("a.md", "remote work policy details"),
            _make_page("b.md", "office snacks and kitchen rules"),
        ],
        "graph": {},
        "embeddings": {},
    }
    results = retriever.find_relevant("remote work policy")
    assert results[0]["path"] == "a.md"


# ── expand_with_graph ────────────────────────────────────────────────────

def test_expand_with_graph_adds_direct_neighbors(retriever):
    retriever._wiki_cache = {
        "pages": [
            _make_page("concepts/a.md", "concept A"),
            _make_page("concepts/b.md", "concept B"),
        ],
        "graph": {"concepts/a.md": ["concepts/b.md"]},
        "embeddings": {},
    }
    expanded = retriever.expand_with_graph(["concepts/a.md"], hops=1)
    assert "concepts/b.md" in expanded


def test_expand_with_graph_no_duplicates(retriever):
    retriever._wiki_cache = {
        "pages": [_make_page("a.md", "a"), _make_page("b.md", "b")],
        "graph": {"a.md": ["b.md"]},
        "embeddings": {},
    }
    expanded = retriever.expand_with_graph(["a.md", "b.md"], hops=1)
    assert expanded.count("b.md") == 1


def test_expand_with_graph_zero_hops_returns_unchanged(retriever):
    retriever._wiki_cache = {
        "pages": [_make_page("a.md", "a"), _make_page("b.md", "b")],
        "graph": {"a.md": ["b.md"]},
        "embeddings": {},
    }
    expanded = retriever.expand_with_graph(["a.md"], hops=0)
    assert expanded == ["a.md"]


# ── retrieve ──────────────────────────────────────────────────────────────

def test_retrieve_returns_list_of_dicts(retriever):
    retriever._wiki_cache = {
        "pages": [_make_page("concepts/a.md", "content about alpha")],
        "graph": {},
        "embeddings": {},
    }
    results = retriever.retrieve("alpha")
    assert isinstance(results, list)
    if results:
        assert {"path", "title", "content", "score"} <= set(results[0].keys())


def test_retrieve_appends_graph_expanded_pages(retriever):
    retriever._wiki_cache = {
        "pages": [
            _make_page("a.md", "primary topic content alpha"),
            _make_page("b.md", "related but different content"),
        ],
        "graph": {"a.md": ["b.md"]},
        "embeddings": {},
    }
    results = retriever.retrieve("alpha", graph_hops=1)
    paths = [r["path"] for r in results]
    assert "a.md" in paths
    assert "b.md" in paths


def test_retrieve_graph_expanded_pages_have_zero_score(retriever):
    retriever._wiki_cache = {
        "pages": [
            _make_page("a.md", "alpha content here"),
            _make_page("b.md", "completely unrelated"),
        ],
        "graph": {"a.md": ["b.md"]},
        "embeddings": {},
    }
    results = retriever.retrieve("alpha", graph_hops=1)
    path_score = {r["path"]: r["score"] for r in results}
    assert path_score.get("b.md", -1) == 0.0


def test_retrieve_empty_wiki_returns_empty(retriever):
    retriever._wiki_cache = {"pages": [], "graph": {}, "embeddings": {}}
    assert retriever.retrieve("anything") == []
