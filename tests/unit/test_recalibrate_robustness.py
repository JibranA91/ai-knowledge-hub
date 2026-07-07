"""Unit tests for recalibration robustness fixes.

Covers three hardening changes:
1. analyze_node caps the flagged shortlist sent to the LLM (_ANALYZE_MAX_SHORTLIST)
   so the prompt stays bounded at any wiki size.
2. analyze_node drops improve/create items that overlap delete/rename targets,
   so apply_deletions → improve_pages can't resurrect a deleted/renamed page.
3. _release_recalibrate_lock persists the 'error' status on the crash/cancel
   path so a failed run doesn't leave the org write-locked.
"""
import json
import time

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def _base_state(**overrides):
    state = {
        "wiki_pages": {},
        "schema": "",
        "index": "",
        "deleted_files": [],
        "fact_instructions": "",
        "triage_candidates": [],
        "improvement_plan": [],
        "new_page_plan": [],
        "delete_plan": [],
        "rename_plan": [],
        "pages_written": [],
        "pages_deleted": [],
        "pages_renamed": [],
        "errors": [],
        "log_entry": "",
        "_t_start": time.time(),
    }
    state.update(overrides)
    return state


def _mock_graph(adj=None):
    """A get_graph() stand-in whose ensure_loaded() yields an object with ._adj."""
    g = MagicMock()
    g._adj = adj or {}
    graph_holder = MagicMock()
    graph_holder.ensure_loaded = AsyncMock(return_value=g)
    return MagicMock(return_value=graph_holder)


def _llm_capturing(response_json: dict):
    """Return (llm_factory, captured) where captured['messages'] holds the last
    ainvoke call's message list and captured['calls'] holds every call's message
    list (analyze now fans out one call per batch). The LLM always responds with
    response_json."""
    captured = {}

    async def _ainvoke(messages):
        captured["messages"] = messages
        captured.setdefault("calls", []).append(messages)
        resp = MagicMock()
        resp.content = json.dumps(response_json)
        resp.response_metadata = {}
        return resp

    llm = MagicMock()
    llm.ainvoke = AsyncMock(side_effect=_ainvoke)
    return MagicMock(return_value=llm), captured


async def _run_analyze(state, *, response_json, adj=None):
    """Build the graph with a captured-LLM + mock graph, run only analyze_node."""
    llm_factory, captured = _llm_capturing(response_json)
    with (
        patch("app.services.recalibrate_agent.make_chat_llm", llm_factory),
        patch("app.services.recalibrate_agent.recalibrate_job") as mock_job_mod,
        patch("app.services.graph.get_graph", _mock_graph(adj)),
    ):
        mock_job_mod.get.return_value = MagicMock()
        from app.services.recalibrate_agent import build_recalibrate_graph
        graph = build_recalibrate_graph()
        result = await graph.nodes["analyze"].bound.afunc(state)
    return result, captured


# ── Fix #2: shortlist cap ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_analyze_caps_shortlist_in_prompt():
    """Far more flagged pages than the cap → prompt carries at most the cap."""
    from app.services.recalibrate_agent import _ANALYZE_MAX_SHORTLIST

    n = _ANALYZE_MAX_SHORTLIST + 50
    pages = {f"topics/p{i}.md": f"# Page {i}\n\nstub" for i in range(n)}
    # Priority-sorted candidates (triage emits them sorted desc).
    candidates = [
        {"path": f"topics/p{i}.md", "reasons": ["stub"], "priority": 1}
        for i in range(n)
    ]
    state = _base_state(wiki_pages=pages, triage_candidates=candidates)

    result, captured = await _run_analyze(state, response_json={})

    # The shortlist is split across batches; the cap bounds the total flagged
    # pages sent to the LLM, summed over every batch prompt.
    flagged_count = sum(m[1].content.count("[FLAGS:") for m in captured["calls"])
    assert flagged_count == _ANALYZE_MAX_SHORTLIST, flagged_count
    # Node still returns cleanly (empty plans from {} response).
    assert result["improvement_plan"] == []


@pytest.mark.asyncio
async def test_analyze_deleted_source_pages_bypass_cap():
    """Deleted-source pages are always included even when the cap is hit."""
    from app.services.recalibrate_agent import _ANALYZE_MAX_SHORTLIST

    n = _ANALYZE_MAX_SHORTLIST + 5
    pages = {f"topics/p{i}.md": "# x\n\nstub" for i in range(n)}
    pages["topics/from_deleted.md"] = "uploaded_file: gone.pdf\n\n# Derived\n\nbody"
    candidates = [
        {"path": f"topics/p{i}.md", "reasons": ["stub"], "priority": 1}
        for i in range(n)
    ]
    state = _base_state(
        wiki_pages=pages, triage_candidates=candidates, deleted_files=["gone.pdf"],
    )

    _, captured = await _run_analyze(state, response_json={})

    # from_deleted bypasses the cap — it must appear in one of the batch prompts.
    assert any("topics/from_deleted.md" in m[1].content for m in captured["calls"])


# ── Content gaps are actionable even with no flagged pages ────────────────

@pytest.mark.asyncio
async def test_analyze_creates_gap_pages_when_no_flagged_pages():
    """Regression: a wiki whose only issue is missing pages (content gaps) must
    NOT be a no-op. With zero flagged pages, analyze still runs a creation pass
    so the missing pages get made — fixing the broken links that point to them
    (previously it short-circuited and reported 'nothing to do')."""
    pages = {"concepts/clustering.md":
             "# Clustering\n\nSee [[hierarchical-clustering]] and [[dbscan]].\n" + "word " * 200}
    gaps = [
        {"slug": "hierarchical-clustering", "ref_count": 2,
         "referenced_by": ["concepts/clustering.md", "concepts/dim-reduction.md"]},
        {"slug": "dbscan", "ref_count": 2,
         "referenced_by": ["concepts/clustering.md", "concepts/dim-reduction.md"]},
    ]
    state = _base_state(wiki_pages=pages, triage_candidates=[], article_candidates=gaps)
    response = {
        "summary": "Create the missing concept pages.",
        "new_pages": [
            {"path": "concepts/hierarchical-clustering.md", "brief": "..."},
            {"path": "concepts/dbscan.md", "brief": "..."},
        ],
    }
    result, captured = await _run_analyze(state, response_json=response)

    assert captured.get("calls"), "expected a gap-creation LLM call (not a no-op)"
    assert sorted(p["path"] for p in result["new_page_plan"]) == [
        "concepts/dbscan.md", "concepts/hierarchical-clustering.md",
    ]


@pytest.mark.asyncio
async def test_analyze_noop_when_no_flags_and_no_gaps():
    """With no flagged pages AND no content gaps, analyze short-circuits without
    an LLM call — the genuine no-op that keeps recalibration idempotent."""
    state = _base_state(wiki_pages={"a.md": "# A\n\n" + "word " * 200},
                        triage_candidates=[], article_candidates=[])
    result, captured = await _run_analyze(state, response_json={})
    assert not captured.get("calls"), "expected NO LLM call (true no-op)"
    assert result["new_page_plan"] == []
    assert result["improvement_plan"] == []


# ── Fix #2b: reverse-adjacency neighbour expansion still correct ───────────

@pytest.mark.asyncio
async def test_analyze_includes_inlink_neighbour():
    """A page that links INTO a flagged page is pulled in as a context neighbour."""
    pages = {
        "topics/flagged.md": "# Flagged\n\nstub",
        "topics/linker.md": "# Linker\n\nlinks to flagged",
    }
    # linker → flagged (so flagged's inlink is linker)
    adj = {"topics/linker.md": {"topics/flagged.md"}}
    candidates = [{"path": "topics/flagged.md", "reasons": ["stub"], "priority": 1}]
    state = _base_state(wiki_pages=pages, triage_candidates=candidates)

    _, captured = await _run_analyze(state, response_json={}, adj=adj)

    human = captured["messages"][1].content
    assert "GRAPH NEIGHBOURS" in human
    assert "topics/linker.md" in human


# ── Fix #3: delete/improve overlap guard ──────────────────────────────────

@pytest.mark.asyncio
async def test_analyze_drops_improve_overlapping_delete():
    pages = {"a.md": "# A", "b.md": "# B"}
    candidates = [
        {"path": "a.md", "reasons": ["stub"], "priority": 1},
        {"path": "b.md", "reasons": ["stub"], "priority": 1},
    ]
    state = _base_state(wiki_pages=pages, triage_candidates=candidates)

    response = {
        "pages_to_delete": [{"path": "a.md", "reason": "redundant"}],
        "pages_to_improve": [
            {"path": "a.md", "instruction": "should be dropped"},
            {"path": "b.md", "instruction": "keep me"},
        ],
        "pages_to_rename": [],
        "new_pages": [],
        "contradictions": [],
        "log_entry": "",
    }
    result, _ = await _run_analyze(state, response_json=response)

    improve_paths = {p["path"] for p in result["improvement_plan"]}
    assert improve_paths == {"b.md"}
    assert "a.md" in result["delete_plan"]


@pytest.mark.asyncio
async def test_analyze_drops_improve_and_create_overlapping_rename_source():
    pages = {"c.md": "# C"}
    candidates = [{"path": "c.md", "reasons": ["stub"], "priority": 1}]
    state = _base_state(wiki_pages=pages, triage_candidates=candidates)

    response = {
        "pages_to_delete": [],
        "pages_to_rename": [{"from": "c.md", "to": "topics/c.md", "reason": "move"}],
        "pages_to_improve": [{"path": "c.md", "instruction": "would resurrect c.md"}],
        "new_pages": [{"path": "c.md", "brief": "would also resurrect"}],
        "contradictions": [],
        "log_entry": "",
    }
    result, _ = await _run_analyze(state, response_json=response)

    assert all(p["path"] != "c.md" for p in result["improvement_plan"])
    assert all(p["path"] != "c.md" for p in result["new_page_plan"])
    assert result["rename_plan"][0]["from"] == "c.md"


@pytest.mark.asyncio
async def test_analyze_malformed_delete_entry_does_not_crash():
    """A delete entry missing 'path' must not blow up the dedup guard."""
    pages = {"a.md": "# A"}
    candidates = [{"path": "a.md", "reasons": ["stub"], "priority": 1}]
    state = _base_state(wiki_pages=pages, triage_candidates=candidates)

    response = {
        "pages_to_delete": [{"reason": "no path key"}, {"path": "a.md"}],
        "pages_to_improve": [],
        "pages_to_rename": [],
        "new_pages": [],
        "contradictions": [],
        "log_entry": "",
    }
    result, _ = await _run_analyze(state, response_json=response)
    assert "a.md" in result["delete_plan"]
    assert "errors" not in result  # analyze did not fall into its except branch


# ── Fix #1: release the DB lock on crash/cancel ────────────────────────────

@pytest.mark.asyncio
async def test_release_recalibrate_lock_persists_error():
    from app.routes import operations

    job = MagicMock()
    job._db_id = 7
    with patch.object(
        operations.recalibrate_job, "persist_finish", new_callable=AsyncMock
    ) as mock_finish:
        await operations._release_recalibrate_lock("org-1", job)
    mock_finish.assert_awaited_once_with("error", "org-1")


@pytest.mark.asyncio
async def test_release_recalibrate_lock_noop_without_db_id():
    from app.routes import operations

    job = MagicMock()
    job._db_id = None
    with patch.object(
        operations.recalibrate_job, "persist_finish", new_callable=AsyncMock
    ) as mock_finish:
        await operations._release_recalibrate_lock("org-1", job)
    mock_finish.assert_not_awaited()


@pytest.mark.asyncio
async def test_release_recalibrate_lock_swallows_persist_failure():
    """A failure to persist must not mask the original error that triggered it."""
    from app.routes import operations

    job = MagicMock()
    job._db_id = 7
    with patch.object(
        operations.recalibrate_job, "persist_finish",
        new_callable=AsyncMock, side_effect=RuntimeError("db down"),
    ):
        # Must not raise.
        await operations._release_recalibrate_lock("org-1", job)


# ── idempotency: actionable-candidate filtering (Layer 3) ─────────────────────
from app.services.recalibrate_agent import _actionable_candidates, _content_sha  # noqa: E402


def test_content_sha_stable_and_sensitive():
    assert _content_sha("x") == _content_sha("x")
    assert _content_sha("x") != _content_sha("y")


def test_actionable_skips_unchanged_reviewed_pages():
    """Skip a flagged page only when BOTH content and context are unchanged — the
    core of idempotency (re-running does no work on already-reviewed pages)."""
    pages = {"a.md": "content A", "b.md": "content B"}
    ctx = {"a.md": "ctx-a", "b.md": "ctx-b"}
    flagged = [
        {"path": "a.md", "reasons": ["stub"], "priority": 1},
        {"path": "b.md", "reasons": ["stub"], "priority": 1},
    ]
    recal = {"a.md": {"content_sha": _content_sha("content A"), "context_sha": "ctx-a",
                      "reviewed_at": None}}
    out = _actionable_candidates(flagged, pages, ctx, recal)
    assert [c["path"] for c in out] == ["b.md"]  # a unchanged → skipped; b never reviewed


def test_actionable_keeps_changed_content():
    pages = {"a.md": "NEW content"}
    ctx = {"a.md": "ctx-a"}
    recal = {"a.md": {"content_sha": _content_sha("OLD content"), "context_sha": "ctx-a",
                      "reviewed_at": None}}
    out = _actionable_candidates([{"path": "a.md", "reasons": ["stub"], "priority": 1}],
                                 pages, ctx, recal)
    assert [c["path"] for c in out] == ["a.md"]  # content changed → reprocess


def test_actionable_keeps_changed_context_even_if_content_unchanged():
    """The relational fix: a neighbour change flips context_sha → re-review, even
    though the page's own content is byte-identical (e.g. a link target vanished)."""
    pages = {"a.md": "content A"}
    ctx = {"a.md": "ctx-NEW"}
    recal = {"a.md": {"content_sha": _content_sha("content A"), "context_sha": "ctx-OLD",
                      "reviewed_at": None}}
    out = _actionable_candidates([{"path": "a.md", "reasons": ["broken_links"], "priority": 1}],
                                 pages, ctx, recal)
    assert [c["path"] for c in out] == ["a.md"]


def test_actionable_keeps_never_reviewed_pages():
    pages = {"a.md": "content A"}
    ctx = {"a.md": "ctx-a"}
    out = _actionable_candidates([{"path": "a.md", "reasons": ["orphan"], "priority": 1}],
                                 pages, ctx, {})
    assert [c["path"] for c in out] == ["a.md"]  # no fingerprint → first-time review
