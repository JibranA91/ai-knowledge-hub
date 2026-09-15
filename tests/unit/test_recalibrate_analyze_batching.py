"""Unit tests for batched analyze in recalibrate_agent.

The analyze step used to send the whole flagged shortlist to the LLM in one
call and expect one JSON plan back — output grew with flagged-page count and
eventually truncated mid-JSON (max_tokens), sinking the entire run. analyze now
batches the shortlist (one bounded LLM call per batch, run concurrently) and
merges the per-batch plans.

Covers:
  - _merge_analyze_plans: bucket concatenation, dedup, contradiction folding,
    delete/rename-vs-improve overlap guard, robustness to junk batches
  - _dedup_by_path
  - analyze_node: fans out into ceil(N / _ANALYZE_BATCH_SIZE) calls, merges
    results, and tolerates a truncated/failed batch without losing the rest
"""
import json

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.recalibrate_agent import (
    _ANALYZE_BATCH_SIZE,
    _assign_batches,
    _contradiction_instruction,
    _dedup_by_path,
    _duplicate_instruction,
    _fold_pair_findings,
    _merge_analyze_plans,
)


# ── _dedup_by_path ─────────────────────────────────────────────────────────

def test_dedup_by_path_keeps_first_per_path():
    items = [
        {"path": "a.md", "instruction": "first"},
        {"path": "a.md", "instruction": "second"},
        {"path": "b.md", "instruction": "b"},
    ]
    out = _dedup_by_path(items)
    assert [i["path"] for i in out] == ["a.md", "b.md"]
    assert out[0]["instruction"] == "first"


def test_dedup_by_path_drops_entries_without_path():
    items = [{"instruction": "no path"}, {"path": "", "instruction": "empty"}, {"path": "a.md"}]
    out = _dedup_by_path(items)
    assert out == [{"path": "a.md"}]


# ── _merge_analyze_plans: basic concatenation ──────────────────────────────

def test_merge_concatenates_buckets_across_batches():
    batch_results = [
        {
            "pages_to_improve": [{"path": "a.md", "instruction": "fix a"}],
            "pages_to_delete": [{"path": "old.md", "reason": "stale"}],
            "summary": "batch 1 summary",
        },
        {
            "pages_to_improve": [{"path": "b.md", "instruction": "fix b"}],
            "new_pages": [{"path": "new.md", "brief": "create"}],
            "pages_to_rename": [{"from": "x.md", "to": "y.md", "reason": "move"}],
            "summary": "batch 2 summary",
        },
    ]
    merged = _merge_analyze_plans(batch_results)

    assert {p["path"] for p in merged["improvement_plan"]} == {"a.md", "b.md"}
    assert merged["new_page_plan"] == [{"path": "new.md", "brief": "create"}]
    assert merged["delete_plan"] == ["old.md"]
    assert merged["rename_plan"] == [{"from": "x.md", "to": "y.md", "reason": "move"}]
    assert merged["summaries"] == ["batch 1 summary", "batch 2 summary"]


def test_merge_dedups_shared_neighbour_improve_across_batches():
    """The same neighbour page suggested by two batches yields one improve entry."""
    batch_results = [
        {"pages_to_improve": [{"path": "shared.md", "instruction": "from batch 1"}]},
        {"pages_to_improve": [{"path": "shared.md", "instruction": "from batch 2"}]},
    ]
    merged = _merge_analyze_plans(batch_results)
    assert len(merged["improvement_plan"]) == 1
    assert merged["improvement_plan"][0]["instruction"] == "from batch 1"


# ── _merge_analyze_plans: robustness ───────────────────────────────────────

def test_merge_ignores_empty_and_non_dict_batches():
    """Failed batches return {} (or, defensively, non-dicts) — these are skipped."""
    batch_results = [
        {},
        None,
        "garbage",
        {"pages_to_improve": [{"path": "a.md", "instruction": "fix"}]},
    ]
    merged = _merge_analyze_plans(batch_results)
    assert merged["improvement_plan"] == [{"path": "a.md", "instruction": "fix"}]


def test_merge_drops_malformed_entries():
    """Entries missing a path or that aren't dicts are dropped, not crashed on."""
    batch_results = [
        {
            "pages_to_improve": [{"instruction": "no path"}, "not a dict", {"path": "ok.md"}],
            "new_pages": [{"brief": "no path"}],
            "pages_to_rename": ["not a dict", {"from": "a.md", "to": "b.md"}],
        }
    ]
    merged = _merge_analyze_plans(batch_results)
    assert merged["improvement_plan"] == [{"path": "ok.md"}]
    assert merged["new_page_plan"] == []
    assert merged["rename_plan"] == [{"from": "a.md", "to": "b.md"}]


# ── _merge_analyze_plans: contradictions ───────────────────────────────────

def test_merge_contradiction_adds_new_improve_entry():
    batch_results = [
        {"contradictions": [{"page_a": "p1.md", "page_b": "p2.md", "description": "X vs Y"}]},
    ]
    merged = _merge_analyze_plans(batch_results)
    paths = {p["path"] for p in merged["improvement_plan"]}
    assert paths == {"p1.md", "p2.md"}
    assert all("CONTRADICTION" in p["instruction"] for p in merged["improvement_plan"])


def test_merge_contradiction_folds_into_existing_instruction():
    batch_results = [
        {
            "pages_to_improve": [{"path": "p1.md", "instruction": "expand stub"}],
            "contradictions": [{"page_a": "p1.md", "page_b": "p2.md", "description": "conflict"}],
        },
    ]
    merged = _merge_analyze_plans(batch_results)
    p1 = next(p for p in merged["improvement_plan"] if p["path"] == "p1.md")
    assert "expand stub" in p1["instruction"]
    assert "CONTRADICTION" in p1["instruction"]
    assert " | " in p1["instruction"]


# ── _merge_analyze_plans: overlap guard ────────────────────────────────────

def test_merge_delete_wins_over_improve():
    """A page in both delete and improve is dropped from improve (delete wins)."""
    batch_results = [
        {
            "pages_to_improve": [{"path": "dupe.md", "instruction": "fix"}, {"path": "keep.md"}],
            "pages_to_delete": [{"path": "dupe.md", "reason": "redundant"}],
        }
    ]
    merged = _merge_analyze_plans(batch_results)
    improve_paths = {p["path"] for p in merged["improvement_plan"]}
    assert "dupe.md" not in improve_paths
    assert "keep.md" in improve_paths
    assert "dupe.md" in merged["delete_plan"]


def test_merge_rename_from_blocks_improve():
    batch_results = [
        {
            "pages_to_improve": [{"path": "old.md", "instruction": "fix"}],
            "pages_to_rename": [{"from": "old.md", "to": "new.md"}],
        }
    ]
    merged = _merge_analyze_plans(batch_results)
    assert merged["improvement_plan"] == []
    assert merged["rename_plan"] == [{"from": "old.md", "to": "new.md"}]


def test_merge_delete_can_be_dict_or_string():
    batch_results = [{"pages_to_delete": [{"path": "a.md", "reason": "x"}, "b.md", "", None]}]
    merged = _merge_analyze_plans(batch_results)
    assert set(merged["delete_plan"]) == {"a.md", "b.md"}


# ── analyze_node: batching fan-out ─────────────────────────────────────────

def _resp(payload: dict, stop: str = "end_turn") -> MagicMock:
    r = MagicMock()
    r.content = json.dumps(payload)
    r.response_metadata = {"stopReason": stop}
    return r


def _reconcile_resp(contradictions=None, duplicates=None) -> MagicMock:
    """A response for the cross-batch reconciliation call (runs once after the
    per-batch calls whenever there is more than one batch)."""
    return _resp({"contradictions": contradictions or [], "duplicates": duplicates or []})


def _base_state(wiki_pages: dict, candidates: list[dict]) -> dict:
    return {
        "wiki_pages": wiki_pages,
        "schema": "schema body",
        "index": "",
        "deleted_files": [],
        "fact_instructions": "",
        "triage_candidates": candidates,
        "improvement_plan": [],
        "new_page_plan": [],
        "delete_plan": [],
        "rename_plan": [],
        "pages_written": [],
        "pages_deleted": [],
        "pages_renamed": [],
        "errors": [],
        "log_entry": "",
        "_t_start": 0.0,
    }


@pytest.mark.asyncio
async def test_analyze_node_batches_and_merges():
    """N flagged pages → ceil(N / batch_size) LLM calls, plans merged."""
    n = _ANALYZE_BATCH_SIZE * 2 + 5          # 45 with default 20 → 3 batches
    paths = [f"concepts/p{i:02d}.md" for i in range(n)]
    wiki_pages = {p: f"# {p}\n\nbroken link to [[missing.md]]" for p in paths}
    candidates = [{"path": p, "reasons": ["broken_links"], "priority": 1} for p in paths]

    # One distinct improve target per batch so we can prove all batches merged.
    expected_batches = (n + _ANALYZE_BATCH_SIZE - 1) // _ANALYZE_BATCH_SIZE
    responses = [
        _resp({"summary": f"batch {i}", "pages_to_improve": [{"path": f"batch{i}.md", "instruction": "fix"}]})
        for i in range(expected_batches)
    ]
    responses.append(_reconcile_resp())  # >1 batch → one trailing reconciliation call
    llm = MagicMock()
    llm.ainvoke = AsyncMock(side_effect=responses)

    graph_obj = MagicMock()
    graph_obj._adj = {}
    graph_obj.ensure_loaded = AsyncMock(return_value=graph_obj)

    with (
        patch("app.model.get_chat", return_value=llm),
        patch("app.services.recalibrate_agent.recalibrate_job") as mock_job_mod,
        patch("app.services.graph.get_graph", return_value=graph_obj),
    ):
        mock_job_mod.get.return_value = MagicMock()

        from app.services.recalibrate_agent import build_recalibrate_graph
        graph = build_recalibrate_graph()
        result = await graph.nodes["analyze"].bound.afunc(_base_state(wiki_pages, candidates))

    assert llm.ainvoke.call_count == expected_batches + 1  # batches + reconciliation
    improve_paths = {p["path"] for p in result["improvement_plan"]}
    assert improve_paths == {f"batch{i}.md" for i in range(expected_batches)}
    # log_entry composed from every batch summary
    for i in range(expected_batches):
        assert f"batch {i}" in result["log_entry"]


@pytest.mark.asyncio
async def test_analyze_node_tolerates_truncated_batch():
    """A batch that truncates (max_tokens → invalid JSON) contributes nothing,
    but the other batches' plans still come through — no exception, no total loss."""
    n = _ANALYZE_BATCH_SIZE + 1              # 21 → 2 batches
    paths = [f"concepts/p{i:02d}.md" for i in range(n)]
    wiki_pages = {p: f"# {p}\n\nstub" for p in paths}
    candidates = [{"path": p, "reasons": ["stub"], "priority": 1} for p in paths]

    good = _resp({"summary": "ok", "pages_to_improve": [{"path": "good.md", "instruction": "fix"}]})
    truncated = MagicMock()
    truncated.content = '{"pages_to_improve": [{"path": "x.md", "instruct'  # cut off mid-JSON
    truncated.response_metadata = {"stopReason": "max_tokens"}

    llm = MagicMock()
    llm.ainvoke = AsyncMock(side_effect=[good, truncated, _reconcile_resp()])

    graph_obj = MagicMock()
    graph_obj._adj = {}
    graph_obj.ensure_loaded = AsyncMock(return_value=graph_obj)

    with (
        patch("app.model.get_chat", return_value=llm),
        patch("app.services.recalibrate_agent.recalibrate_job") as mock_job_mod,
        patch("app.services.graph.get_graph", return_value=graph_obj),
    ):
        mock_job_mod.get.return_value = MagicMock()

        from app.services.recalibrate_agent import build_recalibrate_graph
        graph = build_recalibrate_graph()
        result = await graph.nodes["analyze"].bound.afunc(_base_state(wiki_pages, candidates))

    assert llm.ainvoke.call_count == 3  # 2 batches + reconciliation
    assert "errors" not in result  # partial success, not a hard failure
    assert {p["path"] for p in result["improvement_plan"]} == {"good.md"}


@pytest.mark.asyncio
async def test_analyze_node_all_batches_failing_reports_error():
    """If every batch fails, surface an error instead of a false 'healthy' result."""
    paths = ["concepts/a.md", "concepts/b.md"]
    wiki_pages = {p: "# x\n\nstub" for p in paths}
    candidates = [{"path": p, "reasons": ["stub"], "priority": 1} for p in paths]

    llm = MagicMock()
    llm.ainvoke = AsyncMock(side_effect=RuntimeError("Bedrock down"))

    graph_obj = MagicMock()
    graph_obj._adj = {}
    graph_obj.ensure_loaded = AsyncMock(return_value=graph_obj)

    with (
        patch("app.model.get_chat", return_value=llm),
        patch("app.services.recalibrate_agent.recalibrate_job") as mock_job_mod,
        patch("app.services.graph.get_graph", return_value=graph_obj),
    ):
        mock_job_mod.get.return_value = MagicMock()

        from app.services.recalibrate_agent import build_recalibrate_graph
        graph = build_recalibrate_graph()
        result = await graph.nodes["analyze"].bound.afunc(_base_state(wiki_pages, candidates))

    assert result["errors"]
    assert result["improvement_plan"] == []


# ── _assign_batches: co-location (directions 1 + 2) ────────────────────────

def test_assign_batches_keeps_duplicate_group_atomic():
    """All members of a title-duplicate group land in the same batch, even when
    many unrelated singletons would otherwise fill batches around them."""
    shortlist = {f"p{i:02d}.md" for i in range(25)}  # 25 → 2 batches at size 20
    dup_group = ["p00.md", "p24.md"]                 # would sort into different batches
    batches = _assign_batches(shortlist, {}, [dup_group], _ANALYZE_BATCH_SIZE)

    home = [b for b in batches if "p00.md" in b]
    assert len(home) == 1
    assert "p24.md" in home[0]
    # Nothing lost, nothing duplicated across batches.
    flat = [p for b in batches for p in b]
    assert sorted(flat) == sorted(shortlist)
    assert len(flat) == len(set(flat))


def test_assign_batches_groups_linked_component_together():
    """Linked pages (a connected component) are ordered adjacently, so a small
    component stays within one batch rather than being split by path order."""
    shortlist = {f"p{i:02d}.md" for i in range(40)}
    # p39 links to p00 — without component grouping they'd be in different batches.
    adjacency = {"p00.md": {"p39.md"}, "p39.md": {"p00.md"}}
    batches = _assign_batches(shortlist, adjacency, [], _ANALYZE_BATCH_SIZE)

    home = [b for b in batches if "p00.md" in b]
    assert len(home) == 1
    assert "p39.md" in home[0]


def test_assign_batches_respects_batch_size():
    shortlist = {f"p{i:03d}.md" for i in range(95)}
    batches = _assign_batches(shortlist, {}, [], _ANALYZE_BATCH_SIZE)
    assert all(len(b) <= _ANALYZE_BATCH_SIZE for b in batches)
    assert sum(len(b) for b in batches) == 95


def test_assign_batches_empty_shortlist():
    assert _assign_batches(set(), {}, [], _ANALYZE_BATCH_SIZE) == []


def test_assign_batches_oversized_duplicate_group_gets_own_batch():
    """A pathological duplicate group larger than batch_size is never split."""
    group = [f"d{i:02d}.md" for i in range(_ANALYZE_BATCH_SIZE + 3)]
    shortlist = set(group) | {"other.md"}
    batches = _assign_batches(shortlist, {}, [group], _ANALYZE_BATCH_SIZE)
    home = [b for b in batches if group[0] in b]
    assert len(home) == 1
    assert set(group) <= set(home[0])  # entire group together despite exceeding size


# ── _fold_pair_findings ────────────────────────────────────────────────────

def test_fold_pair_findings_adds_and_appends():
    plan = [{"path": "a.md", "instruction": "expand"}]
    findings = [{"page_a": "a.md", "page_b": "b.md", "description": "X vs Y"}]
    out = _fold_pair_findings(plan, findings, _contradiction_instruction)
    a = next(p for p in out if p["path"] == "a.md")
    b = next(p for p in out if p["path"] == "b.md")
    assert "expand" in a["instruction"] and "CONTRADICTION" in a["instruction"]
    assert "CONTRADICTION" in b["instruction"]


def test_fold_pair_findings_duplicate_instruction():
    plan = []
    findings = [{"page_a": "a.md", "page_b": "b.md", "note": "same topic"}]
    out = _fold_pair_findings(plan, findings, _duplicate_instruction)
    assert {p["path"] for p in out} == {"a.md", "b.md"}
    assert all("DUPLICATE" in p["instruction"] for p in out)


def test_fold_pair_findings_empty_is_noop():
    plan = [{"path": "a.md", "instruction": "x"}]
    assert _fold_pair_findings(plan, [], _contradiction_instruction) == plan


# ── analyze_node: reconciliation behaviour ─────────────────────────────────

@pytest.mark.asyncio
async def test_analyze_node_reconciliation_folds_cross_batch_findings():
    """Reconciliation findings referencing two shortlist pages fold into the plan."""
    n = _ANALYZE_BATCH_SIZE + 5               # 25 → 2 batches
    paths = [f"concepts/p{i:02d}.md" for i in range(n)]
    wiki_pages = {p: f"# {p}\n\nstub body" for p in paths}
    candidates = [{"path": p, "reasons": ["stub"], "priority": 1} for p in paths]

    batch_resps = [_resp({"summary": f"b{i}", "pages_to_improve": []}) for i in range(2)]
    # Reconciliation flags a contradiction between two pages from *different* batches.
    reconcile = _reconcile_resp(
        contradictions=[{"page_a": "concepts/p00.md", "page_b": "concepts/p24.md",
                         "description": "conflicting dates"}],
    )
    llm = MagicMock()
    llm.ainvoke = AsyncMock(side_effect=batch_resps + [reconcile])

    graph_obj = MagicMock()
    graph_obj._adj = {}
    graph_obj.ensure_loaded = AsyncMock(return_value=graph_obj)

    with (
        patch("app.model.get_chat", return_value=llm),
        patch("app.services.recalibrate_agent.recalibrate_job") as mock_job_mod,
        patch("app.services.graph.get_graph", return_value=graph_obj),
    ):
        mock_job_mod.get.return_value = MagicMock()

        from app.services.recalibrate_agent import build_recalibrate_graph
        graph = build_recalibrate_graph()
        result = await graph.nodes["analyze"].bound.afunc(_base_state(wiki_pages, candidates))

    improve = {p["path"]: p["instruction"] for p in result["improvement_plan"]}
    assert "concepts/p00.md" in improve and "concepts/p24.md" in improve
    assert "CONTRADICTION" in improve["concepts/p00.md"]


@pytest.mark.asyncio
async def test_analyze_node_reconciliation_drops_hallucinated_paths():
    """Findings referencing a path not in the shortlist are ignored."""
    n = _ANALYZE_BATCH_SIZE + 1               # 21 → 2 batches
    paths = [f"concepts/p{i:02d}.md" for i in range(n)]
    wiki_pages = {p: "# x\n\nstub" for p in paths}
    candidates = [{"path": p, "reasons": ["stub"], "priority": 1} for p in paths]

    batch_resps = [_resp({"summary": f"b{i}", "pages_to_improve": []}) for i in range(2)]
    reconcile = _reconcile_resp(
        contradictions=[{"page_a": "concepts/p00.md", "page_b": "ghost.md", "description": "bogus"}],
    )
    llm = MagicMock()
    llm.ainvoke = AsyncMock(side_effect=batch_resps + [reconcile])

    graph_obj = MagicMock()
    graph_obj._adj = {}
    graph_obj.ensure_loaded = AsyncMock(return_value=graph_obj)

    with (
        patch("app.model.get_chat", return_value=llm),
        patch("app.services.recalibrate_agent.recalibrate_job") as mock_job_mod,
        patch("app.services.graph.get_graph", return_value=graph_obj),
    ):
        mock_job_mod.get.return_value = MagicMock()

        from app.services.recalibrate_agent import build_recalibrate_graph
        graph = build_recalibrate_graph()
        result = await graph.nodes["analyze"].bound.afunc(_base_state(wiki_pages, candidates))

    plan_paths = {p["path"] for p in result["improvement_plan"]}
    assert "ghost.md" not in plan_paths
    # The finding referenced a hallucinated path, so it's dropped entirely.
    assert "concepts/p00.md" not in plan_paths


@pytest.mark.asyncio
async def test_analyze_node_single_batch_skips_reconciliation():
    """One batch already saw every page — no reconciliation call is made."""
    paths = ["concepts/a.md", "concepts/b.md"]   # 2 → 1 batch
    wiki_pages = {p: "# x\n\nstub" for p in paths}
    candidates = [{"path": p, "reasons": ["stub"], "priority": 1} for p in paths]

    llm = MagicMock()
    llm.ainvoke = AsyncMock(side_effect=[_resp({"summary": "s", "pages_to_improve": []})])

    graph_obj = MagicMock()
    graph_obj._adj = {}
    graph_obj.ensure_loaded = AsyncMock(return_value=graph_obj)

    with (
        patch("app.model.get_chat", return_value=llm),
        patch("app.services.recalibrate_agent.recalibrate_job") as mock_job_mod,
        patch("app.services.graph.get_graph", return_value=graph_obj),
    ):
        mock_job_mod.get.return_value = MagicMock()

        from app.services.recalibrate_agent import build_recalibrate_graph
        graph = build_recalibrate_graph()
        await graph.nodes["analyze"].bound.afunc(_base_state(wiki_pages, candidates))

    assert llm.ainvoke.call_count == 1  # single batch, no reconciliation
