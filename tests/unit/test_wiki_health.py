"""Unit tests for the deterministic wiki-health module.

Covers the two properties the health-score redesign promises:
  * determinism — same input always yields the same score (no LLM, no sampling)
  * monotonicity — fixing any signal can only raise or hold the score, which is
    what guarantees recalibration improves wiki health
plus per-signal detection and the score bounds.
"""
from datetime import datetime, timedelta, timezone

from app.services import wiki_health as wh


_NOW = datetime(2026, 6, 10, tzinfo=timezone.utc)


def _page(title: str, words: int = 200, extra: str = "") -> str:
    """A healthy page (distinct title, above the stub threshold)."""
    body = " ".join(["word"] * words)
    return f"# {title}\n\n{body}" + (f"\n{extra}" if extra else "")


# ── score bounds / determinism ─────────────────────────────────────────────

def test_empty_wiki_scores_100():
    assert wh.compute_health({}, {})["health_score"] == 100


def test_score_is_deterministic():
    pages = {
        "a.md": _page("Alpha", 200, "[[ghost]]"),
        "b.md": "# Beta\n short",
        "c.md": _page("Gamma", 200),
    }
    adj = {"a.md": set(), "c.md": {"a.md"}}
    runs = {wh.compute_health(pages, adj, now=_NOW)["health_score"] for _ in range(5)}
    assert len(runs) == 1  # identical every time


def test_weights_sum_to_100():
    # Guarantees the score stays within [0, 100].
    assert sum(wh.SIGNAL_WEIGHTS.values()) == 100


def test_all_pages_fully_broken_floors_near_zero():
    # Every page: stub + orphan + duplicate title + broken link → heavy penalty.
    pages = {f"p{i}.md": "# Same Title\n short [[ghost]]" for i in range(4)}
    score = wh.compute_health(pages, {}, now=_NOW)["health_score"]
    assert 0 <= score < 40


# ── per-signal detection ────────────────────────────────────────────────────

def test_detects_stub():
    sig = wh.compute_signals({"a.md": "# A\n short"}, {"a.md": {"x"}}, now=_NOW)
    assert "stub" in sig["reasons"]["a.md"]


def test_detects_orphan():
    # Long enough not to be a stub; nothing links to it → orphan.
    sig = wh.compute_signals({"a.md": _page("Alpha", 200)}, {}, now=_NOW)
    assert "orphan" in sig["reasons"]["a.md"]


def test_page_with_inbound_link_not_orphan():
    sig = wh.compute_signals({"a.md": _page("Alpha", 200), "b.md": _page("Beta", 200)},
                             {"b.md": {"a.md"}}, now=_NOW)
    assert "orphan" not in sig["reasons"].get("a.md", [])  # a.md has an inbound link


def test_page_with_no_inbound_is_orphan_even_with_outbound():
    # b.md links out to a.md but nothing links to b.md → orphan (undiscoverable).
    sig = wh.compute_signals({"a.md": _page("Alpha", 200), "b.md": _page("Beta", 200)},
                             {"b.md": {"a.md"}}, now=_NOW)
    assert "orphan" in sig["reasons"].get("b.md", [])


# ── new-article candidates (content-gap detection) ──────────────────────────

def test_article_candidate_when_referenced_by_multiple_pages():
    pages = {
        "a.md": _page("Alpha", 50, "See [[shared-concept]]."),
        "b.md": _page("Beta", 50, "Also [[shared-concept]]."),
    }
    cands = wh.compute_article_candidates(pages, min_refs=2)
    slugs = [c["slug"] for c in cands]
    assert "shared-concept" in slugs
    hit = next(c for c in cands if c["slug"] == "shared-concept")
    assert hit["ref_count"] == 2
    assert hit["referenced_by"] == ["a.md", "b.md"]


def test_article_candidate_ignores_single_reference():
    pages = {"a.md": _page("Alpha", 50, "[[lonely-concept]]"), "b.md": _page("Beta", 50)}
    assert wh.compute_article_candidates(pages, min_refs=2) == []


def test_article_candidate_ignores_existing_pages():
    # [[beta]] resolves to beta.md → not a gap, even if referenced twice.
    pages = {
        "a.md": _page("Alpha", 50, "[[beta]]"),
        "c.md": _page("Gamma", 50, "[[beta]]"),
        "beta.md": _page("Beta", 50),
    }
    assert wh.compute_article_candidates(pages, min_refs=2) == []


def test_article_candidate_counts_distinct_pages_not_mentions():
    # Two mentions on one page is still one referencing page → below min_refs.
    pages = {"a.md": _page("Alpha", 50, "[[x]] and again [[x]]")}
    assert wh.compute_article_candidates(pages, min_refs=2) == []


# ── recent-page selection (stale reconciliation context) ────────────────────

def _dated(name: str, date: str) -> str:
    return f"---\ndate_ingested: {date}\n---\n# {name}\nbody"


def test_select_recent_pages_orders_newest_first():
    pages = {
        "old.md": _dated("Old", "2025-01-01"),
        "new.md": _dated("New", "2026-06-01"),
        "mid.md": _dated("Mid", "2025-12-01"),
    }
    assert wh.select_recent_pages(pages, limit=2) == ["new.md", "mid.md"]


def test_select_recent_pages_undated_sort_last():
    pages = {"dated.md": _dated("D", "2026-01-01"), "undated.md": "# U\nno frontmatter"}
    assert wh.select_recent_pages(pages, limit=1) == ["dated.md"]


def test_detects_broken_wikilink():
    sig = wh.compute_signals({"a.md": _page("Alpha", 200, "[[does-not-exist]]")},
                             {"a.md": {"x"}}, now=_NOW)
    assert "broken_links" in sig["reasons"]["a.md"]


def test_valid_wikilink_not_broken():
    pages = {"a.md": _page("Alpha", 200, "[[beta]]"), "beta.md": _page("Beta", 200)}
    sig = wh.compute_signals(pages, {"a.md": {"beta.md"}, "beta.md": {"a.md"}}, now=_NOW)
    assert "broken_links" not in sig["reasons"].get("a.md", [])


def test_detects_duplicate_titles():
    pages = {"a.md": "# Same\n" + _page("ignored", 200), "b.md": "# Same\n" + _page("ignored", 200)}
    # First heading line wins ("# Same"), so both normalise to the same title.
    sig = wh.compute_signals(pages, {"a.md": {"b.md"}, "b.md": {"a.md"}}, now=_NOW)
    assert "duplicate_candidate" in sig["reasons"]["a.md"]
    assert "duplicate_candidate" in sig["reasons"]["b.md"]
    assert sig["duplicate_groups"] == [["a.md", "b.md"]]


def test_detects_stale():
    old = (_NOW - timedelta(days=120)).strftime("%Y-%m-%d")
    content = f"---\ndate_ingested: {old}\n---\n# A\n" + " ".join(["word"] * 200)
    sig = wh.compute_signals({"a.md": content}, {"a.md": {"x"}}, now=_NOW)
    assert "stale" in sig["reasons"]["a.md"]


def test_recent_page_not_stale():
    recent = (_NOW - timedelta(days=10)).strftime("%Y-%m-%d")
    content = f"---\ndate_ingested: {recent}\n---\n# A\n" + " ".join(["word"] * 200)
    sig = wh.compute_signals({"a.md": content}, {"a.md": {"x"}}, now=_NOW)
    assert "stale" not in sig["reasons"].get("a.md", [])


# ── monotonicity: fixing a signal never lowers the score ────────────────────
# Each pair changes exactly ONE signal (titles/links otherwise identical).

_ADJ = {"a.md": {"b.md"}, "b.md": {"a.md"}}


def test_fixing_broken_link_raises_score():
    broken = {"a.md": _page("Alpha", 200, "[[ghost]]"), "b.md": _page("Beta", 200)}
    fixed = {"a.md": _page("Alpha", 200), "b.md": _page("Beta", 200)}
    before = wh.compute_health(broken, _ADJ, now=_NOW)["health_score"]
    after = wh.compute_health(fixed, _ADJ, now=_NOW)["health_score"]
    assert after >= before
    assert after > before  # this fixture has exactly one issue, so it strictly improves


def test_expanding_stub_raises_score():
    stub = {"a.md": "# Alpha\n short", "b.md": _page("Beta", 200)}
    expanded = {"a.md": _page("Alpha", 200), "b.md": _page("Beta", 200)}
    before = wh.compute_health(stub, _ADJ, now=_NOW)["health_score"]
    after = wh.compute_health(expanded, _ADJ, now=_NOW)["health_score"]
    assert after > before


def test_resolving_duplicate_raises_score():
    dupes = {"a.md": "# Same\n" + " ".join(["w"] * 200), "b.md": "# Same\n" + " ".join(["w"] * 200)}
    resolved = {"a.md": _page("Alpha", 200), "b.md": _page("Beta", 200)}
    before = wh.compute_health(dupes, _ADJ, now=_NOW)["health_score"]
    after = wh.compute_health(resolved, _ADJ, now=_NOW)["health_score"]
    assert after > before


# ── breakdown shape ─────────────────────────────────────────────────────────

def test_breakdown_keys_match_signal_weights():
    report = wh.compute_health({"a.md": "# A\n short"}, {}, now=_NOW)
    assert set(report["breakdown"].keys()) == set(wh.SIGNAL_WEIGHTS.keys())
    assert report["total_pages"] == 1
    assert report["flagged_pages"] == 1


# ── staleness keyed on recalibration review time (reviewed_at) ───────────────
# Staleness = most recent of (date_ingested, last review) older than the window.
# This is what lets recalibration CLEAR staleness so repeated runs converge.

def _dated_page(days_old: int, title: str = "A") -> str:
    d = (_NOW - timedelta(days=days_old)).strftime("%Y-%m-%d")
    return f"---\ndate_ingested: {d}\n---\n# {title}\n" + " ".join(["word"] * 200)


def test_recent_review_clears_staleness_despite_old_ingest():
    # Ingested 200d ago (would be stale on date alone) but reviewed 5d ago → fresh.
    sig = wh.compute_signals({"a.md": _dated_page(200)}, {"a.md": {"x"}}, now=_NOW,
                             reviewed_at={"a.md": _NOW - timedelta(days=5)})
    assert "stale" not in sig["reasons"].get("a.md", [])


def test_old_review_and_old_ingest_is_stale():
    sig = wh.compute_signals({"a.md": _dated_page(200)}, {"a.md": {"x"}}, now=_NOW,
                             reviewed_at={"a.md": _NOW - timedelta(days=120)})
    assert "stale" in sig["reasons"]["a.md"]


def test_no_review_record_falls_back_to_date_ingested():
    # Backward-compatible: with no review time, staleness uses date_ingested alone.
    sig = wh.compute_signals({"a.md": _dated_page(200)}, {"a.md": {"x"}}, now=_NOW,
                             reviewed_at={})
    assert "stale" in sig["reasons"]["a.md"]


def test_recent_ingest_not_stale_even_if_review_is_old():
    # Freshness is the MOST RECENT of ingest/review — a recent ingest wins.
    sig = wh.compute_signals({"a.md": _dated_page(5)}, {"a.md": {"x"}}, now=_NOW,
                             reviewed_at={"a.md": _NOW - timedelta(days=300)})
    assert "stale" not in sig["reasons"].get("a.md", [])


def test_review_time_lifts_health_score():
    # Same old page, mutually linked (neither orphan): a recent review removes the
    # stale penalty, so the deterministic score strictly rises.
    pages = {"a.md": _dated_page(200, "Alpha"), "b.md": _page("Beta", 200)}
    adj = {"a.md": {"b.md"}, "b.md": {"a.md"}}
    before = wh.compute_health(pages, adj, now=_NOW)["health_score"]
    after = wh.compute_health(pages, adj, now=_NOW,
                              reviewed_at={"a.md": _NOW - timedelta(days=2)})["health_score"]
    assert after > before


# ── context signatures: relational-signal idempotency ────────────────────────
# The signature must change when a page's orphan / broken_links / duplicate input
# changes even though its OWN content is untouched — otherwise those fixes defer.

def test_context_signature_changes_when_link_target_removed():
    a = _page("Alpha", 200) + "\n\nSee [Beta](b.md)."   # a.md links to b.md
    before = wh.compute_context_signatures({"a.md": a, "b.md": _page("Beta", 200)})
    after = wh.compute_context_signatures({"a.md": a})   # b.md deleted → broken link
    assert before["a.md"] != after["a.md"]


def test_context_signature_changes_when_inbound_link_removed():
    a = _page("Alpha", 200)
    with_in = wh.compute_context_signatures({"a.md": a, "b.md": _page("Beta", 200) + "\n[A](a.md)"})
    no_in = wh.compute_context_signatures({"a.md": a, "b.md": _page("Beta", 200)})
    assert with_in["a.md"] != no_in["a.md"]   # a.md orphaned without its content changing


def test_context_signature_changes_on_new_title_collision():
    a = "# Shared\n" + " ".join(["w"] * 200)
    alone = wh.compute_context_signatures({"a.md": a})
    collided = wh.compute_context_signatures({"a.md": a, "b.md": "# Shared\n" + " ".join(["w"] * 200)})
    assert alone["a.md"] != collided["a.md"]   # a.md becomes a duplicate candidate


def test_context_signature_stable_when_nothing_relevant_changes():
    pages = {"a.md": _page("Alpha", 200, "[[beta]]"), "beta.md": _page("Beta", 200)}
    assert wh.compute_context_signatures(pages) == wh.compute_context_signatures(dict(pages))


# ── link-gap classification (remediation routing for the health report) ──────

def test_classify_link_gaps_splits_auto_create_and_manual():
    # 'shared' is linked from 2 pages (recalibration creates it); 'solo' from 1
    # (needs a human).
    pages = {
        "a.md": _page("Alpha", 50, "[[shared]] and [[solo]]"),
        "b.md": _page("Beta", 50, "[[shared]]"),
    }
    g = wh.classify_link_gaps(pages, min_refs=2)
    assert [c["slug"] for c in g["auto_create"]] == ["shared"]
    assert [c["slug"] for c in g["manual"]] == ["solo"]


def test_classify_link_gaps_flags_malformed_long_slug():
    long_slug = "a-conceptual-introduction-to-hamiltonian-monte-carlo-michael-betancourt"  # >60 chars
    pages = {"a.md": _page("Alpha", 50, f"[[{long_slug}]] and [[dbscan]]")}
    manual = {c["slug"]: c for c in wh.classify_link_gaps(pages)["manual"]}
    assert manual[long_slug]["malformed"] is True     # garbled title-as-link
    assert manual["dbscan"]["malformed"] is False     # genuine concept, just single-ref


def test_compute_article_candidates_matches_auto_create():
    pages = {"a.md": _page("A", 50, "[[x]]"), "b.md": _page("B", 50, "[[x]]")}
    assert wh.compute_article_candidates(pages) == wh.classify_link_gaps(pages)["auto_create"]
