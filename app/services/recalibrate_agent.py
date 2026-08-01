"""LangGraph master-recalibration agent — with programmatic triage.

Graph flow:
  START → load_content → fix_formatting → triage → analyze → apply_deletions
        → improve_pages → rebuild_index → finalize → END

Fix_formatting (pure Python, no LLM) repairs pages whose stored content was
wrapped in an outer ```markdown ... ``` fence (a known writer-model bug),
persists the fix, and updates state so downstream nodes see clean content.

Triage (pure Python, no LLM) scores every page by five health signals
and builds a shortlist. Analyze only sees the shortlist + its 1-hop
graph neighbours, keeping the LLM context manageable at any wiki size.
Improve_pages is semaphore-capped to avoid Bedrock rate limits.
"""

import asyncio
import hashlib
import json
import operator
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from app.logger import get_logger
from app.services import recalibrate_job, wiki_health
from app import model
from app.utils import (
    page_type_or_infer, parse_llm_json, strip_fence_unconditional, strip_outer_wrapper_fence,
)

log = get_logger(__name__)

_TODAY = lambda: datetime.now().strftime("%Y-%m-%d")  # noqa: E731

# Page types that recalibration must NOT touch. Query pages are user-saved Q&A,
# not LLM-authored knowledge — sweeping them would silently delete user content.
_SKIP_TYPES = {"query_result"}
_SKIP_FILES = {"index.md", "log.md"}

# Triage thresholds — single source of truth lives in wiki_health so triage and
# the (deterministic) health score can never drift apart.
_STALENESS_DAYS = wiki_health.STALENESS_DAYS   # pages not updated in this many days → "stale"
_STUB_WORDS = wiki_health.STUB_WORDS           # pages below this word count → "stub"
_IMPROVE_SEMAPHORE = 20   # max parallel Bedrock calls during improve_pages

# Cap on how many flagged pages go into a single analyze *run*. On a large or
# long-neglected wiki, triage can flag thousands of pages; processing them all
# would blow the context window (and cost). Candidates are priority-sorted, so
# we keep the worst offenders and defer the rest to the next recalibration run.
# Deleted-source pages are always included on top.
_ANALYZE_MAX_SHORTLIST = 200

# The shortlist is processed in batches of this many flagged pages, one LLM call
# per batch (run concurrently under the improve semaphore). This keeps each
# call's OUTPUT bounded — the analyze plan grows with the number of flagged
# pages, so a single call over the whole shortlist will eventually exceed the
# model's max output tokens and truncate mid-JSON. Batching makes output scale
# by adding calls rather than growing one response. Mirrors improve_pages.
_ANALYZE_BATCH_SIZE = 20

# How many content-gap concepts / recent pages to surface in the analyze prompt.
# Bounded so a large wiki can't bloat the prompt.
_GAP_SHORTLIST = 15
_RECENT_SHORTLIST = 10


# ── State ──────────────────────────────────────────────────────────────────

class RecalibrateState(TypedDict):
    wiki_pages: dict[str, str]
    schema: str
    index: str
    deleted_files: list[str]
    fact_instructions: str
    triage_candidates: list[dict]   # [{path, reasons, priority}]
    duplicate_groups: list[list[str]]  # title-duplicate groups (≥2 paths) — kept atomic when batching analyze
    article_candidates: list[dict]  # concepts referenced by ≥2 pages but with no page yet (content gaps)
    recent_pages: list[str]         # newest-ingested page paths, for stale reconciliation
    improvement_plan: list[dict]
    new_page_plan: list[dict]
    delete_plan: list[str]
    rename_plan: list[dict]
    pages_written: Annotated[list, operator.add]
    pages_deleted: Annotated[list, operator.add]
    pages_renamed: Annotated[list, operator.add]
    pages_format_fixed: list[str]   # programmatic markdown repairs applied before LLM analysis
    errors: Annotated[list, operator.add]
    log_entry: str
    health_score: int               # deterministic score at triage time (for the no-op report)
    _t_start: float


# ── Helpers ────────────────────────────────────────────────────────────────

def _content_sha(content: str) -> str:
    """Stable hash of a page's content — tells whether a page has changed since
    its last recalibration review."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _actionable_candidates(
    candidates: list[dict],
    wiki_pages: dict[str, str],
    context_sigs: dict[str, str],
    recal_state: dict[str, dict],
) -> list[dict]:
    """Drop flagged pages already reviewed at their *current* content AND context.

    A candidate survives if it was never recalibrated, its content changed since
    review, or its relational context changed — a neighbour added/removed a link,
    a link target vanished, or another page took its title. Skipping only when
    BOTH are unchanged is provably safe: every input to the page's signals is
    identical, so re-running couldn't produce anything new. This keeps
    recalibration idempotent (10× on a healthy wiki → 9 no-ops) while still
    re-reviewing a page when a neighbour change flips one of its relational
    signals (orphan / broken_links / duplicate)."""
    out: list[dict] = []
    for c in candidates:
        path = c["path"]
        prev = recal_state.get(path)
        if (prev is None
                or prev.get("content_sha") != _content_sha(wiki_pages.get(path, ""))
                or prev.get("context_sha", "") != context_sigs.get(path, "")):
            out.append(c)
    return out


def _page_summary(path: str, content: str, max_chars: int = 280) -> str:
    lines = content.strip().splitlines()
    title = next((l.lstrip("#").strip() for l in lines if l.startswith("#")), path)
    body = " ".join(l for l in lines if l and not l.startswith("#"))
    return f"[{path}] {title} — {body[:max_chars].strip()}"


def _is_safe(path: str, content: str = "") -> bool:
    """A page is safe to touch unless it's a special file or a protected type.

    `content` is optional — when not supplied (e.g. for a rename destination
    path that doesn't yet have content), we fall back to path-prefix
    inference. This matches the legacy behaviour for unmigrated pages.
    """
    if path in _SKIP_FILES:
        return False
    return page_type_or_infer(path, content) not in _SKIP_TYPES


def _dedup_by_path(items: list[dict]) -> list[dict]:
    """Keep the first dict per ``path``. A page can surface in two batches as a
    shared graph-neighbour suggestion; this prevents duplicate improve/create
    entries from the merge."""
    seen: set[str] = set()
    out: list[dict] = []
    for it in items:
        path = it.get("path")
        if not path or path in seen:
            continue
        seen.add(path)
        out.append(it)
    return out


def _assign_batches(
    shortlist: set[str],
    adjacency: dict[str, set[str]],
    duplicate_groups: list[list[str]],
    batch_size: int,
) -> list[list[str]]:
    """Group flagged pages into batches so cross-page judgements survive batching.

    Two co-location signals:
      - **Link components** (direction 1): pages in the same connected component
        of the (undirected) link graph are ordered adjacently, so topically
        linked pages tend to share a batch. ``adjacency`` must already be
        undirected and restricted to the shortlist.
      - **Duplicate groups** (direction 2): all members of a title-duplicate
        group are kept in the same batch (an atomic unit), so the duplicate can
        actually be resolved in one call. Members are also unioned into one
        component for ordering.

    Packing is greedy first-fit over component-ordered units; a unit larger than
    ``batch_size`` (only possible for a pathological duplicate group) gets its
    own batch rather than being split. Pure + deterministic.
    """
    if not shortlist:
        return []

    # ── union-find over the shortlist ──────────────────────────────────────
    parent = {p: p for p in shortlist}

    def find(x: str) -> str:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:      # path compression
            parent[x], x = root, parent[x]
        return root

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for p, neighbours in adjacency.items():
        if p not in parent:
            continue
        for q in neighbours:
            if q in parent:
                union(p, q)
    # Duplicate members share a component even when they aren't linked.
    for grp in duplicate_groups:
        members = [m for m in grp if m in parent]
        for m in members[1:]:
            union(members[0], m)

    # Stable component ordering: first time a component's root is seen in sorted
    # path order fixes its rank, so batch assignment is deterministic.
    root_rank: dict[str, int] = {}
    for p in sorted(shortlist):
        r = find(p)
        if r not in root_rank:
            root_rank[r] = len(root_rank)

    # ── build atomic units: duplicate groups first, then singletons ────────
    units: list[list[str]] = []
    used: set[str] = set()
    for grp in duplicate_groups:
        members = [m for m in grp if m in shortlist and m not in used]
        if members:
            units.append(members)
            used.update(members)
    for p in sorted(shortlist):
        if p not in used:
            units.append([p])
            used.add(p)

    units.sort(key=lambda u: (root_rank[find(u[0])], min(u)))

    # ── greedy first-fit packing ───────────────────────────────────────────
    batches: list[list[str]] = []
    cur: list[str] = []
    for u in units:
        if len(u) > batch_size:           # pathological oversized duplicate group
            if cur:
                batches.append(cur)
                cur = []
            batches.append(u)
            continue
        if cur and len(cur) + len(u) > batch_size:
            batches.append(cur)
            cur = []
        cur.extend(u)
    if cur:
        batches.append(cur)
    return batches


def _fold_pair_findings(plan: list[dict], findings: list[dict], make_instruction) -> list[dict]:
    """Fold cross-page findings (contradictions, duplicates) into the improve plan.

    Each finding has ``page_a``/``page_b``; ``make_instruction(finding)`` returns
    the instruction text. If a referenced page is already in the plan the
    instruction is appended (``|``-joined); otherwise a new improve entry is
    added. Mutates and returns ``plan``."""
    if not findings:
        return plan
    already = {p["path"] for p in plan if p.get("path")}
    for f in findings:
        if not isinstance(f, dict):
            continue
        instr = make_instruction(f)
        for key in ("page_a", "page_b"):
            path = f.get(key)
            if not path:
                continue
            if path in already:
                for p in plan:
                    if p.get("path") == path:
                        p["instruction"] = (p.get("instruction", "") + " | " + instr).strip(" |")
                        break
            else:
                plan.append({"path": path, "instruction": instr})
                already.add(path)
    return plan


def _contradiction_instruction(ct: dict) -> str:
    desc = ct.get("description", "conflicting claims")
    return f"CONTRADICTION: {desc}. Add '## Conflicting Sources' section."


def _duplicate_instruction(dup: dict) -> str:
    note = dup.get("note") or dup.get("description") or "near-identical coverage"
    other = dup.get("page_b") or dup.get("page_a") or "another page"
    return f"POSSIBLE DUPLICATE (overlaps {other}): {note}. Merge the pages or clearly differentiate them."


def _merge_analyze_plans(batch_results: list[dict]) -> dict:
    """Merge per-batch analyze outputs into a single plan.

    Concatenates each bucket across batches, drops malformed entries, dedups
    improve/create by path, folds contradictions into the improve plan, and
    applies the delete/rename-vs-improve overlap guard (deletes and renames win,
    since apply_deletions runs before improve_pages). Pure and testable —
    independent of the LLM, graph, or job state.

    Returns ``{improvement_plan, new_page_plan, delete_plan, rename_plan,
    summaries}``. ``delete_plan`` is a de-duplicated list of paths.
    """
    plan: list[dict] = []
    new_pages: list[dict] = []
    delete_plan: list = []
    rename_plan: list[dict] = []
    contradictions: list[dict] = []
    summaries: list[str] = []
    for data in batch_results:
        if not isinstance(data, dict):
            continue
        plan += [p for p in (data.get("pages_to_improve") or []) if isinstance(p, dict) and p.get("path")]
        new_pages += [p for p in (data.get("new_pages") or []) if isinstance(p, dict) and p.get("path")]
        delete_plan += data.get("pages_to_delete") or []
        rename_plan += [r for r in (data.get("pages_to_rename") or []) if isinstance(r, dict)]
        contradictions += [c for c in (data.get("contradictions") or []) if isinstance(c, dict)]
        if data.get("summary"):
            summaries.append(str(data["summary"]))

    plan = _dedup_by_path(plan)
    new_pages = _dedup_by_path(new_pages)

    if contradictions:
        log.info("Recalibrate | %d contradictions found", len(contradictions))
        plan = _fold_pair_findings(plan, contradictions, _contradiction_instruction)

    # Guard against the LLM listing the same page in conflicting buckets.
    # apply_deletions runs before improve_pages, so a path in both delete (or
    # rename-from) and improve would be deleted and then silently recreated by
    # the improve write. Deletes and renames win; drop the overlapping entries.
    delete_paths = {(d.get("path") if isinstance(d, dict) else d) for d in delete_plan}
    delete_paths.discard(None)
    delete_paths.discard("")
    rename_from = {r.get("from") for r in rename_plan if isinstance(r, dict) and r.get("from")}
    blocked = delete_paths | rename_from
    if blocked:
        dropped = [p for p in plan if p.get("path") in blocked]
        dropped += [p for p in new_pages if p.get("path") in blocked]
        if dropped:
            log.warning(
                "Recalibrate | dropped %d improve/create item(s) overlapping "
                "delete/rename targets: %s",
                len(dropped), sorted({p.get("path") for p in dropped}),
            )
        plan = [p for p in plan if p.get("path") not in blocked]
        new_pages = [p for p in new_pages if p.get("path") not in blocked]

    return {
        "improvement_plan": plan,
        "new_page_plan": new_pages,
        "delete_plan": list(delete_paths),
        "rename_plan": rename_plan,
        "summaries": summaries,
    }


# ── Graph builder ──────────────────────────────────────────────────────────

def build_recalibrate_graph():
    strong_llm = model.get_chat(model.Role.RECALIBRATE, max_tokens=16384, operation="recalibrate_analyze")
    writer_llm = model.get_chat(model.Role.RECALIBRATE, max_tokens=8192, operation="recalibrate_write")
    _MAX_CONTINUATIONS = 2
    _sem = asyncio.Semaphore(_IMPROVE_SEMAPHORE)

    async def _invoke_with_continuation(messages: list) -> str:
        response = await writer_llm.ainvoke(messages)
        parts = [response.content if isinstance(response.content, str) else ""]
        for _ in range(_MAX_CONTINUATIONS):
            stop_reason = (
                response.response_metadata.get("stopReason")
                or response.response_metadata.get("stop_reason")
                or ""
            )
            if stop_reason != "max_tokens":
                break
            log.warning("Recalibrate | writer truncated (max_tokens), requesting continuation")
            continuation_messages = list(messages) + [
                AIMessage(content=parts[-1]),
                HumanMessage(content="Continue exactly where you left off. Do not repeat any content already written."),
            ]
            response = await writer_llm.ainvoke(continuation_messages)
            parts.append(response.content if isinstance(response.content, str) else "")
        return "".join(parts)

    # ── load_content ──────────────────────────────────────────────────────
    async def load_content_node(state: RecalibrateState) -> dict:
        job = recalibrate_job.get()
        job.stage = "Loading wiki content"
        job.progress = 8
        job.details = "Reading all wiki pages from database…"

        from app.services.wiki_db import get_wiki_file, get_compact_index, list_wiki_pages_with_content
        schema = await get_wiki_file("schema/AGENTS.md") or ""
        index = await get_compact_index()

        wiki_pages: dict[str, str] = {}
        for rel, content in await list_wiki_pages_with_content():
            if rel in _SKIP_FILES:
                continue
            if page_type_or_infer(rel, content) in _SKIP_TYPES:
                continue
            if content:
                wiki_pages[rel] = content

        log.info("Recalibrate | load_content: %d pages from DB", len(wiki_pages))
        job.details = f"Loaded {len(wiki_pages)} pages"

        return {
            "wiki_pages": wiki_pages,
            "schema": schema,
            "index": index,
            "_t_start": time.time(),
        }

    # ── fix_formatting ────────────────────────────────────────────────────
    async def fix_formatting_node(state: RecalibrateState) -> dict:
        """Programmatically repair pages whose content was wrapped in a code fence.

        Conservative — only fixes the outer-fence wrapper bug (whole page wrapped
        in ```markdown ... ```). Other formatting issues are left for the
        LLM-driven analyze/improve nodes. Persisted immediately so downstream
        nodes see the fixed content and the fix survives even if a later node
        errors out.
        """
        job = recalibrate_job.get()
        job.stage = "Fixing broken markdown"
        job.progress = 10
        job.details = "Scanning pages for wrapping code fences…"

        from app.services.wiki_db import upsert_wiki_page

        fixed_pages: list[str] = []
        updated_pages = dict(state["wiki_pages"])

        for path, content in state["wiki_pages"].items():
            new_content, changed = strip_outer_wrapper_fence(content)
            if changed:
                await upsert_wiki_page(path, new_content)
                updated_pages[path] = new_content
                fixed_pages.append(path)
                log.info("Recalibrate | format-fix | stripped outer fence | path=%s", path)

        if fixed_pages:
            log.info(
                "Recalibrate | format-fix complete | fixed=%d/%d pages",
                len(fixed_pages), len(state["wiki_pages"]),
            )
            job.details = f"Fixed markdown on {len(fixed_pages)} page(s)"
        else:
            log.info("Recalibrate | format-fix complete | no fixes needed")
            job.details = "All pages had clean markdown"

        return {
            "wiki_pages": updated_pages,
            "pages_format_fixed": fixed_pages,
        }

    # ── triage ────────────────────────────────────────────────────────────
    async def triage_node(state: RecalibrateState) -> dict:
        """Score pages by health signals, or do semantic search in targeted mode."""
        job = recalibrate_job.get()
        job.stage = "Triaging pages"
        job.progress = 12
        wiki_pages = state["wiki_pages"]

        fact_instructions = state.get("fact_instructions", "")
        if fact_instructions:
            job.details = "Targeted mode: finding relevant pages via semantic search…"
            from app.services import embeddings
            from app.services.wiki_db import semantic_search_wiki
            vec = await embeddings.embed_text(fact_instructions)
            candidates: list[dict] = []
            if vec:
                results = await semantic_search_wiki(vec, top_k=15)
                for r in results:
                    path = r.get("path", "")
                    if path in wiki_pages and _is_safe(path):
                        candidates.append({"path": path, "reasons": ["targeted"], "priority": 1})
            log.info("Recalibrate | triage (targeted): %d pages via semantic search", len(candidates))
            job.details = f"Targeted: {len(candidates)} relevant pages found"
            # Targeted runs stay narrowly scoped — skip content-gap/recency scans.
            job.progress = 17
            return {"triage_candidates": candidates, "duplicate_groups": []}

        job.details = "Scanning for staleness, orphans, stubs, broken links, duplicates…"

        # All five structural signals + the duplicate groups come from the shared
        # wiki_health module — the same code the (deterministic) health score uses,
        # so triage and the score can never disagree about what's wrong.
        from app.services.graph import get_graph
        from app.services.wiki_db import get_recalibration_state
        graph = await get_graph().ensure_loaded()

        # Prior recalibration fingerprints: {path: {"sha", "reviewed_at"}}. Review
        # times feed staleness (a reviewed page is fresh again); content hashes let
        # us skip pages already reviewed at their current text.
        recal_state = await get_recalibration_state()
        reviewed_at = {p: info["reviewed_at"] for p, info in recal_state.items()}
        signals = wiki_health.compute_signals(wiki_pages, graph._adj, reviewed_at=reviewed_at)
        context_sigs = wiki_health.compute_context_signatures(wiki_pages)
        counts = signals["counts"]
        health_score = wiki_health.score_from_counts(counts, signals["total"])

        flagged: list[dict] = [
            {"path": path, "reasons": reasons, "priority": len(reasons)}
            for path, reasons in signals["reasons"].items()
        ]
        # Idempotency (Layer 3): drop flagged pages already reviewed at their
        # current content. Combined with review-based staleness, a healthy or
        # already-recalibrated wiki yields an empty shortlist → analyze/improve do
        # no LLM work → the run is a no-op.
        candidates = _actionable_candidates(flagged, wiki_pages, context_sigs, recal_state)
        candidates.sort(key=lambda x: -x["priority"])
        duplicate_groups = signals["duplicate_groups"]

        log.info(
            "Recalibrate | triage: %d flagged, %d actionable / %d pages "
            "(stale=%d orphan=%d stub=%d broken=%d dup=%d) | score=%d",
            len(flagged), len(candidates), len(wiki_pages),
            counts["stale"], counts["orphan"], counts["stub"],
            counts["broken_links"], counts["duplicate_candidate"], health_score,
        )

        # Content gaps (#4) + newest-data context (#2) for the analyze step.
        article_candidates = wiki_health.compute_article_candidates(wiki_pages)
        recent_pages = wiki_health.select_recent_pages(wiki_pages)

        log.info("Recalibrate | content gaps: %d candidate concept(s)", len(article_candidates))
        job.details = (
            f"Triage: {len(candidates)} of {len(flagged)} flagged pages need work "
            f"({len(wiki_pages)} total, health {health_score}/100)"
        )
        job.progress = 17

        return {
            "triage_candidates": candidates,
            "duplicate_groups": duplicate_groups,
            "article_candidates": article_candidates,
            "recent_pages": recent_pages,
            "health_score": health_score,
        }

    # ── analyze ───────────────────────────────────────────────────────────
    async def analyze_node(state: RecalibrateState) -> dict:
        job = recalibrate_job.get()
        job.stage = "Analyzing flagged pages"
        job.progress = 20

        wiki_pages   = state["wiki_pages"]
        candidates   = state.get("triage_candidates", [])
        deleted_files = state.get("deleted_files") or []

        # Force-include pages sourced from deleted files regardless of triage
        deleted_source_pages: set[str] = set()
        if deleted_files:
            for path, content in wiki_pages.items():
                if any(f"uploaded_file: {fn}" in content for fn in deleted_files):
                    deleted_source_pages.add(path)

        # Cap the flagged set so the analyze prompt stays bounded regardless of
        # wiki size. `candidates` is priority-sorted (most signals first), so we
        # keep the worst offenders; the rest get picked up on the next run.
        # Deleted-source pages bypass the cap — they must always be addressed.
        capped = candidates[:_ANALYZE_MAX_SHORTLIST]
        deferred = max(0, len(candidates) - len(capped))
        if deferred:
            log.warning(
                "Recalibrate | analyze: %d candidates flagged, capping shortlist to %d "
                "(%d deferred to next run)",
                len(candidates), len(capped), deferred,
            )
        shortlist: set[str] = {c["path"] for c in capped} | deleted_source_pages

        # Content gaps (missing pages that broken links point to) are real,
        # fixable work even when NO existing page is flagged — the fix is to
        # CREATE the missing target page, not to rewrite the linking page (whose
        # own content is unchanged, so idempotency correctly skips it). The run is
        # therefore only a true no-op when there are no flagged pages AND no gaps;
        # otherwise recalibration would report "nothing to do" while the health
        # check still lists the broken links.
        article_candidates = state.get("article_candidates", [])
        if not shortlist and not article_candidates:
            log.info("Recalibrate | analyze: no flagged pages and no content gaps — skipping LLM analysis")
            job.details = "All pages healthy — nothing to analyze"
            return {
                "improvement_plan": [],
                "new_page_plan": [],
                "delete_plan": [],
                "rename_plan": [],
                "log_entry": "",
            }

        # Expand shortlist with 1-hop graph neighbours for context. Build a
        # reverse-adjacency map in a single pass first — looking up inlinks by
        # rescanning the whole graph per shortlist entry is O(shortlist × edges)
        # and blocks the event loop on large wikis.
        from app.services.graph import get_graph
        graph = await get_graph().ensure_loaded()
        in_adj: dict[str, set[str]] = defaultdict(set)
        for src, targets in graph._adj.items():
            for t in targets:
                in_adj[t].add(src)
        neighbour_paths: set[str] = set()
        for path in shortlist:
            for target in graph._adj.get(path, set()):
                if target in wiki_pages and target not in shortlist:
                    neighbour_paths.add(target)
            for src in in_adj.get(path, set()):
                if src in wiki_pages and src not in shortlist:
                    neighbour_paths.add(src)

        job.details = (
            f"Analyzing {len(shortlist)} flagged pages "
            f"(+ {len(neighbour_paths)} neighbours for context)…"
        )

        # Build prompt content. Per-page flag labels; the flagged/neighbour page
        # bodies are rendered per batch below.
        flag_map = {c["path"]: ", ".join(c["reasons"]) for c in candidates}

        deleted_section = ""
        if deleted_files:
            files_list = "\n".join(f"  - {f}" for f in deleted_files)
            deleted_section = f"""
CRITICAL — Source files deleted by the user:
{files_list}

Any wiki pages whose content was DERIVED FROM these deleted files MUST be listed
in pages_to_delete or pages_to_improve with an explicit instruction to strip all
content sourced exclusively from those files.
"""

        fact_instructions = state.get("fact_instructions", "")
        targeted_directive = ""
        if fact_instructions:
            targeted_directive = (
                f"USER-REPORTED FACTUAL ERROR (PRIMARY TASK):\n"
                f"{fact_instructions}\n\n"
                f"Your PRIMARY task is to find and fix these specific issues in the pages below. "
                f"Do not delete or rename pages unless directly required by the fix.\n\n"
            )

        # Content gaps (#4): concepts many pages link to but that have no page
        # (already read above for the no-op check).
        gaps_section = ""
        if article_candidates:
            gap_lines = "\n".join(
                f"  - [[{c['slug']}]] — referenced by {c['ref_count']} pages "
                f"({', '.join(c['referenced_by'][:5])})"
                for c in article_candidates[:_GAP_SHORTLIST]
            )
            gaps_section = (
                "\nCONTENT GAPS — concepts referenced via [[wikilinks]] from multiple pages "
                "but with no page of their own. Create pages for the well-supported ones "
                "(use new_pages); pick a path matching the schema's directory conventions:\n"
                f"{gap_lines}\n"
            )

        # Newest data (#2): reconcile stale pages against the latest ingests.
        recent_pages = state.get("recent_pages", [])
        recent_section = ""
        if recent_pages:
            recent_section = (
                "\nNEWEST INGESTED PAGES (treat as the most current source of truth when "
                "updating 'stale' pages — prefer their facts on conflict):\n  "
                + ", ".join(recent_pages[:_RECENT_SHORTLIST]) + "\n"
            )

        system = SystemMessage(content=f"""You are an expert wiki editor performing a targeted recalibration.
{targeted_directive}{deleted_section}{gaps_section}{recent_section}
You are shown only pages that were PROGRAMMATICALLY FLAGGED as needing attention.
Each page is labelled with the reason(s) it was flagged:
  - stale         : not updated in {_STALENESS_DAYS}+ days
  - orphan        : no inbound links — nothing links to this page
  - stub          : fewer than {_STUB_WORDS} words
  - broken_links  : contains references to non-existent pages
  - duplicate_candidate : another page has the same or very similar title
  - deleted_source: derived from a deleted source file

Your task:
1. DELETE pages that are redundant, superseded, or only from deleted sources.
2. RENAME/MOVE pages whose path no longer matches their content.
3. IMPROVE pages: fix broken links, expand stubs, resolve duplicates, and update
   'stale' pages to agree with the NEWEST INGESTED PAGES listed above.
4. CREATE new pages for the CONTENT GAPS listed above, or when a clear
   relationship gap exists between flagged pages.
5. DETECT CONTRADICTIONS between any two flagged pages.

Wiki Schema (authoritative reference for directory structure, page formats, and naming — derive all conventions from it):
{state['schema']}

Constraints:
- Only act on flagged pages (or their immediate neighbours). Do not invent work.
- Never delete or rename index.md or log.md.
- Be conservative with deletes.

Return ONLY a valid JSON object — no prose, no code fences:
{{
  "summary": "2-3 sentence assessment",
  "pages_to_delete": [{{"path": "...", "reason": "..."}}],
  "pages_to_rename": [{{"from": "...", "to": "...", "reason": "..."}}],
  "pages_to_improve": [{{"path": "...", "instruction": "..."}}],
  "contradictions": [{{"page_a": "...", "page_b": "...", "description": "..."}}],
  "new_pages": [{{"path": "...", "brief": "..."}}],
  "log_entry": "## [{_TODAY()}] recalibrate\\nSummary."
}}""")

        # Batch the shortlist so each LLM call's OUTPUT stays bounded (the plan
        # grows with flagged count, so one call over the whole shortlist would
        # eventually truncate mid-JSON). _assign_batches co-locates pages that
        # must be judged together: members of a connected link component are
        # ordered adjacently (so contradictions between linked pages stay in one
        # batch), and title-duplicate groups are kept atomic in a single batch.
        # Batches run concurrently under the shared improve semaphore.
        undirected: dict[str, set[str]] = defaultdict(set)
        for p in shortlist:
            for q in graph._adj.get(p, set()):
                if q in shortlist:
                    undirected[p].add(q)
                    undirected[q].add(p)
            for q in in_adj.get(p, set()):
                if q in shortlist:
                    undirected[p].add(q)
                    undirected[q].add(p)
        duplicate_groups = state.get("duplicate_groups", [])
        batches = _assign_batches(shortlist, undirected, duplicate_groups, _ANALYZE_BATCH_SIZE)
        log.info(
            "Recalibrate | analyze: %d flagged + %d neighbours → %d batch(es) of ≤%d "
            "(%d duplicate group(s) kept atomic) → LLM",
            len(shortlist), len(neighbour_paths), len(batches), _ANALYZE_BATCH_SIZE,
            len(duplicate_groups),
        )

        async def _analyze_batch(batch: list[str], idx: int) -> dict:
            """Analyze one batch of flagged pages. Returns the parsed plan buckets,
            or {} if the call/parse fails — a single bad batch must not sink the
            whole run (the old single-call path discarded all analysis on any
            error, which is exactly the truncation failure mode we're fixing)."""
            # Per-batch graph context: neighbours of this batch's pages that are
            # not themselves flagged (flagged pages are handled in their own batch).
            nbrs: set[str] = set()
            for path in batch:
                for target in graph._adj.get(path, set()):
                    if target in wiki_pages and target not in shortlist:
                        nbrs.add(target)
                for src in in_adj.get(path, set()):
                    if src in wiki_pages and src not in shortlist:
                        nbrs.add(src)

            flagged_lines = [
                f"[FLAGS: {flag_map.get(p, 'deleted_source')}]\n{_page_summary(p, wiki_pages.get(p, ''))}"
                for p in batch
            ]
            context_lines = [
                f"[context only — not flagged]\n{_page_summary(p, wiki_pages[p])}"
                for p in sorted(nbrs)
            ]
            secs = ["=== FLAGGED PAGES ==="] + flagged_lines
            if context_lines:
                secs += ["", "=== GRAPH NEIGHBOURS (context only) ==="] + context_lines
            human = HumanMessage(content="\n\n".join(secs))

            try:
                async with _sem:
                    response = await strong_llm.ainvoke([system, human])
                text = response.content if isinstance(response.content, str) else ""
                stop_reason = (
                    response.response_metadata.get("stopReason")
                    or response.response_metadata.get("stop_reason")
                    or ""
                )
                if stop_reason == "max_tokens":
                    log.warning(
                        "Recalibrate | analyze batch %d/%d hit max_tokens — its plan "
                        "may be partial (lower _ANALYZE_BATCH_SIZE if this recurs)",
                        idx + 1, len(batches),
                    )
                return parse_llm_json(text)
            except Exception as e:
                log.error("Recalibrate | analyze batch %d/%d failed: %s", idx + 1, len(batches), e)
                return {}

        async def _analyze_gaps_only() -> dict:
            """Create pages for content gaps when NO existing page is flagged (e.g.
            the only issue is missing pages that broken links point to). The system
            prompt already lists the CONTENT GAPS; this asks only for their creation."""
            human = HumanMessage(content=(
                "No existing pages are flagged this run. Using the CONTENT GAPS listed "
                'above, CREATE the well-supported missing pages (populate "new_pages"). '
                "Leave the other arrays empty. Return the same JSON object."
            ))
            try:
                async with _sem:
                    response = await strong_llm.ainvoke([system, human])
                text = response.content if isinstance(response.content, str) else ""
                return parse_llm_json(text)
            except Exception as e:
                log.error("Recalibrate | gap-only analysis failed: %s", e)
                return {}

        async def _reconcile_cross_batch(card_paths: list[str]) -> tuple[list[dict], list[dict]]:
            """One bounded pass to catch contradictions / near-duplicates that
            SPAN batches — individual batches only ever saw a subset of pages.
            Input is one compact card per flagged page (path + short excerpt);
            output is only the offending *pairs*, so it stays small regardless of
            how many pages are flagged."""
            cards = "\n".join(
                f"- {p} :: {_page_summary(p, wiki_pages.get(p, ''))}" for p in card_paths
            )
            rec_system = SystemMessage(content=(
                "You are auditing a wiki for cross-page issues. Below is one line per "
                "flagged page: its path, then a short excerpt. These pages were analyzed "
                "in SEPARATE batches, so factual CONTRADICTIONS between two pages, or "
                "near-DUPLICATE pages covering the same thing, may have been missed.\n\n"
                "Report ONLY such cross-page issues. Reference pages strictly by the "
                "paths shown — never invent a path. Be conservative: list a pair only "
                "when you are confident.\n\n"
                "Return ONLY valid JSON — no prose, no code fences:\n"
                '{"contradictions": [{"page_a": "...", "page_b": "...", "description": "..."}],\n'
                ' "duplicates": [{"page_a": "...", "page_b": "...", "note": "..."}]}'
            ))
            try:
                async with _sem:
                    response = await strong_llm.ainvoke([rec_system, HumanMessage(content=cards)])
                text = response.content if isinstance(response.content, str) else ""
                data = parse_llm_json(text)
                cons = [c for c in (data.get("contradictions") or []) if isinstance(c, dict)]
                dups = [d for d in (data.get("duplicates") or []) if isinstance(d, dict)]
                return cons, dups
            except Exception as e:
                log.warning("Recalibrate | cross-batch reconciliation failed (skipping): %s", e)
                return [], []

        if batches:
            batch_results = await asyncio.gather(
                *(_analyze_batch(b, i) for i, b in enumerate(batches))
            )
        else:
            # No flagged existing pages, but content gaps exist (guaranteed by the
            # short-circuit above) — run one pass dedicated to creating them.
            log.info("Recalibrate | analyze: no flagged pages — gap-only creation pass (%d gap(s))",
                     len(article_candidates))
            batch_results = [await _analyze_gaps_only()]

        merged = _merge_analyze_plans(batch_results)
        plan        = merged["improvement_plan"]
        new_pages   = merged["new_page_plan"]
        delete_plan = merged["delete_plan"]
        rename_plan = merged["rename_plan"]
        summaries   = merged["summaries"]

        # Every batch failing (e.g. all truncated) yields no plan and no summary.
        # Surface that as an error rather than silently reporting a healthy wiki.
        if not summaries and not any([plan, new_pages, delete_plan, rename_plan]):
            log.error("Recalibrate | analyze: all %d batch(es) returned no usable plan", len(batches))
            job.details = "Analyze produced no plan (all batches failed or empty)"
            return {
                "improvement_plan": [],
                "new_page_plan": [],
                "delete_plan": [],
                "rename_plan": [],
                "errors": ["Analyze failed: no usable plan from any batch"],
            }

        # ── Cross-batch reconciliation ─────────────────────────────────────
        # Per-batch analysis can't see contradictions/duplicates that span
        # batches. Batching co-locates most related pages (link components +
        # atomic duplicate groups), but unlinked pages in different batches can
        # still conflict. One bounded pass over compact cards of the whole
        # shortlist catches the remainder; findings fold into the improve plan.
        # Only meaningful when there is more than one batch.
        if len(batches) > 1:
            rec_cons, rec_dups = await _reconcile_cross_batch(sorted(shortlist))
            # Only trust findings whose BOTH pages are in the shortlist (guards
            # against hallucinated paths folding bogus entries into the plan).
            rec_cons = [c for c in rec_cons if c.get("page_a") in shortlist and c.get("page_b") in shortlist]
            rec_dups = [d for d in rec_dups if d.get("page_a") in shortlist and d.get("page_b") in shortlist]
            if rec_cons or rec_dups:
                log.info(
                    "Recalibrate | cross-batch reconcile: +%d contradiction(s) +%d duplicate(s)",
                    len(rec_cons), len(rec_dups),
                )
                plan = _fold_pair_findings(plan, rec_cons, _contradiction_instruction)
                plan = _fold_pair_findings(plan, rec_dups, _duplicate_instruction)
                # Folding may add improve entries for pages already slated for
                # delete/rename — re-apply the overlap guard so improve_pages
                # can't resurrect a removed page.
                blocked = set(delete_plan) | {
                    r.get("from") for r in rename_plan if isinstance(r, dict) and r.get("from")
                }
                if blocked:
                    plan = [p for p in plan if p.get("path") not in blocked]

        log.info(
            "Recalibrate | plan: %d improve + %d new + %d delete + %d rename (merged from %d batch(es))",
            len(plan), len(new_pages), len(delete_plan), len(rename_plan), len(batches),
        )
        job.details = (
            f"Plan: {len(plan)} improve, {len(new_pages)} create, "
            f"{len(delete_plan)} delete, {len(rename_plan)} rename"
        )
        job.progress = 30
        log_entry = f"## [{_TODAY()}] recalibrate\n" + " ".join(summaries) if summaries else ""
        return {
            "improvement_plan": plan,
            "new_page_plan": new_pages,
            "delete_plan": delete_plan,
            "rename_plan": rename_plan,
            "log_entry": log_entry,
        }

    # ── apply_deletions ───────────────────────────────────────────────────
    async def apply_deletions_node(state: RecalibrateState) -> dict:
        job = recalibrate_job.get()
        job.stage = "Applying deletions & renames"
        job.progress = 32

        pages_deleted: list[str] = []
        pages_renamed: list[dict] = []
        errors: list[str] = []

        from app.services.wiki_db import get_wiki_page_content, upsert_wiki_page, delete_wiki_page, wiki_page_exists

        for entry in state.get("rename_plan", []):
            src_rel = entry.get("from", "")
            dst_rel = entry.get("to", "")
            if not src_rel or not dst_rel:
                continue
            content = await get_wiki_page_content(src_rel)
            if not _is_safe(src_rel, content or "") or not _is_safe(dst_rel):
                log.warning("Recalibrate | rename blocked (protected): %s → %s", src_rel, dst_rel)
                continue
            if not content:
                errors.append(f"Rename source not found: {src_rel}")
                continue
            await upsert_wiki_page(dst_rel, content)
            await delete_wiki_page(src_rel)
            pages_renamed.append({"from": src_rel, "to": dst_rel})
            log.info("Recalibrate | renamed %s → %s", src_rel, dst_rel)

        for path in state.get("delete_plan", []):
            existing = await get_wiki_page_content(path) or ""
            if not _is_safe(path, existing):
                log.warning("Recalibrate | delete blocked (protected): %s", path)
                continue
            if not await wiki_page_exists(path):
                log.warning("Recalibrate | delete target not found (skipped): %s", path)
                continue
            await delete_wiki_page(path)
            pages_deleted.append(path)
            log.info("Recalibrate | deleted %s", path)

        job.details = f"Deleted {len(pages_deleted)}, renamed {len(pages_renamed)}"
        return {
            "pages_deleted": pages_deleted,
            "pages_renamed": pages_renamed,
            "errors": errors,
        }

    # ── improve_pages ─────────────────────────────────────────────────────
    async def improve_pages_node(state: RecalibrateState) -> dict:
        all_work = [
            {"path": p["path"], "instruction": p.get("instruction", ""), "is_new": False}
            for p in state["improvement_plan"]
        ] + [
            {"path": p["path"], "instruction": p.get("brief", ""), "is_new": True}
            for p in state["new_page_plan"]
        ]

        if not all_work:
            log.info("Recalibrate | improve_pages: nothing to do")
            return {"pages_written": [], "errors": []}

        total = len(all_work)
        job = recalibrate_job.get()
        job.stage = "Improving pages"
        job.progress = 33
        job.details = f"Improving {total} pages (max {_IMPROVE_SEMAPHORE} concurrent)…"

        done_count = 0
        count_lock = asyncio.Lock()

        async def improve_one(work: dict) -> tuple[str, str]:
            nonlocal done_count
            path = work["path"]
            instruction = work["instruction"]
            is_new = work["is_new"]
            existing = state["wiki_pages"].get(path, "")

            if is_new:
                sys_content = (
                    f"You are a wiki author. "
                    f"The Wiki Schema below is the authoritative reference for page formats, "
                    f"frontmatter fields, directory conventions, and cross-linking syntax — follow it exactly.\n\n"
                    f"Wiki Schema:\n{state['schema']}\n\n"
                    "Return ONLY raw markdown — no code fences, no explanations."
                )
                human_content = (
                    f"Create a new wiki page.\n\nPath: {path}\n\nBrief: {instruction}\n\n"
                    f"Index context (for linking):\n{state['index'][:2500]}"
                )
            else:
                sys_content = (
                    f"You are a wiki editor improving an existing page. "
                    f"The Wiki Schema below is the authoritative reference for page formats, "
                    f"frontmatter fields, directory conventions, and cross-linking syntax — follow it exactly.\n\n"
                    f"Wiki Schema:\n{state['schema']}\n\n"
                    "Preserve all factual content. Only add/fix links, terminology, and structure.\n"
                    "Return ONLY the improved markdown — no code fences, no explanations."
                )
                human_content = (
                    f"Improve this wiki page.\n\nPath: {path}\n\nInstruction: {instruction}\n\n"
                    f"Current content:\n{existing}"
                )

            invoke_messages = [SystemMessage(content=sys_content), HumanMessage(content=human_content)]

            async with _sem:
                raw = await _invoke_with_continuation(invoke_messages)

            content = strip_fence_unconditional(raw)

            # Source pages need date_ingested preserved across rewrites — key off
            # frontmatter type (with path-prefix fallback for unmigrated pages).
            if page_type_or_infer(path, content) == "source_summary":
                today = _TODAY()
                if is_new:
                    content = re.sub(r"(?m)^date_ingested:.*$", f"date_ingested: {today}", content)
                    content = re.sub(r"(?m)^\*\*Date ingested\*\*:.*$", f"**Date ingested**: {today}", content)
                else:
                    fm = re.search(r"(?m)^date_ingested:\s*(.+)$", existing)
                    orig_fm = fm.group(1).strip() if fm else today
                    body = re.search(r"(?m)^\*\*Date ingested\*\*:\s*(.+)$", existing)
                    orig_body = body.group(1).strip() if body else today
                    content = re.sub(r"(?m)^date_ingested:.*$", f"date_ingested: {orig_fm}", content)
                    content = re.sub(r"(?m)^\*\*Date ingested\*\*:.*$", f"**Date ingested**: {orig_body}", content)

            async with count_lock:
                done_count += 1
                job.progress = 33 + int(done_count / total * 47)
                job.details = f"Improved {done_count}/{total} pages"

            return path, content

        results = await asyncio.gather(*[improve_one(w) for w in all_work], return_exceptions=True)

        errors: list[str] = []
        pages_written: list[str] = []

        for r in results:
            if isinstance(r, Exception):
                log.error("Recalibrate | page write failed: %s", r)
                errors.append(str(r))
                continue
            path, content = r
            from app.services.wiki_db import upsert_wiki_page
            await upsert_wiki_page(path, content)
            pages_written.append(path)
            log.info("Recalibrate | wrote %s", path)

        return {"pages_written": pages_written, "errors": errors}

    # ── rebuild_index ─────────────────────────────────────────────────────
    async def rebuild_index_node(state: RecalibrateState) -> dict:
        """Index is now derived dynamically from wiki_pages table — nothing to write."""
        job = recalibrate_job.get()
        job.stage = "Finalizing index"
        job.progress = 82
        job.details = "Index derived from DB — skipping blob write."
        n_pages = (
            len(state["wiki_pages"])
            + len(state.get("pages_written", []))
            - len(state.get("pages_deleted", []))
        )
        log.info("Recalibrate | rebuild_index: skipped (index is dynamic, ~%d pages in DB)", n_pages)
        return {}

    # ── finalize ──────────────────────────────────────────────────────────
    async def finalize_node(state: RecalibrateState) -> dict:
        job = recalibrate_job.get()
        job.stage = "Finalizing"
        job.progress = 96
        job.details = "Writing log entry…"

        elapsed   = round(time.time() - state["_t_start"], 1)
        n_written = len(state["pages_written"])
        n_deleted = len(state.get("pages_deleted", []))
        n_renamed = len(state.get("pages_renamed", []))
        n_errors  = len(state["errors"])
        n_flagged = len(state.get("triage_candidates", []))
        n_total   = len(state["wiki_pages"])
        health_score = state.get("health_score", 100)
        made_changes = bool(n_written or n_deleted or n_renamed or state.get("pages_format_fixed"))

        fact_instructions = state.get("fact_instructions", "")

        # ── Record recalibration fingerprints (idempotency) ─────────────────
        # Stamp reviewed pages with their current content hash + review time so a
        # re-run over unchanged content does no work and staleness is measured
        # from this review. Full runs review (and prune to) the whole live corpus;
        # targeted runs only the pages they touched. Best-effort — a failure here
        # must not fail the run (worst case: the next run reprocesses). Written to
        # its own table, so it is deliberately NOT change-tracked / revertible.
        try:
            from app.services.wiki_db import list_wiki_pages_with_content, record_recalibration
            final_pages = {
                rel: content
                for rel, content in await list_wiki_pages_with_content()
                if rel not in _SKIP_FILES
                and page_type_or_infer(rel, content) not in _SKIP_TYPES
                and content
            }
            # Context signatures computed over the FINAL corpus (post-writes), the
            # same pure function triage compares against next run.
            ctx = wiki_health.compute_context_signatures(final_pages)
            if fact_instructions:
                touched = set(state["pages_written"])
                fps = [(rel, _content_sha(c), ctx.get(rel, ""))
                       for rel, c in final_pages.items() if rel in touched]
                if fps:
                    await record_recalibration(fps, prune=False)
            else:
                fps = [(rel, _content_sha(c), ctx.get(rel, "")) for rel, c in final_pages.items()]
                await record_recalibration(fps, prune=True)
        except Exception:
            log.exception("Recalibrate | failed to record recalibration fingerprints")
        llm_summary = (state.get("log_entry") or "").strip()
        if fact_instructions:
            heading = f"## [{_TODAY()}] recalibrate (targeted)"
            header_lines: list[str] = [heading, f"> Fact instructions: {fact_instructions}", ""]
        else:
            header_lines = [f"## [{_TODAY()}] master-recalibrate", ""]
        lines: list[str] = header_lines + [
            f"**Duration:** {elapsed}s  |  **Triaged:** {n_flagged}/{n_total}  |  "
            f"**Improved:** {n_written}  |  **Deleted:** {n_deleted}  |  "
            f"**Renamed:** {n_renamed}  |  **Errors:** {n_errors}",
        ]
        if not made_changes and not fact_instructions and not n_errors:
            lines += ["", f"_No changes needed — wiki already in good shape (health {health_score}/100)._"]
        if llm_summary:
            summary_body = re.sub(r"^##.*\n", "", llm_summary, count=1).strip()
            if summary_body:
                lines += ["", summary_body]
        if state.get("pages_format_fixed"):
            lines += ["", "**Markdown formatting auto-fixed (outer code-fence wrapper stripped):**"] + \
                     [f"- `{p}`" for p in sorted(state["pages_format_fixed"])]
        if state.get("pages_written"):
            lines += ["", "**Improved / created:**"] + [f"- `{p}`" for p in sorted(state["pages_written"])]
        if state.get("pages_deleted"):
            lines += ["", "**Deleted:**"] + [f"- `{p}`" for p in sorted(state["pages_deleted"])]
        if state.get("pages_renamed"):
            lines += ["", "**Renamed:**"] + [f"- `{r['from']}` → `{r['to']}`" for r in state["pages_renamed"]]
        if state.get("errors"):
            lines += ["", "**Errors:**"] + [f"- {e}" for e in state["errors"]]

        log_entry = "\n".join(lines)

        from app.services.wiki_db import append_audit_log
        await append_audit_log("recalibrate", log_entry)

        log.info(
            "Recalibrate | finalize | flagged=%d/%d | pages=%d | errors=%d | elapsed=%.1fs",
            n_flagged, n_total, n_written, n_errors, elapsed,
        )

        final_status = "done_with_errors" if state["errors"] else "done"
        job.pages_improved = state["pages_written"]
        job.pages_deleted  = state.get("pages_deleted", [])
        job.pages_renamed  = state.get("pages_renamed", [])
        job.errors         = state["errors"]
        job.progress       = 100
        job.stage          = "Complete"
        if not made_changes and not fact_instructions and not n_errors:
            job.details = (
                f"Wiki already in good shape — reviewed {n_total} pages, "
                f"no changes needed (health {health_score}/100)"
            )
        else:
            job.details = (
                f"Triaged {n_flagged}/{n_total} pages — "
                f"improved {n_written}, deleted {n_deleted}, renamed {n_renamed} in {elapsed}s"
            )
        job.finished_at = datetime.now(timezone.utc).isoformat()
        job.status      = final_status

        await recalibrate_job.persist_finish(final_status)

        return {}

    # ── Wire graph ────────────────────────────────────────────────────────
    builder = StateGraph(RecalibrateState)
    builder.add_node("load_content",    load_content_node)
    builder.add_node("fix_formatting",  fix_formatting_node)
    builder.add_node("triage",          triage_node)
    builder.add_node("analyze",         analyze_node)
    builder.add_node("apply_deletions", apply_deletions_node)
    builder.add_node("improve_pages",   improve_pages_node)
    builder.add_node("rebuild_index",   rebuild_index_node)
    builder.add_node("finalize",        finalize_node)

    builder.add_edge(START,            "load_content")
    builder.add_edge("load_content",   "fix_formatting")
    builder.add_edge("fix_formatting", "triage")
    builder.add_edge("triage",         "analyze")
    builder.add_edge("analyze",        "apply_deletions")
    builder.add_edge("apply_deletions","improve_pages")
    builder.add_edge("improve_pages",  "rebuild_index")
    builder.add_edge("rebuild_index",  "finalize")
    builder.add_edge("finalize",       END)

    return builder.compile()


# ── Public runner ──────────────────────────────────────────────────────────

class RecalibrateAgentRunner:
    def __init__(self):
        self.graph = build_recalibrate_graph()

    async def run(self, deleted_files: list[str] | None = None, fact_instructions: str = "") -> None:
        from app.services.wiki_state import begin_action
        action_type = "targeted_recalibrate" if fact_instructions else "recalibrate"
        summary = (
            f"targeted: {fact_instructions[:80]}" if fact_instructions
            else "full recalibration"
        )
        initial_state: RecalibrateState = {
            "wiki_pages":        {},
            "schema":            "",
            "index":             "",
            "deleted_files":     deleted_files or [],
            "fact_instructions": fact_instructions,
            "triage_candidates": [],
            "duplicate_groups":  [],
            "improvement_plan":  [],
            "new_page_plan":     [],
            "delete_plan":       [],
            "rename_plan":       [],
            "pages_written":     [],
            "pages_deleted":     [],
            "pages_renamed":     [],
            "pages_format_fixed": [],
            "errors":            [],
            "log_entry":         "",
            "health_score":      100,
            "_t_start":          time.time(),
        }
        async with begin_action(action_type, summary=summary,
                                details={"deleted_files": deleted_files or []}):
            await self.graph.ainvoke(initial_state)
