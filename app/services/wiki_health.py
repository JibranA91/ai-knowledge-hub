"""Deterministic structural-health signals and score for the wiki.

Single source of truth for the five structural health signals — stale, orphan,
stub, broken_links, duplicate_candidate. Both the recalibrate *triage* node and
the *health check* (lint) compute from here, so:

  * the health score is fully deterministic (pure computation, no LLM), and
  * recalibration provably moves the score: the score is derived from the very
    signals recalibration repairs, so fixing them can only raise (or hold) it.

The functions are pure — they take the loaded pages plus the link-graph
adjacency and return counts/score, with no DB, LLM, or I/O.
"""
import hashlib
import json
import posixpath
import re
from datetime import datetime, timezone

# Thresholds (shared with recalibrate triage — imported there, never redefined).
STALENESS_DAYS = 90       # pages whose date_ingested is older than this → "stale"
STUB_WORDS = 150          # pages below this word count → "stub"

# Severity weights: points removed from a perfect 100 when *every* page exhibits
# the signal. The actual penalty scales by the fraction of pages affected, so a
# signal present on 10% of pages costs 10% of its weight. Weights sum to 100, so
# the score is bounded to [0, 100]. Ordered worst → least by structural impact.
SIGNAL_WEIGHTS: dict[str, int] = {
    "broken_links": 30,
    "duplicate_candidate": 25,
    "orphan": 20,
    "stub": 15,
    "stale": 10,
}

# Signal order used when building a page's reason list (kept stable so callers
# and tests see a deterministic ordering).
SIGNAL_ORDER = ["stale", "orphan", "stub", "broken_links", "duplicate_candidate"]

_WIKILINK_RE = re.compile(r'\[\[([^\]|#]+?)(?:[|#][^\]]*)?\]\]')
_MDLINK_RE = re.compile(r'\[[^\]]*\]\(([^)#?\s]+\.md)\)')
_DATE_INGESTED_RE = re.compile(r"^date_ingested:\s*(\d{4}-\d{2}-\d{2})", re.MULTILINE)


def normalize_title(title: str) -> str:
    """Lowercase + strip punctuation for duplicate-title detection."""
    return re.sub(r"[^a-z0-9 ]+", "", title.lower()).strip()


def _is_stale(path: str, content: str, now: datetime,
              reviewed_at: dict[str, datetime] | None = None) -> bool:
    """Stale = the page hasn't been *known current* within STALENESS_DAYS.

    Freshness is the most recent of (a) its ``date_ingested`` frontmatter and
    (b) the last master-recalibration review of this page (``reviewed_at``, when
    supplied). Keying off the review time — not just the ingest date — is what
    lets recalibration clear staleness: a reviewed page counts as fresh again,
    so repeated runs converge instead of re-flagging the same old pages forever.
    A page with neither date is never stale (can't tell)."""
    refs: list[datetime] = []
    m = _DATE_INGESTED_RE.search(content)
    if m:
        try:
            refs.append(datetime.strptime(m.group(1), "%Y-%m-%d").replace(tzinfo=timezone.utc))
        except ValueError:
            pass
    if reviewed_at:
        r = reviewed_at.get(path)
        if r is not None:
            refs.append(r)
    if not refs:
        return False
    return (now - max(refs)).days > STALENESS_DAYS


def _has_broken_links(path: str, content: str, pages: dict[str, str], stem_index: set[str]) -> bool:
    base = "/".join(path.split("/")[:-1])
    for m in _WIKILINK_RE.finditer(content):
        slug = m.group(1).strip().lower().replace(" ", "-")
        if slug not in stem_index:
            return True
    for m in _MDLINK_RE.finditer(content):
        target = m.group(1).strip()
        if target.startswith(("http://", "https://")):
            continue
        resolved = posixpath.normpath(posixpath.join(base, target) if base else target)
        if resolved not in pages:
            return True
    return False


def compute_signals(
    pages: dict[str, str],
    adjacency: dict[str, set[str]],
    *,
    now: datetime | None = None,
    reviewed_at: dict[str, datetime] | None = None,
) -> dict:
    """Detect the five structural signals over the wiki.

    ``pages``: {path: full_content}. ``adjacency``: {path: set(outlink targets)}
    from the link graph (e.g. ``WikiGraph._adj``). Returns::

        {
          "total": int,                         # pages considered
          "reasons": {path: [signal, ...]},     # only pages with >=1 signal
          "counts": {signal: n_pages, ...},     # per-signal page count
          "flagged": int,                       # pages with >=1 signal
          "duplicate_groups": [[path, ...], ...]
        }
    """
    now = now or datetime.now(timezone.utc)

    # Inbound link count per page. A page is an "orphan" when nothing links TO
    # it (in_deg == 0), even if it links out — such a page is undiscoverable by
    # navigation and needs reintegrating.
    in_deg: dict[str, int] = {}
    for src, targets in adjacency.items():
        for t in targets:
            in_deg[t] = in_deg.get(t, 0) + 1

    # Duplicate detection by normalised title.
    title_groups: dict[str, list[str]] = {}
    for path, content in pages.items():
        raw_title = next((l.lstrip("#").strip() for l in content.splitlines() if l.startswith("#")), path)
        norm = normalize_title(raw_title)
        if norm:
            title_groups.setdefault(norm, []).append(path)
    duplicate_groups = [sorted(p) for p in title_groups.values() if len(p) > 1]
    duplicate_set = {p for grp in duplicate_groups for p in grp}

    stem_index = {p.split("/")[-1].rsplit(".", 1)[0].lower() for p in pages}

    counts = {sig: 0 for sig in SIGNAL_WEIGHTS}
    reasons: dict[str, list[str]] = {}
    for path, content in pages.items():
        page_reasons: list[str] = []
        if _is_stale(path, content, now, reviewed_at):
            page_reasons.append("stale")
        if in_deg.get(path, 0) == 0:
            page_reasons.append("orphan")
        if len(content.split()) < STUB_WORDS:
            page_reasons.append("stub")
        if _has_broken_links(path, content, pages, stem_index):
            page_reasons.append("broken_links")
        if path in duplicate_set:
            page_reasons.append("duplicate_candidate")
        if page_reasons:
            reasons[path] = page_reasons
            for r in page_reasons:
                counts[r] += 1

    return {
        "total": len(pages),
        "reasons": reasons,
        "counts": counts,
        "flagged": len(reasons),
        "duplicate_groups": duplicate_groups,
    }


def compute_context_signatures(pages: dict[str, str]) -> dict[str, str]:
    """Per-page hash of the *relational* inputs to the health signals.

    Orphan, broken_links and duplicate_candidate can flip for a page WITHOUT its
    own content changing — because a neighbour was added/removed/relinked, a link
    target was deleted, or another page took the same title. Recalibration keys
    idempotency off content hashes, so those changes would otherwise be missed.
    This captures exactly those inputs so a page is re-reviewed when its context
    changes, and only then:

      * inbound link sources   → orphan
      * outbound targets + whether each resolves → broken_links
      * same-normalised-title group → duplicate_candidate

    Pure — mirrors ``compute_signals``' link/title parsing (no DB, LLM, graph
    cache), so the signature computed at review time and at the next triage match
    whenever nothing relevant changed."""
    paths = set(pages)
    stem_to_paths: dict[str, list[str]] = {}
    for p in paths:
        stem_to_paths.setdefault(p.split("/")[-1].rsplit(".", 1)[0].lower(), []).append(p)
    stem_index = set(stem_to_paths)

    # Normalised-title groups (duplicate signal).
    norm_of: dict[str, str] = {}
    groups: dict[str, list[str]] = {}
    for path, content in pages.items():
        raw_title = next((l.lstrip("#").strip() for l in content.splitlines() if l.startswith("#")), path)
        norm = normalize_title(raw_title)
        norm_of[path] = norm
        if norm:
            groups.setdefault(norm, []).append(path)

    # Outbound targets (+ resolution) and inbound sources, from the same link
    # parsing compute_signals uses for broken_links.
    outbound: dict[str, list] = {}
    inbound: dict[str, set[str]] = {p: set() for p in paths}
    for path, content in pages.items():
        base = "/".join(path.split("/")[:-1])
        targets: list[tuple[str, bool]] = []
        for m in _WIKILINK_RE.finditer(content):
            slug = m.group(1).strip().lower().replace(" ", "-")
            targets.append((f"[[{slug}]]", slug in stem_index))
            for tp in stem_to_paths.get(slug, []):
                inbound[tp].add(path)
        for m in _MDLINK_RE.finditer(content):
            target = m.group(1).strip()
            if target.startswith(("http://", "https://")):
                continue
            resolved = posixpath.normpath(posixpath.join(base, target) if base else target)
            exists = resolved in paths
            targets.append((target, exists))
            if exists:
                inbound[resolved].add(path)
        outbound[path] = sorted(set(targets))

    sigs: dict[str, str] = {}
    for path in pages:
        dup_group = sorted(g for g in groups.get(norm_of[path], []) if g != path)
        payload = json.dumps(
            {"in": sorted(inbound[path]), "out": outbound[path], "dup": dup_group},
            sort_keys=True,
        )
        sigs[path] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return sigs


def score_from_counts(counts: dict[str, int], total: int) -> int:
    """Deterministic 0–100 health score.

    100 minus a severity-weighted penalty proportional to the fraction of pages
    exhibiting each signal. Monotonic: reducing any signal's count can only raise
    (or hold) the score — which is what guarantees recalibration improves it.
    An empty wiki scores 100 (nothing wrong).
    """
    if total <= 0:
        return 100
    penalty = sum(SIGNAL_WEIGHTS[sig] * (counts.get(sig, 0) / total) for sig in SIGNAL_WEIGHTS)
    return max(0, min(100, round(100 - penalty)))


_MALFORMED_SLUG_CHARS = 60  # a wikilink target longer than this is almost certainly a
                            # document title/sentence pasted as a link, not a concept slug.


def classify_link_gaps(pages: dict[str, str], *, min_refs: int = 2) -> dict:
    """Split broken ``[[wikilink]]`` targets by how they should be remediated.

    Returns ``{"auto_create": [...], "manual": [...]}``. Each entry is
    ``{"slug", "ref_count", "referenced_by": [paths]}`` and, for manual entries,
    a ``"malformed": bool`` flag:

      * ``auto_create`` — target referenced by >= ``min_refs`` distinct pages;
        master recalibration creates a page for these (the content-gap path).
      * ``manual`` — referenced by fewer pages, so recalibration leaves them.
        ``malformed`` is True when the slug looks like a title/sentence rather
        than a concept (fix the link) vs. a genuine missing concept (create it).

    Pure — uses the same wikilink parsing as compute_signals. No DB/LLM.
    """
    stem_index = {p.split("/")[-1].rsplit(".", 1)[0].lower() for p in pages}
    refs: dict[str, set[str]] = {}
    for path, content in pages.items():
        targets_here: set[str] = set()
        for m in _WIKILINK_RE.finditer(content):
            slug = m.group(1).strip().lower().replace(" ", "-")
            if slug and slug not in stem_index:
                targets_here.add(slug)
        for slug in targets_here:
            refs.setdefault(slug, set()).add(path)

    auto_create: list[dict] = []
    manual: list[dict] = []
    for slug, srcs in refs.items():
        entry = {"slug": slug, "ref_count": len(srcs), "referenced_by": sorted(srcs)}
        if len(srcs) >= min_refs:
            auto_create.append(entry)
        else:
            entry["malformed"] = len(slug) > _MALFORMED_SLUG_CHARS
            manual.append(entry)
    auto_create.sort(key=lambda c: (-c["ref_count"], c["slug"]))
    manual.sort(key=lambda c: (c["malformed"], c["slug"]))
    return {"auto_create": auto_create, "manual": manual}


def compute_article_candidates(pages: dict[str, str], *, min_refs: int = 2) -> list[dict]:
    """Concepts referenced by several pages but with no page of their own.

    Uses `[[wikilinks]]` as the "mention" signal: a target slug that doesn't
    resolve to an existing page (same check as broken_links) and is referenced
    by ``min_refs`` or more *distinct* pages is a content gap worth creating.

    Returns ``[{"slug", "ref_count", "referenced_by": [paths]}]`` sorted by
    ref_count desc then slug (the ``auto_create`` half of ``classify_link_gaps``).
    Pure — no DB/LLM.
    """
    return classify_link_gaps(pages, min_refs=min_refs)["auto_create"]


def select_recent_pages(pages: dict[str, str], *, limit: int = 8) -> list[str]:
    """Return the paths of the most recently ingested pages, newest first.

    Ordered by the `date_ingested` frontmatter field (pages without one sort
    last). Used to give recalibration the "newest data" to reconcile stale
    pages against. Pure — no DB/LLM.
    """
    def _key(item: tuple[str, str]):
        m = _DATE_INGESTED_RE.search(item[1])
        # Missing/invalid dates sort last (epoch); valid dates sort by recency.
        return (m.group(1) if m else "", item[0])

    ordered = sorted(pages.items(), key=_key, reverse=True)
    return [path for path, _ in ordered[:limit]]


def compute_health(
    pages: dict[str, str],
    adjacency: dict[str, set[str]],
    *,
    now: datetime | None = None,
    reviewed_at: dict[str, datetime] | None = None,
) -> dict:
    """Full deterministic health report: ``{health_score, total_pages,
    flagged_pages, breakdown}`` where ``breakdown`` is the per-signal page count.

    ``reviewed_at`` (path → last recalibration review time) makes staleness clear
    once a page is reviewed — pass it so the score matches recalibrate's triage."""
    sig = compute_signals(pages, adjacency, now=now, reviewed_at=reviewed_at)
    return {
        "health_score": score_from_counts(sig["counts"], sig["total"]),
        "total_pages": sig["total"],
        "flagged_pages": sig["flagged"],
        "breakdown": sig["counts"],
    }
