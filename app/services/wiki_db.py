"""DB helpers for wiki content — all queries scoped by org_id from request context.

wiki_pages  — one row per generated wiki page; metadata + full content for FTS.
wiki_files  — special files: index.md, log.md, schema/AGENTS.md, wiki/.graph.json.
audit_log   — append-only operation log.
wiki_links  — structured link graph (from_path → to_path per org).  [Phase 5]
"""
import datetime
import json
import re

import sqlalchemy as sa

from app import model
from app.context import get_org_id


def _json_default(obj):
    """Serialize types that PyYAML produces but stdlib json doesn't handle."""
    if isinstance(obj, (datetime.date, datetime.datetime)):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
from app.db import get_db
from app.logger import get_logger

log = get_logger(__name__)

# Lazy-cached flag: True only when the embedding column actually exists in the DB.
# Set on first call to _embedding_col_exists(); never re-checked after that.
_embedding_col_ready: bool | None = None


async def _embedding_col_exists() -> bool | None:
    """Return True when wiki_pages.embedding column is present in the DB.

    Cached after the first successful check so we don't hit information_schema
    on every page write. The result is reset to None on process restart.
    """
    global _embedding_col_ready
    if _embedding_col_ready is not None:
        log.debug("embedding_col_exists | cache_hit=%s", _embedding_col_ready)
        return _embedding_col_ready
    try:
        async with get_db() as db:
            result = await db.execute(sa.text("""
                SELECT COUNT(*) FROM information_schema.columns
                WHERE table_name = 'wiki_pages' AND column_name = 'embedding'
            """))
            _embedding_col_ready = (result.scalar() or 0) > 0
    except Exception as e:
        log.warning("embedding_col_exists | check_failed: %s", e)
        _embedding_col_ready = False
    if _embedding_col_ready:
        log.info("embedding_col_exists | column present — semantic search enabled")
    else:
        log.warning(
            "embedding_col_exists | column absent — BEDROCK_EMBEDDING_MODEL_ID is set "
            "but wiki_pages.embedding does not exist. Run 'alembic upgrade head' with "
            "pgvector installed in PostgreSQL to enable semantic search."
        )
    return _embedding_col_ready


# ── wiki_pages ─────────────────────────────────────────────────────────────

_UPSERT = sa.text("""
    INSERT INTO wiki_pages (org_id, path, title, tags, summary, content, frontmatter, ingested_from, s3_key)
    VALUES (CAST(:org_id AS UUID), :path, :title, CAST(:tags AS JSONB), :summary, :content,
            CAST(:frontmatter AS JSONB), :ingested_from, :s3_key)
    ON CONFLICT (org_id, path) DO UPDATE SET
        title         = EXCLUDED.title,
        tags          = EXCLUDED.tags,
        summary       = EXCLUDED.summary,
        content       = EXCLUDED.content,
        frontmatter   = EXCLUDED.frontmatter,
        ingested_from = EXCLUDED.ingested_from,
        s3_key        = EXCLUDED.s3_key,
        embedding_space = NULL,
        updated_at    = NOW()
""")

# Phase 5: upsert that also stores a precomputed embedding vector (as string literal).
# The :embedding parameter must be a pgvector literal string '[f1,f2,...]' or NULL.
_UPSERT_WITH_EMBEDDING = sa.text("""
    INSERT INTO wiki_pages (org_id, path, title, tags, summary, content, frontmatter, ingested_from, s3_key, embedding, embedding_space)
    VALUES (CAST(:org_id AS UUID), :path, :title, CAST(:tags AS JSONB), :summary, :content,
            CAST(:frontmatter AS JSONB), :ingested_from, :s3_key,
            CAST(:embedding AS vector), :embedding_space)
    ON CONFLICT (org_id, path) DO UPDATE SET
        title         = EXCLUDED.title,
        tags          = EXCLUDED.tags,
        summary       = EXCLUDED.summary,
        content       = EXCLUDED.content,
        frontmatter   = EXCLUDED.frontmatter,
        ingested_from = EXCLUDED.ingested_from,
        s3_key        = EXCLUDED.s3_key,
        embedding     = EXCLUDED.embedding,
        embedding_space = EXCLUDED.embedding_space,
        updated_at    = NOW()
""")

_DELETE       = sa.text("DELETE FROM wiki_pages WHERE org_id = CAST(:org_id AS UUID) AND path = :path")
_LIST         = sa.text("SELECT path FROM wiki_pages WHERE org_id = CAST(:org_id AS UUID) ORDER BY path")
_LIST_CONTENT = sa.text("SELECT path, content FROM wiki_pages WHERE org_id = CAST(:org_id AS UUID) ORDER BY path")
_GET_CONTENT  = sa.text("SELECT content FROM wiki_pages WHERE org_id = CAST(:org_id AS UUID) AND path = :path")
_EXISTS       = sa.text("SELECT 1 FROM wiki_pages WHERE org_id = CAST(:org_id AS UUID) AND path = :path")

# Phase 5: paginated listing with optional tag filter.
_LIST_PAGINATED = sa.text("""
    SELECT path, title, tags, summary, updated_at
    FROM wiki_pages
    WHERE org_id = CAST(:org_id AS UUID)
    AND (:tag IS NULL OR CAST(:tag AS TEXT) = ANY(
            SELECT jsonb_array_elements_text(tags)
        ))
    ORDER BY path
    LIMIT :limit OFFSET :offset
""")

_COUNT_PAGES = sa.text("""
    SELECT COUNT(*) FROM wiki_pages
    WHERE org_id = CAST(:org_id AS UUID)
    AND (:tag IS NULL OR CAST(:tag AS TEXT) = ANY(
            SELECT jsonb_array_elements_text(tags)
        ))
""")

_SEARCH = sa.text("""
    SELECT path, title, content,
        ts_rank(search_vector, plainto_tsquery('english', :q)) AS rank
    FROM wiki_pages
    WHERE org_id = CAST(:org_id AS UUID)
    AND (
            search_vector @@ plainto_tsquery('english', :q)
        OR lower(path)  LIKE :q_like
        OR lower(title) LIKE :q_like
    )
    ORDER BY rank DESC, lower(title) LIKE :q_like DESC
    LIMIT 20
""")

# Phase 5: hybrid search — BM25 weighted 0.4, cosine similarity weighted 0.6.
# ts_rank and cosine similarity live on different scales (ts_rank is typically a
# small fraction, cosine is a clean 0–1), so weighting the raw values lets cosine
# dominate regardless of the nominal weights. We min-max normalize the BM25 term
# against the best-matching page for this query (MAX(bm25_raw) OVER ()) so both
# components are 0–1 and the 0.4/0.6 split is real. Non-matching pages get
# bm25_raw = 0 → bm25_norm = 0; if nothing matches the query, MAX is 0 and NULLIF
# yields NULL → COALESCE back to 0.
#
# Only pages scoring above min_score (default 0.45) are returned so that low-relevance
# pages don't inflate the LLM context. Pages with NULL embeddings (ingested before
# pgvector was enabled) or from another model use full keyword-only scoring.
_HYBRID_SEARCH = sa.text("""
    WITH base AS (
        SELECT path,
               ts_rank(search_vector, plainto_tsquery('english', :q)) AS bm25_raw,
               (embedding IS NOT NULL AND embedding_space = :embedding_space) IS TRUE AS compatible,
               CASE WHEN embedding_space = :embedding_space
                    THEN COALESCE(1.0 - (embedding <=> CAST(:embedding AS vector)), 0.0)
                    ELSE 0.0 END AS cosine
        FROM wiki_pages
        WHERE org_id = CAST(:org_id AS UUID)
        AND path NOT IN ('index.md', 'log.md')
    ),
    scored AS (
        SELECT path,
               COALESCE(bm25_raw / NULLIF(MAX(bm25_raw) OVER (), 0), 0.0)
               * CASE WHEN compatible THEN 0.4 ELSE 1.0 END
               + cosine * 0.6
                AS score
        FROM base
    )
    SELECT path, score FROM scored
    WHERE score > :min_score
    ORDER BY score DESC
    LIMIT :limit
""")

# Phase 5: BM25-only page relevance (no embedding required).
_BM25_FIND = sa.text("""
    SELECT path, title, content,
        ts_rank(search_vector, plainto_tsquery('english', :q)) AS score
    FROM wiki_pages
    WHERE org_id = CAST(:org_id AS UUID)
    AND path NOT IN ('index.md', 'log.md')
    AND (
            search_vector @@ plainto_tsquery('english', :q)
        OR lower(path)  LIKE :q_like
        OR lower(title) LIKE :q_like
    )
    ORDER BY score DESC
    LIMIT :limit
""")

_COMPACT_INDEX = sa.text("""
    SELECT path, title FROM wiki_pages
    WHERE org_id = CAST(:org_id AS UUID)
    ORDER BY path
""")

# Pure cosine similarity — used when we have a document vector and want the
# most conceptually similar existing pages (no query string needed).
_VECTOR_SEARCH = sa.text("""
    SELECT path, title, content,
        1.0 - (embedding <=> CAST(:embedding AS vector)) AS score
    FROM wiki_pages
    WHERE org_id = CAST(:org_id AS UUID)
      AND embedding IS NOT NULL
      AND embedding_space = :embedding_space
    ORDER BY embedding <=> CAST(:embedding AS vector)
    LIMIT :top_k
""")

_MD_NOISE_RE = re.compile(
    r'```.*?```'           # fenced code blocks
    r'|`[^`]*`'            # inline code
    r'|\[\[.*?\]\]'        # wikilinks
    r'|\[([^\]]*)\]\([^)]*\)'  # markdown links → keep label
    r'|!\[[^\]]*\]\([^)]*\)'   # images
    r'|^\s*#+\s*'          # heading markers
    r'|^\s*[-*>|]\s*',     # list bullets, blockquotes, table pipes
    re.DOTALL | re.MULTILINE,
)


def _build_embed_text(title: str, summary: str, content: str, fm_end: int) -> str:
    """Build a clean text string for embedding: title + summary + stripped body."""
    body_raw = content[fm_end:]
    body_clean = _MD_NOISE_RE.sub(' ', body_raw)
    body_clean = ' '.join(body_clean.split())  # collapse whitespace
    return f"{title}\n\n{summary}\n\n{body_clean[:6000]}"


def _parse_meta(path: str, content: str) -> dict:
    # Frontmatter + title resolution are shared with the graph (app.utils) so the
    # two can't drift; tags/summary/fm_end are store-specific.
    from app.utils import parse_frontmatter, resolve_title, frontmatter_end
    fm = parse_frontmatter(content)
    title = resolve_title(content, path)
    tags = [str(t) for t in fm["tags"]] if isinstance(fm.get("tags"), list) else []
    fm_end = frontmatter_end(content)
    body = " ".join(content[fm_end:].split())
    summary = body[:500]
    return {"title": title, "tags": tags, "frontmatter": fm, "summary": summary, "fm_end": fm_end}


async def upsert_wiki_page(path: str, content: str, ingested_from: str = "",
                           embedding: list[float] | None = None) -> None:
    """Upsert a wiki page. When embedding is enabled AND the column exists, stores a vector.

    A caller-supplied `embedding` (e.g. a wiki import reusing exported vectors)
    must be validated for model compatibility by the importer; this helper
    checks vector shape and stamps current provenance. Otherwise a vector is
    generated when embeddings are enabled.

    Emits a `wiki_revisions` row if a tracked action is open in the current
    context (see `app.services.wiki_state`).
    """
    org_id = get_org_id()
    meta = _parse_meta(path, content)

    content_before = await get_wiki_page_content(path)
    op_kind = "update" if content_before is not None else "create"

    from app.services import embeddings
    embedding_vec: list[float] | None = None
    has_col = await _embedding_col_exists()
    if embedding is not None and not model.valid_embedding(embedding):
        raise ValueError("Supplied embedding must be a finite, nonzero 1536-dimensional vector")
    if embedding is not None and has_col and embeddings.is_enabled():
        embedding_vec = embedding  # reuse caller-supplied vector (import path)
        log.debug("upsert_wiki_page | path=%s | embedding=reused | dims=%d", path, len(embedding))
    elif embeddings.is_enabled() and has_col:
        embed_text = _build_embed_text(meta["title"], meta["summary"], content, meta["fm_end"])
        embedding_vec = await embeddings.embed_text(embed_text)
        if embedding_vec is not None:
            log.debug("upsert_wiki_page | path=%s | embedding=generated | dims=%d", path, len(embedding_vec))
        else:
            log.warning("upsert_wiki_page | path=%s | embedding=failed (embed_text returned None)", path)
    else:
        log.debug("upsert_wiki_page | path=%s | embedding=disabled", path)

    if embedding_vec is not None and not model.valid_embedding(embedding_vec):
        log.warning("upsert_wiki_page | invalid embedding discarded | path=%s", path)
        embedding_vec = None

    base_params = {
        "org_id": org_id,
        "path": path,
        "title": meta["title"],
        "tags": json.dumps(meta["tags"], default=_json_default),
        "summary": meta["summary"],
        "content": content,
        "frontmatter": json.dumps(meta["frontmatter"], default=_json_default),
        "ingested_from": ingested_from,
        "s3_key": "",
    }

    from app.services.wiki_state import record_revision
    async with get_db() as db:
        if embedding_vec is not None:
            await db.execute(_UPSERT_WITH_EMBEDDING, {
                **base_params,
                "embedding": embeddings.vec_to_pg(embedding_vec),
                "embedding_space": model.embedding_identity(),
            })
            log.debug("upsert_wiki_page | path=%s | stored with embedding", path)
        else:
            await db.execute(_UPSERT, base_params)
            log.debug("upsert_wiki_page | path=%s | stored without embedding", path)
        # Same transaction as the write → a crash can't keep the page change
        # while losing its revision (which would make it un-revertible).
        await record_revision(
            db=db,
            target_kind="page",
            target_key=path,
            op=op_kind,
            content_before=content_before,
            content_after=content,
        )


async def delete_wiki_page(path: str) -> None:
    org_id = get_org_id()
    content_before = await get_wiki_page_content(path)
    if content_before is None:
        return
    from app.services.wiki_state import record_revision
    async with get_db() as db:
        await db.execute(_DELETE, {"org_id": org_id, "path": path})
        await record_revision(
            db=db,
            target_kind="page",
            target_key=path,
            op="delete",
            content_before=content_before,
            content_after=None,
        )


async def get_wiki_page_content(path: str) -> str | None:
    org_id = get_org_id()
    async with get_db() as db:
        result = await db.execute(_GET_CONTENT, {"org_id": org_id, "path": path})
        row = result.fetchone()
    return row.content if row else None


async def wiki_page_exists(path: str) -> bool:
    org_id = get_org_id()
    async with get_db() as db:
        result = await db.execute(_EXISTS, {"org_id": org_id, "path": path})
        return result.fetchone() is not None


async def list_wiki_paths() -> list[str]:
    org_id = get_org_id()
    async with get_db() as db:
        result = await db.execute(_LIST, {"org_id": org_id})
        return [row.path for row in result.fetchall()]


async def get_compact_index() -> str:
    """Return all wiki pages as a compact 'path — title' list for routing decisions.

    Much cheaper than the full index.md: ~50 chars/page vs hundreds. Callers use
    this to decide create-vs-update without blowing token budgets.
    """
    org_id = get_org_id()
    async with get_db() as db:
        result = await db.execute(_COMPACT_INDEX, {"org_id": org_id})
        rows = result.fetchall()
    if not rows:
        return "No pages yet."
    return "\n".join(f"{r.path} — {r.title}" for r in rows)


async def semantic_search_wiki(query_vec: list[float], top_k: int = 8) -> list[dict]:
    """Return top_k pages by cosine similarity to query_vec, with full content.

    Returns an empty list when the embedding column doesn't exist yet.
    Each item: {path, title, content, score} where score ∈ [0, 1].
    """
    if not await _embedding_col_exists():
        return []
    if not model.valid_embedding(query_vec) or not model.embedding_enabled():
        return []
    from app.services import embeddings
    org_id = get_org_id()
    async with get_db() as db:
        result = await db.execute(_VECTOR_SEARCH, {
            "org_id": org_id,
            "embedding": embeddings.vec_to_pg(query_vec),
            "embedding_space": model.embedding_identity(),
            "top_k": top_k,
        })
        rows = result.fetchall()
    return [
        {"path": r.path, "title": r.title, "content": r.content or "", "score": float(r.score)}
        for r in rows
    ]


_EXPORT_NO_EMBEDDINGS = sa.text("""
    SELECT path, content
    FROM wiki_pages
    WHERE org_id = CAST(:org_id AS UUID)
    ORDER BY path
""")

_EXPORT_WITH_EMBEDDINGS = sa.text("""
    SELECT path, content,
           CASE WHEN embedding_space = :embedding_space
                THEN embedding::text ELSE NULL END AS embedding_str
    FROM wiki_pages
    WHERE org_id = CAST(:org_id AS UUID)
    ORDER BY path
""")


async def list_wiki_pages_for_export(include_embeddings: bool) -> list[dict]:
    """Return all wiki pages for export. Each item: {path, content, embedding: list[float]|None}."""
    org_id = get_org_id()
    has_emb_col = include_embeddings and await _embedding_col_exists()

    async with get_db() as db:
        if has_emb_col:
            result = await db.execute(_EXPORT_WITH_EMBEDDINGS, {
                "org_id": org_id, "embedding_space": model.embedding_identity(),
            })
        else:
            result = await db.execute(_EXPORT_NO_EMBEDDINGS, {"org_id": org_id})
        rows = result.fetchall()

    pages = []
    for row in rows:
        embedding = None
        if has_emb_col and getattr(row, "embedding_str", None):
            try:
                embedding = json.loads(row.embedding_str)
            except Exception:
                pass
        pages.append({"path": row.path, "content": row.content or "", "embedding": embedding})
    return pages


async def list_wiki_pages_with_content() -> list[tuple[str, str]]:
    org_id = get_org_id()
    async with get_db() as db:
        result = await db.execute(_LIST_CONTENT, {"org_id": org_id})
        return [(row.path, row.content or "") for row in result.fetchall()]


async def list_wiki_pages_paginated(
    page: int = 1, limit: int = 50, tag: str | None = None
) -> dict:
    """Return a paginated page listing with total count. Pages are 1-indexed."""
    org_id = get_org_id()
    offset = (page - 1) * limit
    async with get_db() as db:
        total_result = await db.execute(_COUNT_PAGES, {"org_id": org_id, "tag": tag})
        total = total_result.scalar() or 0

        rows_result = await db.execute(
            _LIST_PAGINATED,
            {"org_id": org_id, "tag": tag, "limit": limit, "offset": offset},
        )
        rows = rows_result.fetchall()

    items = [
        {
            "path": r.path,
            "title": r.title,
            "tags": json.loads(r.tags) if isinstance(r.tags, str) else (r.tags or []),
            "summary": r.summary or "",
            "updated_at": r.updated_at.isoformat() if r.updated_at else None,
        }
        for r in rows
    ]
    return {"items": items, "total": total, "page": page, "limit": limit}


async def search_wiki(q: str) -> list[dict]:
    org_id = get_org_id()
    q_like = f"%{q.lower()}%"
    async with get_db() as db:
        result = await db.execute(_SEARCH, {"org_id": org_id, "q": q, "q_like": q_like})
        rows = result.fetchall()

    q_lower = q.lower()
    results: list[dict] = []
    for row in rows:
        content = row.content or ""
        idx = content.lower().find(q_lower)
        snippet = content[max(0, idx - 60):idx + 120].strip() if idx >= 0 else content[:180].strip()
        stem = row.path.split("/")[-1].rsplit(".", 1)[0]
        is_exact = stem == q_lower.replace(" ", "-")
        is_title = q_lower in (row.title or "").lower()
        results.append({"path": row.path, "snippet": snippet,
                        "_exact": is_exact, "_title": is_title})

    results.sort(key=lambda r: (not r.pop("_exact"), not r.pop("_title")))
    return results


async def find_relevant_pages(question: str, limit: int = 8) -> list[str]:
    """Return up to *limit* page paths most relevant to *question*.

    Uses hybrid BM25 + cosine search when embedding is configured, BM25-only
    otherwise. Returns an empty list when no pages match (callers should fall
    back to the LLM-driven approach).
    """
    org_id = get_org_id()
    q_short = question[:60]

    from app.services import embeddings
    if embeddings.is_enabled() and await _embedding_col_exists():
        log.debug("find_relevant_pages | mode=hybrid | q=%r", q_short)
        vec = await embeddings.embed_text(question)
        if vec is not None and model.valid_embedding(vec):
            async with get_db() as db:
                result = await db.execute(_HYBRID_SEARCH, {
                    "org_id": org_id,
                    "q": question,
                    "embedding": embeddings.vec_to_pg(vec),
                    "embedding_space": model.embedding_identity(),
                    "limit": limit,
                    "min_score": 0.45,
                })
                rows = result.fetchall()
                
            log.debug("find_relevant_pages | hybrid_search returned %d rows for q=%r", len(rows), q_short)
            paths = [row.path for row in rows]
            for row in rows:
                log.debug("find_relevant_pages | hybrid_hit | path=%s | score=%.4f", row.path, row.score)
            log.info("find_relevant_pages | mode=hybrid | q=%r | hits=%d | pages=%s", q_short, len(paths), paths)
            return paths
        log.warning("find_relevant_pages | embed_text returned None for q=%r — falling back to BM25", q_short)
    else:
        log.debug("find_relevant_pages | mode=bm25 | embedding_enabled=%s | q=%r", embeddings.is_enabled(), q_short)

    # BM25 fallback
    q_like = f"%{question.lower()}%"
    async with get_db() as db:
        result = await db.execute(_BM25_FIND, {
            "org_id": org_id,
            "q": question,
            "q_like": q_like,
            "limit": limit,
        })
        rows = result.fetchall()
    paths = [row.path for row in rows]
    log.info("find_relevant_pages | mode=bm25 | q=%r | hits=%d | pages=%s", q_short, len(paths), paths)
    return paths


# ── wiki_files ─────────────────────────────────────────────────────────────
# Composite PK: (org_id, key)
# Keys: "wiki/index.md", "wiki/log.md", "schema/AGENTS.md", "wiki/.graph.json"

_GET_FILE = sa.text(
    "SELECT content FROM wiki_files WHERE org_id = CAST(:org_id AS UUID) AND key = :key"
)

_SET_FILE = sa.text("""
    INSERT INTO wiki_files (org_id, key, content, updated_at)
    VALUES (CAST(:org_id AS UUID), :key, :content, NOW())
    ON CONFLICT (org_id, key) DO UPDATE SET
        content    = EXCLUDED.content,
        updated_at = NOW()
""")

_PREPEND_FILE = sa.text("""
    INSERT INTO wiki_files (org_id, key, content, updated_at)
    VALUES (CAST(:org_id AS UUID), :key, :new_text, NOW())
    ON CONFLICT (org_id, key) DO UPDATE SET
        content    = :new_text || E'\\n\\n' || wiki_files.content,
        updated_at = NOW()
""")


async def get_wiki_file(key: str) -> str | None:
    org_id = get_org_id()
    async with get_db() as db:
        result = await db.execute(_GET_FILE, {"org_id": org_id, "key": key})
        row = result.fetchone()
    return row.content if row else None


_DERIVED_FILE_KEYS = {"wiki/.graph.json"}


async def set_wiki_file(key: str, content: str) -> None:
    """Set the content of a special wiki file (e.g. schema/AGENTS.md).

    Derived caches (`wiki/.graph.json`) are not tracked — they're rebuildable
    from page content and emitting revisions for them would balloon storage.
    """
    org_id = get_org_id()
    content_before = await get_wiki_file(key) if key not in _DERIVED_FILE_KEYS else None
    tracked = key not in _DERIVED_FILE_KEYS
    op_kind = "update" if content_before is not None else "create"
    from app.services.wiki_state import record_revision
    async with get_db() as db:
        await db.execute(_SET_FILE, {"org_id": org_id, "key": key, "content": content})
        if tracked:
            # Same transaction as the write (see upsert_wiki_page).
            await record_revision(
                db=db,
                target_kind="file",
                target_key=key,
                op=op_kind,
                content_before=content_before,
                content_after=content,
            )


async def prepend_wiki_file(key: str, new_text: str) -> None:
    org_id = get_org_id()
    async with get_db() as db:
        await db.execute(_PREPEND_FILE, {"org_id": org_id, "key": key, "new_text": new_text})


# ── audit_log ──────────────────────────────────────────────────────────────

async def get_rendered_log(limit: int = 100) -> str:
    """Render the most recent audit_log entries as markdown.

    Replaces reading wiki/log.md. Always current, never unbounded.
    """
    org_id = get_org_id()
    async with get_db() as db:
        result = await db.execute(sa.text("""
            SELECT raw_text FROM audit_log
            WHERE org_id = CAST(:org_id AS UUID)
            ORDER BY created_at DESC
            LIMIT :limit
        """), {"org_id": org_id, "limit": limit})
        rows = result.fetchall()
    return "\n\n".join(r.raw_text for r in rows) if rows else ""


async def append_audit_log(operation: str, raw_text: str, details: dict | None = None) -> None:
    org_id = get_org_id()
    from app.context import current_user
    ctx = current_user.get(None)
    user_id = ctx.user_id if ctx and ctx.user_id else None
    async with get_db() as db:
        await db.execute(sa.text("""
            INSERT INTO audit_log (org_id, user_id, operation, raw_text, details)
            VALUES (CAST(:org_id AS UUID), CAST(:user_id AS UUID), :operation, :raw_text, CAST(:details AS JSONB))
        """), {
            "org_id": org_id,
            "user_id": user_id,
            "operation": operation,
            "raw_text": raw_text,
            "details": json.dumps(details or {}),
        })


# ── wiki_links (Phase 5) ───────────────────────────────────────────────────

async def replace_wiki_links(from_path: str, to_paths: list[str]) -> None:
    """Replace all outgoing links from *from_path* with *to_paths*.

    Called by WikiGraph after each page update so the DB table stays in sync
    with the in-memory adjacency dict.
    """
    org_id = get_org_id()
    async with get_db() as db:
        await db.execute(
            sa.text("DELETE FROM wiki_links WHERE org_id = CAST(:org_id AS UUID) AND from_path = :from_path"),
            {"org_id": org_id, "from_path": from_path},
        )
        for to_path in to_paths:
            await db.execute(
                sa.text("""
                    INSERT INTO wiki_links (org_id, from_path, to_path)
                    VALUES (CAST(:org_id AS UUID), :from_path, :to_path)
                    ON CONFLICT DO NOTHING
                """),
                {"org_id": org_id, "from_path": from_path, "to_path": to_path},
            )


async def get_wiki_links(path: str) -> dict[str, list[str]]:
    """Return outgoing and incoming links for *path* from the DB."""
    org_id = get_org_id()
    async with get_db() as db:
        out_result = await db.execute(
            sa.text("SELECT to_path FROM wiki_links WHERE org_id = CAST(:org_id AS UUID) AND from_path = :path"),
            {"org_id": org_id, "path": path},
        )
        in_result = await db.execute(
            sa.text("SELECT from_path FROM wiki_links WHERE org_id = CAST(:org_id AS UUID) AND to_path = :path"),
            {"org_id": org_id, "path": path},
        )
    return {
        "outgoing": [r.to_path for r in out_result.fetchall()],
        "incoming": [r.from_path for r in in_result.fetchall()],
    }


# ── recalibration fingerprints ─────────────────────────────────────────────
# Per-page (content hash + last-reviewed) state that lets master recalibration
# run idempotently: staleness is measured from reviewed_at, and a page unchanged
# since its last review is skipped. Internal bookkeeping — NOT change-tracked.

_GET_RECALIB = sa.text("""
    SELECT path, content_sha, context_sha, reviewed_at
    FROM wiki_recalibration
    WHERE org_id = CAST(:org_id AS UUID)
""")

_UPSERT_RECALIB = sa.text("""
    INSERT INTO wiki_recalibration (org_id, path, content_sha, context_sha, reviewed_at)
    VALUES (CAST(:org_id AS UUID), :path, :content_sha, :context_sha, NOW())
    ON CONFLICT (org_id, path) DO UPDATE
       SET content_sha = EXCLUDED.content_sha,
           context_sha = EXCLUDED.context_sha,
           reviewed_at = NOW()
""")

_PRUNE_RECALIB = sa.text("""
    DELETE FROM wiki_recalibration
    WHERE org_id = CAST(:org_id AS UUID)
      AND path NOT IN (SELECT jsonb_array_elements_text(CAST(:paths AS JSONB)))
""")


async def get_recalibration_state() -> dict[str, dict]:
    """Return ``{path: {"content_sha", "context_sha", "reviewed_at"}}`` for the org.

    Feeds recalibration idempotency: staleness is measured from ``reviewed_at``,
    and a flagged page is skipped when BOTH its content hash and its context hash
    (inbound links / target existence / title group) are unchanged since review.
    """
    org_id = get_org_id()
    async with get_db() as db:
        rows = (await db.execute(_GET_RECALIB, {"org_id": org_id})).fetchall()
    return {
        r.path: {"content_sha": r.content_sha, "context_sha": r.context_sha,
                 "reviewed_at": r.reviewed_at}
        for r in rows
    }


async def record_recalibration(fingerprints: list[tuple[str, str, str]], *, prune: bool = True) -> None:
    """Stamp ``(path, content_sha, context_sha)`` with ``reviewed_at = NOW()``.

    ``prune=True`` also drops rows for paths not in *fingerprints* (deleted pages
    / full-run cleanup); pass ``prune=False`` for targeted runs that touched only
    a subset. Writes straight to ``wiki_recalibration`` (not via the tracked
    choke-point writers), so it is deliberately excluded from change-tracking.
    """
    org_id = get_org_id()
    async with get_db() as db:
        if fingerprints:
            await db.execute(_UPSERT_RECALIB, [
                {"org_id": org_id, "path": p, "content_sha": cs, "context_sha": xs}
                for p, cs, xs in fingerprints
            ])
        if prune:
            await db.execute(
                _PRUNE_RECALIB,
                {"org_id": org_id, "paths": json.dumps([p for p, _, _ in fingerprints])},
            )
