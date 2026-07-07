"""Member-facing wiki activity & per-page provenance.

Surfaces the change-tracking data (wiki_actions / wiki_revisions) to anyone who
can view the wiki — not just supervisors/admins (who use the richer /api/admin
History tab). Two read-only, org-scoped views:

  GET /api/activity/recent       — org "recent changes" feed
  GET /api/activity/page?path=…  — per-page contributors + last-edited

Lives in its own module (not wiki.py) because wiki.py's `/{path:path}` route is
greedy and would shadow any sibling path here.
"""
import sqlalchemy as sa
from fastapi import APIRouter, Depends, HTTPException, Query, Request

from app.db import get_db
from app.logger import get_logger
from app.services.permissions import require_permission

log = get_logger(__name__)
router = APIRouter(tags=["activity"])


def _org_id(request: Request) -> str:
    return getattr(request.state, "org_id", "") or ""


# Actions that touched ≥1 wiki page, newest first, with a small sample of the
# affected page paths for click-through.
_RECENT = sa.text("""
    SELECT a.id::text AS id, a.action_type, a.summary, a.started_at,
           u.email AS user_email,
           p.page_count, p.pages
    FROM wiki_actions a
    LEFT JOIN users u ON u.id = a.user_id
    JOIN LATERAL (
        SELECT COUNT(*) AS page_count,
               (array_agg(target_key ORDER BY target_key))[1:8] AS pages
        FROM (
            SELECT DISTINCT r.target_key
            FROM wiki_revisions r
            WHERE r.action_id = a.id AND r.target_kind = 'page'
        ) d
    ) p ON TRUE
    WHERE a.org_id = CAST(:org_id AS UUID)
      AND a.status = 'done'
      AND p.page_count > 0
    ORDER BY a.started_at DESC
    LIMIT :limit
""")

# Every tracked revision of one page, newest first. Aggregated in Python — a
# single page's revision count is small.
_PAGE_HISTORY = sa.text("""
    SELECT r.id AS revision_id, u.email AS user_email, a.action_type,
           a.started_at, r.op
    FROM wiki_revisions r
    JOIN wiki_actions a ON a.id = r.action_id
    LEFT JOIN users u ON u.id = a.user_id
    WHERE a.org_id = CAST(:org_id AS UUID)
      AND a.status = 'done'
      AND r.target_kind = 'page'
      AND r.target_key = :path
    ORDER BY a.started_at DESC
    LIMIT 500
""")

# Before/after content of one page revision (org-scoped; pages only — never
# exposes raw-file or schema revision blobs through this member-facing route).
_REVISION = sa.text("""
    SELECT r.id AS revision_id, r.op, r.target_key,
           r.content_before, r.content_after
    FROM wiki_revisions r
    WHERE r.id = :rev_id
      AND r.org_id = CAST(:org_id AS UUID)
      AND r.target_kind = 'page'
""")


@router.get("/recent", dependencies=[Depends(require_permission("can_view_wiki"))])
async def recent_changes(request: Request, limit: int = Query(default=30, ge=1, le=100)):
    """Org-wide feed of recent wiki changes (page-affecting actions only)."""
    org_id = _org_id(request)
    if not org_id:
        raise HTTPException(400, "No organization context")
    async with get_db() as db:
        rows = (await db.execute(_RECENT, {"org_id": org_id, "limit": limit})).fetchall()
    return {
        "changes": [
            {
                "id": r.id,
                "action_type": r.action_type,
                "summary": r.summary or "",
                "user_email": r.user_email,
                "when": r.started_at.isoformat() if r.started_at else None,
                "page_count": r.page_count or 0,
                "pages": list(r.pages) if r.pages else [],
            }
            for r in rows
        ]
    }


@router.get("/page", dependencies=[Depends(require_permission("can_view_wiki"))])
async def page_activity(request: Request, path: str = Query(..., min_length=1)):
    """Per-page provenance: who last edited it, when, and the contributor list."""
    org_id = _org_id(request)
    if not org_id:
        raise HTTPException(400, "No organization context")
    async with get_db() as db:
        rows = (await db.execute(_PAGE_HISTORY, {"org_id": org_id, "path": path})).fetchall()

    if not rows:
        # Page exists but predates change-tracking, or was never tracked.
        return {
            "path": path, "change_count": 0,
            "last_edited_by": None, "last_edited_at": None,
            "created_at": None, "contributors": [],
        }

    counts: dict[str, int] = {}
    for r in rows:
        email = r.user_email or "system"
        counts[email] = counts.get(email, 0) + 1
    contributors = sorted(
        ({"email": e, "count": c} for e, c in counts.items()),
        key=lambda x: (-x["count"], x["email"]),
    )

    return {
        "path": path,
        "change_count": len(rows),
        "last_edited_by": rows[0].user_email,
        "last_edited_at": rows[0].started_at.isoformat() if rows[0].started_at else None,
        "created_at": rows[-1].started_at.isoformat() if rows[-1].started_at else None,
        "contributors": contributors,
    }


@router.get("/page/history", dependencies=[Depends(require_permission("can_view_wiki"))])
async def page_history(
    request: Request,
    path: str = Query(..., min_length=1),
    limit: int = Query(default=100, ge=1, le=500),
):
    """Full per-page change timeline (newest first) for the history view.

    Separate from ``/page`` so the provenance byline — fetched on every page
    navigation — stays a lean aggregate; the timeline is only pulled when a
    member opens the history panel.
    """
    org_id = _org_id(request)
    if not org_id:
        raise HTTPException(400, "No organization context")
    async with get_db() as db:
        rows = (await db.execute(_PAGE_HISTORY, {"org_id": org_id, "path": path})).fetchall()
    return {
        "path": path,
        "change_count": len(rows),
        "changes": [
            {
                "id": r.revision_id,
                "user_email": r.user_email,
                "action_type": r.action_type,
                "op": r.op,
                "when": r.started_at.isoformat() if r.started_at else None,
            }
            for r in rows[:limit]
        ],
    }


@router.get("/revision/{revision_id}", dependencies=[Depends(require_permission("can_view_wiki"))])
async def revision_content(request: Request, revision_id: int):
    """Before/after content of a single page revision, for the diff view.

    Scoped to the caller's org and to ``target_kind = 'page'`` so this
    member-facing route can never surface raw-upload or schema revision blobs.
    """
    org_id = _org_id(request)
    if not org_id:
        raise HTTPException(400, "No organization context")
    async with get_db() as db:
        row = (await db.execute(_REVISION, {"rev_id": revision_id, "org_id": org_id})).fetchone()
    if row is None:
        raise HTTPException(404, "Revision not found")
    return {
        "id": row.revision_id,
        "op": row.op,
        "path": row.target_key,
        "content_before": row.content_before,
        "content_after": row.content_after,
    }
