"""DB-backed notification service.

Notifications are user-scoped and persist across sessions so users see
pending-review alerts and ingest outcomes when they log back in.
"""
import json

import sqlalchemy as sa

from app.db import get_db
from app.logger import get_logger

log = get_logger(__name__)

_INSERT = sa.text("""
    INSERT INTO notifications (user_id, org_id, type, title, body, link, metadata)
    VALUES (
        CAST(:user_id AS UUID),
        CAST(:org_id  AS UUID),
        :type, :title, :body, :link,
        CAST(:metadata AS JSONB)
    )
    RETURNING id
""")

_SELECT_UNREAD = sa.text("""
    SELECT id, type, title, body, link, metadata, created_at
    FROM notifications
    WHERE user_id = CAST(:user_id AS UUID) AND is_read = FALSE
    ORDER BY created_at DESC
    LIMIT :limit
""")

_MARK_READ = sa.text("""
    UPDATE notifications
    SET is_read = TRUE
    WHERE id = CAST(:id AS UUID) AND user_id = CAST(:user_id AS UUID)
""")

_MARK_ALL_READ = sa.text("""
    UPDATE notifications SET is_read = TRUE
    WHERE user_id = CAST(:user_id AS UUID) AND is_read = FALSE
""")

_MARK_READ_BY_LINK = sa.text("""
    UPDATE notifications SET is_read = TRUE
    WHERE user_id = CAST(:user_id AS UUID) AND link = :link AND is_read = FALSE
""")


async def create(
    user_id: str,
    org_id: str,
    type: str,
    title: str,
    body: str = "",
    link: str = "",
    metadata: dict | None = None,
) -> str:
    """Insert a notification and return its UUID."""
    if not user_id:
        return ""
    from app.services.notif_stream import CHANNEL
    async with get_db() as db:
        result = await db.execute(_INSERT, {
            "user_id":  user_id,
            "org_id":   org_id or None,
            "type":     type,
            "title":    title,
            "body":     body,
            "link":     link,
            "metadata": json.dumps(metadata or {}),
        })
        row = result.fetchone()
        if row:
            # Signal the recipient's live SSE stream (fires on commit). Payload is
            # just the user_id — the stream pulls the row from the DB.
            await db.execute(
                sa.text("SELECT pg_notify(:ch, :payload)"),
                {"ch": CHANNEL, "payload": str(user_id)},
            )
    notif_id = str(row.id) if row else ""
    log.debug("notification created | id=%s | type=%s | user=%s", notif_id, type, user_id)
    return notif_id


async def get_unread(user_id: str, limit: int = 50) -> list[dict]:
    """Return unread notifications for a user, newest first."""
    if not user_id:
        return []
    async with get_db() as db:
        result = await db.execute(_SELECT_UNREAD, {"user_id": user_id, "limit": limit})
        rows = result.fetchall()
    return [
        {
            "id":         str(r.id),
            "type":       r.type,
            "title":      r.title,
            "body":       r.body,
            "link":       r.link,
            "metadata":   r.metadata or {},
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in rows
    ]


async def mark_read(notif_id: str, user_id: str) -> None:
    async with get_db() as db:
        await db.execute(_MARK_READ, {"id": notif_id, "user_id": user_id})


async def mark_all_read(user_id: str) -> None:
    async with get_db() as db:
        await db.execute(_MARK_ALL_READ, {"user_id": user_id})


async def mark_read_by_link(user_id: str, link: str) -> None:
    """Mark all unread notifications for a given link (e.g. filename) as read."""
    if not user_id or not link:
        return
    async with get_db() as db:
        await db.execute(_MARK_READ_BY_LINK, {"user_id": user_id, "link": link})


# Source of truth for per-org role is org_memberships (since migration 023) —
# users.org_id/users.role are legacy rollback columns and are NOT kept in sync
# when a user is granted/promoted to supervisor via membership, so querying
# `users` here silently missed those supervisors. Suspended supervisors are
# excluded. Global admins (org_id IS NULL, no membership) are intentionally not
# included to avoid cross-org notification spam.
_GET_ORG_SUPERVISORS = sa.text("""
    SELECT u.id, u.email
    FROM org_memberships m
    JOIN users u ON u.id = m.user_id
    WHERE m.org_id = CAST(:org_id AS UUID)
      AND m.role = 'supervisor'
      AND COALESCE(m.is_suspended, false) = false
""")


async def get_org_supervisors(org_id: str) -> list[dict]:
    """Return id+email for every (non-suspended) supervisor in the org."""
    if not org_id:
        return []
    async with get_db() as db:
        result = await db.execute(_GET_ORG_SUPERVISORS, {"org_id": org_id})
        rows = result.fetchall()
    return [{"id": str(r.id), "email": r.email} for r in rows]


async def create_for_supervisors(
    org_id: str,
    type: str,
    title: str,
    body: str = "",
    link: str = "",
    metadata: dict | None = None,
    exclude_user_id: str = "",
) -> None:
    """Send a notification to every supervisor/admin in the org, skipping exclude_user_id."""
    supervisors = await get_org_supervisors(org_id)
    for sup in supervisors:
        if sup["id"] == exclude_user_id:
            continue
        await create(sup["id"], org_id, type, title, body, link, metadata)
