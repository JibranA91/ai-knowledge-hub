"""Async DB-backed chat session store with auto-summarize support — org-scoped.

Writer mode (`mode='writer'`) extends the same table: rows additionally carry
a generated `draft_content` markdown body and the user-chosen `draft_filename`.
The writer draft picker lists sessions per user, filtered by TTL.
"""
import json
import re
import uuid
from dataclasses import dataclass, field

import sqlalchemy as sa

from app.context import current_user, get_org_id
from app.db import get_db

SUMMARIZE_AFTER = 10
KEEP_RECENT = 4

_UPSERT = sa.text("""
    INSERT INTO chat_sessions (
        org_id, session_id, messages, summary, mode,
        draft_content, draft_filename, draft_ready, user_id, last_active_at
    )
    VALUES (
        CAST(:org_id AS UUID), :session_id, CAST(:messages AS JSONB), :summary, :mode,
        :draft_content, :draft_filename, :draft_ready, CAST(:user_id AS UUID), NOW()
    )
    ON CONFLICT (session_id) DO UPDATE SET
        messages       = CAST(:messages AS JSONB),
        summary        = :summary,
        draft_content  = :draft_content,
        draft_filename = :draft_filename,
        draft_ready    = :draft_ready,
        last_active_at = NOW()
""")
_SELECT = sa.text("""
    SELECT session_id, messages, summary, mode, draft_content, draft_filename,
           draft_ready, user_id
    FROM chat_sessions
    WHERE org_id = CAST(:org_id AS UUID) AND session_id = :session_id
""")
_DELETE = sa.text(
    "DELETE FROM chat_sessions WHERE org_id = CAST(:org_id AS UUID) AND session_id = :session_id"
)
_LIST_WRITER = sa.text("""
    SELECT session_id, draft_filename, draft_content, draft_ready, last_active_at
    FROM chat_sessions
    WHERE org_id = CAST(:org_id AS UUID)
      AND user_id = CAST(:user_id AS UUID)
      AND mode = 'writer'
      AND last_active_at > NOW() - make_interval(days => :ttl_days)
    ORDER BY last_active_at DESC
""")
_PRUNE_WRITER = sa.text("""
    DELETE FROM chat_sessions
    WHERE mode = 'writer'
      AND last_active_at < NOW() - make_interval(days => :ttl_days)
""")


@dataclass
class ChatSession:
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    messages: list[dict] = field(default_factory=list)
    summary: str = ""
    mode: str = "chat"
    draft_content: str = ""
    draft_filename: str = ""
    draft_ready: bool = False
    user_id: str | None = None


def _ctx_user_id() -> str | None:
    ctx = current_user.get(None)
    return ctx.user_id if ctx and ctx.user_id else None


async def get_or_create(session_id: str | None, mode: str = "chat") -> ChatSession:
    org_id = get_org_id()
    if session_id:
        async with get_db() as db:
            result = await db.execute(_SELECT, {"org_id": org_id, "session_id": session_id})
            row = result.fetchone()
        if row:
            return ChatSession(
                session_id=row.session_id,
                messages=row.messages or [],
                summary=row.summary or "",
                mode=row.mode or "chat",
                draft_content=row.draft_content or "",
                draft_filename=row.draft_filename or "",
                draft_ready=bool(getattr(row, "draft_ready", False)),
                user_id=str(row.user_id) if row.user_id else None,
            )
    session = ChatSession(mode=mode, user_id=_ctx_user_id())
    async with get_db() as db:
        await db.execute(_UPSERT, {
            "org_id": org_id,
            "session_id": session.session_id,
            "messages": json.dumps(session.messages),
            "summary": session.summary,
            "mode": session.mode,
            "draft_content": session.draft_content,
            "draft_filename": session.draft_filename,
            "draft_ready": session.draft_ready,
            "user_id": session.user_id,
        })
    return session


async def save(session: ChatSession) -> None:
    org_id = get_org_id()
    async with get_db() as db:
        await db.execute(_UPSERT, {
            "org_id": org_id,
            "session_id": session.session_id,
            "messages": json.dumps(session.messages),
            "summary": session.summary,
            "mode": session.mode,
            "draft_content": session.draft_content,
            "draft_filename": session.draft_filename,
            "draft_ready": session.draft_ready,
            "user_id": session.user_id,
        })


async def clear(session_id: str) -> bool:
    org_id = get_org_id()
    async with get_db() as db:
        result = await db.execute(_DELETE, {"org_id": org_id, "session_id": session_id})
        return result.rowcount > 0


# ── Writer-mode helpers ────────────────────────────────────────────────────

async def get_draft(session_id: str) -> dict | None:
    """Return {draft_content, draft_filename, draft_ready} for a writer session, or None."""
    org_id = get_org_id()
    async with get_db() as db:
        result = await db.execute(_SELECT, {"org_id": org_id, "session_id": session_id})
        row = result.fetchone()
    if not row:
        return None
    return {
        "draft_content": row.draft_content or "",
        "draft_filename": row.draft_filename or "",
        "draft_ready": bool(getattr(row, "draft_ready", False)),
    }


def _excerpt(content: str, max_chars: int = 100) -> str:
    if not content:
        return ""
    flat = re.sub(r"\s+", " ", content.strip())
    return flat[:max_chars]


async def list_writer_sessions(user_id: str, ttl_days: int) -> list[dict]:
    """All writer-mode drafts for the current org + given user, within TTL."""
    org_id = get_org_id()
    async with get_db() as db:
        result = await db.execute(_LIST_WRITER, {
            "org_id": org_id, "user_id": user_id, "ttl_days": ttl_days,
        })
        rows = result.fetchall()
    return [
        {
            "session_id": r.session_id,
            "draft_filename": r.draft_filename or "",
            "last_active_at": r.last_active_at.isoformat() if r.last_active_at else None,
            "excerpt": _excerpt(r.draft_content or ""),
            "draft_ready": bool(getattr(r, "draft_ready", False)),
        }
        for r in rows
    ]


# Section locating/patching lives in md_sections (shared with the inline AI
# editor). Kept as a module-local alias so existing call sites stay unchanged.
from app.services.md_sections import find_section_bounds as _find_section_bounds


async def patch_draft_section(
    session_id: str, heading: str, new_section: str
) -> tuple[str | None, str | None]:
    """Replace the section under *heading* with *new_section*.

    Returns (new_full_content, error). On success error is None and the new
    content is persisted to the row. On ambiguity / not-found the draft is
    left untouched.
    """
    draft = await get_draft(session_id)
    if draft is None:
        return None, "session not found"
    content = draft["draft_content"]
    if not content.strip():
        return None, (
            "the draft is empty — emit a full [DRAFT_START]...[DRAFT_END] block "
            "instead of a section patch"
        )
    matches, err = _find_section_bounds(content, heading)
    if err:
        return None, err

    start, body_end, _ = matches[0]
    # Preserve a single trailing newline boundary if present
    replacement = new_section.rstrip() + "\n"
    new_content = content[:start] + replacement + content[body_end:]

    org_id = get_org_id()
    async with get_db() as db:
        await db.execute(sa.text("""
            UPDATE chat_sessions SET draft_content = :c, last_active_at = NOW()
            WHERE org_id = CAST(:org_id AS UUID) AND session_id = :session_id
        """), {"c": new_content, "org_id": org_id, "session_id": session_id})
    return new_content, None


async def prune_expired_writer_sessions(ttl_days: int) -> int:
    """Delete writer-mode sessions whose `last_active_at` is older than TTL."""
    async with get_db() as db:
        result = await db.execute(_PRUNE_WRITER, {"ttl_days": ttl_days})
        return result.rowcount or 0
