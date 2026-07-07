"""Permission templates and access-control enforcement.

Roles vs. templates
-------------------
- role="admin"  → bypasses all template checks; has every permission implicitly.
- role="member" → governed by the permission_template assigned to the user row.
  If no template is assigned the user is treated as "read_only" (safest default).

Dependency factory
------------------
    from app.services.permissions import require_permission, check_upload_quota

    @router.post("/upload", dependencies=[Depends(require_permission("can_upload")),
                                          Depends(check_upload_quota())])
    async def upload(...):
        ...
"""
import uuid
from typing import Any

import sqlalchemy as sa
from fastapi import HTTPException, Request

from app.db import get_db
from app.logger import get_logger

log = get_logger(__name__)

# ── Column list shared by read queries ────────────────────────────────────────

_TPL_COLS = """
    id, org_id, name, description, is_builtin,
    can_upload, can_upload_writer_draft, can_delete_files, can_download_files,
    max_upload_size_mb, max_uploads_per_day, max_uploads_per_week,
    can_view_wiki, can_edit_wiki, can_delete_wiki_pages,
    can_query, can_chat, can_use_writer, max_queries_per_day, max_chat_messages_per_day,
    max_tokens_per_day, max_tokens_per_week,
    can_recalibrate, can_run_lint, can_manage_schema,
    can_view_audit_log, can_view_graph, can_rebuild_graph,
    can_manage_workspace, can_approve_ingest, can_cancel_ingest,
    created_at, updated_at
"""

# ── SQL statements ────────────────────────────────────────────────────────────

# Effective permissions are resolved from the user's membership in the active
# org (org_memberships), not the legacy users.* columns — a user may have a
# different template / suspension / limits in each org they belong to.
_GET_USER_PERMS = sa.text(f"""
    SELECT m.is_suspended, m.permission_template_id,
           m.max_tokens_per_day        AS user_max_tokens_per_day,
           m.max_chat_messages_per_day AS user_max_chat_messages_per_day,
           pt.can_upload, pt.can_upload_writer_draft, pt.can_delete_files, pt.can_download_files,
           pt.max_upload_size_mb, pt.max_uploads_per_day, pt.max_uploads_per_week,
           pt.can_view_wiki, pt.can_edit_wiki, pt.can_delete_wiki_pages,
           pt.can_query, pt.can_chat, pt.can_use_writer,
           pt.max_queries_per_day, pt.max_chat_messages_per_day,
           pt.max_tokens_per_day, pt.max_tokens_per_week,
           pt.can_recalibrate, pt.can_run_lint, pt.can_manage_schema,
           pt.can_view_audit_log, pt.can_view_graph, pt.can_rebuild_graph,
           pt.can_manage_workspace, pt.can_approve_ingest, pt.can_cancel_ingest
    FROM org_memberships m
    LEFT JOIN permission_templates pt ON m.permission_template_id = pt.id
    WHERE m.user_id = CAST(:user_id AS UUID) AND m.org_id = CAST(:org_id AS UUID)
""")

# Built-in templates have org_id = NULL (global); custom templates have a specific org_id.

_LIST_TEMPLATES = sa.text(f"""
    SELECT {_TPL_COLS}
    FROM permission_templates
    WHERE org_id = CAST(:org_id AS UUID) OR org_id IS NULL
    ORDER BY is_builtin DESC, name
""")

_GET_TEMPLATE = sa.text(f"""
    SELECT {_TPL_COLS}
    FROM permission_templates
    WHERE id = CAST(:id AS UUID)
      AND (org_id = CAST(:org_id AS UUID) OR org_id IS NULL)
""")

_GET_TEMPLATE_ANY = sa.text(f"""
    SELECT {_TPL_COLS}
    FROM permission_templates
    WHERE id = CAST(:id AS UUID)
""")

_GET_TEMPLATE_BY_NAME = sa.text(f"""
    SELECT {_TPL_COLS}
    FROM permission_templates
    WHERE name = :name
      AND (org_id = CAST(:org_id AS UUID) OR org_id IS NULL)
    ORDER BY org_id NULLS LAST
    LIMIT 1
""")

_INSERT_TEMPLATE = sa.text("""
    INSERT INTO permission_templates
        (id, org_id, name, description, is_builtin,
         can_upload, can_upload_writer_draft, can_delete_files, can_download_files,
         max_upload_size_mb, max_uploads_per_day, max_uploads_per_week,
         can_view_wiki, can_edit_wiki, can_delete_wiki_pages,
         can_query, can_chat, can_use_writer, max_queries_per_day, max_chat_messages_per_day,
         max_tokens_per_day, max_tokens_per_week,
         can_recalibrate, can_run_lint, can_manage_schema,
         can_view_audit_log, can_view_graph, can_rebuild_graph,
         can_manage_workspace, can_approve_ingest, can_cancel_ingest)
    VALUES
        (CAST(:id AS UUID), CAST(:org_id AS UUID), :name, :description, :is_builtin,
         :can_upload, :can_upload_writer_draft, :can_delete_files, :can_download_files,
         :max_upload_size_mb, :max_uploads_per_day, :max_uploads_per_week,
         :can_view_wiki, :can_edit_wiki, :can_delete_wiki_pages,
         :can_query, :can_chat, :can_use_writer, :max_queries_per_day, :max_chat_messages_per_day,
         :max_tokens_per_day, :max_tokens_per_week,
         :can_recalibrate, :can_run_lint, :can_manage_schema,
         :can_view_audit_log, :can_view_graph, :can_rebuild_graph,
         :can_manage_workspace, :can_approve_ingest, :can_cancel_ingest)
    ON CONFLICT DO NOTHING
    RETURNING id
""")

_INSERT_CUSTOM_GLOBAL_TEMPLATE = sa.text("""
    INSERT INTO permission_templates
        (id, org_id, name, description, is_builtin,
         can_upload, can_upload_writer_draft, can_delete_files, can_download_files,
         max_upload_size_mb, max_uploads_per_day, max_uploads_per_week,
         can_view_wiki, can_edit_wiki, can_delete_wiki_pages,
         can_query, can_chat, can_use_writer, max_queries_per_day, max_chat_messages_per_day,
         max_tokens_per_day, max_tokens_per_week,
         can_recalibrate, can_run_lint, can_manage_schema,
         can_view_audit_log, can_view_graph, can_rebuild_graph,
         can_manage_workspace, can_approve_ingest, can_cancel_ingest)
    VALUES
        (CAST(:id AS UUID), NULL, :name, :description, :is_builtin,
         :can_upload, :can_upload_writer_draft, :can_delete_files, :can_download_files,
         :max_upload_size_mb, :max_uploads_per_day, :max_uploads_per_week,
         :can_view_wiki, :can_edit_wiki, :can_delete_wiki_pages,
         :can_query, :can_chat, :can_use_writer, :max_queries_per_day, :max_chat_messages_per_day,
         :max_tokens_per_day, :max_tokens_per_week,
         :can_recalibrate, :can_run_lint, :can_manage_schema,
         :can_view_audit_log, :can_view_graph, :can_rebuild_graph,
         :can_manage_workspace, :can_approve_ingest, :can_cancel_ingest)
    RETURNING id
""")

_INSERT_GLOBAL_TEMPLATE = sa.text("""
    INSERT INTO permission_templates
        (id, org_id, name, description, is_builtin,
         can_upload, can_upload_writer_draft, can_delete_files, can_download_files,
         max_upload_size_mb, max_uploads_per_day, max_uploads_per_week,
         can_view_wiki, can_edit_wiki, can_delete_wiki_pages,
         can_query, can_chat, can_use_writer, max_queries_per_day, max_chat_messages_per_day,
         max_tokens_per_day, max_tokens_per_week,
         can_recalibrate, can_run_lint, can_manage_schema,
         can_view_audit_log, can_view_graph, can_rebuild_graph,
         can_manage_workspace, can_approve_ingest, can_cancel_ingest)
    VALUES
        (CAST(:id AS UUID), NULL, :name, :description, true,
         :can_upload, :can_upload_writer_draft, :can_delete_files, :can_download_files,
         :max_upload_size_mb, :max_uploads_per_day, :max_uploads_per_week,
         :can_view_wiki, :can_edit_wiki, :can_delete_wiki_pages,
         :can_query, :can_chat, :can_use_writer, :max_queries_per_day, :max_chat_messages_per_day,
         :max_tokens_per_day, :max_tokens_per_week,
         :can_recalibrate, :can_run_lint, :can_manage_schema,
         :can_view_audit_log, :can_view_graph, :can_rebuild_graph,
         :can_manage_workspace, :can_approve_ingest, :can_cancel_ingest)
    ON CONFLICT DO NOTHING
    RETURNING id
""")

_UPDATE_TEMPLATE = sa.text("""
    UPDATE permission_templates SET
        name                    = :name,
        description             = :description,
        can_upload              = :can_upload,
        can_upload_writer_draft = :can_upload_writer_draft,
        can_delete_files        = :can_delete_files,
        can_download_files      = :can_download_files,
        max_upload_size_mb      = :max_upload_size_mb,
        max_uploads_per_day     = :max_uploads_per_day,
        max_uploads_per_week    = :max_uploads_per_week,
        can_view_wiki           = :can_view_wiki,
        can_edit_wiki           = :can_edit_wiki,
        can_delete_wiki_pages   = :can_delete_wiki_pages,
        can_query               = :can_query,
        can_chat                = :can_chat,
        can_use_writer          = :can_use_writer,
        max_queries_per_day     = :max_queries_per_day,
        max_chat_messages_per_day = :max_chat_messages_per_day,
        max_tokens_per_day      = :max_tokens_per_day,
        max_tokens_per_week     = :max_tokens_per_week,
        can_recalibrate         = :can_recalibrate,
        can_run_lint            = :can_run_lint,
        can_manage_schema       = :can_manage_schema,
        can_view_audit_log      = :can_view_audit_log,
        can_view_graph          = :can_view_graph,
        can_rebuild_graph       = :can_rebuild_graph,
        can_manage_workspace    = :can_manage_workspace,
        can_approve_ingest      = :can_approve_ingest,
        can_cancel_ingest       = :can_cancel_ingest,
        updated_at              = NOW()
    WHERE id = CAST(:id AS UUID)
      AND org_id = CAST(:org_id AS UUID)
      AND is_builtin = false
""")

_UPDATE_GLOBAL_TEMPLATE = sa.text("""
    UPDATE permission_templates SET
        name                    = :name,
        description             = :description,
        can_upload              = :can_upload,
        can_upload_writer_draft = :can_upload_writer_draft,
        can_delete_files        = :can_delete_files,
        can_download_files      = :can_download_files,
        max_upload_size_mb      = :max_upload_size_mb,
        max_uploads_per_day     = :max_uploads_per_day,
        max_uploads_per_week    = :max_uploads_per_week,
        can_view_wiki           = :can_view_wiki,
        can_edit_wiki           = :can_edit_wiki,
        can_delete_wiki_pages   = :can_delete_wiki_pages,
        can_query               = :can_query,
        can_chat                = :can_chat,
        can_use_writer          = :can_use_writer,
        max_queries_per_day     = :max_queries_per_day,
        max_chat_messages_per_day = :max_chat_messages_per_day,
        max_tokens_per_day      = :max_tokens_per_day,
        max_tokens_per_week     = :max_tokens_per_week,
        can_recalibrate         = :can_recalibrate,
        can_run_lint            = :can_run_lint,
        can_manage_schema       = :can_manage_schema,
        can_view_audit_log      = :can_view_audit_log,
        can_view_graph          = :can_view_graph,
        can_rebuild_graph       = :can_rebuild_graph,
        can_manage_workspace    = :can_manage_workspace,
        can_approve_ingest      = :can_approve_ingest,
        can_cancel_ingest       = :can_cancel_ingest,
        updated_at              = NOW()
    WHERE id = CAST(:id AS UUID)
      AND org_id IS NULL
      AND is_builtin = false
""")

_DELETE_TEMPLATE = sa.text("""
    DELETE FROM permission_templates
    WHERE id = CAST(:id AS UUID)
      AND org_id = CAST(:org_id AS UUID)
      AND is_builtin = false
""")

_DELETE_GLOBAL_TEMPLATE = sa.text("""
    DELETE FROM permission_templates
    WHERE id = CAST(:id AS UUID)
      AND org_id IS NULL
      AND is_builtin = false
""")

_COUNT_TEMPLATE_USERS = sa.text("""
    SELECT COUNT(*) FROM org_memberships
    WHERE permission_template_id = CAST(:template_id AS UUID)
""")

_COUNT_UPLOADS_TODAY = sa.text("""
    SELECT COUNT(*) FROM ingest_jobs
    WHERE user_id = CAST(:user_id AS UUID)
      AND created_at >= CURRENT_DATE
""")

_COUNT_UPLOADS_THIS_WEEK = sa.text("""
    SELECT COUNT(*) FROM ingest_jobs
    WHERE user_id = CAST(:user_id AS UUID)
      AND created_at >= DATE_TRUNC('week', NOW())
""")

_COUNT_ORG_UPLOADS_TODAY = sa.text("""
    SELECT COUNT(*) FROM ingest_jobs
    WHERE org_id = CAST(:org_id AS UUID)
      AND created_at >= CURRENT_DATE
""")

_SUM_TOKENS_TODAY = sa.text("""
    SELECT COALESCE(SUM(tokens_in + tokens_out), 0) FROM usage_log
    WHERE user_id = CAST(:user_id AS UUID)
      AND created_at >= CURRENT_DATE
""")

_SUM_TOKENS_THIS_WEEK = sa.text("""
    SELECT COALESCE(SUM(tokens_in + tokens_out), 0) FROM usage_log
    WHERE user_id = CAST(:user_id AS UUID)
      AND created_at >= DATE_TRUNC('week', NOW())
""")

_SUM_ORG_TOKENS_TODAY = sa.text("""
    SELECT COALESCE(SUM(tokens_in + tokens_out), 0) FROM usage_log
    WHERE org_id = CAST(:org_id AS UUID)
      AND created_at >= CURRENT_DATE
""")

_GET_ORG_LIMITS = sa.text("""
    SELECT max_uploads_per_day_org, max_tokens_per_day_org, max_members
    FROM organizations
    WHERE id = CAST(:org_id AS UUID)
""")

_COUNT_QUERIES_TODAY = sa.text("""
    SELECT COUNT(*) FROM usage_log
    WHERE user_id = CAST(:user_id AS UUID)
      AND operation = 'query'
      AND created_at >= CURRENT_DATE
""")

_COUNT_CHATS_TODAY = sa.text("""
    SELECT COUNT(*) FROM usage_log
    WHERE user_id = CAST(:user_id AS UUID)
      AND operation IN ('chat', 'chat_stream')
      AND created_at >= CURRENT_DATE
""")

# ── Defaults used when no template is assigned ────────────────────────────────

_READ_ONLY_DEFAULTS: dict[str, Any] = {
    "is_suspended": False,
    "can_upload": False,
    "can_upload_writer_draft": False,
    "can_delete_files": False,
    "can_download_files": True,
    "max_upload_size_mb": None,
    "max_uploads_per_day": None,
    "max_uploads_per_week": None,
    "can_view_wiki": True,
    "can_edit_wiki": False,
    "can_delete_wiki_pages": False,
    "can_query": False,
    "can_chat": False,
    "can_use_writer": False,
    "max_queries_per_day": None,
    "max_chat_messages_per_day": None,
    "max_tokens_per_day": None,
    "max_tokens_per_week": None,
    "can_recalibrate": False,
    "can_run_lint": False,
    "can_manage_schema": False,
    "can_view_audit_log": False,
    "can_view_graph": True,
    "can_rebuild_graph": False,
    "can_manage_workspace": False,
    "can_approve_ingest": False,
    "can_cancel_ingest": False,
}

_ADMIN_FULL: dict[str, Any] = {k: True if isinstance(v, bool) else None
                                for k, v in _READ_ONLY_DEFAULTS.items()}
_ADMIN_FULL["is_suspended"] = False


# ── Template helpers ──────────────────────────────────────────────────────────

def _row_to_template(row) -> dict:
    return {
        "id": str(row.id),
        "org_id": str(row.org_id) if row.org_id else None,
        "name": row.name,
        "description": row.description,
        "is_builtin": row.is_builtin,
        "can_upload": row.can_upload,
        "can_upload_writer_draft": row.can_upload_writer_draft,
        "can_delete_files": row.can_delete_files,
        "can_download_files": row.can_download_files,
        "max_upload_size_mb": row.max_upload_size_mb,
        "max_uploads_per_day": row.max_uploads_per_day,
        "max_uploads_per_week": row.max_uploads_per_week,
        "can_view_wiki": row.can_view_wiki,
        "can_edit_wiki": row.can_edit_wiki,
        "can_delete_wiki_pages": row.can_delete_wiki_pages,
        "can_query": row.can_query,
        "can_chat": row.can_chat,
        "can_use_writer": row.can_use_writer,
        "max_queries_per_day": row.max_queries_per_day,
        "max_chat_messages_per_day": row.max_chat_messages_per_day,
        "max_tokens_per_day": row.max_tokens_per_day,
        "max_tokens_per_week": row.max_tokens_per_week,
        "can_recalibrate": row.can_recalibrate,
        "can_run_lint": row.can_run_lint,
        "can_manage_schema": row.can_manage_schema,
        "can_view_audit_log": row.can_view_audit_log,
        "can_view_graph": row.can_view_graph,
        "can_rebuild_graph": row.can_rebuild_graph,
        "can_manage_workspace": row.can_manage_workspace,
        "can_approve_ingest": row.can_approve_ingest,
        "can_cancel_ingest": row.can_cancel_ingest,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
    }


def _template_params(data: dict) -> dict:
    """Extract template field values from a dict (e.g. request body)."""
    return {
        "name": data.get("name", ""),
        "description": data.get("description", ""),
        "is_builtin": data.get("is_builtin", False),
        "can_upload": data.get("can_upload", False),
        "can_upload_writer_draft": data.get("can_upload_writer_draft", False),
        "can_delete_files": data.get("can_delete_files", False),
        "can_download_files": data.get("can_download_files", True),
        "max_upload_size_mb": data.get("max_upload_size_mb"),
        "max_uploads_per_day": data.get("max_uploads_per_day"),
        "max_uploads_per_week": data.get("max_uploads_per_week"),
        "can_view_wiki": data.get("can_view_wiki", True),
        "can_edit_wiki": data.get("can_edit_wiki", False),
        "can_delete_wiki_pages": data.get("can_delete_wiki_pages", False),
        "can_query": data.get("can_query", False),
        "can_chat": data.get("can_chat", False),
        "can_use_writer": data.get("can_use_writer", False),
        "max_queries_per_day": data.get("max_queries_per_day"),
        "max_chat_messages_per_day": data.get("max_chat_messages_per_day"),
        "max_tokens_per_day": data.get("max_tokens_per_day"),
        "max_tokens_per_week": data.get("max_tokens_per_week"),
        "can_recalibrate": data.get("can_recalibrate", False),
        "can_run_lint": data.get("can_run_lint", False),
        "can_manage_schema": data.get("can_manage_schema", False),
        "can_view_audit_log": data.get("can_view_audit_log", False),
        "can_view_graph": data.get("can_view_graph", True),
        "can_rebuild_graph": data.get("can_rebuild_graph", False),
        "can_manage_workspace": data.get("can_manage_workspace", False),
        "can_approve_ingest": data.get("can_approve_ingest", False),
        "can_cancel_ingest": data.get("can_cancel_ingest", False),
    }


# ── Public API — template CRUD ────────────────────────────────────────────────

async def list_templates(org_id: str) -> list[dict]:
    async with get_db() as db:
        result = await db.execute(_LIST_TEMPLATES, {"org_id": org_id})
        rows = result.fetchall()
    return [_row_to_template(r) for r in rows]


async def get_template(template_id: str, org_id: str) -> dict | None:
    async with get_db() as db:
        result = await db.execute(_GET_TEMPLATE, {"id": template_id, "org_id": org_id})
        row = result.fetchone()
    return _row_to_template(row) if row else None


async def get_template_any(template_id: str) -> dict | None:
    """Look up a template by id without org restriction (admin use only)."""
    async with get_db() as db:
        result = await db.execute(_GET_TEMPLATE_ANY, {"id": template_id})
        row = result.fetchone()
    return _row_to_template(row) if row else None


async def create_template(org_id: str | None, data: dict) -> str | None:
    """Create a custom template. Returns new UUID or None on name conflict.

    Pass org_id=None to create a global template visible to all orgs (admin only).
    """
    new_id = str(uuid.uuid4())
    params = _template_params(data)
    params["id"] = new_id
    async with get_db() as db:
        if org_id is None:
            # UNIQUE(org_id, name) doesn't catch NULL duplicates in PostgreSQL
            existing = await db.execute(
                sa.text("SELECT id FROM permission_templates WHERE org_id IS NULL AND name = :name"),
                {"name": params["name"]},
            )
            if existing.fetchone():
                return None
            result = await db.execute(_INSERT_CUSTOM_GLOBAL_TEMPLATE, params)
        else:
            params["org_id"] = org_id
            result = await db.execute(_INSERT_TEMPLATE, params)
        row = result.fetchone()
    if row:
        log.info("Permission template created | id=%s | org=%s | name=%s", new_id, org_id, params["name"])
        return str(row.id)
    return None


async def update_template(template_id: str, org_id: str | None, data: dict) -> bool:
    """Update a non-builtin template. Returns False if not found or is builtin.

    Pass org_id=None to update a global template (admin only).
    """
    params = _template_params(data)
    params["id"] = template_id
    async with get_db() as db:
        if org_id is None:
            result = await db.execute(_UPDATE_GLOBAL_TEMPLATE, params)
        else:
            params["org_id"] = org_id
            result = await db.execute(_UPDATE_TEMPLATE, params)
    updated = result.rowcount > 0
    if updated:
        log.info("Permission template updated | id=%s | org=%s", template_id, org_id)
    return updated


async def delete_template(template_id: str, org_id: str | None) -> bool:
    """Delete a non-builtin template. Returns False if not found or is builtin.

    Pass org_id=None to delete a global template (admin only).
    """
    async with get_db() as db:
        if org_id is None:
            result = await db.execute(_DELETE_GLOBAL_TEMPLATE, {"id": template_id})
        else:
            result = await db.execute(_DELETE_TEMPLATE, {"id": template_id, "org_id": org_id})
    deleted = result.rowcount > 0
    if deleted:
        log.info("Permission template deleted | id=%s | org=%s", template_id, org_id)
    return deleted


async def count_template_users(template_id: str) -> int:
    """Return how many users currently have this template assigned."""
    async with get_db() as db:
        result = await db.execute(_COUNT_TEMPLATE_USERS, {"template_id": template_id})
        return result.scalar() or 0


async def seed_builtin_templates(_org_id: str = "") -> None:
    """Ensure the three global built-in templates exist (org_id = NULL).

    Idempotent — safe to call on every startup. The `_org_id` parameter is
    accepted but ignored; it exists only for backwards-compatibility with the
    startup call in main.py.
    """
    builtins = [
        {
            "name": "read_only", "description": "View-only access to wiki and graph",
            "can_upload": False, "can_upload_writer_draft": False,
            "can_delete_files": False, "can_download_files": False,
            "max_upload_size_mb": None, "max_uploads_per_day": None, "max_uploads_per_week": None,
            "can_view_wiki": True, "can_edit_wiki": False, "can_delete_wiki_pages": False,
            "can_query": False, "can_chat": False, "can_use_writer": False,
            "max_queries_per_day": None, "max_chat_messages_per_day": None,
            "max_tokens_per_day": None, "max_tokens_per_week": None,
            "can_recalibrate": False, "can_run_lint": False, "can_manage_schema": False,
            "can_view_audit_log": False, "can_view_graph": True, "can_rebuild_graph": False,
            "can_manage_workspace": False, "can_approve_ingest": False, "can_cancel_ingest": False,
        },
        {
            "name": "contributor", "description": "Upload documents, query and chat with the wiki",
            "can_upload": True, "can_upload_writer_draft": True,
            "can_delete_files": False, "can_download_files": True,
            "max_upload_size_mb": None, "max_uploads_per_day": 20, "max_uploads_per_week": None,
            "can_view_wiki": True, "can_edit_wiki": False, "can_delete_wiki_pages": False,
            "can_query": True, "can_chat": True, "can_use_writer": False,
            "max_queries_per_day": None, "max_chat_messages_per_day": 200,
            "max_tokens_per_day": None, "max_tokens_per_week": None,
            "can_recalibrate": False, "can_run_lint": False, "can_manage_schema": False,
            "can_view_audit_log": False, "can_view_graph": True, "can_rebuild_graph": False,
            "can_manage_workspace": False, "can_approve_ingest": True, "can_cancel_ingest": True,
        },
        {
            "name": "power_user", "description": "Full access except recalibration",
            "can_upload": True, "can_upload_writer_draft": True,
            "can_delete_files": True, "can_download_files": True,
            "max_upload_size_mb": None, "max_uploads_per_day": None, "max_uploads_per_week": None,
            "can_view_wiki": True, "can_edit_wiki": True, "can_delete_wiki_pages": True,
            "can_query": True, "can_chat": True, "can_use_writer": True,
            "max_queries_per_day": None, "max_chat_messages_per_day": None,
            "max_tokens_per_day": None, "max_tokens_per_week": None,
            "can_recalibrate": False, "can_run_lint": True, "can_manage_schema": True,
            "can_view_audit_log": True, "can_view_graph": True, "can_rebuild_graph": True,
            "can_manage_workspace": True, "can_approve_ingest": True, "can_cancel_ingest": True,
        },
    ]
    for tpl_data in builtins:
        new_id = str(uuid.uuid4())
        params = {**tpl_data, "id": new_id}
        async with get_db() as db:
            await db.execute(_INSERT_GLOBAL_TEMPLATE, params)
    log.debug("seed_builtin_templates | global built-ins ensured")


# ── User permissions resolution ───────────────────────────────────────────────

async def get_user_permissions(user_id: str, org_id: str) -> dict:
    """Return the effective permission dict for a member user.

    Falls back to read_only defaults if no template is assigned.
    """
    try:
        async with get_db() as db:
            result = await db.execute(_GET_USER_PERMS, {"user_id": user_id, "org_id": org_id})
            row = result.fetchone()
    except Exception as exc:
        log.warning("get_user_permissions | DB error | %s", exc)
        return dict(_READ_ONLY_DEFAULTS)

    if not row:
        return dict(_READ_ONLY_DEFAULTS)

    if row.permission_template_id is None:
        return {"is_suspended": row.is_suspended, **{k: v for k, v in _READ_ONLY_DEFAULTS.items() if k != "is_suspended"}}

    return {
        "is_suspended": row.is_suspended,
        "can_upload": row.can_upload,
        "can_upload_writer_draft": row.can_upload_writer_draft,
        "can_delete_files": row.can_delete_files,
        "can_download_files": row.can_download_files,
        "max_upload_size_mb": row.max_upload_size_mb,
        "max_uploads_per_day": row.max_uploads_per_day,
        "max_uploads_per_week": row.max_uploads_per_week,
        "can_view_wiki": row.can_view_wiki,
        "can_edit_wiki": row.can_edit_wiki,
        "can_delete_wiki_pages": row.can_delete_wiki_pages,
        "can_query": row.can_query,
        "can_chat": row.can_chat,
        "can_use_writer": row.can_use_writer,
        "max_queries_per_day": row.max_queries_per_day,
        # Per-user overrides take precedence over template limits when set
        "max_chat_messages_per_day": row.user_max_chat_messages_per_day
                                     if row.user_max_chat_messages_per_day is not None
                                     else row.max_chat_messages_per_day,
        "max_tokens_per_day": row.user_max_tokens_per_day
                              if row.user_max_tokens_per_day is not None
                              else row.max_tokens_per_day,
        "max_tokens_per_week": row.max_tokens_per_week,
        "can_recalibrate": row.can_recalibrate,
        "can_run_lint": row.can_run_lint,
        "can_manage_schema": row.can_manage_schema,
        "can_view_audit_log": row.can_view_audit_log,
        "can_view_graph": row.can_view_graph,
        "can_rebuild_graph": row.can_rebuild_graph,
        "can_manage_workspace": row.can_manage_workspace,
        "can_approve_ingest": row.can_approve_ingest,
        "can_cancel_ingest": row.can_cancel_ingest,
    }


# ── FastAPI dependency factories ──────────────────────────────────────────────

async def _request_user_permissions(request, user_id: str, org_id: str) -> dict:
    """get_user_permissions memoized for the lifetime of one request.

    A single request often stacks several permission/quota dependencies (e.g.
    /chat = require_permission + check_chat_quota + check_token_quota), each of
    which previously re-ran the same permissions query. Permissions can't change
    mid-request, so we cache the result on request.state keyed by (user_id, org_id).
    """
    cache = getattr(request.state, "_perms_cache", None)
    if not isinstance(cache, dict):   # also guards MagicMock request.state in tests
        cache = {}
        request.state._perms_cache = cache
    key = (user_id, org_id)
    if key not in cache:
        cache[key] = await get_user_permissions(user_id, org_id)
    return cache[key]


def require_permission(flag: str):
    """Dependency factory: raise 403 if the authenticated user lacks *flag*.

    Admin role always passes. Suspended users are blocked on any permission.

    Usage::

        @router.post("/upload", dependencies=[Depends(require_permission("can_upload"))])
    """
    async def _dep(request: Request) -> None:
        role = getattr(request.state, "role", "member")
        if role in ("admin", "supervisor"):
            log.debug("permission_check | flag=%s | role=%s | status=bypassed", flag, role)
            return
        user_id = getattr(request.state, "user_id", "")
        org_id  = getattr(request.state, "org_id",  "")
        perms = await _request_user_permissions(request, user_id, org_id)
        if perms.get("is_suspended", False):
            log.warning("permission_check | flag=%s | user=%s | status=SUSPENDED", flag, user_id)
            raise HTTPException(403, "Your account has been suspended")
        if not perms.get(flag, False):
            log.warning("permission_check | flag=%s | user=%s | status=DENIED", flag, user_id)
            raise HTTPException(403, f"Permission denied: '{flag}' is not enabled for your account")
        log.debug("permission_check | flag=%s | user=%s | status=pass", flag, user_id)
    return _dep


def check_upload_quota():
    """Dependency: enforces per-user daily/weekly upload limits and org-wide daily cap."""
    async def _dep(request: Request) -> None:
        role = getattr(request.state, "role", "member")
        if role in ("admin", "supervisor"):
            log.debug("upload_quota | role=%s | status=bypassed", role)
            return
        user_id = getattr(request.state, "user_id", "")
        org_id  = getattr(request.state, "org_id",  "")
        perms = await _request_user_permissions(request, user_id, org_id)

        max_day  = perms.get("max_uploads_per_day")
        max_week = perms.get("max_uploads_per_week")

        async with get_db() as db:
            if max_day is not None:
                res = await db.execute(_COUNT_UPLOADS_TODAY, {"user_id": user_id})
                used_day = res.scalar() or 0
                log.info("upload_quota | user=%s | daily limit=%d | used=%d | status=%s",
                         user_id, max_day, used_day, "BLOCKED" if used_day >= max_day else "pass")
                if used_day >= max_day:
                    raise HTTPException(429, f"Daily upload limit reached ({max_day}/day)")

            if max_week is not None:
                res = await db.execute(_COUNT_UPLOADS_THIS_WEEK, {"user_id": user_id})
                used_week = res.scalar() or 0
                log.info("upload_quota | user=%s | weekly limit=%d | used=%d | status=%s",
                         user_id, max_week, used_week, "BLOCKED" if used_week >= max_week else "pass")
                if used_week >= max_week:
                    raise HTTPException(429, f"Weekly upload limit reached ({max_week}/week)")

        if max_day is None and max_week is None:
            log.debug("upload_quota | user=%s | status=no_limit", user_id)

        # Org-level cap
        try:
            async with get_db() as db:
                org_res = await db.execute(_GET_ORG_LIMITS, {"org_id": org_id})
                org_row = org_res.fetchone()
            if org_row and org_row.max_uploads_per_day_org is not None:
                async with get_db() as db:
                    res = await db.execute(_COUNT_ORG_UPLOADS_TODAY, {"org_id": org_id})
                    org_used = res.scalar() or 0
                    log.info("upload_quota | org=%s | org daily limit=%d | used=%d | status=%s",
                             org_id, org_row.max_uploads_per_day_org, org_used,
                             "BLOCKED" if org_used >= org_row.max_uploads_per_day_org else "pass")
                    if org_used >= org_row.max_uploads_per_day_org:
                        raise HTTPException(
                            429, f"Organization daily upload limit reached ({org_row.max_uploads_per_day_org}/day)")
        except HTTPException:
            raise
        except Exception:
            pass  # Non-fatal; don't block upload if org-limit query fails
    return _dep


def check_query_quota():
    """Dependency: enforces per-user daily query limit."""
    async def _dep(request: Request) -> None:
        role = getattr(request.state, "role", "member")
        if role in ("admin", "supervisor"):
            log.debug("query_quota | role=%s | status=bypassed", role)
            return
        user_id = getattr(request.state, "user_id", "")
        org_id  = getattr(request.state, "org_id",  "")
        perms = await _request_user_permissions(request, user_id, org_id)
        max_day = perms.get("max_queries_per_day")
        if max_day is None:
            log.debug("query_quota | user=%s | status=no_limit", user_id)
            return
        async with get_db() as db:
            res = await db.execute(_COUNT_QUERIES_TODAY, {"user_id": user_id})
            used = res.scalar() or 0
        log.info("query_quota | user=%s | limit=%d | used=%d | status=%s",
                 user_id, max_day, used, "BLOCKED" if used >= max_day else "pass")
        if used >= max_day:
            raise HTTPException(429, f"Daily query limit reached ({max_day}/day)")
    return _dep


def check_chat_quota():
    """Dependency: enforces per-user daily chat message limit."""
    async def _dep(request: Request) -> None:
        role = getattr(request.state, "role", "member")
        if role in ("admin", "supervisor"):
            log.debug("chat_quota | role=%s | status=bypassed", role)
            return
        user_id = getattr(request.state, "user_id", "")
        org_id  = getattr(request.state, "org_id",  "")
        perms = await _request_user_permissions(request, user_id, org_id)
        max_day = perms.get("max_chat_messages_per_day")
        if max_day is None:
            log.debug("chat_quota | user=%s | status=no_limit", user_id)
            return
        async with get_db() as db:
            res = await db.execute(_COUNT_CHATS_TODAY, {"user_id": user_id})
            used = res.scalar() or 0
        log.info("chat_quota | user=%s | limit=%d | used=%d | status=%s",
                 user_id, max_day, used, "BLOCKED" if used >= max_day else "pass")
        if used >= max_day:
            raise HTTPException(429, f"Daily chat message limit reached ({max_day}/day)")
    return _dep


def check_token_quota():
    """Dependency: enforces per-user and per-org daily/weekly token budget."""
    async def _dep(request: Request) -> None:
        role = getattr(request.state, "role", "member")
        if role in ("admin", "supervisor"):
            log.debug("token_quota | role=%s | status=bypassed", role)
            return
        user_id = getattr(request.state, "user_id", "")
        org_id  = getattr(request.state, "org_id",  "")
        perms = await _request_user_permissions(request, user_id, org_id)

        async with get_db() as db:
            max_day = perms.get("max_tokens_per_day")
            if max_day is not None:
                res = await db.execute(_SUM_TOKENS_TODAY, {"user_id": user_id})
                used_day = res.scalar() or 0
                log.info("token_quota | user=%s | daily limit=%d | used=%d | status=%s",
                         user_id, max_day, used_day, "BLOCKED" if used_day >= max_day else "pass")
                if used_day >= max_day:
                    raise HTTPException(429, f"Daily token budget exhausted ({max_day:,} tokens/day)")
            else:
                log.debug("token_quota | user=%s | daily limit=none", user_id)

            max_week = perms.get("max_tokens_per_week")
            if max_week is not None:
                res = await db.execute(_SUM_TOKENS_THIS_WEEK, {"user_id": user_id})
                used_week = res.scalar() or 0
                log.info("token_quota | user=%s | weekly limit=%d | used=%d | status=%s",
                         user_id, max_week, used_week, "BLOCKED" if used_week >= max_week else "pass")
                if used_week >= max_week:
                    raise HTTPException(429, f"Weekly token budget exhausted ({max_week:,} tokens/week)")

        try:
            async with get_db() as db:
                org_res = await db.execute(_GET_ORG_LIMITS, {"org_id": org_id})
                org_row = org_res.fetchone()
            if org_row and org_row.max_tokens_per_day_org is not None:
                async with get_db() as db:
                    res = await db.execute(_SUM_ORG_TOKENS_TODAY, {"org_id": org_id})
                    org_used = res.scalar() or 0
                    log.info("token_quota | org=%s | org daily limit=%d | used=%d | status=%s",
                             org_id, org_row.max_tokens_per_day_org, org_used,
                             "BLOCKED" if org_used >= org_row.max_tokens_per_day_org else "pass")
                    if org_used >= org_row.max_tokens_per_day_org:
                        raise HTTPException(
                            429, f"Organization daily token budget exhausted ({org_row.max_tokens_per_day_org:,} tokens/day)")
        except HTTPException:
            raise
        except Exception:
            pass
    return _dep
