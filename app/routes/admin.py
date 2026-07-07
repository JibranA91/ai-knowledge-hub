"""Admin dashboard API — all endpoints require role=admin.

Prefix: /api/admin

Sections:
  GET/POST/PUT/DELETE /api/admin/templates           — permission template CRUD
  POST                /api/admin/templates/{id}/clone — clone a template
  GET                 /api/admin/users               — list users with template + stats
  PUT                 /api/admin/users/{id}          — edit template / workspace / suspend
  POST                /api/admin/users/{id}/reset-password
  GET/POST            /api/admin/workspaces          — workspace list + create
  POST                /api/admin/workspaces/{id}/copy — copy workspace
  DELETE              /api/admin/workspaces/{id}     — delete workspace
  GET                 /api/admin/usage               — token/upload usage (filterable)
  GET/PUT             /api/admin/org-limits          — org-level caps
  GET                 /api/admin/audit-log           — paginated + searchable audit log
"""
import sqlalchemy as sa
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app.db import get_db
from app.logger import get_logger
from app.services import auth as auth_svc
from app.services import orgs as orgs_svc
from app.services import permissions as perm_svc
from app.services import workspaces as ws_svc

log = get_logger(__name__)
router = APIRouter(tags=["admin"])


# ── Auth guard ────────────────────────────────────────────────────────────────

def _is_admin(request: Request) -> bool:
    return getattr(request.state, "role", "") == "admin"


def _require_admin(request: Request) -> None:
    """Admin-only: org creation/deletion, supervisor assignment, cross-org ops."""
    if not _is_admin(request):
        raise HTTPException(403, "Admin role required")


def _require_supervisor_access(request: Request) -> None:
    """Admin or supervisor: CRUD within their own org."""
    if getattr(request.state, "role", "") not in ("admin", "supervisor"):
        raise HTTPException(403, "Admin or supervisor role required")


def _org_id(request: Request) -> str:
    return getattr(request.state, "org_id", "")


# ── Pydantic models ───────────────────────────────────────────────────────────

class TemplateCreate(BaseModel):
    org_id: str | None = None  # required when admin creates a template in a specific org
    name: str
    description: str = ""
    can_upload: bool = False
    can_upload_writer_draft: bool = False
    can_delete_files: bool = False
    can_download_files: bool = True
    max_upload_size_mb: int | None = None
    max_uploads_per_day: int | None = None
    max_uploads_per_week: int | None = None
    can_view_wiki: bool = True
    can_edit_wiki: bool = False
    can_delete_wiki_pages: bool = False
    can_query: bool = False
    can_chat: bool = False
    can_use_writer: bool = False
    max_queries_per_day: int | None = None
    max_chat_messages_per_day: int | None = None
    max_tokens_per_day: int | None = None
    max_tokens_per_week: int | None = None
    can_recalibrate: bool = False
    can_run_lint: bool = False
    can_manage_schema: bool = False
    can_view_audit_log: bool = False
    can_view_graph: bool = True
    can_rebuild_graph: bool = False
    can_manage_workspace: bool = False
    can_approve_ingest: bool = False
    can_cancel_ingest: bool = False


class TemplateUpdate(TemplateCreate):
    pass


class UserUpdate(BaseModel):
    # The membership being edited. Admins must pass org_id to target a specific
    # org; supervisors operate within their own org (org_id is ignored).
    org_id: str | None = None
    permission_template_id: str | None = None
    workspace_id: str | None = None
    is_suspended: bool | None = None
    role: str | None = None
    max_tokens_per_day: int | None = None
    max_chat_messages_per_day: int | None = None


class MembershipCreate(BaseModel):
    org_id: str
    role: str = "member"
    permission_template_id: str | None = None
    workspace_id: str | None = None
    is_suspended: bool = False
    max_tokens_per_day: int | None = None
    max_chat_messages_per_day: int | None = None


class MembershipUpdate(BaseModel):
    role: str | None = None
    permission_template_id: str | None = None
    workspace_id: str | None = None
    is_suspended: bool | None = None
    max_tokens_per_day: int | None = None
    max_chat_messages_per_day: int | None = None


class ResetPasswordRequest(BaseModel):
    new_password: str


class WorkspaceCreate(BaseModel):
    org_id: str | None = None  # required when admin creates a workspace in a specific org
    name: str
    owner_user_id: str | None = None


class WorkspaceCopy(BaseModel):
    new_name: str
    owner_user_id: str | None = None


class OrgLimitsUpdate(BaseModel):
    org_id: str | None = None  # required when called by admin to target a specific org
    max_uploads_per_day_org: int | None = None
    max_tokens_per_day_org: int | None = None
    max_members: int | None = None
    revision_retention_days: int | None = None


class OrgCreate(BaseModel):
    name: str
    supervisor_email: str | None = None  # optional; can be assigned later via PUT /organizations/{id}/supervisor


class ChangeSupervisorRequest(BaseModel):
    supervisor_email: str  # existing user to designate as supervisor


class OrgCloneRequest(BaseModel):
    new_name: str


class GlobalUserCreate(BaseModel):
    email: str
    password: str
    role: str = "member"
    org_id: str | None = None
    permission_template_id: str | None = None
    max_tokens_per_day: int | None = None
    max_chat_messages_per_day: int | None = None


class OrgUserCreate(BaseModel):
    email: str
    password: str
    role: str = "member"
    permission_template_id: str | None = None


class TransferOrgRequest(BaseModel):
    new_org_id: str


# ── Admin-wide SQL (no org filter) ───────────────────────────────────────────

_LIST_TEMPLATES_ALL = sa.text("""
    SELECT pt.id, pt.org_id, pt.name, pt.description, pt.is_builtin,
           pt.can_upload, pt.can_upload_writer_draft, pt.can_delete_files, pt.can_download_files,
           pt.max_upload_size_mb, pt.max_uploads_per_day, pt.max_uploads_per_week,
           pt.can_view_wiki, pt.can_edit_wiki, pt.can_delete_wiki_pages,
           pt.can_query, pt.can_chat, pt.can_use_writer,
           pt.max_queries_per_day, pt.max_chat_messages_per_day,
           pt.max_tokens_per_day, pt.max_tokens_per_week,
           pt.can_recalibrate, pt.can_run_lint, pt.can_manage_schema,
           pt.can_view_audit_log, pt.can_view_graph, pt.can_rebuild_graph,
           pt.can_manage_workspace, pt.can_approve_ingest, pt.can_cancel_ingest,
           pt.created_at, pt.updated_at,
           o.name AS org_name
    FROM permission_templates pt
    LEFT JOIN organizations o ON pt.org_id = o.id
    ORDER BY o.name, pt.is_builtin DESC, pt.name
""")

_LIST_WORKSPACES_ALL = sa.text("""
    SELECT w.id, w.org_id, w.name, w.owner_user_id, w.s3_prefix,
           w.created_from_workspace_id, w.created_at,
           u.email AS owner_email,
           o.name  AS org_name
    FROM workspaces w
    LEFT JOIN users u ON w.owner_user_id = u.id
    LEFT JOIN organizations o ON w.org_id = o.id
    ORDER BY o.name, w.created_at DESC
""")

_USAGE_SUMMARY_ALL = sa.text("""
    SELECT
        u.id AS user_id, u.email,
        ul.model_id,
        ul.operation,
        COALESCE(SUM(ul.tokens_in), 0)  AS tokens_in,
        COALESCE(SUM(ul.tokens_out), 0) AS tokens_out,
        COALESCE(SUM(ul.tokens_in + ul.tokens_out), 0) AS tokens_total,
        COUNT(*) AS call_count
    FROM usage_log ul
    LEFT JOIN users u ON ul.user_id = u.id
    WHERE ul.created_at >= NOW() - (:days * INTERVAL '1 day')
    GROUP BY u.id, u.email, ul.model_id, ul.operation
    ORDER BY tokens_total DESC
    LIMIT 200
""")

_UPLOAD_COUNTS_ALL = sa.text("""
    SELECT
        u.id AS user_id, u.email,
        COUNT(*) AS upload_count,
        MAX(ij.created_at) AS last_upload_at
    FROM ingest_jobs ij
    LEFT JOIN users u ON ij.user_id = u.id
    WHERE ij.created_at >= NOW() - (:days * INTERVAL '1 day')
    GROUP BY u.id, u.email
    ORDER BY upload_count DESC
    LIMIT 100
""")

# Audit-log SQL templates. The {sort} placeholder is substituted at runtime
# from a whitelist (never user input directly), so no SQL injection risk.

_AUDIT_LOG_ALL_TPL = """
    SELECT al.id, al.operation, al.raw_text, al.details, al.org_id, al.created_at,
           u.email AS user_email,
           o.name  AS org_name
    FROM audit_log al
    LEFT JOIN users u ON al.user_id = u.id
    LEFT JOIN organizations o ON al.org_id = o.id
    WHERE (CAST(:org_filter AS TEXT) IS NULL OR al.org_id = CAST(:org_filter AS UUID))
      AND (:operation = '' OR al.operation = :operation)
      AND (:search    = '' OR al.operation ILIKE :search_like OR al.raw_text ILIKE :search_like)
    ORDER BY al.created_at {sort}
    LIMIT :limit OFFSET :offset
"""

_AUDIT_LOG_COUNT_ALL = sa.text("""
    SELECT COUNT(*) FROM audit_log al
    WHERE (CAST(:org_filter AS TEXT) IS NULL OR al.org_id = CAST(:org_filter AS UUID))
      AND (:operation = '' OR al.operation = :operation)
      AND (:search    = '' OR al.operation ILIKE :search_like OR al.raw_text ILIKE :search_like)
""")

_AUDIT_LOG_OPERATIONS_ALL = sa.text("""
    SELECT DISTINCT operation FROM audit_log
    WHERE (CAST(:org_filter AS TEXT) IS NULL OR org_id = CAST(:org_filter AS UUID))
    ORDER BY operation
""")

_AUDIT_LOG_OPERATIONS_ONE = sa.text("""
    SELECT DISTINCT operation FROM audit_log
    WHERE org_id = CAST(:org_id AS UUID)
    ORDER BY operation
""")


# ── Permission templates ──────────────────────────────────────────────────────

@router.get("/me")
async def get_me(request: Request):
    """Return the current user's role and org — used by the admin SPA for access control.

    `memberships` lists the orgs this (non-admin) user belongs to, so the SPA can
    offer a supervisor of multiple orgs an org switcher scoped to the orgs they
    supervise. Empty for admins, who already switch across all orgs.
    """
    role = getattr(request.state, "role", "")
    user_id = getattr(request.state, "user_id", "")
    memberships: list[dict] = []
    if role != "admin" and user_id:
        memberships = [
            {"org_id": m["org_id"], "org_name": m["org_name"], "role": m["role"]}
            for m in await orgs_svc.list_memberships_for_user(user_id)
        ]
    return {
        "email": getattr(request.state, "user", ""),
        "role": role,
        "org_id": getattr(request.state, "org_id", ""),
        "user_id": user_id,
        "memberships": memberships,
    }


@router.get("/templates")
async def list_templates(request: Request):
    _require_supervisor_access(request)
    if _is_admin(request):
        async with get_db() as db:
            result = await db.execute(_LIST_TEMPLATES_ALL)
            rows = result.fetchall()
        return [
            {
                **perm_svc._row_to_template(r),
                "org_name": r.org_name,
            }
            for r in rows
        ]
    return await perm_svc.list_templates(_org_id(request))


@router.post("/templates", status_code=201)
async def create_template(request: Request, body: TemplateCreate):
    _require_supervisor_access(request)
    if _is_admin(request):
        org_id = body.org_id  # None → global (all orgs); explicit org_id → org-scoped
    else:
        org_id = _org_id(request)
        if not org_id:
            raise HTTPException(400, "org_id required")
    data = body.model_dump(exclude={"org_id"})
    template_id = await perm_svc.create_template(org_id, data)
    if not template_id:
        raise HTTPException(409, f"A template named '{body.name}' already exists")
    return {"id": template_id, "name": body.name}


@router.get("/templates/{template_id}")
async def get_template(request: Request, template_id: str):
    _require_supervisor_access(request)
    if _is_admin(request):
        # Admin can view any template regardless of org
        tpl = await perm_svc.get_template_any(template_id)
    else:
        tpl = await perm_svc.get_template(template_id, _org_id(request))
    if not tpl:
        raise HTTPException(404, "Template not found")
    return tpl


@router.put("/templates/{template_id}")
async def update_template(request: Request, template_id: str, body: TemplateUpdate):
    _require_supervisor_access(request)
    if _is_admin(request):
        tpl = await perm_svc.get_template_any(template_id)
        if tpl and tpl.get("is_builtin"):
            raise HTTPException(403, "Built-in templates cannot be edited")
        # global templates have org_id=None; preserve None rather than falling back to ""
        org_id = tpl["org_id"] if tpl else _org_id(request)
    else:
        org_id = _org_id(request)
    data = body.model_dump(exclude={"org_id"})
    updated = await perm_svc.update_template(template_id, org_id, data)
    if not updated:
        raise HTTPException(404, "Template not found or is a built-in template (cannot edit)")
    return {"updated": template_id}


@router.delete("/templates/{template_id}")
async def delete_template(request: Request, template_id: str):
    _require_supervisor_access(request)
    if _is_admin(request):
        tpl = await perm_svc.get_template_any(template_id)
        org_id = tpl["org_id"] if tpl else ""
    else:
        org_id = _org_id(request)
    deleted = await perm_svc.delete_template(template_id, org_id)
    if not deleted:
        raise HTTPException(404, "Template not found or is a built-in template (cannot delete)")
    return {"deleted": template_id}


@router.get("/templates/{template_id}/user-count")
async def template_user_count(request: Request, template_id: str):
    _require_supervisor_access(request)
    count = await perm_svc.count_template_users(template_id)
    return {"count": count}


@router.post("/templates/{template_id}/clone", status_code=201)
async def clone_template(request: Request, template_id: str, body: dict):
    _require_supervisor_access(request)
    if _is_admin(request):
        source = await perm_svc.get_template_any(template_id)
        target_org = body.get("org_id")  # None → global (all orgs); explicit org_id → org-scoped
    else:
        target_org = _org_id(request)
        if not target_org:
            raise HTTPException(400, "org_id required")
        source = await perm_svc.get_template(template_id, target_org)
    if not source:
        raise HTTPException(404, "Source template not found")
    new_name = body.get("name", f"{source['name']}_copy")
    clone_data = {**source, "name": new_name, "is_builtin": False}
    new_id = await perm_svc.create_template(target_org, clone_data)
    if not new_id:
        raise HTTPException(409, f"A template named '{new_name}' already exists")
    return {"id": new_id, "name": new_name}


# ── User management ───────────────────────────────────────────────────────────

# One row per (user, org) membership. Admin users have no membership, so they
# appear once with org_id = NULL (LEFT JOIN) and their global role. Per-org
# fields (role, template, suspension, limits) come from org_memberships.
_LIST_USERS_ALL = sa.text("""
    SELECT u.id, m.org_id, u.email,
           COALESCE(m.role, u.role) AS role,
           COALESCE(m.is_suspended, false) AS is_suspended,
           m.permission_template_id, m.workspace_id,
           u.created_at, u.last_login_at,
           m.max_tokens_per_day, m.max_chat_messages_per_day,
           pt.name AS template_name,
           o.name AS org_name,
           (SELECT COUNT(*) FROM ingest_jobs ij
            WHERE ij.user_id = u.id
              AND ij.created_at >= CURRENT_DATE
              AND (m.org_id IS NULL OR ij.org_id = m.org_id)) AS uploads_today,
           (SELECT COALESCE(SUM(ul.tokens_in + ul.tokens_out), 0) FROM usage_log ul
            WHERE ul.user_id = u.id
              AND ul.created_at >= CURRENT_DATE
              AND (m.org_id IS NULL OR ul.org_id = m.org_id)) AS tokens_today
    FROM users u
    LEFT JOIN org_memberships m ON m.user_id = u.id
    LEFT JOIN permission_templates pt ON m.permission_template_id = pt.id
    LEFT JOIN organizations o ON m.org_id = o.id
    ORDER BY o.name NULLS FIRST, u.email
""")

_LIST_USERS_ORG = sa.text("""
    SELECT u.id, m.org_id, u.email, m.role,
           m.is_suspended,
           m.permission_template_id, m.workspace_id,
           u.created_at, u.last_login_at,
           m.max_tokens_per_day, m.max_chat_messages_per_day,
           pt.name AS template_name,
           NULL AS org_name,
           (SELECT COUNT(*) FROM ingest_jobs ij
            WHERE ij.user_id = u.id
              AND ij.created_at >= CURRENT_DATE
              AND ij.org_id = m.org_id) AS uploads_today,
           (SELECT COALESCE(SUM(ul.tokens_in + ul.tokens_out), 0) FROM usage_log ul
            WHERE ul.user_id = u.id
              AND ul.created_at >= CURRENT_DATE
              AND ul.org_id = m.org_id) AS tokens_today
    FROM org_memberships m
    JOIN users u ON u.id = m.user_id
    LEFT JOIN permission_templates pt ON m.permission_template_id = pt.id
    WHERE m.org_id = CAST(:org_id AS UUID)
    ORDER BY u.email
""")

_RESET_PASSWORD = sa.text("""
    UPDATE users SET password_hash = :password_hash
    WHERE id = CAST(:id AS UUID)
      AND (CAST(:org_id AS TEXT) IS NULL OR org_id = CAST(:org_id AS UUID))
""")


@router.get("/users")
async def list_users(request: Request):
    _require_supervisor_access(request)
    async with get_db() as db:
        if _is_admin(request):
            result = await db.execute(_LIST_USERS_ALL)
        else:
            result = await db.execute(_LIST_USERS_ORG, {"org_id": _org_id(request)})
        rows = result.fetchall()
    return [
        {
            "id": str(r.id),
            "org_id": str(r.org_id) if r.org_id else None,
            "org_name": r.org_name,
            "email": r.email,
            "role": r.role,
            "is_suspended": r.is_suspended,
            "permission_template_id": str(r.permission_template_id) if r.permission_template_id else None,
            "template_name": r.template_name,
            "workspace_id": str(r.workspace_id) if r.workspace_id else None,
            "max_tokens_per_day": r.max_tokens_per_day,
            "max_chat_messages_per_day": r.max_chat_messages_per_day,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "last_login_at": r.last_login_at.isoformat() if r.last_login_at else None,
            "uploads_today": r.uploads_today,
            "tokens_today": r.tokens_today,
        }
        for r in rows
    ]


_LIST_IDENTITIES = sa.text("""
    SELECT u.id, u.email, u.role,
           (SELECT COUNT(*) FROM org_memberships m WHERE m.user_id = u.id) AS org_count
    FROM users u
    ORDER BY u.email
""")


@router.get("/identities")
async def list_identities(request: Request):
    """Flat list of global user identities — used to populate the member-picker
    in the Organizations tab. One row per user (not per membership)."""
    _require_admin(request)
    async with get_db() as db:
        rows = (await db.execute(_LIST_IDENTITIES)).fetchall()
    return [
        {
            "id": str(r.id),
            "email": r.email,
            "is_admin": r.role == "admin",
            "org_count": r.org_count,
        }
        for r in rows
    ]


@router.post("/users", status_code=201)
async def create_user_global(request: Request, body: GlobalUserCreate):
    """Create a user identity, or grant an existing user access to an org.

    Identity-only creation (the Add User modal):
      - admin flag/role → global admin, org_id=NULL, no membership.
      - otherwise, with no org_id → a "floating" identity (credentials only, no
        membership). Org access is granted later from the Organizations tab.

    Org-scoped creation (org_id supplied) is still supported: a new email gets a
    user + membership; an existing email simply gains a membership."""
    _require_admin(request)
    if len(body.password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters")
    if body.role not in ("admin", "supervisor", "member"):
        raise HTTPException(400, "role must be 'admin', 'supervisor', or 'member'")

    if body.role == "admin":
        user_id = await orgs_svc.create_user(None, body.email, body.password, role="admin")
        if not user_id:
            raise HTTPException(409, f"User '{body.email}' already exists")
        log.info("Admin user created | user=%s | by=%s", user_id, getattr(request.state, "user_id", ""))
        return {"id": user_id, "email": body.email, "org_id": None, "role": "admin"}

    # Only an EXPLICIT org_id makes this org-scoped. We deliberately do not fall
    # back to the admin's active org (X-Org-Context) — the Add User modal sends
    # no org_id and must create a floating identity regardless of what org the
    # admin happens to be viewing.
    effective_org = body.org_id
    if not effective_org:
        # Floating identity: credentials only, no org access yet. create_user
        # with org_id=None inserts the user row and skips membership creation.
        user_id = await orgs_svc.create_user(None, body.email, body.password, role="member")
        if not user_id:
            raise HTTPException(409, f"User '{body.email}' already exists")
        log.info("Floating user created | user=%s | by=%s", user_id, getattr(request.state, "user_id", ""))
        return {"id": user_id, "email": body.email, "org_id": None, "role": "member"}

    existing = await orgs_svc.get_user_by_email(body.email)
    if existing:
        if existing["role"] == "admin":
            raise HTTPException(400, "That email belongs to a global admin and cannot be added to an org")
        user_id = existing["id"]
        added = await orgs_svc.add_membership(
            user_id, effective_org, body.role,
            permission_template_id=body.permission_template_id,
            max_tokens_per_day=body.max_tokens_per_day,
            max_chat_messages_per_day=body.max_chat_messages_per_day,
        )
        if not added:
            raise HTTPException(409, f"User '{body.email}' is already a member of this organization")
    else:
        user_id = await orgs_svc.create_user(effective_org, body.email, body.password, role=body.role)
        if not user_id:
            raise HTTPException(409, f"User '{body.email}' already exists")
        # create_user seeds a membership with defaults; apply the requested extras.
        if body.permission_template_id or body.max_tokens_per_day or body.max_chat_messages_per_day:
            await orgs_svc.update_membership(user_id, effective_org, {
                "role": body.role,
                "permission_template_id": body.permission_template_id,
                "max_tokens_per_day": body.max_tokens_per_day,
                "max_chat_messages_per_day": body.max_chat_messages_per_day,
            })

    if body.role == "supervisor":
        async with get_db() as db:
            await db.execute(_SET_ORG_SUPERVISOR, {"org_id": effective_org, "user_id": user_id})
    log.info("User created/assigned | user=%s | org=%s | role=%s | by=%s",
             user_id, effective_org, body.role, getattr(request.state, "user_id", ""))
    return {"id": user_id, "email": body.email, "org_id": effective_org, "role": body.role}


@router.put("/users/{user_id}")
async def update_user(request: Request, user_id: str, body: UserUpdate):
    """Edit a user's membership in one org (template, workspace, suspension,
    role, limit overrides). Admins target any org via body.org_id; supervisors
    operate within their own org."""
    _require_supervisor_access(request)

    # Membership roles are supervisor/member only — 'admin' is a global identity
    # role, set when the user is created, never via a membership edit.
    if body.role is not None and body.role not in ("supervisor", "member"):
        raise HTTPException(400, "membership role must be 'supervisor' or 'member'")

    target_org = body.org_id if _is_admin(request) else _org_id(request)
    # Convenience: an admin who omits org_id targets the user's only membership.
    if not target_org and _is_admin(request):
        memberships = await orgs_svc.list_memberships_for_user(user_id)
        if len(memberships) == 1:
            target_org = memberships[0]["org_id"]
    if not target_org:
        raise HTTPException(400, "org_id required")

    # Prevent self-suspension in the org you're acting in
    if body.is_suspended and user_id == getattr(request.state, "user_id", ""):
        raise HTTPException(400, "Cannot suspend your own account")

    membership = await orgs_svc.get_membership(user_id, target_org)
    if not membership:
        raise HTTPException(404, "User is not a member of this organization")

    updated = await orgs_svc.update_membership(user_id, target_org, {
        "role": body.role or membership["role"],
        "permission_template_id": body.permission_template_id,
        "workspace_id": body.workspace_id,
        "is_suspended": body.is_suspended if body.is_suspended is not None else False,
        "max_tokens_per_day": body.max_tokens_per_day,
        "max_chat_messages_per_day": body.max_chat_messages_per_day,
    })
    if not updated:
        raise HTTPException(404, "User is not a member of this organization")
    log.info("Membership updated | user=%s | org=%s | by=%s",
             user_id, target_org, getattr(request.state, "user_id", ""))
    return {"updated": user_id}


# ── Membership management (assign a user to multiple orgs) ─────────────────────

@router.get("/users/{user_id}/memberships")
async def list_user_memberships(request: Request, user_id: str):
    """List every org a user belongs to (with their role in each)."""
    _require_supervisor_access(request)
    memberships = await orgs_svc.list_memberships_for_user(user_id)
    if not _is_admin(request):
        # Supervisors only see the membership in their own org.
        memberships = [m for m in memberships if m["org_id"] == _org_id(request)]
    return memberships


@router.post("/users/{user_id}/memberships", status_code=201)
async def add_user_membership(request: Request, user_id: str, body: MembershipCreate):
    """Assign an existing user to an org. This is how one identity gains access
    to multiple orgs with a different role/profile in each."""
    _require_supervisor_access(request)
    if body.role not in ("supervisor", "member"):
        raise HTTPException(400, "role must be 'supervisor' or 'member'")
    target_org = body.org_id if _is_admin(request) else _org_id(request)
    if not target_org:
        raise HTTPException(400, "org_id required")
    if not _is_admin(request) and body.org_id and body.org_id != _org_id(request):
        raise HTTPException(403, "Supervisors can only assign within their own org")

    user = await orgs_svc.get_user(user_id)
    if not user:
        raise HTTPException(404, "User not found")
    if user["role"] == "admin":
        raise HTTPException(400, "Admin users are global and cannot be assigned to an org")

    created = await orgs_svc.add_membership(
        user_id, target_org, body.role,
        permission_template_id=body.permission_template_id,
        workspace_id=body.workspace_id,
        is_suspended=body.is_suspended,
        max_tokens_per_day=body.max_tokens_per_day,
        max_chat_messages_per_day=body.max_chat_messages_per_day,
    )
    if not created:
        raise HTTPException(409, "User is already a member of this organization")
    log.info("Membership added | user=%s | org=%s | role=%s | by=%s",
             user_id, target_org, body.role, getattr(request.state, "user_id", ""))
    return {"user_id": user_id, "org_id": target_org, "role": body.role}


@router.put("/users/{user_id}/memberships/{org_id}")
async def update_user_membership(request: Request, user_id: str, org_id: str, body: MembershipUpdate):
    _require_supervisor_access(request)
    if not _is_admin(request) and org_id != _org_id(request):
        raise HTTPException(403, "Supervisors can only edit memberships in their own org")
    if body.role is not None and body.role not in ("supervisor", "member"):
        raise HTTPException(400, "role must be 'supervisor' or 'member'")
    if body.is_suspended and user_id == getattr(request.state, "user_id", ""):
        raise HTTPException(400, "Cannot suspend your own account")

    membership = await orgs_svc.get_membership(user_id, org_id)
    if not membership:
        raise HTTPException(404, "User is not a member of this organization")
    updated = await orgs_svc.update_membership(user_id, org_id, {
        "role": body.role or membership["role"],
        "permission_template_id": body.permission_template_id,
        "workspace_id": body.workspace_id,
        "is_suspended": body.is_suspended if body.is_suspended is not None else membership["is_suspended"],
        "max_tokens_per_day": body.max_tokens_per_day,
        "max_chat_messages_per_day": body.max_chat_messages_per_day,
    })
    if not updated:
        raise HTTPException(404, "User is not a member of this organization")
    return {"updated": user_id, "org_id": org_id}


@router.delete("/users/{user_id}/memberships/{org_id}")
async def remove_user_membership(request: Request, user_id: str, org_id: str):
    _require_supervisor_access(request)
    if not _is_admin(request) and org_id != _org_id(request):
        raise HTTPException(403, "Supervisors can only remove memberships in their own org")
    if user_id == getattr(request.state, "user_id", "") and org_id == _org_id(request):
        raise HTTPException(400, "Cannot remove your own membership in the org you're acting in")
    removed = await orgs_svc.remove_membership(user_id, org_id)
    if not removed:
        raise HTTPException(404, "User is not a member of this organization")
    # Revoke sessions so a removed user can't keep acting on the org via a cached token.
    await auth_svc.revoke_all_tokens_for_user(user_id)
    log.info("Membership removed | user=%s | org=%s | by=%s",
             user_id, org_id, getattr(request.state, "user_id", ""))
    return {"removed": user_id, "org_id": org_id}


@router.post("/users/{user_id}/reset-password")
async def reset_password(request: Request, user_id: str, body: ResetPasswordRequest):
    _require_supervisor_access(request)
    if not body.new_password or len(body.new_password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters")
    user = await orgs_svc.get_user(user_id)
    if not user:
        raise HTTPException(404, "User not found")
    if not _is_admin(request) and user.get("org_id") != _org_id(request):
        raise HTTPException(404, "User not found in this organization")
    actual_org_id = user.get("org_id") if _is_admin(request) else _org_id(request)
    import bcrypt
    hashed = bcrypt.hashpw(body.new_password.encode(), bcrypt.gensalt()).decode()
    async with get_db() as db:
        result = await db.execute(_RESET_PASSWORD, {
            "id": user_id, "org_id": actual_org_id, "password_hash": hashed
        })
    if result.rowcount == 0:
        raise HTTPException(404, "User not found in this organization")
    log.info("Password reset | user=%s | by=%s", user_id, getattr(request.state, "user_id", ""))
    return {"reset": user_id}


# ── Workspaces ────────────────────────────────────────────────────────────────

@router.get("/workspaces")
async def list_workspaces(request: Request):
    _require_supervisor_access(request)
    if _is_admin(request):
        async with get_db() as db:
            result = await db.execute(_LIST_WORKSPACES_ALL)
            rows = result.fetchall()
        return [
            {
                "id": str(r.id),
                "org_id": str(r.org_id),
                "org_name": r.org_name,
                "name": r.name,
                "owner_user_id": str(r.owner_user_id) if r.owner_user_id else None,
                "owner_email": r.owner_email,
                "s3_prefix": r.s3_prefix,
                "created_from_workspace_id": str(r.created_from_workspace_id) if r.created_from_workspace_id else None,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ]
    return await ws_svc.list_workspaces(_org_id(request))


@router.post("/workspaces", status_code=201)
async def create_workspace(request: Request, body: WorkspaceCreate):
    _require_supervisor_access(request)
    org_id = (body.org_id if _is_admin(request) and body.org_id else None) or _org_id(request)
    if not org_id:
        raise HTTPException(400, "org_id required")
    ws_id = await ws_svc.create_workspace(org_id, body.name, body.owner_user_id)
    if not ws_id:
        raise HTTPException(409, f"A workspace named '{body.name}' already exists")
    return {"id": ws_id, "name": body.name}


@router.post("/workspaces/{workspace_id}/copy", status_code=201)
async def copy_workspace(request: Request, workspace_id: str, body: WorkspaceCopy):
    _require_supervisor_access(request)
    if _is_admin(request):
        # Look up the source workspace to get its org
        _ws_lookup = sa.text("SELECT org_id FROM workspaces WHERE id = CAST(:id AS UUID)")
        async with get_db() as db:
            row = (await db.execute(_ws_lookup, {"id": workspace_id})).fetchone()
        if not row:
            raise HTTPException(404, "Source workspace not found or name conflict")
        org_id = str(row.org_id)
    else:
        org_id = _org_id(request)
    new_id = await ws_svc.copy_workspace(workspace_id, org_id, body.new_name, body.owner_user_id)
    if not new_id:
        raise HTTPException(404, "Source workspace not found or name conflict")
    return {"id": new_id, "name": body.new_name}


@router.delete("/workspaces/{workspace_id}")
async def delete_workspace(request: Request, workspace_id: str):
    _require_supervisor_access(request)
    if _is_admin(request):
        _ws_lookup = sa.text("SELECT org_id FROM workspaces WHERE id = CAST(:id AS UUID)")
        async with get_db() as db:
            row = (await db.execute(_ws_lookup, {"id": workspace_id})).fetchone()
        if not row:
            raise HTTPException(404, "Workspace not found")
        org_id = str(row.org_id)
    else:
        org_id = _org_id(request)
    deleted = await ws_svc.delete_workspace(workspace_id, org_id)
    if not deleted:
        raise HTTPException(404, "Workspace not found")
    return {"deleted": workspace_id}


# ── Usage analytics ───────────────────────────────────────────────────────────

_USAGE_BY_USER = sa.text("""
    SELECT
        u.id AS user_id, u.email,
        COALESCE(SUM(ul.tokens_in), 0)  AS tokens_in,
        COALESCE(SUM(ul.tokens_out), 0) AS tokens_out,
        COALESCE(SUM(ul.tokens_in + ul.tokens_out), 0) AS tokens_total,
        COUNT(*) AS call_count
    FROM usage_log ul
    JOIN users u ON ul.user_id = u.id
    WHERE ul.org_id = CAST(:org_id AS UUID)
      AND ul.created_at >= NOW() - INTERVAL ':days days'
    GROUP BY u.id, u.email
    ORDER BY tokens_total DESC
    LIMIT 100
""")

_USAGE_SUMMARY = sa.text("""
    SELECT
        u.id AS user_id, u.email,
        ul.model_id,
        ul.operation,
        COALESCE(SUM(ul.tokens_in), 0)  AS tokens_in,
        COALESCE(SUM(ul.tokens_out), 0) AS tokens_out,
        COALESCE(SUM(ul.tokens_in + ul.tokens_out), 0) AS tokens_total,
        COUNT(*) AS call_count
    FROM usage_log ul
    LEFT JOIN users u ON ul.user_id = u.id
    WHERE ul.org_id = CAST(:org_id AS UUID)
      AND ul.created_at >= NOW() - (:days * INTERVAL '1 day')
    GROUP BY u.id, u.email, ul.model_id, ul.operation
    ORDER BY tokens_total DESC
    LIMIT 200
""")

_UPLOAD_COUNTS = sa.text("""
    SELECT
        u.id AS user_id, u.email,
        COUNT(*) AS upload_count,
        MAX(ij.created_at) AS last_upload_at
    FROM ingest_jobs ij
    LEFT JOIN users u ON ij.user_id = u.id
    WHERE ij.org_id = CAST(:org_id AS UUID)
      AND ij.created_at >= NOW() - (:days * INTERVAL '1 day')
    GROUP BY u.id, u.email
    ORDER BY upload_count DESC
    LIMIT 100
""")


@router.get("/usage")
async def get_usage(request: Request, days: int = 30):
    _require_supervisor_access(request)

    async with get_db() as db:
        if _is_admin(request):
            token_res = await db.execute(_USAGE_SUMMARY_ALL, {"days": days})
            upload_res = await db.execute(_UPLOAD_COUNTS_ALL, {"days": days})
        else:
            org_id = _org_id(request)
            token_res = await db.execute(_USAGE_SUMMARY, {"org_id": org_id, "days": days})
            upload_res = await db.execute(_UPLOAD_COUNTS, {"org_id": org_id, "days": days})
        token_rows = token_res.fetchall()
        upload_rows = upload_res.fetchall()

    token_data = [
        {
            "user_id": str(r.user_id) if r.user_id else None,
            "email": r.email,
            "model_id": r.model_id,
            "operation": r.operation,
            "tokens_in": r.tokens_in,
            "tokens_out": r.tokens_out,
            "tokens_total": r.tokens_total,
            "call_count": r.call_count,
        }
        for r in token_rows
    ]
    upload_data = [
        {
            "user_id": str(r.user_id) if r.user_id else None,
            "email": r.email,
            "upload_count": r.upload_count,
            "last_upload_at": r.last_upload_at.isoformat() if r.last_upload_at else None,
        }
        for r in upload_rows
    ]
    return {"days": days, "token_usage": token_data, "upload_counts": upload_data}


# ── Organization management ───────────────────────────────────────────────────

# user_count and supervisors are derived from org_memberships. supervisor_email
# (singular) is kept for back-compat (the org's "primary" supervisor pointer);
# supervisor_emails (array) lists all supervisors now that an org can have many.
_ORG_SELECT_COLS = """
    o.id, o.name, o.slug, o.created_at,
    o.max_uploads_per_day_org, o.max_tokens_per_day_org, o.max_members,
    o.revision_retention_days,
    o.supervisor_user_id,
    su.email AS supervisor_email,
    (SELECT COUNT(*) FROM org_memberships m WHERE m.org_id = o.id) AS user_count,
    (SELECT COALESCE(array_agg(u2.email ORDER BY u2.email), ARRAY[]::text[])
     FROM org_memberships m2 JOIN users u2 ON u2.id = m2.user_id
     WHERE m2.org_id = o.id AND m2.role = 'supervisor') AS supervisor_emails
"""

_LIST_ORGS_ALL = sa.text(f"""
    SELECT {_ORG_SELECT_COLS}
    FROM organizations o
    LEFT JOIN users su ON su.id = o.supervisor_user_id
    ORDER BY o.created_at
""")

_LIST_ORGS_ONE = sa.text(f"""
    SELECT {_ORG_SELECT_COLS}
    FROM organizations o
    LEFT JOIN users su ON su.id = o.supervisor_user_id
    WHERE o.id = CAST(:org_id AS UUID)
""")

# Designate a supervisor by upserting a supervisor membership (multiple allowed).
_ASSIGN_SUPERVISOR = sa.text("""
    INSERT INTO org_memberships (user_id, org_id, role)
    VALUES (CAST(:user_id AS UUID), CAST(:org_id AS UUID), 'supervisor')
    ON CONFLICT (user_id, org_id) DO UPDATE SET role = 'supervisor'
""")

# Optional "primary supervisor" pointer kept for back-compat.
_SET_ORG_SUPERVISOR = sa.text("""
    UPDATE organizations SET supervisor_user_id = CAST(:user_id AS UUID)
    WHERE id = CAST(:org_id AS UUID)
""")

_DELETE_ORG = sa.text("""
    DELETE FROM organizations
    WHERE id = CAST(:id AS UUID)
      AND id != CAST('00000000-0000-0000-0000-000000000001' AS UUID)
""")


def _org_row_to_dict(r) -> dict:
    return {
        "id": str(r.id),
        "name": r.name,
        "slug": r.slug,
        "created_at": r.created_at.isoformat() if r.created_at else None,
        "max_uploads_per_day_org": r.max_uploads_per_day_org,
        "max_tokens_per_day_org": r.max_tokens_per_day_org,
        "max_members": r.max_members,
        "revision_retention_days": r.revision_retention_days,
        "supervisor_user_id": str(r.supervisor_user_id) if r.supervisor_user_id else None,
        "supervisor_email": r.supervisor_email,
        "supervisor_emails": list(r.supervisor_emails) if r.supervisor_emails else [],
        "user_count": r.user_count,
    }


@router.get("/organizations")
async def list_organizations(request: Request):
    _require_supervisor_access(request)
    async with get_db() as db:
        if _is_admin(request):
            result = await db.execute(_LIST_ORGS_ALL)
        else:
            result = await db.execute(_LIST_ORGS_ONE, {"org_id": _org_id(request)})
        rows = result.fetchall()
    return [_org_row_to_dict(r) for r in rows]


@router.post("/organizations", status_code=201)
async def create_organization(request: Request, body: OrgCreate):
    _require_admin(request)
    import re as _re
    _slug = _re.sub(r"[^a-z0-9]+", "-", body.name.lower()).strip("-")
    async with get_db() as db:
        _existing = (await db.execute(
            sa.text("SELECT id FROM organizations WHERE slug = :slug"),
            {"slug": _slug},
        )).fetchone()
    if _existing:
        raise HTTPException(409, f"An organization named '{body.name}' already exists")
    supervisor_id = None
    if body.supervisor_email:
        supervisor = await orgs_svc.get_user_by_email(body.supervisor_email)
        if not supervisor:
            raise HTTPException(404, f"No user found with email '{body.supervisor_email}'")
        supervisor_id = supervisor["id"]

    org_id = await orgs_svc.create_org(body.name)
    if not org_id:
        raise HTTPException(409, f"An organization named '{body.name}' already exists")

    await ws_svc.create_workspace(org_id, "default")

    if supervisor_id:
        async with get_db() as db:
            await db.execute(_ASSIGN_SUPERVISOR, {"org_id": org_id, "user_id": supervisor_id})
            await db.execute(_SET_ORG_SUPERVISOR, {"org_id": org_id, "user_id": supervisor_id})

    try:
        from app.services.permissions import seed_builtin_templates
        await seed_builtin_templates(org_id)
    except Exception as exc:
        log.warning("Organization created | template seeding failed | %s", exc)

    log.info("Organization created | id=%s | name=%s | supervisor=%s | by=%s",
             org_id, body.name, supervisor_id, getattr(request.state, "user_id", ""))
    return {"id": org_id, "name": body.name, "supervisor_user_id": supervisor_id}


@router.delete("/organizations/{org_id}")
async def delete_organization(request: Request, org_id: str):
    _require_admin(request)
    if org_id == "00000000-0000-0000-0000-000000000001":
        raise HTTPException(400, "Cannot delete the default organization")
    async with get_db() as db:
        result = await db.execute(_DELETE_ORG, {"id": org_id})
    if result.rowcount == 0:
        raise HTTPException(404, "Organization not found or is protected")
    log.info("Organization deleted | id=%s | by=%s", org_id, getattr(request.state, "user_id", ""))
    return {"deleted": org_id}


@router.post("/organizations/{org_id}/clone", status_code=201)
async def clone_organization(request: Request, org_id: str, body: OrgCloneRequest):
    """Admin-only: create a copy of an org's wiki under a new name.

    Copies wiki pages (incl. embeddings) + schema; regenerates index/graph.
    Does not copy members, custom templates, workspaces, raw files, or history.
    """
    _require_admin(request)
    new_name = (body.new_name or "").strip()
    if not new_name:
        raise HTTPException(400, "New organization name is required")
    new_id = await orgs_svc.clone_org(org_id, new_name)
    if not new_id:
        raise HTTPException(
            409,
            "Source organization not found, or an organization with that name already exists",
        )
    log.info("Organization cloned | src=%s | new=%s | by=%s",
             org_id, new_id, getattr(request.state, "user_id", ""))
    return {"id": new_id, "name": new_name, "cloned_from": org_id}


@router.post("/organizations/{org_id}/users", status_code=201)
async def create_user_in_org(request: Request, org_id: str, body: OrgUserCreate):
    """Provision a user in a specific org. If the email already exists, the
    user gains a membership in this org instead of erroring."""
    _require_admin(request)
    if len(body.password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters")
    if body.role not in ("supervisor", "member"):
        raise HTTPException(400, "role must be 'supervisor' or 'member'")

    existing = await orgs_svc.get_user_by_email(body.email)
    if existing:
        if existing["role"] == "admin":
            raise HTTPException(400, "That email belongs to a global admin and cannot be added to an org")
        user_id = existing["id"]
        added = await orgs_svc.add_membership(
            user_id, org_id, body.role,
            permission_template_id=body.permission_template_id,
        )
        if not added:
            raise HTTPException(409, f"User '{body.email}' is already a member of this organization")
    else:
        user_id = await orgs_svc.create_user(org_id, body.email, body.password, role=body.role)
        if not user_id:
            raise HTTPException(409, f"User '{body.email}' already exists")
        if body.permission_template_id:
            await orgs_svc.update_membership(user_id, org_id, {
                "role": body.role,
                "permission_template_id": body.permission_template_id,
            })

    if body.role == "supervisor":
        async with get_db() as db:
            await db.execute(_SET_ORG_SUPERVISOR, {"org_id": org_id, "user_id": user_id})
    log.info("User created/assigned in org | user=%s | org=%s | by=%s", user_id, org_id,
             getattr(request.state, "user_id", ""))
    return {"id": user_id, "email": body.email, "org_id": org_id}


@router.put("/organizations/{org_id}/supervisor")
async def change_supervisor(request: Request, org_id: str, body: ChangeSupervisorRequest):
    """Designate a supervisor for an org. Adds a supervisor membership (orgs may
    have multiple supervisors) and points the org's primary supervisor at them."""
    _require_admin(request)
    org = await orgs_svc.get_org(org_id)
    if not org:
        raise HTTPException(404, "Organization not found")
    supervisor = await orgs_svc.get_user_by_email(body.supervisor_email)
    if not supervisor:
        raise HTTPException(404, f"No user found with email '{body.supervisor_email}'")
    if supervisor["role"] == "admin":
        raise HTTPException(400, "A global admin cannot be made an org supervisor")
    async with get_db() as db:
        await db.execute(_ASSIGN_SUPERVISOR, {"org_id": org_id, "user_id": supervisor["id"]})
        await db.execute(_SET_ORG_SUPERVISOR, {"org_id": org_id, "user_id": supervisor["id"]})
    log.info("Supervisor changed | org=%s | supervisor=%s | by=%s",
             org_id, supervisor["id"], getattr(request.state, "user_id", ""))
    return {"org_id": org_id, "supervisor_user_id": supervisor["id"]}


_MOVE_MEMBERSHIP = sa.text("""
    UPDATE org_memberships SET org_id = CAST(:new_org_id AS UUID)
    WHERE user_id = CAST(:user_id AS UUID) AND org_id = CAST(:old_org_id AS UUID)
""")


@router.put("/users/{user_id}/transfer-org")
async def transfer_user_org(request: Request, user_id: str, body: TransferOrgRequest):
    """Move a user's membership from one org to another, and revoke their
    sessions. The source org is the admin's active org (X-Org-Context); if none
    is set and the user belongs to exactly one org, that one is moved."""
    _require_admin(request)
    if user_id == getattr(request.state, "user_id", ""):
        raise HTTPException(400, "Cannot transfer your own account")

    source_org = _org_id(request)
    if not source_org:
        memberships = await orgs_svc.list_memberships_for_user(user_id)
        if len(memberships) == 1:
            source_org = memberships[0]["org_id"]
        else:
            raise HTTPException(400, "Specify the source org via the X-Org-Context header")

    if await orgs_svc.get_membership(user_id, body.new_org_id):
        raise HTTPException(409, "User is already a member of the target organization")

    async with get_db() as db:
        result = await db.execute(_MOVE_MEMBERSHIP, {
            "user_id": user_id, "old_org_id": source_org, "new_org_id": body.new_org_id,
        })
    if result.rowcount == 0:
        raise HTTPException(404, "User is not a member of the source organization")
    # Revoke sessions so the user is forced to re-login with the new org context
    await auth_svc.revoke_all_tokens_for_user(user_id)
    log.info("Membership moved | user=%s | %s → %s | by=%s", user_id, source_org,
             body.new_org_id, getattr(request.state, "user_id", ""))
    return {"transferred": user_id, "new_org_id": body.new_org_id}


# ── Org-level limits ──────────────────────────────────────────────────────────

_GET_ORG_FULL = sa.text("""
    SELECT id, name, slug, created_at,
           max_uploads_per_day_org, max_tokens_per_day_org, max_members,
           revision_retention_days
    FROM organizations WHERE id = CAST(:id AS UUID)
""")

_UPDATE_ORG_LIMITS = sa.text("""
    UPDATE organizations SET
        max_uploads_per_day_org = :max_uploads_per_day_org,
        max_tokens_per_day_org  = :max_tokens_per_day_org,
        max_members             = :max_members,
        revision_retention_days = COALESCE(:revision_retention_days, revision_retention_days)
    WHERE id = CAST(:id AS UUID)
""")


@router.get("/org-limits")
async def get_org_limits(request: Request):
    _require_supervisor_access(request)
    if _is_admin(request):
        # Admin: return all orgs as a list
        async with get_db() as db:
            result = await db.execute(_LIST_ORGS_ALL)
            rows = result.fetchall()
        return [
            {
                "id": str(r.id),
                "name": r.name,
                "slug": r.slug,
                "created_at": r.created_at.isoformat() if r.created_at else None,
                "max_uploads_per_day_org": r.max_uploads_per_day_org,
                "max_tokens_per_day_org": r.max_tokens_per_day_org,
                "max_members": r.max_members,
            }
            for r in rows
        ]
    async with get_db() as db:
        result = await db.execute(_GET_ORG_FULL, {"id": _org_id(request)})
        row = result.fetchone()
    if not row:
        raise HTTPException(404, "Organization not found")
    return {
        "id": str(row.id),
        "name": row.name,
        "slug": row.slug,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "max_uploads_per_day_org": row.max_uploads_per_day_org,
        "max_tokens_per_day_org": row.max_tokens_per_day_org,
        "max_members": row.max_members,
        "revision_retention_days": row.revision_retention_days,
    }


@router.put("/org-limits")
async def update_org_limits(request: Request, body: OrgLimitsUpdate):
    _require_supervisor_access(request)
    target_org = body.org_id if (_is_admin(request) and body.org_id) else _org_id(request)
    if not target_org:
        raise HTTPException(400, "org_id required")
    if body.revision_retention_days is not None and body.revision_retention_days < 1:
        raise HTTPException(422, "revision_retention_days must be >= 1")
    async with get_db() as db:
        result = await db.execute(_UPDATE_ORG_LIMITS, {
            "id": target_org,
            "max_uploads_per_day_org": body.max_uploads_per_day_org,
            "max_tokens_per_day_org": body.max_tokens_per_day_org,
            "max_members": body.max_members,
            "revision_retention_days": body.revision_retention_days,
        })
    if result.rowcount == 0:
        raise HTTPException(404, "Organization not found")
    log.info("Org limits updated | org=%s | by=%s", target_org, getattr(request.state, "user_id", ""))
    return {"updated": True}


# ── Audit log ─────────────────────────────────────────────────────────────────

_AUDIT_LOG_TPL = """
    SELECT al.id, al.operation, al.raw_text, al.details, al.org_id, al.created_at,
           u.email AS user_email
    FROM audit_log al
    LEFT JOIN users u ON al.user_id = u.id
    WHERE al.org_id = CAST(:org_id AS UUID)
      AND (:operation = '' OR al.operation = :operation)
      AND (:search    = '' OR al.operation ILIKE :search_like OR al.raw_text ILIKE :search_like)
    ORDER BY al.created_at {sort}
    LIMIT :limit OFFSET :offset
"""

_AUDIT_LOG_COUNT = sa.text("""
    SELECT COUNT(*) FROM audit_log al
    WHERE al.org_id = CAST(:org_id AS UUID)
      AND (:operation = '' OR al.operation = :operation)
      AND (:search    = '' OR al.operation ILIKE :search_like OR al.raw_text ILIKE :search_like)
""")


_AUDIT_LOG_SORT_WHITELIST = {"desc": "DESC", "asc": "ASC"}


@router.get("/audit-log")
async def get_audit_log(
    request: Request,
    page: int = 1,
    limit: int = 50,
    search: str = "",
    operation: str = "",
    sort: str = "desc",
    org_filter: str = "",
):
    _require_supervisor_access(request)
    limit = min(limit, 200)
    offset = (max(page, 1) - 1) * limit
    search_like = f"%{search}%" if search else "%"
    sort_sql = _AUDIT_LOG_SORT_WHITELIST.get(sort.lower(), "DESC")

    async with get_db() as db:
        if _is_admin(request):
            params = {
                "org_filter": org_filter if org_filter else None,
                "operation": operation,
                "search": search,
                "search_like": search_like,
                "limit": limit,
                "offset": offset,
            }
            count_res = await db.execute(_AUDIT_LOG_COUNT_ALL, params)
            total = count_res.scalar() or 0
            rows_res = await db.execute(sa.text(_AUDIT_LOG_ALL_TPL.format(sort=sort_sql)), params)
        else:
            params = {
                "org_id": _org_id(request),
                "operation": operation,
                "search": search,
                "search_like": search_like,
                "limit": limit,
                "offset": offset,
            }
            count_res = await db.execute(_AUDIT_LOG_COUNT, params)
            total = count_res.scalar() or 0
            rows_res = await db.execute(sa.text(_AUDIT_LOG_TPL.format(sort=sort_sql)), params)
        rows = rows_res.fetchall()

    is_admin = _is_admin(request)
    return {
        "total": total,
        "page": page,
        "limit": limit,
        "entries": [
            {
                "id": r.id,
                "operation": r.operation,
                "raw_text": r.raw_text,
                "details": r.details,
                "user_email": r.user_email,
                "org_name": r.org_name if is_admin else None,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ],
    }


@router.get("/audit-log/operations")
async def list_audit_log_operations(request: Request, org_filter: str = ""):
    """Distinct operation values for populating the filter dropdown."""
    _require_supervisor_access(request)
    async with get_db() as db:
        if _is_admin(request):
            result = await db.execute(_AUDIT_LOG_OPERATIONS_ALL, {
                "org_filter": org_filter if org_filter else None,
            })
        else:
            result = await db.execute(_AUDIT_LOG_OPERATIONS_ONE, {
                "org_id": _org_id(request),
            })
        return [r.operation for r in result.fetchall()]


# ─────────────────────────────────────────────────────────────────────────────
# Change history + revert
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/history")
async def list_history(
    request: Request,
    page: int = 1,
    limit: int = 50,
    action_type: str | None = None,
    user_id: str | None = None,
):
    """List recent actions for the current org, newest first."""
    _require_supervisor_access(request)
    org_id = _org_id(request)
    if not org_id:
        raise HTTPException(400, "No organization context")
    offset = max(0, (page - 1) * limit)

    filters = ["a.org_id = CAST(:org_id AS UUID)"]
    params: dict = {"org_id": org_id, "limit": limit, "offset": offset}
    if action_type:
        filters.append("a.action_type = :action_type")
        params["action_type"] = action_type
    if user_id:
        filters.append("a.user_id = CAST(:user_id AS UUID)")
        params["user_id"] = user_id
    where_clause = " AND ".join(filters)

    async with get_db() as db:
        total = (await db.execute(
            sa.text(f"SELECT COUNT(*) FROM wiki_actions a WHERE {where_clause}"),
            params,
        )).scalar() or 0

        rows = (await db.execute(sa.text(f"""
            SELECT a.id::text AS id, a.action_type, a.summary, a.status,
                   a.started_at, a.finished_at, a.revert_of_id::text AS revert_of_id,
                   u.email AS user_email,
                   (SELECT COUNT(*) FROM wiki_revisions r WHERE r.action_id = a.id) AS revision_count
            FROM wiki_actions a
            LEFT JOIN users u ON u.id = a.user_id
            WHERE {where_clause}
            ORDER BY a.started_at DESC, a.id DESC
            LIMIT :limit OFFSET :offset
        """), params)).fetchall()

    return {
        "total": total,
        "page": page,
        "limit": limit,
        "entries": [
            {
                "id": r.id,
                "action_type": r.action_type,
                "summary": r.summary,
                "status": r.status,
                "started_at": r.started_at.isoformat() if r.started_at else None,
                "finished_at": r.finished_at.isoformat() if r.finished_at else None,
                "revert_of_id": r.revert_of_id,
                "user_email": r.user_email,
                "revision_count": r.revision_count,
            }
            for r in rows
        ],
    }


@router.get("/history/{action_id}")
async def get_history_action(request: Request, action_id: str):
    """Return one action plus its revision list (without content payloads)."""
    _require_supervisor_access(request)
    org_id = _org_id(request)
    async with get_db() as db:
        action = (await db.execute(sa.text("""
            SELECT a.id::text AS id, a.action_type, a.summary, a.status,
                   a.started_at, a.finished_at, a.details,
                   u.email AS user_email
            FROM wiki_actions a
            LEFT JOIN users u ON u.id = a.user_id
            WHERE a.id = CAST(:id AS UUID) AND a.org_id = CAST(:org_id AS UUID)
        """), {"id": action_id, "org_id": org_id})).fetchone()
        if action is None:
            raise HTTPException(404, "Action not found")

        revisions = (await db.execute(sa.text("""
            SELECT id, target_kind, target_key, op,
                   octet_length(COALESCE(content_before, '')) AS size_before,
                   octet_length(COALESCE(content_after, ''))  AS size_after
            FROM wiki_revisions
            WHERE action_id = CAST(:id AS UUID)
            ORDER BY id
        """), {"id": action_id})).fetchall()

    return {
        "id": action.id,
        "action_type": action.action_type,
        "summary": action.summary,
        "status": action.status,
        "user_email": action.user_email,
        "started_at": action.started_at.isoformat() if action.started_at else None,
        "finished_at": action.finished_at.isoformat() if action.finished_at else None,
        "details": action.details or {},
        "revisions": [
            {
                "id": r.id,
                "target_kind": r.target_kind,
                "target_key": r.target_key,
                "op": r.op,
                "size_before": r.size_before,
                "size_after": r.size_after,
            }
            for r in revisions
        ],
    }


@router.get("/history/{action_id}/diff")
async def get_history_diff(request: Request, action_id: str, target_kind: str, target_key: str):
    """Return content_before / content_after for one specific revision."""
    _require_supervisor_access(request)
    org_id = _org_id(request)
    async with get_db() as db:
        row = (await db.execute(sa.text("""
            SELECT r.target_kind, r.target_key, r.op, r.content_before, r.content_after
            FROM wiki_revisions r
            JOIN wiki_actions a ON a.id = r.action_id
            WHERE r.action_id = CAST(:id AS UUID)
              AND a.org_id = CAST(:org_id AS UUID)
              AND r.target_kind = :target_kind
              AND r.target_key = :target_key
            ORDER BY r.id DESC
            LIMIT 1
        """), {
            "id": action_id, "org_id": org_id,
            "target_kind": target_kind, "target_key": target_key,
        })).fetchone()
    if row is None:
        raise HTTPException(404, "Revision not found")
    return {
        "target_kind": row.target_kind,
        "target_key": row.target_key,
        "op": row.op,
        "content_before": row.content_before,
        "content_after": row.content_after,
    }


@router.get("/history/{action_id}/preview-revert")
async def preview_revert(request: Request, action_id: str):
    """Show which actions would be discarded if this revert ran now."""
    _require_supervisor_access(request)
    org_id = _org_id(request)
    async with get_db() as db:
        target = (await db.execute(sa.text("""
            SELECT id::text AS id, started_at, action_type, summary, status
            FROM wiki_actions
            WHERE id = CAST(:id AS UUID) AND org_id = CAST(:org_id AS UUID)
        """), {"id": action_id, "org_id": org_id})).fetchone()
        if target is None:
            raise HTTPException(404, "Action not found")

        rows = (await db.execute(sa.text("""
            SELECT a.id::text AS id, a.action_type, a.summary, a.started_at
            FROM wiki_actions a
            WHERE a.org_id = CAST(:org_id AS UUID)
              AND a.started_at >= :since
              AND a.status IN ('done', 'error')
            ORDER BY a.started_at DESC, a.id DESC
        """), {"org_id": org_id, "since": target.started_at})).fetchall()

    return {
        "target": {
            "id": target.id,
            "action_type": target.action_type,
            "summary": target.summary,
            "status": target.status,
        },
        "actions_to_discard": [
            {
                "id": r.id,
                "action_type": r.action_type,
                "summary": r.summary,
                "started_at": r.started_at.isoformat() if r.started_at else None,
            }
            for r in rows
        ],
    }


@router.post("/history/{action_id}/revert", status_code=202)
async def revert_history_action(request: Request, action_id: str):
    """Kick off a background revert job. Returns 202 with the new
    revert_action_id; clients poll `/history/revert-job/{id}` for progress.

    Reverting many actions (e.g. 100 ingests) can take well over the typical
    proxy timeout, so the work is fire-and-forget. The HTTP response is fast
    even if the actual revert takes minutes.
    """
    _require_supervisor_access(request)
    if not _org_id(request):
        raise HTTPException(400, "No organization context")

    from app.services.wiki_state import (
        RevertError, RevertJobAlreadyRunning, start_revert_job,
    )
    try:
        job = await start_revert_job(action_id)
    except RevertJobAlreadyRunning as exc:
        raise HTTPException(409, str(exc))
    except RevertError as exc:
        raise HTTPException(409, str(exc))
    return job


# ─────────────────────────────────────────────────────────────────────────────
# Jobs — unified view of all active background work (ingests, recalibrations,
# reverts). Admin picks an org via X-Org-Context; supervisors are scoped to
# their own org — same model as the History tab.
# ─────────────────────────────────────────────────────────────────────────────

_ACTIVE_INGEST_JOBS = sa.text("""
    SELECT ij.filename, ij.status, ij.cancel_requested,
           ij.created_at, ij.updated_at,
           u.email AS user_email
    FROM ingest_jobs ij
    LEFT JOIN users u ON u.id = ij.user_id
    WHERE ij.org_id = CAST(:org_id AS UUID)
      AND ij.status IN ('queued', 'processing', 'pending_review', 'queued_write', 'writing')
    ORDER BY ij.created_at
""")

_RUNNING_RECALIBRATE_JOBS = sa.text("""
    SELECT id, stage, progress, started_at
    FROM recalibrate_jobs
    WHERE org_id = CAST(:org_id AS UUID) AND status = 'running'
    ORDER BY started_at DESC NULLS LAST
""")

_RUNNING_REVERT_JOBS = sa.text("""
    SELECT a.id::text AS id, a.summary, a.started_at,
           a.progress_done, a.progress_total,
           u.email AS user_email
    FROM wiki_actions a
    LEFT JOIN users u ON u.id = a.user_id
    WHERE a.org_id = CAST(:org_id AS UUID)
      AND a.action_type = 'revert' AND a.status = 'running'
    ORDER BY a.started_at DESC
""")


@router.get("/jobs")
async def list_jobs(request: Request):
    """Unified active-job list across ingests, recalibrations, and reverts."""
    _require_supervisor_access(request)
    org_id = _org_id(request)
    if not org_id:
        raise HTTPException(400, "Select an organization to view its jobs")

    from app.services import ingest_queue

    async with get_db() as db:
        ingest_rows  = (await db.execute(_ACTIVE_INGEST_JOBS,      {"org_id": org_id})).fetchall()
        recal_rows   = (await db.execute(_RUNNING_RECALIBRATE_JOBS, {"org_id": org_id})).fetchall()
        revert_rows  = (await db.execute(_RUNNING_REVERT_JOBS,      {"org_id": org_id})).fetchall()

    jobs: list[dict] = []

    for r in ingest_rows:
        pos = await ingest_queue.queue_position(r.filename, org_id) if r.status == "queued_write" else None
        when = r.updated_at or r.created_at
        jobs.append({
            "type": "ingest",
            "id": r.filename,
            "target": r.filename,
            "user_email": r.user_email,
            "status": r.status,
            "cancel_requested": bool(r.cancel_requested),
            "queue_position": pos,
            "started_at": when.isoformat() if when else None,
            "cancellable": True,
        })

    for r in recal_rows:
        jobs.append({
            "type": "recalibrate",
            "id": "recalibrate",
            "target": r.stage or "Recalibration",
            "user_email": None,
            "status": "running",
            "cancel_requested": False,
            "queue_position": None,
            "started_at": r.started_at.isoformat() if r.started_at else None,
            "cancellable": True,
        })

    for r in revert_rows:
        progress = None
        if r.progress_total:
            progress = f"{r.progress_done}/{r.progress_total}"
        jobs.append({
            "type": "revert",
            "id": r.id,
            "target": r.summary or "Revert",
            "user_email": r.user_email,
            "status": "running",
            "progress": progress,
            "cancel_requested": False,
            "queue_position": None,
            "started_at": r.started_at.isoformat() if r.started_at else None,
            # Cancelling a half-applied restore is unsafe — surfaced read-only.
            "cancellable": False,
            "cancel_reason": "A revert in progress cannot be cancelled — interrupting a half-applied restore could leave the wiki inconsistent.",
        })

    return {"jobs": jobs}


@router.post("/jobs/ingest/{filename:path}/cancel")
async def cancel_job_ingest(request: Request, filename: str):
    """Request cancellation of an ingest job (any non-terminal state)."""
    _require_supervisor_access(request)
    if not _org_id(request):
        raise HTTPException(400, "No organization context")
    from app.services import jobs as job_store
    ok = await job_store.request_cancel(filename)
    if not ok:
        raise HTTPException(404, "No ingest job found for this file")
    log.info("Jobs | ingest cancel requested | file=%s | by=%s", filename,
             getattr(request.state, "user_id", ""))
    return {"cancelled": filename}


_CANCEL_RUNNING_RECALIBRATE = sa.text("""
    UPDATE recalibrate_jobs
       SET status = 'error',
           errors = errors || '["Cancelled by admin"]'::jsonb,
           finished_at = NOW()
     WHERE org_id = CAST(:org_id AS UUID) AND status = 'running'
""")


@router.post("/jobs/recalibrate/cancel")
async def cancel_job_recalibrate(request: Request):
    """Cancel the org's running recalibration and release its write lock."""
    _require_supervisor_access(request)
    org_id = _org_id(request)
    if not org_id:
        raise HTTPException(400, "No organization context")

    from app.services import recalibrate_job
    from app.routes.operations import _release_recalibrate_lock

    job = recalibrate_job.get(org_id)
    # Fast path: cancel the local task if the run is on this replica.
    if job.task and not job.task.done():
        job.task.cancel()
    # Release the in-memory replica's DB lock (best-effort; no-op if not local).
    await _release_recalibrate_lock(org_id, job)
    job.status = "error"
    job.details = "Cancelled by admin"
    # Cross-replica safety net: clear any running row for this org so the
    # write-lock middleware stops blocking even if the run is on another replica.
    async with get_db() as db:
        await db.execute(_CANCEL_RUNNING_RECALIBRATE, {"org_id": org_id})
    log.info("Jobs | recalibrate cancel requested | org=%s | by=%s", org_id,
             getattr(request.state, "user_id", ""))
    return {"cancelled": "recalibrate"}


@router.get("/history/revert-job/{revert_action_id}")
async def get_revert_job(request: Request, revert_action_id: str):
    """Poll the status of an in-flight or finished revert job.

    Response shape:
      {revert_action_id, status: 'running'|'done'|'error',
       progress_done, progress_total, error_message?,
       summary, target_action_id, started_at, finished_at?}
    """
    _require_supervisor_access(request)
    if not _org_id(request):
        raise HTTPException(400, "No organization context")
    from app.services.wiki_state import get_revert_job_status
    job = await get_revert_job_status(revert_action_id)
    if job is None:
        raise HTTPException(404, "Revert job not found")
    return job
