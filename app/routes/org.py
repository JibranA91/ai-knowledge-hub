"""Organization management routes.

GET  /api/orgs/me            — current org info
GET  /api/orgs/me/usage      — Bedrock usage summary for current org
GET  /api/orgs/me/users      — list users in current org (admin only)
POST /api/orgs/me/users      — create a user in current org (admin only)
DELETE /api/orgs/me/users/{user_id} — remove user (admin only)

GET  /api/orgs               — list all orgs (admin only)
POST /api/orgs               — create a new org (admin only)
"""
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app.logger import get_logger
from app.services import orgs as orgs_service
from app.services import usage_log

log = get_logger(__name__)
router = APIRouter(tags=["orgs"])


def _require_admin(request: Request) -> None:
    if getattr(request.state, "role", "") != "admin":
        raise HTTPException(403, "Admin role required")


def _org_id(request: Request) -> str:
    return getattr(request.state, "org_id", "")


# ── Current-org endpoints (any authenticated user) ─────────────────────────

@router.get("/me")
async def get_my_org(request: Request):
    org = await orgs_service.get_org(_org_id(request))
    if not org:
        raise HTTPException(404, "Organization not found")
    return org


@router.get("/me/usage")
async def get_my_usage(request: Request, days: int = 30):
    summary = await usage_log.get_org_summary(_org_id(request), days=days)
    return {"org_id": _org_id(request), "days": days, "usage": summary}


# ── User management (admin only) ───────────────────────────────────────────

@router.get("/me/users")
async def list_org_users(request: Request):
    _require_admin(request)
    return await orgs_service.list_members(_org_id(request))


class CreateUserRequest(BaseModel):
    email: str
    password: str
    role: str = "member"


@router.post("/me/users", status_code=201)
async def create_org_user(request: Request, req: CreateUserRequest):
    """Add a user to the current org. An existing identity gains a membership;
    a new email gets a user row + membership."""
    _require_admin(request)
    if req.role not in ("supervisor", "member"):
        raise HTTPException(400, "role must be 'supervisor' or 'member'")
    org_id = _org_id(request)
    existing = await orgs_service.get_user_by_email(req.email)
    if existing:
        if existing["role"] == "admin":
            raise HTTPException(400, "That email belongs to a global admin")
        added = await orgs_service.add_membership(existing["id"], org_id, req.role)
        if not added:
            raise HTTPException(409, f"User already a member of this org: {req.email}")
        user_id = existing["id"]
    else:
        user_id = await orgs_service.create_user(org_id, req.email, req.password, req.role)
        if not user_id:
            raise HTTPException(409, f"Email already registered: {req.email}")
    log.info("User added via API | email=%s | org=%s", req.email, org_id)
    return {"id": user_id, "email": req.email, "role": req.role}


@router.delete("/me/users/{user_id}")
async def delete_org_user(request: Request, user_id: str):
    """Remove a user's membership in the current org (the identity is preserved
    if they still belong to other orgs)."""
    _require_admin(request)
    # Prevent self-deletion
    if user_id == getattr(request.state, "user_id", ""):
        raise HTTPException(400, "Cannot delete your own account")
    removed = await orgs_service.remove_membership(user_id, _org_id(request))
    if not removed:
        raise HTTPException(404, "User not found in this organization")
    return {"deleted": user_id}


# ── System-level org management (admin only) ───────────────────────────────

@router.get("")
async def list_all_orgs(request: Request):
    _require_admin(request)
    return await orgs_service.list_orgs()


class CreateOrgRequest(BaseModel):
    name: str
    slug: str | None = None
    admin_email: str
    admin_password: str


@router.post("", status_code=201)
async def create_org(request: Request, req: CreateOrgRequest):
    _require_admin(request)
    org_id = await orgs_service.create_org(req.name, req.slug)
    user_id = await orgs_service.create_user(
        org_id, req.admin_email, req.admin_password, role="admin"
    )
    if not user_id:
        raise HTTPException(409, f"Email already registered: {req.admin_email}")
    log.info("Org created via API | org_id=%s | admin=%s", org_id, req.admin_email)
    return {"org_id": org_id, "admin_user_id": user_id}
