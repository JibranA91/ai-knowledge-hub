from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app import model
from app.config import settings
from app.logger import get_logger
from app.services import auth as auth_service
from app.services.orgs import verify_user, get_user, list_memberships_for_user

log = get_logger(__name__)
router = APIRouter(tags=["auth"])


class LoginRequest(BaseModel):
    username: str   # accepts email or legacy username
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str


@router.get("/config")
async def get_config():
    return {
        "company_name": settings.COMPANY_NAME,
        "chat_stream": settings.CHAT_STREAM,
        "max_upload_size_mb": settings.MAX_UPLOAD_SIZE_MB,
        "embedding_enabled": model.embedding_enabled(),
    }


@router.post("/login")
async def login(req: LoginRequest):
    """Authenticate via the users table. Returns JWT access + opaque refresh tokens.

    A user may belong to several orgs, so the response carries their full
    `memberships` list and an `active_org_id` (auto-selected when there is
    exactly one membership, else null — the client then prompts the user to
    pick one and sends it back via the X-Org-Context header). Admins are global
    and return an empty memberships list.
    """
    user = await verify_user(req.username, req.password)
    if not user:
        log.warning("Failed login attempt | email=%s", req.username)
        return JSONResponse(status_code=401, content={"detail": "Invalid credentials"})

    is_admin = user["role"] == "admin"
    memberships = [] if is_admin else await list_memberships_for_user(user["id"])

    # Choose an initial active org: admins have none; single-membership users
    # auto-select; multi-membership users must choose client-side.
    if is_admin:
        active_org_id = None
    elif len(memberships) == 1:
        active_org_id = memberships[0]["org_id"]
    else:
        active_org_id = None

    access_token = auth_service.create_access_token(
        email=user["email"],
        user_id=user["id"],
        org_id=active_org_id or "",
        role=user["role"],
    )
    refresh_token = await auth_service.create_refresh_token(
        email=user["email"],
        user_id=user["id"],
        org_id=active_org_id or "",
    )
    log.info("Login successful | email=%s | role=%s | memberships=%d",
             user["email"], user["role"], len(memberships))
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "user_id": user["id"],
        "email": user["email"],
        "role": user["role"],
        "memberships": memberships,
        "active_org_id": active_org_id,
        # Back-compat fields (single-org clients): the active org, if any.
        "org_id": active_org_id,
        "org_name": memberships[0]["org_name"] if len(memberships) == 1 else None,
    }


@router.post("/refresh")
async def refresh(req: RefreshRequest):
    """Exchange a valid refresh token for a new short-lived access token."""
    token_data = await auth_service.validate_refresh_token(req.refresh_token)
    if not token_data:
        return JSONResponse(status_code=401, content={"detail": "Invalid or expired refresh token"})
    # Look up current role from DB — avoids hardcoding "member" for admins/supervisors
    user = await get_user(token_data["user_id"]) if token_data.get("user_id") else None
    role = user["role"] if user else "member"
    # Prefer the refresh token's org (the session's active org at login) over the
    # user's home org, so a multi-org member's refreshed access token doesn't
    # silently revert to a different org. (Normal requests are scoped by the
    # X-Org-Context header regardless; this token org_id only matters on the
    # auth-middleware DB-failure fallback path.)
    org_id = token_data.get("org_id") or (user["org_id"] if user else "")
    access_token = auth_service.create_access_token(
        email=token_data["email"],
        user_id=token_data["user_id"],
        org_id=org_id,
        role=role,
    )
    return {"access_token": access_token, "token_type": "bearer", "role": role}


@router.post("/logout")
async def logout(req: dict):
    refresh_token = req.get("refresh_token", "")
    await auth_service.revoke_token(refresh_token)
    return {"ok": True}
