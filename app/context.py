"""Request-scoped context variables.

Set by auth_middleware on every authenticated request.
All service-layer functions read org_id from here instead of
being passed it explicitly — keeps function signatures clean and
works correctly with asyncio.create_task (tasks inherit the context
of their creator).
"""
from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True)
class UserContext:
    user_id: str
    org_id: str   # the ACTIVE org for this request; "" when none is selected
    email: str
    role: str   # role IN THE ACTIVE ORG: "admin" | "supervisor" | "member".
                # Resolved per-request from org_memberships (see auth_middleware)
                # since a user may have a different role in each org.


# Default: None (unset outside request context — e.g. during startup seeding)
current_user: ContextVar[UserContext | None] = ContextVar("current_user", default=None)

# Set by `wiki_state.begin_action()` while a tracked action is open. Every wiki
# write that runs inside the block emits a `wiki_revisions` row attributed to
# this action id. Propagates through `asyncio.create_task` (standard ContextVar
# behaviour), which is how background ingest + recalibrate jobs inherit it.
current_action: ContextVar[str | None] = ContextVar("current_action", default=None)


def get_org_id() -> str:
    """Return the current request's org_id. Raises if not in a request context."""
    ctx = current_user.get()
    if ctx is None:
        raise RuntimeError("No user context — call outside a request or startup context")
    if not ctx.org_id:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=400,
            detail="No organization context. Select an organization and send it "
                   "via the X-Org-Context header (users in multiple orgs and "
                   "admins must choose which org to act on).",
        )
    return ctx.org_id


def get_user_id() -> str:
    ctx = current_user.get()
    if ctx is None:
        raise RuntimeError("No user context")
    return ctx.user_id


def set_startup_context(org_id: str) -> None:
    """Set a minimal context for startup seeding operations (no real user)."""
    current_user.set(UserContext(
        user_id="00000000-0000-0000-0000-000000000000",
        org_id=org_id,
        email="system",
        role="admin",
    ))
