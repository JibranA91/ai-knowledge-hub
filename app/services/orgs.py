"""Organization and user management.

organizations   — top-level tenant containers
users           — global identities; one row per email. Admins (role='admin')
                  are global and org-less.
org_memberships — a user's membership in an org, carrying the per-org access
                  profile (role, template, workspace, suspension, limits). A user
                  may belong to many orgs with a different role in each.

Passwords are hashed with bcrypt via passlib.
"""
import hashlib
import uuid

import sqlalchemy as sa

from app.db import get_db
from app.logger import get_logger

log = get_logger(__name__)

# Default org seeded by migration 004 — deterministic UUID.
DEFAULT_ORG_ID = "00000000-0000-0000-0000-000000000001"

# Starting token-budget for the Default Organization on a fresh install.
# Admins can change this via /api/admin/org-limits at any time; this value is
# only applied at first creation (and by migration 019 for existing deployments
# whose cap was still NULL).
DEFAULT_ORG_MAX_TOKENS_PER_DAY = 10_000_000

# ── Queries ────────────────────────────────────────────────────────────────

_ORG_BY_ID   = sa.text("SELECT id, name, slug, max_uploads_per_day_org, max_tokens_per_day_org, max_members FROM organizations WHERE id = CAST(:id AS UUID)")
_ORG_BY_SLUG = sa.text("SELECT id, name, slug FROM organizations WHERE slug = :slug")
_ORG_ALL     = sa.text("SELECT id, name, slug, created_at FROM organizations ORDER BY name")

_USER_BY_EMAIL = sa.text("""
    SELECT u.id, u.org_id, u.email, u.password_hash, u.role,
           o.name AS org_name
    FROM users u
    LEFT JOIN organizations o ON u.org_id = o.id
    WHERE u.email = :email
""")
_USER_BY_ID = sa.text("""
    SELECT id, org_id, email, role FROM users WHERE id = CAST(:id AS UUID)
""")
_USERS_IN_ORG = sa.text("""
    SELECT id, email, role, created_at, last_login_at
    FROM users WHERE org_id = CAST(:org_id AS UUID) ORDER BY email
""")

_INSERT_ORG = sa.text("""
    INSERT INTO organizations (id, name, slug)
    VALUES (CAST(:id AS UUID), :name, :slug)
    ON CONFLICT (slug) DO NOTHING
    RETURNING id
""")

_INSERT_USER = sa.text("""
    INSERT INTO users (id, org_id, email, password_hash, role)
    VALUES (CAST(:id AS UUID), CAST(:org_id AS UUID), :email, :password_hash, :role)
    ON CONFLICT (email) DO NOTHING
    RETURNING id
""")

_DELETE_USER = sa.text("""
    DELETE FROM users WHERE id = CAST(:id AS UUID) AND org_id = CAST(:org_id AS UUID)
""")

_TOUCH_LOGIN = sa.text("""
    UPDATE users SET last_login_at = NOW() WHERE id = CAST(:id AS UUID)
""")

_DEFAULT_ORG_EXISTS = sa.text(
    "SELECT 1 FROM organizations WHERE id = CAST(:id AS UUID)"
)

# ── Membership queries ───────────────────────────────────────────────────────

_INSERT_MEMBERSHIP = sa.text("""
    INSERT INTO org_memberships
        (user_id, org_id, role, permission_template_id, workspace_id,
         is_suspended, max_tokens_per_day, max_chat_messages_per_day)
    VALUES
        (CAST(:user_id AS UUID), CAST(:org_id AS UUID), :role,
         CAST(:permission_template_id AS UUID), CAST(:workspace_id AS UUID),
         :is_suspended, :max_tokens_per_day, :max_chat_messages_per_day)
    ON CONFLICT (user_id, org_id) DO NOTHING
    RETURNING id
""")

_GET_MEMBERSHIP = sa.text("""
    SELECT id, user_id, org_id, role, permission_template_id, workspace_id,
           is_suspended, max_tokens_per_day, max_chat_messages_per_day
    FROM org_memberships
    WHERE user_id = CAST(:user_id AS UUID) AND org_id = CAST(:org_id AS UUID)
""")

_LIST_MEMBERSHIPS_FOR_USER = sa.text("""
    SELECT m.org_id, m.role, m.permission_template_id, m.workspace_id,
           m.is_suspended, o.name AS org_name
    FROM org_memberships m
    LEFT JOIN organizations o ON m.org_id = o.id
    WHERE m.user_id = CAST(:user_id AS UUID)
    ORDER BY o.name
""")

_LIST_MEMBERS_IN_ORG = sa.text("""
    SELECT u.id, u.email, m.role, m.created_at, u.last_login_at
    FROM org_memberships m
    JOIN users u ON u.id = m.user_id
    WHERE m.org_id = CAST(:org_id AS UUID)
    ORDER BY u.email
""")

_UPDATE_MEMBERSHIP = sa.text("""
    UPDATE org_memberships SET
        role                      = :role,
        permission_template_id    = CAST(:permission_template_id AS UUID),
        workspace_id              = CAST(:workspace_id AS UUID),
        is_suspended              = :is_suspended,
        max_tokens_per_day        = :max_tokens_per_day,
        max_chat_messages_per_day = :max_chat_messages_per_day
    WHERE user_id = CAST(:user_id AS UUID) AND org_id = CAST(:org_id AS UUID)
""")

_DELETE_MEMBERSHIP = sa.text("""
    DELETE FROM org_memberships
    WHERE user_id = CAST(:user_id AS UUID) AND org_id = CAST(:org_id AS UUID)
""")


# ── Password helpers ───────────────────────────────────────────────────────

def _hash_password(password: str) -> str:
    import bcrypt
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def _verify_password(plain: str, hashed: str) -> bool:
    # Legacy SHA-256 hashes created before bcrypt migration
    if hashed.startswith("sha256:"):
        import hmac
        expected = "sha256:" + hashlib.sha256(plain.encode()).hexdigest()
        return hmac.compare_digest(hashed, expected)
    import bcrypt
    try:
        return bcrypt.checkpw(plain.encode(), hashed.encode())
    except Exception:
        return False


# ── Public API ─────────────────────────────────────────────────────────────

async def get_org(org_id: str) -> dict | None:
    async with get_db() as db:
        result = await db.execute(_ORG_BY_ID, {"id": org_id})
        row = result.fetchone()
    if not row:
        return None
    return {
        "id": str(row.id),
        "name": row.name,
        "slug": row.slug,
        "max_uploads_per_day_org": getattr(row, "max_uploads_per_day_org", None),
        "max_tokens_per_day_org": getattr(row, "max_tokens_per_day_org", None),
        "max_members": getattr(row, "max_members", None),
    }


async def list_orgs() -> list[dict]:
    async with get_db() as db:
        result = await db.execute(_ORG_ALL)
        rows = result.fetchall()
    return [{"id": str(r.id), "name": r.name, "slug": r.slug,
             "created_at": r.created_at.isoformat() if r.created_at else None}
            for r in rows]


async def create_org(name: str, slug: str | None = None) -> str:
    """Insert a new organization. Returns the new org's UUID string."""
    if slug is None:
        import re
        slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    new_id = str(uuid.uuid4())
    async with get_db() as db:
        result = await db.execute(_INSERT_ORG, {"id": new_id, "name": name, "slug": slug})
        row = result.fetchone()
    if row:
        log.info("Org created | id=%s | name=%s", new_id, name)
        org_id = str(row.id)
    else:
        # Slug conflict — fetch existing
        async with get_db() as db:
            result = await db.execute(_ORG_BY_SLUG, {"slug": slug})
            row = result.fetchone()
        org_id = str(row.id) if row else new_id

    # Seed built-in permission templates for the new org (best-effort)
    try:
        from app.services.permissions import seed_builtin_templates
        await seed_builtin_templates(org_id)
    except Exception as exc:
        log.warning("Could not seed built-in templates for org %s: %s", org_id, exc)

    # Seed default AGENTS.md schema for the new org (best-effort)
    try:
        from pathlib import Path as _Path
        from app.context import current_user, UserContext
        from app.services.wiki_db import get_wiki_file, set_wiki_file
        _BUNDLED = _Path(__file__).parent.parent.parent / "data" / "schema" / "AGENTS.md"
        _token = current_user.set(UserContext(
            user_id="00000000-0000-0000-0000-000000000000",
            org_id=org_id,
            email="system",
            role="admin",
        ))
        try:
            if not await get_wiki_file("schema/AGENTS.md") and _BUNDLED.exists():
                bundled = _BUNDLED.read_text(encoding="utf-8")
                await set_wiki_file("schema/AGENTS.md", bundled)
                log.info("Org schema seeded from bundled default | org_id=%s", org_id)
        finally:
            current_user.reset(_token)
    except Exception as exc:
        log.warning("Could not seed AGENTS.md for org %s: %s", org_id, exc)

    return org_id


async def _rebuild_graph_for_org(org_id: str) -> None:
    """Regenerate the link graph + wiki_links + graph cache for one org.

    The graph singleton is context-scoped, so we temporarily set a system
    context for the target org (mirrors the AGENTS.md seeding above) and restore
    it afterward.
    """
    from app.context import current_user, UserContext
    from app.services.graph import get_graph
    token = current_user.set(UserContext(
        user_id="00000000-0000-0000-0000-000000000000",
        org_id=org_id, email="system", role="admin",
    ))
    try:
        await get_graph().rebuild()
    finally:
        current_user.reset(token)


async def clone_org(source_org_id: str, new_name: str) -> str | None:
    """Create a copy of an org's wiki under a new name.

    Copies wiki pages (incl. embeddings) and the schema (AGENTS.md); the index
    and link graph are regenerated from the copied pages. Does NOT copy members,
    custom permission templates, workspaces, raw source files, or history — the
    new org gets the standard built-in templates + a default workspace, like any
    freshly created org.

    Returns the new org id, or None if the source is missing or the name/slug is
    already taken (create_org silently returns an existing org on slug collision,
    which would copy INTO it — so we guard against that explicitly here).
    """
    import re

    src = await get_org(source_org_id)
    if not src:
        return None

    slug = re.sub(r"[^a-z0-9]+", "-", new_name.lower()).strip("-")
    if not slug:
        return None
    async with get_db() as db:
        existing = (await db.execute(
            sa.text("SELECT id FROM organizations WHERE slug = :slug"), {"slug": slug}
        )).fetchone()
    if existing:
        return None

    new_org_id = await create_org(new_name, slug=slug)

    # Default workspace, for parity with normal org creation.
    try:
        from app.services import workspaces as ws_svc
        await ws_svc.create_workspace(new_org_id, "default")
    except Exception as exc:
        log.warning("clone_org | default workspace creation failed | %s", exc)

    from app.services.wiki_db import _embedding_col_exists
    page_cols = ["path", "title", "tags", "summary", "content", "frontmatter",
                 "ingested_from", "s3_key", "created_at", "updated_at", "embedding_space"]
    if await _embedding_col_exists():
        page_cols.append("embedding")
    col_list = ", ".join(page_cols)

    async with get_db() as db:
        # Copy the schema (overwrites the default AGENTS.md create_org just seeded).
        await db.execute(sa.text("""
            INSERT INTO wiki_files (org_id, key, content, updated_at)
            SELECT CAST(:new AS UUID), key, content, NOW()
            FROM wiki_files
            WHERE org_id = CAST(:src AS UUID) AND key LIKE 'schema/%'
            ON CONFLICT (org_id, key) DO UPDATE
                SET content = EXCLUDED.content, updated_at = NOW()
        """), {"new": new_org_id, "src": source_org_id})

        # Copy all wiki pages (one set-based INSERT … SELECT).
        await db.execute(sa.text(f"""
            INSERT INTO wiki_pages (org_id, {col_list})
            SELECT CAST(:new AS UUID), {col_list}
            FROM wiki_pages
            WHERE org_id = CAST(:src AS UUID)
        """), {"new": new_org_id, "src": source_org_id})

    # Regenerate index / wiki_links / graph cache for the clone.
    await _rebuild_graph_for_org(new_org_id)

    log.info("Org cloned | src=%s -> new=%s | name=%s", source_org_id, new_org_id, new_name)
    return new_org_id


async def get_user_by_email(email: str) -> dict | None:
    async with get_db() as db:
        result = await db.execute(_USER_BY_EMAIL, {"email": email})
        row = result.fetchone()
    if not row:
        return None
    return {"id": str(row.id), "org_id": str(row.org_id) if row.org_id else None,
            "email": row.email, "role": row.role}


async def get_user(user_id: str) -> dict | None:
    async with get_db() as db:
        result = await db.execute(_USER_BY_ID, {"id": user_id})
        row = result.fetchone()
    if not row:
        return None
    return {"id": str(row.id), "org_id": str(row.org_id) if row.org_id else None,
            "email": row.email, "role": row.role}


async def list_users(org_id: str) -> list[dict]:
    async with get_db() as db:
        result = await db.execute(_USERS_IN_ORG, {"org_id": org_id})
        rows = result.fetchall()
    return [
        {
            "id": str(r.id),
            "email": r.email,
            "role": r.role,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "last_login_at": r.last_login_at.isoformat() if r.last_login_at else None,
        }
        for r in rows
    ]


async def create_user(org_id: str | None, email: str, password: str,
                      role: str = "member") -> str | None:
    """Hash password and insert a global user identity. Returns new user_id or
    None on email conflict.

    For non-admin roles with an org, also creates the org membership. The legacy
    `users.org_id`/`users.role` columns are still populated for rollback safety,
    but `org_memberships` is the source of truth.
    """
    new_id = str(uuid.uuid4())
    hashed = _hash_password(password)
    async with get_db() as db:
        result = await db.execute(_INSERT_USER, {
            "id": new_id, "org_id": org_id, "email": email,
            "password_hash": hashed, "role": role,
        })
        row = result.fetchone()
    if not row:
        log.warning("User email conflict | email=%s", email)
        return None
    user_id = str(row.id)
    log.info("User created | id=%s | org=%s | email=%s | role=%s", user_id, org_id, email, role)
    if role != "admin" and org_id:
        await add_membership(user_id, org_id, "supervisor" if role == "supervisor" else "member")
    return user_id


# ── Membership management ────────────────────────────────────────────────────

async def add_membership(user_id: str, org_id: str, role: str = "member",
                         permission_template_id: str | None = None,
                         workspace_id: str | None = None,
                         is_suspended: bool = False,
                         max_tokens_per_day: int | None = None,
                         max_chat_messages_per_day: int | None = None) -> bool:
    """Assign a user to an org. Idempotent — returns False if already a member.

    This is the path for giving an existing identity access to another org.
    """
    role = "supervisor" if role == "supervisor" else "member"
    async with get_db() as db:
        result = await db.execute(_INSERT_MEMBERSHIP, {
            "user_id": user_id, "org_id": org_id, "role": role,
            "permission_template_id": permission_template_id,
            "workspace_id": workspace_id, "is_suspended": is_suspended,
            "max_tokens_per_day": max_tokens_per_day,
            "max_chat_messages_per_day": max_chat_messages_per_day,
        })
        row = result.fetchone()
    created = row is not None
    if created:
        log.info("Membership added | user=%s | org=%s | role=%s", user_id, org_id, role)
    return created


async def get_membership(user_id: str, org_id: str) -> dict | None:
    async with get_db() as db:
        result = await db.execute(_GET_MEMBERSHIP, {"user_id": user_id, "org_id": org_id})
        row = result.fetchone()
    if not row:
        return None
    return {
        "user_id": str(row.user_id),
        "org_id": str(row.org_id),
        "role": row.role,
        "permission_template_id": str(row.permission_template_id) if row.permission_template_id else None,
        "workspace_id": str(row.workspace_id) if row.workspace_id else None,
        "is_suspended": row.is_suspended,
        "max_tokens_per_day": row.max_tokens_per_day,
        "max_chat_messages_per_day": row.max_chat_messages_per_day,
    }


async def list_memberships_for_user(user_id: str) -> list[dict]:
    async with get_db() as db:
        result = await db.execute(_LIST_MEMBERSHIPS_FOR_USER, {"user_id": user_id})
        rows = result.fetchall()
    return [
        {
            "org_id": str(r.org_id),
            "org_name": r.org_name,
            "role": r.role,
            "permission_template_id": str(r.permission_template_id) if r.permission_template_id else None,
            "workspace_id": str(r.workspace_id) if r.workspace_id else None,
            "is_suspended": r.is_suspended,
        }
        for r in rows
    ]


async def update_membership(user_id: str, org_id: str, data: dict) -> bool:
    """Update a membership's per-org access profile. Returns False if not found."""
    role = data.get("role") or "member"
    role = "supervisor" if role == "supervisor" else "member"
    async with get_db() as db:
        result = await db.execute(_UPDATE_MEMBERSHIP, {
            "user_id": user_id, "org_id": org_id, "role": role,
            "permission_template_id": data.get("permission_template_id"),
            "workspace_id": data.get("workspace_id"),
            "is_suspended": data.get("is_suspended", False),
            "max_tokens_per_day": data.get("max_tokens_per_day"),
            "max_chat_messages_per_day": data.get("max_chat_messages_per_day"),
        })
    updated = result.rowcount > 0
    if updated:
        log.info("Membership updated | user=%s | org=%s | role=%s", user_id, org_id, role)
    return updated


async def remove_membership(user_id: str, org_id: str) -> bool:
    async with get_db() as db:
        result = await db.execute(_DELETE_MEMBERSHIP, {"user_id": user_id, "org_id": org_id})
    deleted = result.rowcount > 0
    if deleted:
        log.info("Membership removed | user=%s | org=%s", user_id, org_id)
    return deleted


async def list_members(org_id: str) -> list[dict]:
    """List users who are members of an org (via org_memberships)."""
    async with get_db() as db:
        result = await db.execute(_LIST_MEMBERS_IN_ORG, {"org_id": org_id})
        rows = result.fetchall()
    return [
        {
            "id": str(r.id),
            "email": r.email,
            "role": r.role,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "last_login_at": r.last_login_at.isoformat() if r.last_login_at else None,
        }
        for r in rows
    ]


async def resolve_active_membership(user_id: str, requested_org_id: str = "") -> dict:
    """Resolve the active org + role for a request.

    Returns {"org_id": str, "role": str}. `org_id` is "" when no org can be
    determined (admin without an X-Org-Context header, or a user who belongs to
    several orgs and sent no header) — callers treat that as "no org context".

    - admin            → role 'admin', org_id = requested (X-Org-Context) or "".
    - member/supervisor:
        * requested org they belong to → that membership's role.
        * no header + exactly one membership → that membership.
        * no header + several memberships → org_id "" with the user's highest
          role across orgs. A supervisor of multiple orgs keeps role
          'supervisor' so they still reach the admin dashboard; they then pick
          an org (X-Org-Context) to scope actions, just like an admin. Picking
          an org re-resolves the role per that membership above, so they can
          only act as supervisor in orgs where they actually are one.
    """
    user = await get_user(user_id)
    role = user["role"] if user else "member"

    if role == "admin":
        return {"org_id": requested_org_id or "", "role": "admin"}

    if requested_org_id:
        membership = await get_membership(user_id, requested_org_id)
        if membership:
            return {"org_id": requested_org_id, "role": membership["role"]}
        return {"org_id": "", "role": "member"}

    memberships = await list_memberships_for_user(user_id)
    if len(memberships) == 1:
        return {"org_id": memberships[0]["org_id"], "role": memberships[0]["role"]}
    highest = "supervisor" if any(m["role"] == "supervisor" for m in memberships) else "member"
    return {"org_id": "", "role": highest}


async def delete_user(user_id: str, org_id: str) -> bool:
    async with get_db() as db:
        result = await db.execute(_DELETE_USER, {"id": user_id, "org_id": org_id})
    deleted = result.rowcount > 0
    if deleted:
        log.info("User deleted | id=%s | org=%s", user_id, org_id)
    return deleted


async def verify_user(email: str, password: str) -> dict | None:
    """Return {id, org_id, email, role} if credentials valid, else None.

    org_id is None for admin users (they have no organization).
    """
    async with get_db() as db:
        result = await db.execute(_USER_BY_EMAIL, {"email": email})
        row = result.fetchone()
    if not row:
        return None
    if not _verify_password(password, row.password_hash):
        return None
    # Touch last_login_at
    async with get_db() as db:
        await db.execute(_TOUCH_LOGIN, {"id": str(row.id)})
    return {"id": str(row.id), "org_id": str(row.org_id) if row.org_id else None,
            "email": row.email, "role": row.role,
            "org_name": row.org_name if row.org_name else None}


async def ensure_default_org_and_admin(username: str, password: str) -> tuple[str, str]:
    """Idempotently create the default org + admin user. Returns (org_id, user_id)."""
    # Check if default org exists
    async with get_db() as db:
        result = await db.execute(_DEFAULT_ORG_EXISTS, {"id": DEFAULT_ORG_ID})
        org_exists = result.fetchone() is not None

    if not org_exists:
        async with get_db() as db:
            await db.execute(sa.text("""
                INSERT INTO organizations (id, name, slug, max_tokens_per_day_org)
                VALUES (CAST(:id AS UUID), :name, :slug, :max_tokens)
                ON CONFLICT DO NOTHING
            """), {
                "id": DEFAULT_ORG_ID, "name": "Default Organization", "slug": "default",
                "max_tokens": DEFAULT_ORG_MAX_TOKENS_PER_DAY,
            })
        log.info("Default org seeded | id=%s | max_tokens_per_day_org=%d",
                 DEFAULT_ORG_ID, DEFAULT_ORG_MAX_TOKENS_PER_DAY)

    # Ensure admin user exists
    async with get_db() as db:
        result = await db.execute(_USER_BY_EMAIL, {"email": username})
        row = result.fetchone()

    if row:
        # Idempotent: ensure existing admin has no org (migration may not have run yet)
        if row.org_id is not None:
            async with get_db() as db:
                await db.execute(
                    sa.text("UPDATE users SET org_id = NULL WHERE id = CAST(:id AS UUID)"),
                    {"id": str(row.id)},
                )
        return DEFAULT_ORG_ID, str(row.id)

    # Create admin user with no org
    user_id = await create_user(None, username, password, role="admin")
    if not user_id:
        # User may have been created by a concurrent replica — fetch it
        async with get_db() as db:
            result = await db.execute(_USER_BY_EMAIL, {"email": username})
            row = result.fetchone()
        user_id = str(row.id) if row else "unknown"
    return DEFAULT_ORG_ID, user_id
