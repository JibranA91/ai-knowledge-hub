"""Workspace management.

Workspaces are isolated namespaces within an org. Each workspace maps to an S3
prefix so raw documents are stored separately. Wiki pages and ingest jobs can
optionally be scoped to a workspace (workspace_id FK) in a future phase.

For Phase 7 the focus is on the DB entity and CRUD operations used by the admin
dashboard. S3-prefix provisioning is included; copy is a server-side S3 copy.
"""
import uuid

import sqlalchemy as sa

from app.db import get_db
from app.logger import get_logger
from app.services import s3

log = get_logger(__name__)

# ── SQL ───────────────────────────────────────────────────────────────────────

_LIST = sa.text("""
    SELECT w.id, w.org_id, w.name, w.owner_user_id, w.s3_prefix,
           w.created_from_workspace_id, w.created_at,
           u.email AS owner_email
    FROM workspaces w
    LEFT JOIN users u ON w.owner_user_id = u.id
    WHERE w.org_id = CAST(:org_id AS UUID)
    ORDER BY w.created_at DESC
""")

_GET = sa.text("""
    SELECT id, org_id, name, owner_user_id, s3_prefix,
           created_from_workspace_id, created_at
    FROM workspaces
    WHERE id = CAST(:id AS UUID) AND org_id = CAST(:org_id AS UUID)
""")

_INSERT = sa.text("""
    INSERT INTO workspaces (id, org_id, name, owner_user_id, s3_prefix, created_from_workspace_id)
    VALUES (CAST(:id AS UUID), CAST(:org_id AS UUID), :name,
            CAST(:owner_user_id AS UUID), :s3_prefix,
            CAST(:source_id AS UUID))
    ON CONFLICT (org_id, name) DO NOTHING
    RETURNING id
""")

_DELETE = sa.text("""
    DELETE FROM workspaces
    WHERE id = CAST(:id AS UUID) AND org_id = CAST(:org_id AS UUID)
""")

_COUNT_MEMBERS = sa.text("""
    SELECT COUNT(*) FROM users
    WHERE org_id = CAST(:org_id AS UUID)
""")


# ── Public API ────────────────────────────────────────────────────────────────

def _row_to_dict(row) -> dict:
    return {
        "id": str(row.id),
        "org_id": str(row.org_id),
        "name": row.name,
        "owner_user_id": str(row.owner_user_id) if row.owner_user_id else None,
        "owner_email": getattr(row, "owner_email", None),
        "s3_prefix": row.s3_prefix,
        "created_from_workspace_id": (
            str(row.created_from_workspace_id) if row.created_from_workspace_id else None
        ),
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


async def list_workspaces(org_id: str) -> list[dict]:
    async with get_db() as db:
        result = await db.execute(_LIST, {"org_id": org_id})
        rows = result.fetchall()
    return [_row_to_dict(r) for r in rows]


async def get_workspace(workspace_id: str, org_id: str) -> dict | None:
    async with get_db() as db:
        result = await db.execute(_GET, {"id": workspace_id, "org_id": org_id})
        row = result.fetchone()
    return _row_to_dict(row) if row else None


async def create_workspace(org_id: str, name: str, owner_user_id: str | None = None) -> str | None:
    """Create a blank workspace with its own S3 prefix. Returns new UUID or None on conflict."""
    new_id = str(uuid.uuid4())
    prefix = f"{org_id}/workspaces/{new_id}/"
    async with get_db() as db:
        result = await db.execute(_INSERT, {
            "id": new_id,
            "org_id": org_id,
            "name": name,
            "owner_user_id": owner_user_id,  # NULL when no owner — nullable FK
            "s3_prefix": prefix,
            "source_id": None,               # NULL — not copied from another workspace
        })
        row = result.fetchone()
    if row:
        log.info("Workspace created | id=%s | org=%s | name=%s", new_id, org_id, name)
        return str(row.id)
    return None


async def copy_workspace(source_id: str, org_id: str, new_name: str,
                         owner_user_id: str | None = None) -> str | None:
    """Create a new workspace by copying the source's S3 objects.

    Returns the new workspace UUID or None on name conflict.
    """
    source = await get_workspace(source_id, org_id)
    if not source:
        return None

    new_id = str(uuid.uuid4())
    new_prefix = f"{org_id}/workspaces/{new_id}/"

    async with get_db() as db:
        result = await db.execute(_INSERT, {
            "id": new_id,
            "org_id": org_id,
            "name": new_name,
            "owner_user_id": owner_user_id,  # NULL when no owner — nullable FK
            "s3_prefix": new_prefix,
            "source_id": source_id,
        })
        row = result.fetchone()

    if not row:
        return None

    # Best-effort S3 copy — silently skip if local storage or S3 unavailable
    try:
        src_prefix = source["s3_prefix"]
        for key in s3.list_keys(src_prefix):
            dest_key = new_prefix + key[len(src_prefix):]
            data = s3.read_bytes(key)
            if data is not None:
                s3.write_bytes(dest_key, data)
    except Exception as exc:
        log.warning("Workspace copy | S3 copy skipped | %s", exc)

    log.info("Workspace copied | src=%s → dst=%s | org=%s", source_id, new_id, org_id)
    return new_id


async def delete_workspace(workspace_id: str, org_id: str) -> bool:
    ws = await get_workspace(workspace_id, org_id)
    if not ws:
        return False

    # Best-effort S3 cleanup
    try:
        for key in s3.list_keys(ws["s3_prefix"]):
            s3.delete(key)
    except Exception as exc:
        log.warning("Workspace delete | S3 cleanup skipped | %s", exc)

    async with get_db() as db:
        result = await db.execute(_DELETE, {"id": workspace_id, "org_id": org_id})
    deleted = result.rowcount > 0
    if deleted:
        log.info("Workspace deleted | id=%s | org=%s", workspace_id, org_id)
    return deleted


async def check_member_limit(org_id: str) -> None:
    """Raise 409 if the org has reached its max_members limit."""
    from app.services import orgs as orgs_svc
    org = await orgs_svc.get_org(org_id)
    if not org:
        return
    max_members = org.get("max_members")
    if max_members is None:
        return
    async with get_db() as db:
        res = await db.execute(_COUNT_MEMBERS, {"org_id": org_id})
        count = res.scalar() or 0
    if count >= max_members:
        from fastapi import HTTPException
        raise HTTPException(409, f"Organization member limit reached ({max_members} users)")
