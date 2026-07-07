"""Change tracking + revert infrastructure.

Every wiki-state mutation flows through one of five choke-point functions
(`upsert_wiki_page`, `delete_wiki_page`, `set_wiki_file`, `s3.write_bytes`,
`s3.delete`). Those functions call `record_revision()` here, which emits a
`wiki_revisions` row attributed to the currently open action (if any).

Actions are opened with `begin_action()` (context manager) or `@tracked_action`
(route decorator). The action id flows through a ContextVar so background
tasks created via `asyncio.create_task` inherit it.

When no action is open (startup seeding, migrations, internal cleanup), the
write still happens — `record_revision()` is a no-op. That keeps every
existing code path working without modification.
"""
import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from functools import wraps
from typing import Any, Callable

import sqlalchemy as sa

from app.context import UserContext, current_action, current_user
from app.db import get_db
from app.logger import get_logger
from app.services import s3

log = get_logger(__name__)

# classid for the revert-start advisory lock (distinct from ingest_queue's 42).
_REVERT_LOCK_CLASS = 43


# ── Action lifecycle ───────────────────────────────────────────────────────

_INSERT_ACTION = sa.text("""
    INSERT INTO wiki_actions (id, org_id, user_id, action_type, summary, status, details)
    VALUES (CAST(:id AS UUID), CAST(:org_id AS UUID), CAST(:user_id AS UUID),
            :action_type, :summary, 'running', CAST(:details AS JSONB))
""")

_FINALIZE_ACTION = sa.text("""
    UPDATE wiki_actions
       SET status = :status,
           finished_at = NOW()
     WHERE id = CAST(:id AS UUID)
""")


async def _insert_action_row(
    action_id: str,
    org_id: str,
    user_id: str | None,
    action_type: str,
    summary: str,
    details: dict,
) -> None:
    async with get_db() as db:
        await db.execute(_INSERT_ACTION, {
            "id": action_id,
            "org_id": org_id,
            "user_id": user_id,
            "action_type": action_type,
            "summary": summary,
            "details": json.dumps(details),
        })


async def _finalize_action(action_id: str, status: str) -> None:
    async with get_db() as db:
        await db.execute(_FINALIZE_ACTION, {"id": action_id, "status": status})


@asynccontextmanager
async def begin_action(action_type: str, summary: str = "", details: dict | None = None):
    """Open a tracked action. Every write inside the block emits a revision.

    Nested calls are a no-op: if an action is already open in the current
    context, the inner call reuses the outer id and does not insert a new row.
    This prevents fragmenting a single user action (upload + auto-ingest) into
    multiple disconnected entries.

    On clean exit: status='done'. On exception: status='error'. The exception
    re-raises so the caller's error handling is unchanged.
    """
    outer = current_action.get()
    if outer is not None:
        yield outer
        return

    ctx = current_user.get(None)
    org_id = ctx.org_id if ctx and ctx.org_id else None
    user_id = ctx.user_id if ctx and ctx.user_id else None
    if not org_id:
        # No org context — can't attribute the action. Fall through without
        # tracking. This matches the "writes outside an action are untracked"
        # contract used by startup seeding.
        yield None
        return

    action_id = uuid.uuid4().hex
    await _insert_action_row(action_id, org_id, user_id, action_type, summary, details or {})
    token = current_action.set(action_id)
    try:
        yield action_id
    except Exception:
        await _finalize_action(action_id, "error")
        current_action.reset(token)
        raise
    else:
        await _finalize_action(action_id, "done")
    finally:
        if current_action.get() == action_id:
            current_action.reset(token)


def tracked_action(action_type: str, summary_fn: Callable[..., str] | None = None):
    """FastAPI route decorator: wrap a handler in `begin_action()`.

    `summary_fn` receives the handler's args/kwargs and returns the action
    summary string. Defaults to an empty string when not provided.
    """
    def deco(fn):
        @wraps(fn)
        async def wrapper(*args, **kwargs):
            summary = summary_fn(*args, **kwargs) if summary_fn else ""
            async with begin_action(action_type, summary):
                return await fn(*args, **kwargs)
        return wrapper
    return deco


# ── Revision recording ────────────────────────────────────────────────────

_INSERT_REVISION = sa.text("""
    INSERT INTO wiki_revisions (action_id, org_id, target_kind, target_key, op,
                                content_before, content_after,
                                bytes_size_before, bytes_size_after)
    VALUES (CAST(:action_id AS UUID), CAST(:org_id AS UUID),
            :target_kind, :target_key, :op,
            :content_before, :content_after,
            :bytes_size_before, :bytes_size_after)
""")


async def record_revision(
    *,
    target_kind: str,
    target_key: str,
    op: str,
    content_before: str | None = None,
    content_after: str | None = None,
    bytes_size_before: int | None = None,
    bytes_size_after: int | None = None,
    db=None,
) -> None:
    """Emit a revision row for the currently open action.

    No-op when no action is open. This is the single point that ties writes
    to actions — every wrapped choke-point function calls this.

    Pass `db` (an open session) to record the revision in the SAME transaction
    as the write it describes — so a crash can't durably commit the write but
    lose its revision (which would make the change un-revertible). When omitted,
    a separate transaction is used (callers whose write isn't a DB row, e.g. S3).
    """
    action_id = current_action.get()
    if action_id is None:
        return

    ctx = current_user.get(None)
    org_id = ctx.org_id if ctx and ctx.org_id else None
    if not org_id:
        log.warning("record_revision | action=%s but no org_id in context — skipping", action_id)
        return

    params = {
        "action_id": action_id,
        "org_id": org_id,
        "target_kind": target_kind,
        "target_key": target_key,
        "op": op,
        "content_before": content_before,
        "content_after": content_after,
        "bytes_size_before": bytes_size_before,
        "bytes_size_after": bytes_size_after,
    }
    if db is not None:
        await db.execute(_INSERT_REVISION, params)   # caller's transaction commits it
    else:
        async with get_db() as own:
            await own.execute(_INSERT_REVISION, params)


# ── S3 archive key helper ─────────────────────────────────────────────────

def archive_key_for(action_id: str, live_key: str) -> str:
    """Build the archive S3 key for a soon-to-be-overwritten/deleted object.

    Live keys are already org-prefixed (e.g. `<org_id>/raw/foo.pdf`); the
    archive key adds an `archive/<action_id>/` segment so a prune job can
    sweep an entire action's archived blobs by prefix.
    """
    return f"archive/{action_id}/{live_key}"


# ── Tracked S3 writes ─────────────────────────────────────────────────────
#
# The raw `s3.write_bytes` / `s3.delete` calls stay sync and untracked so
# rollback/cleanup paths (e.g. cancelled-ingest cleanup) can use them without
# producing fake revert entries. Callers that want tracking use these
# async wrappers, which archive the prior bytes and emit a revision row.

async def tracked_write_bytes(key: str, data: bytes) -> None:
    """Write bytes to storage and emit a `s3_raw` revision.

    If `key` already has an object, copy it to an archive key first so a
    future revert can restore it. The revision's `content_before` field
    holds the archive key string (not the bytes).
    """
    action_id = current_action.get()

    archive_path: str | None = None
    size_before: int | None = None
    op_kind = "create"

    if s3.exists(key):
        op_kind = "update"
        size_before = s3.get_object_size(key)
        if action_id is not None:
            archive_path = archive_key_for(action_id, key)
            existing = s3.read_bytes(key)
            s3.write_bytes(archive_path, existing)

    s3.write_bytes(key, data)

    await record_revision(
        target_kind="s3_raw",
        target_key=key,
        op=op_kind,
        content_before=archive_path,
        content_after=key,
        bytes_size_before=size_before,
        bytes_size_after=len(data),
    )


async def tracked_delete(key: str) -> None:
    """Delete an object from storage and emit a `s3_raw` revision.

    Archives the prior bytes before deletion so revert can restore them.
    No-op (and no revision) when the key doesn't exist.
    """
    if not s3.exists(key):
        return

    action_id = current_action.get()
    archive_path: str | None = None
    size_before = s3.get_object_size(key)
    if action_id is not None:
        archive_path = archive_key_for(action_id, key)
        existing = s3.read_bytes(key)
        s3.write_bytes(archive_path, existing)

    s3.delete(key)

    await record_revision(
        target_kind="s3_raw",
        target_key=key,
        op="delete",
        content_before=archive_path,
        content_after=None,
        bytes_size_before=size_before,
        bytes_size_after=None,
    )


# ── Revert ────────────────────────────────────────────────────────────────

_GET_ACTION = sa.text("""
    SELECT id::text AS id, action_type, status, started_at, summary, org_id::text AS org_id
    FROM wiki_actions
    WHERE id = CAST(:id AS UUID) AND org_id = CAST(:org_id AS UUID)
""")

_ACTIONS_SINCE = sa.text("""
    SELECT id::text AS id, action_type, started_at, status
    FROM wiki_actions
    WHERE org_id = CAST(:org_id AS UUID)
      AND started_at >= :since
      AND status IN ('done', 'error')
    ORDER BY started_at DESC, id DESC
""")

# All revisions across every action in the revert window, oldest first. We
# pull them in one query (rather than per-action) and coalesce in Python: see
# `_coalesce_revisions`.
_REVISIONS_SINCE = sa.text("""
    SELECT r.id, r.target_kind, r.target_key, r.op, r.content_before, r.content_after
    FROM wiki_revisions r
    JOIN wiki_actions a ON a.id = r.action_id
    WHERE a.org_id = CAST(:org_id AS UUID)
      AND a.started_at >= :since
      AND a.status IN ('done', 'error')
    ORDER BY r.id ASC
""")

# Number of distinct objects the revert will touch — i.e. how many restore
# operations the coalesced revert performs. Drives the progress bar.
_COUNT_DISTINCT_TARGETS_SINCE = sa.text("""
    SELECT COUNT(*) AS n FROM (
        SELECT DISTINCT r.target_kind, r.target_key
        FROM wiki_revisions r
        JOIN wiki_actions a ON a.id = r.action_id
        WHERE a.org_id = CAST(:org_id AS UUID)
          AND a.started_at >= :since
          AND a.status IN ('done', 'error')
    ) t
""")

# `_ACTIONS_SINCE` returns ids already cast to text (canonical dashed form),
# so we compare on `id::text` and expand the list into a text IN clause.
_MARK_REVERTED_BULK = sa.text("""
    UPDATE wiki_actions
       SET status = 'reverted'
     WHERE id::text IN :ids
""").bindparams(sa.bindparam("ids", expanding=True))

# Record, on the revert's own row, which actions it reverted and what their
# status was beforehand — so a later revert-of-this-revert (a redo) can flip
# those actions' stale 'reverted' status back to what it was.
_SET_REVERTED_ACTIONS = sa.text("""
    UPDATE wiki_actions
       SET details = jsonb_set(COALESCE(details, '{}'::jsonb),
                               '{reverted_actions}', CAST(:val AS JSONB))
     WHERE id = CAST(:id AS UUID)
""")

# Pull the stored reverted-action lists for a set of revert actions.
_GET_REVERTED_ACTIONS_FOR = sa.text("""
    SELECT details
    FROM wiki_actions
    WHERE action_type = 'revert' AND id::text IN :ids
""").bindparams(sa.bindparam("ids", expanding=True))

# Restore a set of actions to a given prior status — but only if they're still
# sitting at 'reverted', so we never clobber a status that changed since.
_RESTORE_STATUS = sa.text("""
    UPDATE wiki_actions
       SET status = :status
     WHERE status = 'reverted' AND id::text IN :ids
""").bindparams(sa.bindparam("ids", expanding=True))


class RevertError(Exception):
    """Raised when a revert cannot proceed (target not found, already reverted, etc.)."""


async def revert_action(target_action_id: str) -> dict:
    """Revert a target action — and every action that happened after it.

    Git-reset semantics: the wiki ends up in the state it was in right after
    the action immediately preceding `target_action_id`. The target plus all
    subsequent done/error actions get `status='reverted'`. The revert itself
    is recorded as a new `revert` action so it is itself revertable — reverting
    a revert is the natural "redo": a revert records its own inverse ops as
    revisions, so undoing it restores each object to its exact pre-revert state.

    Raises `RevertError` if the target doesn't exist for this org, is still
    running, or has already been reverted.
    """
    from app.context import get_org_id
    org_id = get_org_id()

    async with get_db() as db:
        result = await db.execute(_GET_ACTION, {"id": target_action_id, "org_id": org_id})
        target = result.fetchone()

    if target is None:
        raise RevertError(f"Action {target_action_id} not found")
    if target.status == "reverted":
        raise RevertError("Action has already been reverted")
    if target.status == "running":
        raise RevertError("Cannot revert an action that is still running")

    async with begin_action(
        "revert",
        summary=f"revert {target.action_type}: {target.summary}",
        details={"target_action_id": target_action_id, "target_action_type": target.action_type},
    ) as revert_action_id:
        async with get_db() as db:
            await db.execute(sa.text("""
                UPDATE wiki_actions SET revert_of_id = CAST(:target AS UUID)
                 WHERE id = CAST(:id AS UUID)
            """), {"target": target_action_id, "id": revert_action_id})

            actions = (await db.execute(_ACTIONS_SINCE, {
                "org_id": org_id,
                "since": target.started_at,
            })).fetchall()

        reverted = [{"id": a.id, "status": a.status}
                    for a in actions if a.id != revert_action_id]
        reverted_action_ids = [a["id"] for a in reverted]
        touched_pages = await _apply_coalesced_revert(org_id, target.started_at)
        await _finalize_reverted_marks(revert_action_id, reverted)
        await _restore_redone_revert_statuses(actions)

        # Log to audit_log so reverts are visible alongside other operations.
        # `append_audit_log` reads the current user from the request context.
        from app.services.wiki_db import append_audit_log
        await append_audit_log(
            operation="revert",
            raw_text=(
                f"Reverted action `{target_action_id}` ({target.action_type}: "
                f"{target.summary or '(no summary)'}) — undid {len(reverted_action_ids)} "
                f"action(s), touched {len(touched_pages)} page(s)"
            ),
            details={
                "target_action_id": target_action_id,
                "target_action_type": target.action_type,
                "target_summary": target.summary,
                "revert_action_id": revert_action_id,
                "reverted_action_ids": reverted_action_ids,
                "pages_touched": sorted(touched_pages),
            },
        )

    if touched_pages:
        await _rebuild_graph_for(touched_pages)

    return {
        "revert_action_id": revert_action_id,
        "reverted_action_ids": reverted_action_ids,
        "pages_touched": sorted(touched_pages),
    }


def _coalesce_revisions(revs: list) -> list:
    """Collapse a window of revisions to one restore op per distinct object.

    A point-in-time revert only needs to put each object back to the state it
    had right before the window began — every intermediate edit is irrelevant.
    That target state is the `content_before` of the *oldest* revision touching
    the object in the window, so we keep exactly that revision per
    (target_kind, target_key). Applying its inverse (`_restore_revision`)
    yields the correct end state regardless of how many times the object
    changed: oldest op `create` → object didn't exist → delete; oldest op
    `update`/`delete` → restore its `content_before`.

    `revs` must be ordered by id ASC, so the first row seen per key is oldest.
    """
    oldest: dict[tuple[str, str], Any] = {}
    for rev in revs:
        key = (rev.target_kind, rev.target_key)
        if key not in oldest:
            oldest[key] = rev
    return list(oldest.values())


async def _apply_coalesced_revert(
    org_id: str, since, progress_cb: Callable[[int, int], Any] | None = None,
) -> set[str]:
    """Restore every object touched since `since` to its pre-window state.

    Loads all revisions in the window in one query, coalesces them to one op
    per distinct object, and applies the inverse. Cost is O(distinct objects),
    not O(total revisions) — this is what makes a large revert fast.

    Calls `progress_cb(done, total)` after each object if provided. Returns the
    set of touched page paths so the caller can rebuild the link graph.
    """
    async with get_db() as db:
        revs = (await db.execute(
            _REVISIONS_SINCE, {"org_id": org_id, "since": since}
        )).fetchall()

    coalesced = _coalesce_revisions(revs)
    touched_pages: set[str] = set()
    for idx, rev in enumerate(coalesced):
        await _restore_revision(rev)
        if rev.target_kind == "page":
            touched_pages.add(rev.target_key)
        if progress_cb is not None:
            await progress_cb(idx + 1, len(coalesced))
    return touched_pages


async def _finalize_reverted_marks(revert_action_id: str, reverted: list[dict]) -> None:
    """Flip the reverted actions to status='reverted' AND stash the redo record
    on the revert's own row — in ONE transaction.

    Doing these as two separate transactions meant a crash between them could
    leave actions marked 'reverted' with no recorded `reverted_actions`, so a
    later redo couldn't restore their original statuses (or vice-versa). One
    transaction keeps the status flip and the redo record consistent.
    """
    action_ids = [a["id"] for a in reverted]
    async with get_db() as db:
        if action_ids:
            await db.execute(_MARK_REVERTED_BULK, {"ids": action_ids})
        await db.execute(_SET_REVERTED_ACTIONS, {
            "id": revert_action_id, "val": json.dumps(reverted),
        })


async def _restore_redone_revert_statuses(window_actions: list) -> None:
    """Undo the status side effect of any revert actions we just reverted.

    When this revert undoes an earlier revert R, the actions R had reverted are
    live again — but they still read `status='reverted'`. R recorded those
    actions (with their pre-R status) in `details.reverted_actions`, so we flip
    each back to what it was. Content is already handled by the coalesced
    restore; this only fixes the History label.
    """
    revert_ids = [a.id for a in window_actions if a.action_type == "revert"]
    if not revert_ids:
        return

    async with get_db() as db:
        rows = (await db.execute(_GET_REVERTED_ACTIONS_FOR, {"ids": revert_ids})).fetchall()

    ids_by_status: dict[str, list[str]] = {}
    for row in rows:
        details = row.details or {}
        for entry in details.get("reverted_actions", []):
            ids_by_status.setdefault(entry.get("status", "done"), []).append(entry["id"])

    for status, ids in ids_by_status.items():
        async with get_db() as db:
            await db.execute(_RESTORE_STATUS, {"status": status, "ids": ids})


async def _restore_revision(rev) -> None:
    """Undo a single revision by applying the inverse operation."""
    if rev.target_kind == "page":
        await _restore_page_revision(rev)
    elif rev.target_kind == "file":
        await _restore_file_revision(rev)
    elif rev.target_kind == "s3_raw":
        await _restore_s3_revision(rev)
    else:
        log.warning("_restore_revision | unknown target_kind=%s | skipping", rev.target_kind)


async def _restore_page_revision(rev) -> None:
    from app.services import wiki_db
    if rev.op == "create":
        await wiki_db.delete_wiki_page(rev.target_key)
    elif rev.op in ("update", "delete"):
        await wiki_db.upsert_wiki_page(rev.target_key, rev.content_before or "")


async def _restore_file_revision(rev) -> None:
    from app.context import get_org_id
    from app.services import wiki_db
    if rev.op == "create":
        async with get_db() as db:
            await db.execute(
                sa.text("DELETE FROM wiki_files WHERE org_id = CAST(:org_id AS UUID) AND key = :key"),
                {"org_id": get_org_id(), "key": rev.target_key},
            )
        return
    await wiki_db.set_wiki_file(rev.target_key, rev.content_before or "")


async def _restore_s3_revision(rev) -> None:
    archive_key = rev.content_before
    live_key = rev.target_key
    if rev.op == "create":
        s3.delete(live_key)
        return
    if not archive_key or not s3.exists(archive_key):
        log.error("_restore_s3_revision | archive missing | key=%s | archive=%s", live_key, archive_key)
        return
    data = s3.read_bytes(archive_key)
    s3.write_bytes(live_key, data)


async def _rebuild_graph_for(paths: set[str]) -> None:
    """Refresh the link graph for the given pages after revert."""
    try:
        # Use the per-org cached instance (not a throwaway WikiGraph()), so the
        # live graph that serves /graph and retrieval is updated, and so
        # update_pages re-resolves links against the full loaded graph rather
        # than an empty one (which would drop every existing edge).
        from app.services.graph import get_graph
        await get_graph().update_pages(list(paths))
    except Exception as exc:
        log.warning("_rebuild_graph_for | best-effort failed: %s", exc)


# ── Retention pruning ─────────────────────────────────────────────────────

_PICK_PRUNABLE_ACTIONS = sa.text("""
    SELECT a.id::text AS id
    FROM wiki_actions a
    JOIN organizations o ON o.id = a.org_id
    WHERE a.started_at < NOW() - (o.revision_retention_days || ' days')::interval
      AND a.status IN ('done', 'error', 'reverted')
""")

_PICK_S3_ARCHIVES_FOR_ACTION = sa.text("""
    SELECT content_before, content_after
    FROM wiki_revisions
    WHERE action_id = CAST(:id AS UUID)
      AND target_kind = 's3_raw'
""")

_DELETE_ACTION = sa.text("DELETE FROM wiki_actions WHERE id = CAST(:id AS UUID)")


async def prune_expired_revisions() -> dict:
    """Delete actions and their revisions older than each org's retention.

    Also removes the archived S3 objects each revision points to so we don't
    leak storage. Returns counts for monitoring.
    """
    async with get_db() as db:
        rows = (await db.execute(_PICK_PRUNABLE_ACTIONS)).fetchall()

    action_ids = [r.id for r in rows]
    archives_deleted = 0
    for action_id in action_ids:
        async with get_db() as db:
            archive_rows = (await db.execute(
                _PICK_S3_ARCHIVES_FOR_ACTION, {"id": action_id}
            )).fetchall()
        for row in archive_rows:
            for key in (row.content_before, row.content_after):
                if key and key.startswith("archive/"):
                    try:
                        s3.delete(key)
                        archives_deleted += 1
                    except Exception as exc:
                        log.warning("prune | s3.delete failed | key=%s | err=%s", key, exc)
        async with get_db() as db:
            await db.execute(_DELETE_ACTION, {"id": action_id})

    if action_ids:
        log.info("prune_expired_revisions | actions=%d | archives=%d",
                 len(action_ids), archives_deleted)
    return {"actions_pruned": len(action_ids), "archives_deleted": archives_deleted}


async def is_revert_running(org_id: str) -> bool:
    """True when a revert action is currently in flight for this org.

    Used by the middleware to block concurrent writes during a revert.
    """
    async with get_db() as db:
        result = await db.execute(sa.text("""
            SELECT 1 FROM wiki_actions
            WHERE org_id = CAST(:org_id AS UUID)
              AND action_type = 'revert'
              AND status = 'running'
            LIMIT 1
        """), {"org_id": org_id})
        return result.fetchone() is not None


# ── Background revert job ─────────────────────────────────────────────────
# For large reverts (e.g. undoing 100 ingest actions) we spawn the loop as
# a background task and let the client poll progress, rather than tying up
# the HTTP request. The new wiki_actions row carries progress_done /
# progress_total so the frontend can render an X/Y indicator.

_revert_tasks: dict[str, asyncio.Task] = {}

_UPDATE_PROGRESS = sa.text("""
    UPDATE wiki_actions SET progress_done = :done
    WHERE id = CAST(:id AS UUID)
""")

_SET_PROGRESS_TOTAL = sa.text("""
    UPDATE wiki_actions SET progress_total = :total
    WHERE id = CAST(:id AS UUID)
""")

_GET_REVERT_JOB = sa.text("""
    SELECT id::text AS id, action_type, status, summary, started_at, finished_at,
           progress_done, progress_total, error_message, details,
           revert_of_id::text AS revert_of_id
    FROM wiki_actions
    WHERE id = CAST(:id AS UUID) AND org_id = CAST(:org_id AS UUID)
""")

_MARK_FAILED_REVERT = sa.text("""
    UPDATE wiki_actions
       SET status = 'error',
           error_message = :msg,
           finished_at = NOW()
     WHERE id = CAST(:id AS UUID)
""")

_RECOVER_STALE_RUNNING_REVERTS = sa.text("""
    UPDATE wiki_actions
       SET status = 'error',
           error_message = 'Interrupted by server restart',
           finished_at = NOW()
     WHERE action_type = 'revert' AND status = 'running'
""")


class RevertJobAlreadyRunning(RevertError):
    """Another revert is already in flight for this org."""


async def start_revert_job(target_action_id: str) -> dict:
    """Validate + create a revert job and spawn the worker. Returns immediately.

    The returned dict carries `revert_action_id` (UUID hex) which the caller
    polls via `get_revert_job_status`. The revert itself runs in a background
    `asyncio.Task` and updates `progress_done` as it walks the action queue.

    Raises `RevertError` for invalid targets (not found, already reverted,
    still running) — these cases return synchronously. Reverting a `revert`
    action is allowed and behaves as a redo.

    Raises `RevertJobAlreadyRunning` if another revert is already in flight
    for this org (we serialize per-org so the link graph and audit log stay
    coherent).
    """
    from app.context import get_org_id
    org_id = get_org_id()

    async with get_db() as db:
        result = await db.execute(_GET_ACTION, {"id": target_action_id, "org_id": org_id})
        target = result.fetchone()

    if target is None:
        raise RevertError(f"Action {target_action_id} not found")
    if target.status == "reverted":
        raise RevertError("Action has already been reverted")
    if target.status == "running":
        raise RevertError("Cannot revert an action that is still running")

    # Pre-count the distinct objects in scope so the UI can show progress_total
    # upfront. The coalesced revert performs one restore op per distinct object,
    # so this — not the action count — is the real unit of work. (Read; safe to
    # do before the lock.)
    async with get_db() as db:
        progress_total = (await db.execute(_COUNT_DISTINCT_TARGETS_SINCE, {
            "org_id": org_id, "since": target.started_at,
        })).scalar() or 0

    ctx = current_user.get(None)
    user_id = ctx.user_id if ctx and ctx.user_id else None
    revert_action_id = uuid.uuid4().hex

    # Atomically claim the org's single revert slot. A transaction-level advisory
    # lock serializes concurrent revert-starts across replicas, so the
    # "is a revert already running?" check and the running-row insert cannot
    # interleave — without it, two requests could both pass the check and start
    # overlapping reverts that corrupt each other's restores. The lock is held
    # only for this short transaction (released on commit); the inserted
    # status='running' row is what blocks further reverts for the worker's life.
    # We manage the row lifecycle manually (the worker outlives the HTTP request,
    # so begin_action's context-manager model doesn't fit).
    async with get_db() as db:
        await db.execute(
            sa.text("SELECT pg_advisory_xact_lock(:cls, hashtext(:org))"),
            {"cls": _REVERT_LOCK_CLASS, "org": org_id},
        )
        running = (await db.execute(sa.text("""
            SELECT 1 FROM wiki_actions
             WHERE org_id = CAST(:org_id AS UUID)
               AND action_type = 'revert' AND status = 'running'
             LIMIT 1
        """), {"org_id": org_id})).fetchone()
        if running is not None:
            raise RevertJobAlreadyRunning(
                "Another revert is already in progress for this organization. "
                "Wait for it to finish before starting a new one."
            )
        await db.execute(_INSERT_ACTION, {
            "id": revert_action_id, "org_id": org_id, "user_id": user_id,
            "action_type": "revert",
            "summary": f"revert {target.action_type}: {target.summary}",
            "details": json.dumps({"target_action_id": target_action_id,
                                   "target_action_type": target.action_type}),
        })
        await db.execute(sa.text("""
            UPDATE wiki_actions SET revert_of_id = CAST(:target AS UUID)
            WHERE id = CAST(:id AS UUID)
        """), {"target": target_action_id, "id": revert_action_id})
        await db.execute(_SET_PROGRESS_TOTAL, {"id": revert_action_id, "total": progress_total})

    task = asyncio.create_task(
        _run_revert_job(revert_action_id, target_action_id, org_id, user_id),
        name=f"revert-job-{revert_action_id}",
    )
    _revert_tasks[revert_action_id] = task
    task.add_done_callback(lambda t: _revert_tasks.pop(revert_action_id, None))
    log.info("start_revert_job | revert=%s | target=%s | total=%d",
             revert_action_id, target_action_id, progress_total)

    return {
        "revert_action_id": revert_action_id,
        "status": "running",
        "progress_done": 0,
        "progress_total": progress_total,
    }


async def _run_revert_job(
    revert_action_id: str, target_action_id: str, org_id: str, user_id: str | None,
) -> None:
    """Background worker — restore revisions, update progress, finalize the row.

    Failures are swallowed (we can't propagate them to the HTTP client) and
    written to `wiki_actions.error_message` so the polling client can show
    them. Idempotent on partial failure: the client can call `start_revert_job`
    again with the same target to resume the loop — anything already marked
    `reverted` is skipped by the algorithm (`_ACTIONS_SINCE` filters by
    `status IN ('done', 'error')`).
    """
    # Re-establish the ContextVars that the wrapped writes rely on. The
    # request-scoped vars from the originating HTTP call are NOT visible in
    # this background task because asyncio.create_task captures the context
    # at task-creation time — which usually works, but ASGI middleware may
    # have already torn down the request context by the time we run, so we
    # set them explicitly for safety.
    if user_id:
        current_user.set(UserContext(user_id=user_id, org_id=org_id, email="", role=""))
    current_action.set(revert_action_id)

    try:
        async with get_db() as db:
            target = (await db.execute(
                _GET_ACTION, {"id": target_action_id, "org_id": org_id}
            )).fetchone()
        if target is None:
            await _finalize_revert_failure(
                revert_action_id, f"Target action {target_action_id} disappeared mid-revert"
            )
            return

        async with get_db() as db:
            actions = (await db.execute(_ACTIONS_SINCE, {
                "org_id": org_id, "since": target.started_at,
            })).fetchall()
        reverted = [{"id": a.id, "status": a.status}
                    for a in actions if a.id != revert_action_id]
        reverted_action_ids = [a["id"] for a in reverted]

        async def _report(done: int, _total: int) -> None:
            async with get_db() as db:
                await db.execute(_UPDATE_PROGRESS, {"id": revert_action_id, "done": done})

        touched_pages = await _apply_coalesced_revert(
            org_id, target.started_at, progress_cb=_report,
        )
        await _finalize_reverted_marks(revert_action_id, reverted)
        await _restore_redone_revert_statuses(actions)

        from app.services.wiki_db import append_audit_log
        await append_audit_log(
            operation="revert",
            raw_text=(
                f"Reverted action `{target_action_id}` ({target.action_type}: "
                f"{target.summary or '(no summary)'}) — undid {len(reverted_action_ids)} "
                f"action(s), touched {len(touched_pages)} page(s)"
            ),
            details={
                "target_action_id": target_action_id,
                "target_action_type": target.action_type,
                "target_summary": target.summary,
                "revert_action_id": revert_action_id,
                "reverted_action_ids": reverted_action_ids,
                "pages_touched": sorted(touched_pages),
            },
        )
        if touched_pages:
            await _rebuild_graph_for(touched_pages)
        await _finalize_action(revert_action_id, "done")
        log.info("revert_job done | revert=%s | undid=%d | pages=%d",
                 revert_action_id, len(reverted_action_ids), len(touched_pages))
    except asyncio.CancelledError:
        await _finalize_revert_failure(revert_action_id, "Revert cancelled")
        raise
    except Exception as exc:
        log.exception("revert_job failed | revert=%s", revert_action_id)
        await _finalize_revert_failure(revert_action_id, str(exc) or repr(exc))


async def _finalize_revert_failure(revert_action_id: str, message: str) -> None:
    """Mark the revert wiki_actions row as 'error' with a user-facing message."""
    try:
        async with get_db() as db:
            await db.execute(_MARK_FAILED_REVERT, {
                "id": revert_action_id, "msg": message[:500],
            })
    except Exception:
        log.exception("_finalize_revert_failure | could not mark %s", revert_action_id)


async def get_revert_job_status(revert_action_id: str) -> dict | None:
    """Return the current state of a revert job, or None if it doesn't exist."""
    from app.context import get_org_id
    org_id = get_org_id()
    async with get_db() as db:
        row = (await db.execute(
            _GET_REVERT_JOB, {"id": revert_action_id, "org_id": org_id}
        )).fetchone()
    if row is None:
        return None
    details = row.details or {}
    return {
        "revert_action_id": row.id,
        "status": row.status,
        "summary": row.summary,
        "progress_done": row.progress_done or 0,
        "progress_total": row.progress_total or 0,
        "error_message": row.error_message,
        "target_action_id": details.get("target_action_id"),
        "started_at": row.started_at.isoformat() if row.started_at else None,
        "finished_at": row.finished_at.isoformat() if row.finished_at else None,
    }


async def recover_stale_revert_jobs() -> int:
    """Called at startup: any 'running' revert from a previous process is dead.

    We can't resume an in-flight revert (the worker task is gone), so we mark
    these as 'error' so the client polling them sees a clear failure rather
    than waiting forever. The user can re-trigger the revert — the algorithm
    is idempotent over already-reverted actions.

    Returns the number of rows updated, for logging.
    """
    async with get_db() as db:
        result = await db.execute(_RECOVER_STALE_RUNNING_REVERTS)
        return result.rowcount


__all__ = [
    "begin_action",
    "tracked_action",
    "record_revision",
    "archive_key_for",
    "tracked_write_bytes",
    "tracked_delete",
    "revert_action",
    "start_revert_job",
    "get_revert_job_status",
    "recover_stale_revert_jobs",
    "is_revert_running",
    "prune_expired_revisions",
    "RevertError",
    "RevertJobAlreadyRunning",
]
