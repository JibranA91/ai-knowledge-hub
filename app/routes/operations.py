import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

from fastapi import APIRouter, Depends, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from app.sse import sse_response

from app.logger import get_logger
from app.utils import PAGE_TYPES
from app.services import jobs as job_store
from app.services.ingest_agent import IngestCancelledError
from app.services.permissions import (
    check_chat_quota, check_query_quota, check_token_quota, check_upload_quota,
    require_permission,
)
from app.services.rate_limit import rate_limit_chat, rate_limit_query, rate_limit_recalibrate
from app.services import chat_sessions
from app.services import recalibrate_job
from app.services import s3
from app.services.recalibrate_agent import RecalibrateAgentRunner
from app.services.wiki_db import get_wiki_file, set_wiki_file, get_rendered_log, append_audit_log
from app.services.wiki_engine import WikiEngine
from app.services.wiki_state import begin_action, tracked_action

log = get_logger(__name__)
router = APIRouter(tags=["operations"])
_engine = WikiEngine()


class IngestRequest(BaseModel):
    filename: str


class PlanChatRequest(BaseModel):
    message: str


class ApproveIngestRequest(BaseModel):
    notes: str = ""
    plan: list[dict] | None = None


class QueryRequest(BaseModel):
    question: str
    save_to_wiki: bool = False


class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str
    save_to_wiki: bool = False


class SchemaUpdate(BaseModel):
    content: str


class RecalibrateRequest(BaseModel):
    deleted_files: list[str] = []
    fact_instructions: str = ""


@router.get("/permissions")
async def get_permissions(request: Request):
    """Return the effective permission flags for the current user.

    Used by the frontend to show/hide UI elements before any action is attempted.
    Admins and supervisors receive full permissions implicitly.
    """
    from app.services.permissions import get_user_permissions, _ADMIN_FULL
    role = getattr(request.state, "role", "member")
    if role in ("admin", "supervisor"):
        return {"role": role, **{k: v for k, v in _ADMIN_FULL.items() if k != "is_suspended"}}
    user_id = getattr(request.state, "user_id", "")
    org_id  = getattr(request.state, "org_id",  "")
    perms = await get_user_permissions(user_id, org_id)
    return {"role": role, **perms}


@router.post("/ingest", dependencies=[Depends(check_token_quota())])
async def ingest_document(req: IngestRequest):
    if not s3.exists(s3.org_prefix(f"raw/{req.filename}")):
        raise HTTPException(404, f"Document not found: {req.filename}")
    log.info("Manual ingest triggered | file=%s", req.filename)
    result = await _engine.ingest(req.filename)
    return result


@router.get("/status/{filename:path}")
async def ingest_status(filename: str):
    job = await job_store.get(filename)
    if not job:
        raise HTTPException(404, "No ingest job found for this file")
    resp = {
        "filename": job.filename,
        "status": job.status,
        "message": job.message,
        "pages_created": job.pages_created,
        "pages_updated": job.pages_updated,
    }
    if job.status == "pending_review":
        resp["plan"] = job.plan
        resp["plan_chat_history"] = job.plan_chat_history
        if job.conflicts:
            resp["conflicts"] = job.conflicts
    return resp


@router.delete("/ingest/{filename:path}",
               dependencies=[Depends(require_permission("can_cancel_ingest"))])
async def cancel_ingest(filename: str):
    job = await job_store.get(filename)
    if not job:
        raise HTTPException(404, "No ingest job found for this file")
    # Cooperative, replica-safe cancel — works in any state, including 'writing'
    # (the write loop aborts between pages; partial writes stay recoverable via
    # History/revert). Replaces the old hard 409 while writing.
    status_before = job.status
    cancelled = await job_store.request_cancel(filename)
    if not cancelled:
        raise HTTPException(404, "No ingest job found for this file")
    # Only remove the raw upload once nothing is mid-write — a job still writing
    # may re-read it; the worker deletes nothing, so leaving it is safe.
    if status_before != "writing":
        raw_key = s3.org_prefix(f"raw/{filename}")
        if s3.exists(raw_key):
            s3.delete(raw_key)
    log.info("Ingest cancel requested | file=%s | status_before=%s", filename, status_before)
    return {"cancelled": filename}


async def _execute_ingest(filename: str, user_notes: str = "") -> None:
    job = await job_store.get(filename)
    if not job:
        return
    job.status = "writing"
    await job_store.save(job)
    log.info("Ingest execution started | file=%s | notes=%r", filename, user_notes[:80] if user_notes else "")
    try:
        async with begin_action("upload_ingest", summary=filename,
                                details={"filename": filename}):
            result = await _engine.execute_ingest(
                filename=filename,
                plan=job.plan,
                log_entry=job.log_entry,
                user_notes=user_notes,
                doc_text=job.doc_text,
            )
        job = await job_store.get(filename) or job
        job.status = "done"
        job.pages_created = result["pages_created"]
        job.pages_updated = result["pages_updated"]
        job.message = result.get("log_entry", "")
        job.plan = []
        job.doc_text = ""
        await job_store.save(job)
        log.info("Ingest complete | file=%s | created=%d | updated=%d",
                 filename, len(job.pages_created), len(job.pages_updated))
        from app.services import notif_svc
        notif_title = f"Ingest complete: {filename}"
        notif_body  = f"Created {len(job.pages_created)}, updated {len(job.pages_updated)} wiki pages."
        notif_meta  = {"filename": filename, "pages_created": len(job.pages_created), "pages_updated": len(job.pages_updated)}
        if job.user_id:
            await notif_svc.create(
                user_id=job.user_id,
                org_id=job.org_id,
                type="ingest_done",
                title=notif_title,
                body=notif_body,
                link=filename,
                metadata=notif_meta,
            )
        await notif_svc.create_for_supervisors(
            org_id=job.org_id,
            type="ingest_done",
            title=notif_title,
            body=notif_body,
            link=filename,
            metadata=notif_meta,
            exclude_user_id=job.user_id or "",
        )
    except IngestCancelledError:
        # Cooperative cancel mid-write (the job's cancel flag was set). The job
        # is already flagged; mark it cancelled and return normally so the
        # per-org queue worker keeps running for the next job. Pages written
        # before the abort remain recoverable via History/revert.
        try:
            job = await job_store.get(filename) or job
            job.status = "cancelled"
            job.message = "Ingest cancelled by user"
            await job_store.save(job)
        except Exception as save_err:
            log.warning("Failed to save cancelled status | file=%s | err=%s", filename, save_err)
        log.info("Ingest execution cancelled (cooperative) | file=%s", filename)
        return
    except asyncio.CancelledError:
        try:
            job = await job_store.get(filename) or job
            job.status = "cancelled"
            job.message = "Ingest cancelled by user"
            await job_store.save(job)
        except Exception as save_err:
            log.warning("Failed to save cancelled status | file=%s | err=%s", filename, save_err)
        log.info("Ingest execution cancelled | file=%s", filename)
        raise
    except Exception as e:
        try:
            job = await job_store.get(filename) or job
            job.status = "error"
            job.message = str(e)
            await job_store.save(job)
            if job.user_id:
                from app.services import notif_svc
                await notif_svc.create(
                    user_id=job.user_id,
                    org_id=job.org_id,
                    type="ingest_error",
                    title=f"Ingest failed: {filename}",
                    body=str(e),
                    link=filename,
                    metadata={"filename": filename},
                )
        except Exception as save_err:
            log.warning("Failed to save error status | file=%s | err=%s", filename, save_err)
        log.error("Ingest execution failed | file=%s | error=%s", filename, e)


@router.post("/ingest/{filename:path}/plan-chat",
             dependencies=[Depends(require_permission("can_approve_ingest")), Depends(check_token_quota())])
async def plan_chat(filename: str, req: PlanChatRequest):
    job = await job_store.get(filename)
    if not job or job.status != "pending_review":
        raise HTTPException(400, "No plan pending review for this file")
    if not req.message.strip():
        raise HTTPException(400, "Message cannot be empty")

    result = await _engine.plan_chat(
        filename=filename,
        message=req.message.strip(),
        current_plan=job.plan,
        history=job.plan_chat_history,
        doc_text=job.doc_text,
        conflicts=job.conflicts,
    )

    job.plan_chat_history.append({"role": "user", "content": req.message.strip()})
    job.plan_chat_history.append({"role": "assistant", "content": result["reply"]})

    if result.get("updated_plan") is not None:
        job.plan = result["updated_plan"]
    if result.get("log_entry"):
        job.log_entry = result["log_entry"]

    await job_store.save(job)
    log.info("Plan chat | file=%s | plan_updated=%s", filename, result.get("updated_plan") is not None)
    return {"reply": result["reply"], "plan": job.plan}


@router.post("/ingest/{filename:path}/approve",
             dependencies=[Depends(require_permission("can_approve_ingest")), Depends(check_token_quota())])
async def approve_ingest(filename: str, req: ApproveIngestRequest):
    job = await job_store.get(filename)
    if not job or job.status != "pending_review":
        raise HTTPException(400, "No plan pending review for this file")
    if req.plan is not None:
        job.plan = req.plan
        await job_store.save(job)  # persist the edited plan before queueing
    # Hand the write phase to the org's durable, serial queue. submit() flips the
    # job to 'queued_write' (persisting notes + queue time) and kicks the org's
    # drain; the drain claims it FIFO under a per-org advisory lock so writes are
    # serialised per org (fixes the same-page lost-update race) and survive
    # restarts / move across replicas.
    from app.services import ingest_queue
    await ingest_queue.submit(filename, req.notes, job.org_id, job.user_id)
    log.info("Ingest approved | file=%s | pages=%d | has_notes=%s | queued_write", filename, len(job.plan), bool(req.notes))
    if job.user_id:
        from app.services import notif_svc
        await notif_svc.mark_read_by_link(job.user_id, filename)
    return {"approved": filename}


@router.post("/ingest/{filename:path}/reject",
             dependencies=[Depends(require_permission("can_approve_ingest"))])
async def reject_ingest(filename: str):
    job = await job_store.get(filename)
    if not job or job.status != "pending_review":
        raise HTTPException(400, "No plan pending review for this file")
    job.status = "cancelled"
    job.message = "Plan rejected by user"
    await job_store.save(job)
    raw_key = s3.org_prefix(f"raw/{filename}")
    if s3.exists(raw_key):
        s3.delete(raw_key)
    log.info("Ingest plan rejected | file=%s", filename)
    return {"rejected": filename}


@router.post("/query", dependencies=[Depends(rate_limit_query), Depends(require_permission("can_query")), Depends(check_query_quota()), Depends(check_token_quota())])
async def query_wiki(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty")
    log.info("Query | question=%r | save=%s", req.question[:80], req.save_to_wiki)
    result = await _engine.query(req.question.strip(), req.save_to_wiki)
    log.info("Query complete | sources=%d | saved_to=%s", len(result.get("sources", [])), result.get("saved_to"))
    return result


@router.post("/chat", dependencies=[Depends(rate_limit_chat), Depends(require_permission("can_chat")), Depends(check_chat_quota()), Depends(check_token_quota())])
async def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(400, "Message cannot be empty")
    log.info("Chat | session=%s | message=%r", req.session_id, req.message[:80])
    result = await _engine.chat(req.session_id, req.message.strip(), req.save_to_wiki)
    return result


@router.post("/chat/stream", dependencies=[Depends(rate_limit_chat), Depends(require_permission("can_chat")), Depends(check_chat_quota()), Depends(check_token_quota())])
async def chat_stream(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(400, "Message cannot be empty")
    log.info("Chat stream | session=%s | message=%r", req.session_id, req.message[:80])
    return sse_response(
        _engine.chat_stream(req.session_id, req.message.strip(), req.save_to_wiki)
    )


@router.get("/chat/{session_id}/history")
async def get_chat_history(session_id: str):
    """Return the conversation messages for a session so the frontend can
    rehydrate the chat panel after a reload (or when resuming a writer draft).
    Works for both `mode='chat'` and `mode='writer'` sessions — same backing store.
    """
    from app.context import get_org_id
    import sqlalchemy as sa
    from app.db import get_db
    org_id = get_org_id()
    async with get_db() as db:
        result = await db.execute(sa.text("""
            SELECT messages, summary, mode FROM chat_sessions
            WHERE org_id = CAST(:org_id AS UUID) AND session_id = :session_id
        """), {"org_id": org_id, "session_id": session_id})
        row = result.fetchone()
    if not row:
        raise HTTPException(404, "Chat session not found")
    return {
        "messages": row.messages or [],
        "summary": row.summary or "",
        "mode": row.mode or "chat",
    }


@router.delete("/chat/{session_id}")
async def clear_chat(session_id: str):
    await chat_sessions.clear(session_id)
    return {"cleared": session_id}


@router.post("/lint", dependencies=[Depends(require_permission("can_run_lint")), Depends(check_token_quota())])
async def lint_wiki():
    log.info("Lint requested")
    result = await _engine.lint()
    log.info("Lint complete | score=%s | issues=%d", result.get("health_score"), len(result.get("issues", [])))
    return result


@router.get("/log")
async def get_log(limit: int = 100):
    """Return the most recent audit log entries rendered as markdown."""
    content = await get_rendered_log(limit=limit)
    return {"content": content}


@router.get("/schema", dependencies=[Depends(require_permission("can_manage_schema"))])
async def get_schema():
    content = await get_wiki_file("schema/AGENTS.md") or ""
    return {"content": content}


@router.get("/schema/default", dependencies=[Depends(require_permission("can_manage_schema"))])
async def get_schema_default():
    """Return the bundled default AGENTS.md shipped with the image."""
    from pathlib import Path
    bundled = Path(__file__).parent.parent.parent / "data" / "schema" / "AGENTS.md"
    if bundled.exists():
        return {"content": bundled.read_text(encoding="utf-8")}
    return {"content": ""}


# ── Schema acceptance rules ──────────────────────────────────────────────────
# Single source of truth: both validate_schema (enforcement) and the
# GET /schema/rules endpoint (which feeds the admin UI's guidance panel) read
# from SCHEMA_RULES, so the rules shown to users can never drift from what's
# actually enforced. Each rule's `check(content, current)` returns failure
# messages (empty = passed); `description` is the human guidance (backtick spans
# render as inline code in the UI).

_REQUIRED_SECTIONS = [
    "## Page Types",
    "## Directory Structure",
    "## Page Naming",
    "## Cross-linking",
    "## Audit Log Entry Format",
    "## Ingest Checklist",
    "## Contradictions",
]


@dataclass(frozen=True)
class _SchemaRule:
    severity: str   # "error" → blocks the save | "warning" → advisory
    description: str
    check: Callable[[str, str], list[str]]


def _rule_top_heading(content: str, current: str) -> list[str]:
    first_line = content.splitlines()[0] if content else ""
    if not first_line.startswith("# "):
        return ["File must start with a top-level heading (`# ...`)."]
    return []


def _rule_required_sections(content: str, current: str) -> list[str]:
    return [f"Required section missing: `{s}`" for s in _REQUIRED_SECTIONS if s not in content]


def _rule_page_types(content: str, current: str) -> list[str]:
    # The planner picks one of these values for every page it plans, and the
    # server keys provenance/cleanup/recalibration off them. If the schema
    # doesn't document a value, the LLM may emit one the server can't recognise.
    out: list[str] = []
    for type_name in sorted(PAGE_TYPES):
        if type_name not in content:
            out.append(
                f"Required page type `{type_name}` is not documented in the schema. "
                f"The server recognises this value in page frontmatter to drive provenance "
                f"and cleanup — orgs may rename folders but must keep the type vocabulary intact."
            )
    return out


def _rule_min_length(content: str, current: str) -> list[str]:
    if current and len(content) < len(current) * 0.5:
        return [
            f"Content is more than 50% shorter than the current schema "
            f"({len(content)} vs {len(current)} chars). This looks like an accidental deletion."
        ]
    return []


def _rule_fences(content: str, current: str) -> list[str]:
    if content.count("\n```") % 2 != 0:
        return ["Odd number of ``` fences detected — a code block may be unclosed."]
    return []


def _rule_cross_link(content: str, current: str) -> list[str]:
    if "cross-link" not in content.lower():
        return ["No mention of cross-linking — ingest agent may stop inserting `[[wikilinks]]` between pages."]
    return []


def _rule_long_lines(content: str, current: str) -> list[str]:
    for i, line in enumerate(content.splitlines(), 1):
        if len(line) > 500:
            return [f"Line {i} is {len(line)} characters long — this may be a paste error."]
    return []


SCHEMA_RULES: list[_SchemaRule] = [
    _SchemaRule(
        "error",
        "The file must start with a top-level heading (`# …`).",
        _rule_top_heading,
    ),
    _SchemaRule(
        "error",
        "Must contain every required section heading, spelled exactly: "
        + ", ".join(f"`{s}`" for s in _REQUIRED_SECTIONS) + ".",
        _rule_required_sections,
    ),
    _SchemaRule(
        "error",
        "Must document all page-type keywords the server keys off in page frontmatter: "
        + ", ".join(f"`{t}`" for t in sorted(PAGE_TYPES))
        + ". You can rename folders/directories freely, but these type names must stay intact.",
        _rule_page_types,
    ),
    _SchemaRule(
        "error",
        "Must not be more than 50% shorter than the currently-saved schema — this guards against "
        "an accidental deletion. For a large rewrite, trim in stages or revert to default first.",
        _rule_min_length,
    ),
    _SchemaRule(
        "warning",
        "Close every code fence — an odd number of fence markers means a code block is left open.",
        _rule_fences,
    ),
    _SchemaRule(
        "warning",
        "Keep cross-linking guidance — if the text never mentions “cross-link”, the ingest agent "
        "may stop inserting `[[wikilinks]]` between pages.",
        _rule_cross_link,
    ),
    _SchemaRule(
        "warning",
        "Avoid very long lines — any single line over 500 characters is flagged as a likely paste error.",
        _rule_long_lines,
    ),
]


@router.post("/schema/validate", dependencies=[Depends(require_permission("can_manage_schema"))])
async def validate_schema(update: SchemaUpdate):
    """Validate a candidate AGENTS.md before saving.

    Runs every rule in SCHEMA_RULES. Errors block the save (the frontend keeps
    the Save button disabled while any exist); warnings are advisory. The
    frontend shows this report before the user confirms the save.
    """
    content = update.content.strip()
    current = await get_wiki_file("schema/AGENTS.md") or ""
    errors: list[str] = []
    warnings: list[str] = []
    for rule in SCHEMA_RULES:
        messages = rule.check(content, current)
        (errors if rule.severity == "error" else warnings).extend(messages)

    log.info("Schema validate | errors=%d | warnings=%d", len(errors), len(warnings))
    return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}


@router.get("/schema/rules", dependencies=[Depends(require_permission("can_manage_schema"))])
async def schema_rules():
    """Describe the acceptance rules so the admin UI can render guidance that
    can't drift from the validator — both read from SCHEMA_RULES."""
    return {"rules": [{"severity": r.severity, "description": r.description} for r in SCHEMA_RULES]}


@router.put("/schema", dependencies=[Depends(require_permission("can_manage_schema"))])
@tracked_action("schema_update", summary_fn=lambda request, update: "schema (AGENTS.md) updated")
async def update_schema(request: Request, update: SchemaUpdate):
    old_content = await get_wiki_file("schema/AGENTS.md") or ""
    await set_wiki_file("schema/AGENTS.md", update.content)
    log.info("Wiki schema updated")
    actor = getattr(request.state, "user", getattr(request.state, "user_id", "unknown"))
    await append_audit_log(
        operation="schema_updated",
        raw_text=f"Wiki schema (AGENTS.md) updated by {actor}",
        details={
            "actor": actor,
            "chars_before": len(old_content),
            "chars_after": len(update.content),
        },
    )
    return {"updated": True}


@router.get("/schema/orgs")
async def list_org_schemas(request: Request):
    """Admin-only: return AGENTS.md content for every organization."""
    role = getattr(request.state, "role", "member")
    if role != "admin":
        raise HTTPException(403, "Admin access required")
    import sqlalchemy as _sa
    from app.services.wiki_db import get_db
    async with get_db() as db:
        rows = await db.execute(_sa.text("""
            SELECT o.id AS org_id, o.name AS org_name, wf.content
            FROM organizations o
            LEFT JOIN wiki_files wf
                   ON wf.org_id = o.id AND wf.key = 'schema/AGENTS.md'
            ORDER BY o.name
        """))
        orgs = rows.fetchall()
    return [
        {"org_id": str(r.org_id), "org_name": r.org_name, "content": r.content or ""}
        for r in orgs
    ]


async def _release_recalibrate_lock(org_id: str, job) -> None:
    """Mark the recalibrate_jobs DB row finished when the run crashed/was cancelled.

    On the happy path `finalize_node` persists the terminal status, but if any
    earlier node raises (or the task is cancelled) `finalize` never runs and the
    row is left 'running' — which keeps the write-lock middleware blocking every
    write for the org until the next restart's stale-sweep. We close that gap
    here. Best-effort: a failure to persist must not mask the original error.
    """
    if job is None or getattr(job, "_db_id", None) is None:
        return
    try:
        await recalibrate_job.persist_finish("error", org_id)
    except Exception:
        log.exception("Recalibrate | failed to release DB lock for org=%s", org_id)


@router.post("/recalibrate", status_code=202,
             dependencies=[Depends(rate_limit_recalibrate), Depends(require_permission("can_recalibrate")), Depends(check_token_quota())])
async def start_recalibrate(request: Request, req: RecalibrateRequest = None):
    if req is None:
        req = RecalibrateRequest()
    org_id = getattr(request.state, "org_id", "")
    job = recalibrate_job.get(org_id)
    if job.status == "running":
        raise HTTPException(409, "Recalibration is already running")

    recalibrate_job.reset(org_id)
    job = recalibrate_job.get(org_id)
    job.status = "running"
    job.stage = "Starting"
    job.progress = 0
    job.fact_instructions = req.fact_instructions
    job.started_at = datetime.now(timezone.utc).isoformat()

    async def _run():
        try:
            await recalibrate_job.persist_start(org_id)
            runner = RecalibrateAgentRunner()
            await runner.run(
                deleted_files=req.deleted_files,
                fact_instructions=req.fact_instructions,
            )
            from app.services.graph import get_graph
            await get_graph().rebuild()
            log.info("Recalibrate | knowledge graph rebuilt")
            j = recalibrate_job.get(org_id)
            parts = []
            if j.pages_improved: parts.append(f"{len(j.pages_improved)} improved")
            if j.pages_deleted:  parts.append(f"{len(j.pages_deleted)} deleted")
            if j.pages_renamed:  parts.append(f"{len(j.pages_renamed)} renamed")
            if j.errors:         parts.append(f"{len(j.errors)} errors")
            targeted = bool(req.fact_instructions)
            from app.services import notif_svc
            await notif_svc.create_for_supervisors(
                org_id=org_id,
                type="recalib_done",
                title="Targeted recalibration complete" if targeted else "Master recalibration complete",
                body=", ".join(parts) or "No changes made.",
                metadata={
                    "pages_improved": len(j.pages_improved),
                    "pages_deleted": len(j.pages_deleted),
                    "pages_renamed": len(j.pages_renamed),
                    "errors": len(j.errors),
                    "targeted": targeted,
                    "fact_instructions": req.fact_instructions,
                },
            )
        except asyncio.CancelledError:
            j = recalibrate_job.get(org_id)
            j.status = "error"
            j.details = "Cancelled"
            j.finished_at = datetime.now(timezone.utc).isoformat()
            # Clear the DB lock — otherwise the recalibrate_jobs row stays
            # 'running' and the write-lock middleware blocks the org until restart.
            await _release_recalibrate_lock(org_id, j)
        except Exception as exc:
            log.error("Recalibrate | unhandled exception: %s", exc)
            j = recalibrate_job.get(org_id)
            j.status = "error"
            j.details = str(exc)
            j.errors.append(str(exc))
            j.finished_at = datetime.now(timezone.utc).isoformat()
            await _release_recalibrate_lock(org_id, j)

    task = asyncio.create_task(_run())
    recalibrate_job.get(org_id).task = task
    log.info("Recalibration started | org=%s", org_id)
    return {"started": True, "status": "running"}


@router.get("/graph", dependencies=[Depends(require_permission("can_view_graph"))])
async def get_graph_data(clusters: bool = False):
    """Return graph data.

    - `clusters=false` (default): full node/edge list for the pyvis visualisation.
    - `clusters=true`: Louvain cluster summary — lighter payload for large wikis.
    """
    from app.services.graph import get_graph
    graph = await get_graph().ensure_loaded()
    if clusters:
        return graph.get_clusters()
    return graph.as_dict()


@router.get("/graph/html", response_class=HTMLResponse,
            dependencies=[Depends(require_permission("can_view_graph"))])
async def get_graph_html():
    from app.services.graph import get_graph
    graph = await get_graph().ensure_loaded()
    return HTMLResponse(content=graph.generate_html())


@router.post("/graph/rebuild", dependencies=[Depends(require_permission("can_rebuild_graph"))])
async def rebuild_graph():
    from app.services.graph import get_graph
    graph = get_graph()
    await graph.rebuild()
    data = graph.as_dict()
    log.info("Graph rebuilt | pages=%d edges=%d", len(data["nodes"]), len(data["edges"]))
    return {"nodes": len(data["nodes"]), "edges": len(data["edges"])}


@router.get("/export")
async def export_wiki(request: Request, include_embeddings: bool = True):
    role = getattr(request.state, "role", "member")
    if role not in ("admin", "supervisor"):
        raise HTTPException(403, "Supervisor or admin access required")

    zip_bytes, page_count, has_embeddings = await _engine.build_wiki_export(include_embeddings)
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    log.info("Wiki export | role=%s | embeddings=%s | pages=%d", role, has_embeddings, page_count)
    await append_audit_log(
        operation="wiki_export",
        raw_text=f"wiki_export | role={role} | pages={page_count} | embeddings={has_embeddings}",
        details={"role": role, "page_count": page_count, "has_embeddings": has_embeddings},
    )
    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=wiki-export-{date_str}.zip"},
    )


@router.post("/import")
async def import_wiki(request: Request, file: UploadFile = File(...)):
    """Import a wiki export bundle into the current org. Empty-org only.

    Admin or supervisor of the active org. The bundle's AGENTS.md (if present)
    overwrites the org's schema; its pages are added and the graph rebuilt.
    """
    role = getattr(request.state, "role", "member")
    if role not in ("admin", "supervisor"):
        raise HTTPException(403, "Supervisor or admin access required")
    org_id = getattr(request.state, "org_id", "")
    if not org_id:
        raise HTTPException(400, "No organization context")

    from app.services.wiki_db import list_wiki_paths
    if await list_wiki_paths():
        raise HTTPException(
            409,
            "Import is only allowed into an empty wiki. This organization already "
            "has pages — create or select an empty organization to import into.",
        )

    zip_bytes = await file.read()
    if not zip_bytes:
        raise HTTPException(400, "Uploaded file is empty")

    from app.services import wiki_import
    try:
        result = await wiki_import.import_bundle(zip_bytes)
    except wiki_import.WikiImportError as e:
        raise HTTPException(400, str(e))

    log.info("Wiki import | role=%s | pages=%d | schema=%s | embeddings_reused=%d",
             role, result["pages_imported"], result["schema_imported"], result["embeddings_reused"])
    await append_audit_log(
        operation="wiki_import",
        raw_text=f"wiki_import | role={role} | pages={result['pages_imported']} | schema={result['schema_imported']}",
        details=result,
    )
    return result


# ── Inline AI editing ──────────────────────────────────────────────────────

class WikiEditRequest(BaseModel):
    path: str
    scope: str = "page"            # "page" | "section"
    action: str = "improve"        # expand | summarize | improve | reconcile | custom
    heading: str | None = None     # required when scope == "section"
    instruction: str = ""          # free-text directive (required for action="custom")
    reconcile_with: str | None = None  # other page path for action="reconcile"


@router.post(
    "/wiki/edit/stream",
    dependencies=[
        Depends(rate_limit_chat),
        Depends(require_permission("can_edit_wiki")),
        Depends(check_token_quota()),
    ],
)
async def wiki_edit_stream(req: WikiEditRequest, request: Request):
    """Stream an AI-proposed rewrite of a page/section. Read-only: the client
    applies the accepted result via PUT /api/wiki/{path} (tracked + revertible)."""
    if not req.path.strip():
        raise HTTPException(400, "A page path is required")
    if req.scope == "section" and not (req.heading or "").strip():
        raise HTTPException(400, "A section heading is required for a section edit")
    if req.action == "custom" and not req.instruction.strip():
        raise HTTPException(400, "A custom edit needs an instruction")
    if req.action == "reconcile" and not (req.reconcile_with or "").strip():
        raise HTTPException(400, "Reconcile needs another page to reconcile with")
    log.info("Wiki AI edit | path=%s | scope=%s | action=%s", req.path, req.scope, req.action)
    return sse_response(
        _engine.edit_stream(
            path=req.path.strip(),
            scope=req.scope,
            action=req.action,
            heading=(req.heading or None),
            instruction=req.instruction,
            reconcile_with=(req.reconcile_with or None),
        )
    )


# ── Writer Mode ──────────────────────────────────────────────────────────

class WriterChatRequest(BaseModel):
    session_id: str | None = None
    message: str


class WriterFilenameUpdate(BaseModel):
    filename: str


class WriterIngestRequest(BaseModel):
    filename: str


async def _require_writer_role(request: Request) -> None:
    """Allow admins/supervisors unconditionally; members need the can_use_writer flag."""
    role = getattr(request.state, "role", "member")
    if role in ("admin", "supervisor"):
        return
    user_id = getattr(request.state, "user_id", "")
    org_id  = getattr(request.state, "org_id",  "")
    from app.services.permissions import get_user_permissions
    perms = await get_user_permissions(user_id, org_id)
    if perms.get("is_suspended", False):
        raise HTTPException(403, "Your account has been suspended")
    if not perms.get("can_use_writer", False):
        raise HTTPException(403, "Writer mode is not enabled for your account")


def _validate_md_filename(name: str) -> str:
    name = (name or "").strip()
    if not name:
        raise HTTPException(400, "Filename cannot be empty")
    if len(name) > 200:
        raise HTTPException(400, "Filename is too long (max 200 characters)")
    if ".." in name or "/" in name or "\\" in name:
        raise HTTPException(400, "Filename must not contain path separators")
    if not name.lower().endswith(".md"):
        raise HTTPException(400, "Filename must end in .md")
    return name


@router.get("/writer/sessions")
async def writer_list_sessions(request: Request):
    from app.config import settings
    await _require_writer_role(request)
    user_id = getattr(request.state, "user_id", "")
    if not user_id:
        raise HTTPException(400, "No user context")
    items = await chat_sessions.list_writer_sessions(user_id, settings.WRITER_DRAFT_TTL_DAYS)
    return {"sessions": items}


@router.post(
    "/writer/chat/stream",
    dependencies=[Depends(rate_limit_chat), Depends(check_chat_quota()), Depends(check_token_quota())],
)
async def writer_chat_stream(req: WriterChatRequest, request: Request):
    await _require_writer_role(request)
    if not req.message.strip():
        raise HTTPException(400, "Message cannot be empty")
    log.info("Writer chat stream | session=%s | message=%r", req.session_id, req.message[:80])
    return sse_response(
        _engine.writer_chat_stream(req.session_id, req.message.strip())
    )


@router.get("/writer/{session_id}/draft")
async def writer_get_draft(session_id: str, request: Request):
    await _require_writer_role(request)
    draft = await chat_sessions.get_draft(session_id)
    if draft is None:
        raise HTTPException(404, "Draft not found")
    return draft


@router.put("/writer/{session_id}/draft/filename")
async def writer_set_filename(session_id: str, body: WriterFilenameUpdate, request: Request):
    await _require_writer_role(request)
    safe = _validate_md_filename(body.filename)
    session = await chat_sessions.get_or_create(session_id, mode="writer")
    session.draft_filename = safe
    await chat_sessions.save(session)
    return {"filename": safe}


@router.post(
    "/writer/{session_id}/ingest",
    dependencies=[
        Depends(require_permission("can_upload_writer_draft")),
        Depends(check_upload_quota()),
        Depends(check_token_quota()),
    ],
)
async def writer_ingest(session_id: str, body: WriterIngestRequest, request: Request):
    await _require_writer_role(request)
    safe_name = _validate_md_filename(body.filename)

    draft = await chat_sessions.get_draft(session_id)
    if draft is None:
        raise HTTPException(404, "Draft session not found")
    content = draft["draft_content"]
    if not content.strip():
        raise HTTPException(400, "Draft is empty — ask the writer agent to produce a draft first")
    if not draft.get("draft_ready", False):
        raise HTTPException(
            409,
            "The writer agent has not marked this draft as ready. "
            "Continue the conversation until the agent confirms the draft is complete; "
            "the 'Save & Ingest' button will be enabled once the agent signals readiness.",
        )

    key = s3.org_prefix(f"raw/{safe_name}")
    if s3.exists(key):
        raise HTTPException(409, f"A file named '{safe_name}' already exists. Please choose a different name.")

    async with begin_action("writer_ingest", summary=safe_name):
        from app.services.wiki_state import tracked_write_bytes
        await tracked_write_bytes(key, content.encode("utf-8"))

    job = await job_store.enqueue(safe_name)

    from app.routes.documents import run_ingest_queued
    task = asyncio.create_task(run_ingest_queued(safe_name))
    task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
    job_store.set_task(safe_name, task)

    await append_audit_log(
        "writer_ingest",
        f"file={safe_name} | session={session_id}",
        {"filename": safe_name, "session_id": session_id, "chars": len(content)},
    )

    return {"filename": safe_name, "job_id": getattr(job, "id", None)}


@router.delete("/writer/{session_id}")
async def writer_delete_draft(session_id: str, request: Request):
    await _require_writer_role(request)
    deleted = await chat_sessions.clear(session_id)
    if not deleted:
        raise HTTPException(404, "Draft not found")
    await append_audit_log(
        "writer_draft_deleted",
        f"session={session_id}",
        {"session_id": session_id},
    )
    return {"deleted": True}


@router.get("/recalibrate/status")
async def recalibrate_status(request: Request):
    org_id = getattr(request.state, "org_id", "")
    job = recalibrate_job.get(org_id)
    return {
        "status": job.status,
        "stage": job.stage,
        "progress": job.progress,
        "details": job.details,
        "fact_instructions": job.fact_instructions,
        "pages_improved": job.pages_improved,
        "pages_deleted": job.pages_deleted,
        "pages_renamed": job.pages_renamed,
        "errors": job.errors,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
    }
