import asyncio
import mimetypes
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File
from fastapi.responses import Response

from app.config import settings
from app.logger import get_logger
from app.services import jobs as job_store
from app.services import s3
from app.services.permissions import check_token_quota, check_upload_quota, require_permission
from app.services.rate_limit import rate_limit_upload
from app.services.wiki_engine import WikiEngine
from app.services.wiki_state import begin_action, tracked_delete, tracked_write_bytes

log = get_logger(__name__)
router = APIRouter(tags=["documents"])
_engine = WikiEngine()

ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx", ".doc"}

_ingest_semaphore: asyncio.Semaphore | None = None


def _get_ingest_semaphore() -> asyncio.Semaphore:
    global _ingest_semaphore
    if _ingest_semaphore is None:
        _ingest_semaphore = asyncio.Semaphore(settings.INGEST_CONCURRENCY)
    return _ingest_semaphore


async def _run_ingest(filename: str) -> None:
    """Planning phase — stops at pending_review for user to approve before writing."""
    job = None
    try:
        job = await job_store.get(filename)
        if not job:
            return
        job.status = "processing"
        await job_store.save(job)
        log.info("Ingest planning started | file=%s", filename)

        plan_result = await _engine.plan(filename)
        job = await job_store.get(filename)
        job.plan = plan_result["plan"]
        job.conflicts = plan_result["conflicts"]
        job.log_entry = plan_result["log_entry"]
        job.doc_text = plan_result["doc_text"]

        # If planning errored AND produced no pages, surface as a proper error
        # rather than dropping the user into an empty review popup.
        plan_errors = plan_result.get("errors", [])
        if plan_errors and not job.plan:
            job.status = "error"
            job.message = "Planning failed: " + "; ".join(plan_errors)
            await job_store.save(job)
            log.error("Plan failed | file=%s | errors=%s", filename, plan_errors)
            if job.user_id:
                from app.services import notif_svc
                await notif_svc.create(
                    user_id=job.user_id,
                    org_id=job.org_id,
                    type="ingest_error",
                    title=f"Plan failed: {filename}",
                    body=job.message[:300],
                    link=filename,
                    metadata={"filename": filename},
                )
            return

        if job.conflicts:
            lines = ["⚠️ **I found conflicting information that needs your input before I can proceed.**\n"]
            for i, c in enumerate(job.conflicts, 1):
                lines.append(
                    f"**Conflict {i} — `{c['path']}`**\n"
                    f"- **Existing wiki says:** {c['existing_claim']}\n"
                    f"- **New document says:** {c['new_claim']}\n"
                )
            lines.append(
                "\nFor each conflict, please tell me which version is accurate, "
                "or whether you'd like to keep both as a 'Conflicting Sources' note."
            )
            job.plan_chat_history.append({"role": "assistant", "content": "\n".join(lines)})

        job.status = "pending_review"
        await job_store.save(job)
        log.info("Plan ready for review | file=%s | pages=%d | conflicts=%d",
                 filename, len(job.plan), len(job.conflicts))
        if job.user_id:
            from app.services import notif_svc
            await notif_svc.create(
                user_id=job.user_id,
                org_id=job.org_id,
                type="pending_review",
                title=f"Review needed: {filename}",
                body=f"The ingest plan for '{filename}' is ready. Please review and approve before writing proceeds.",
                link=filename,
                metadata={"filename": filename, "pages": len(job.plan), "conflicts": len(job.conflicts)},
            )
    except asyncio.CancelledError:
        try:
            job = await job_store.get(filename) if job is None else job
            job.status = "cancelled"
            job.message = "Planning cancelled by user"
            await job_store.save(job)
        except Exception as save_err:
            log.warning("Failed to save cancelled status | file=%s | err=%s", filename, save_err)
        log.info("Ingest planning cancelled | file=%s", filename)
        raise
    except Exception as e:
        try:
            job = (await job_store.get(filename) or job) if job is None else job
            if job:
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
        log.error("Ingest planning failed | file=%s | error=%s", filename, e)


@router.get("")
async def list_documents():
    prefix = s3.org_prefix("raw/")
    files = []
    for key in sorted(s3.list_keys(prefix)):
        # Strip the org-prefixed "raw/" to get just the filename
        name = key[len(prefix):]
        if not name or name.startswith("."):
            continue
        if Path(name).suffix.lower() not in ALLOWED_EXTENSIONS:
            continue
        job = await job_store.get(name)
        files.append({
            "name": name,
            "size": s3.get_object_size(key),
            "modified": s3.get_object_mtime(key),
            "ingest_status": job.status if job else None,
        })
    return files


@router.post(
    "/upload",
    dependencies=[
        Depends(rate_limit_upload),
        Depends(require_permission("can_upload")),
        Depends(check_upload_quota()),
        Depends(check_token_quota()),
    ],
)
async def upload_document(request: Request, file: UploadFile = File(...)):
    if Path(file.filename).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")

    # Honour per-template file-size override when stricter than the global limit
    from app.services.permissions import get_user_permissions
    role = getattr(request.state, "role", "member")
    max_mb = settings.MAX_UPLOAD_SIZE_MB
    if role != "admin":
        user_id = getattr(request.state, "user_id", "")
        org_id  = getattr(request.state, "org_id",  "")
        perms = await get_user_permissions(user_id, org_id)
        tpl_max = perms.get("max_upload_size_mb")
        if tpl_max is not None:
            max_mb = min(max_mb, tpl_max)

    max_bytes = max_mb * 1024 * 1024
    content = await file.read()
    if len(content) > max_bytes:
        raise HTTPException(413, f"File exceeds {max_mb} MB limit")

    safe_name = Path(file.filename).name
    async with begin_action("document_upload", summary=safe_name):
        await tracked_write_bytes(s3.org_prefix(f"raw/{safe_name}"), content)

    log.info("Upload received | file=%s | size=%d bytes", safe_name, len(content))

    job = await job_store.enqueue(safe_name)
    task = asyncio.create_task(run_ingest_queued(safe_name))
    task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
    job_store.set_task(safe_name, task)

    return {"filename": safe_name, "size": len(content)}


async def run_ingest_queued(filename: str) -> None:
    """Wrap _run_ingest with the per-process ingest semaphore.

    Module-level so other route handlers (writer mode) can reuse it
    instead of redefining the closure.
    """
    async with _get_ingest_semaphore():
        await _run_ingest(filename)


@router.get("/{filename}", dependencies=[Depends(require_permission("can_download_files"))])
async def get_document(filename: str):
    key = s3.org_prefix(f"raw/{filename}")
    if not s3.exists(key):
        raise HTTPException(404, "Document not found")
    content_bytes = s3.read_bytes(key)
    mime, _ = mimetypes.guess_type(filename)
    return Response(
        content=content_bytes,
        media_type=mime or "application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.delete("/{filename}", dependencies=[Depends(require_permission("can_delete_files"))])
async def delete_document(filename: str):
    raw_key = s3.org_prefix(f"raw/{filename}")
    if not s3.exists(raw_key):
        raise HTTPException(404, "Document not found")

    async with begin_action("document_delete", summary=filename):
        await tracked_delete(raw_key)

        # Remove any source-summary wiki pages stamped with this filename.
        # Identifies source pages by frontmatter `type: source_summary`, with a
        # path-prefix fallback for legacy pages that pre-date the migration.
        from app.services.wiki_db import list_wiki_pages_with_content, delete_wiki_page
        from app.utils import page_type_or_infer
        removed_source_paths: list[str] = []
        try:
            for rel, text in await list_wiki_pages_with_content():
                if page_type_or_infer(rel, text) != "source_summary":
                    continue
                if f"uploaded_file: {filename}" in text:
                    await delete_wiki_page(rel)
                    removed_source_paths.append(rel)
                    log.info("Source wiki page removed | page=%s | file=%s", rel, filename)
        except Exception as exc:
            log.warning("Could not scan/remove source pages: %s", exc)

    log.info("Document deleted | file=%s | wiki_pages_removed=%d", filename, len(removed_source_paths))
    return {"deleted": filename, "wiki_pages_removed": removed_source_paths}
