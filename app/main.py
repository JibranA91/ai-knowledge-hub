import sqlalchemy as sa
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.logger import get_logger
from app.routes import admin as admin_router_module
from app.routes import activity, auth, documents, notifications as notifications_router, operations, wiki
from app.services import auth as auth_service
from app.services import jobs as job_store
from app.services import recalibrate_job

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from app import db
    from app.context import set_startup_context
    from app.services import aws_auth, s3
    from app.services.orgs import ensure_default_org_and_admin, DEFAULT_ORG_ID
    from app.services.wiki_db import get_wiki_file, set_wiki_file

    # Resolve every configured model name against the active provider before
    # anything else. A typo here would otherwise stay silent until the first
    # request that happens to use that role, and arrive as an opaque vendor 400.
    from app import model as model_layer
    resolved = model_layer.validate_configuration()
    log.info("startup | llm provider=%s | %s", model_layer.provider_name(),
             " | ".join(f"{role.value}={model_layer.model_name_for(role)}" for role in resolved))

    log.info("startup | running DB migrations")
    await db.run_migrations()
    log.info("startup | migrations complete")
    await job_store.mark_stale_as_error()
    await recalibrate_job.mark_stale_as_error()
    # Start the durable ingest queue: a failover poller + an immediate scan that
    # drains any jobs left 'queued_write' (and reclaims any orphaned 'writing')
    # by a prior process. The queue lives in the DB, so no in-memory re-enqueue.
    from app.services import ingest_queue
    ingest_queue.start()
    log.info("startup | ingest queue started")
    # Background heartbeat that keeps the LISTEN/NOTIFY connection alive so
    # real-time notification delivery self-heals after a connection drop.
    from app.services import notif_stream
    notif_stream.hub().start()
    log.info("startup | notification stream heartbeat started")
    # Any revert task that was running when the previous process died is
    # gone — mark its wiki_actions row 'error' so the client polling it
    # doesn't wait forever. Idempotent: the user can re-trigger the revert.
    from app.services.wiki_state import recover_stale_revert_jobs
    recovered = await recover_stale_revert_jobs()
    if recovered:
        log.info("startup | marked %d stale revert job(s) as error", recovered)
    s3.ensure_bucket()
    log.info("startup | S3 bucket ready")
    await aws_auth.start_refresh_task()
    log.info("startup | AWS credential refresh task started")

    # Seed the default org + admin user, then set context for startup file seeding
    org_id, _admin_id = await ensure_default_org_and_admin(
        settings.AUTH_USERNAME, settings.AUTH_PASSWORD
    )
    log.info("startup | default org seeded | org_id=%s", org_id)
    set_startup_context(org_id)

    # Bundled schema file shipped inside the Docker image at build time.
    # DATA_DIR may point to a mounted volume that shadows ./data/schema/, so we
    # always keep this absolute path as a reliable fallback.
    _BUNDLED_AGENTS_MD = Path(__file__).parent.parent / "data" / "schema" / "AGENTS.md"

    # Only AGENTS.md is seeded — index.md and log.md are derived at query time
    # from wiki_pages and audit_log respectively, so they no longer need a blob.
    if not await get_wiki_file("schema/AGENTS.md"):
        existing = s3.read("schema/AGENTS.md")
        if not existing and _BUNDLED_AGENTS_MD.exists():
            existing = _BUNDLED_AGENTS_MD.read_text(encoding="utf-8")
            log.info("startup | seeded schema/AGENTS.md from bundled image file")
        await set_wiki_file("schema/AGENTS.md", existing or "")
        if existing:
            log.info("startup | migrated schema/AGENTS.md from storage to DB")

    from app.services.graph import get_graph
    log.info("startup | loading wiki graph")
    await get_graph().ensure_loaded()
    log.info("startup | wiki graph ready")

    # Seed built-in permission templates for the default org (idempotent)
    try:
        from app.services.permissions import seed_builtin_templates
        await seed_builtin_templates(org_id)
        log.info("startup | built-in permission templates seeded | org_id=%s", org_id)
    except Exception as exc:
        log.warning("startup | permission template seeding failed: %s", exc)

    import asyncio
    from app.services.wiki_state import prune_expired_revisions
    from app.services.chat_sessions import prune_expired_writer_sessions

    async def _prune_loop() -> None:
        # Run every 24h. First sweep ~60s after startup so a fresh deploy
        # doesn't hammer the DB during boot.
        await asyncio.sleep(60)
        while True:
            try:
                await prune_expired_revisions()
            except Exception as exc:
                log.warning("prune_loop | revisions error: %s", exc)
            try:
                pruned = await prune_expired_writer_sessions(settings.WRITER_DRAFT_TTL_DAYS)
                if pruned:
                    log.info("prune_loop | writer_drafts_pruned=%d", pruned)
            except Exception as exc:
                log.warning("prune_loop | writer drafts error: %s", exc)
            await asyncio.sleep(24 * 60 * 60)

    prune_task = asyncio.create_task(_prune_loop())
    log.info("startup | revision prune loop started")

    log.info("startup | application ready")
    try:
        yield
    finally:
        log.info("shutdown | cancelling prune task")
        prune_task.cancel()
        log.info("shutdown | draining ingest queue workers")
        from app.services import ingest_queue
        await ingest_queue.shutdown()
        log.info("shutdown | closing notification stream listener")
        from app.services import notif_stream
        await notif_stream.hub().shutdown()
        log.info("shutdown | stopping AWS credential refresh task")
        await aws_auth.stop_refresh_task()


app = FastAPI(title=settings.APP_TITLE, lifespan=lifespan)


@app.middleware("http")
async def recalibrate_lock_middleware(request: Request, call_next):
    """Block write operations while recalibration or a revert runs for this org.

    Registered first so auth_middleware (registered second) runs first as the
    outermost wrapper, ensuring request.state.org_id is set before we read it.
    """
    org_id = getattr(request.state, "org_id", "")
    local_running = recalibrate_job.get(org_id).status == "running" if org_id else False
    if not local_running and org_id:
        try:
            local_running = await recalibrate_job.is_running_in_db(org_id)
        except Exception:
            pass

    revert_running = False
    if org_id and not local_running:
        try:
            from app.services.wiki_state import is_revert_running
            revert_running = await is_revert_running(org_id)
        except Exception:
            pass

    if local_running or revert_running:
        path = request.url.path
        passthrough = (
            request.method in ("GET", "HEAD", "OPTIONS")
            or path.startswith("/api/ops/recalibrate")
            or path.startswith("/api/ops/query")
            or path.startswith("/api/ops/chat")
            or path == "/api/ops/writer/chat/stream"
            or path.startswith("/api/auth/")
            or path.startswith("/static/")
            or path == "/"
        )
        if not passthrough:
            reason = "Recalibration" if local_running else "Revert"
            log.warning("write_lock | 503 | org=%s | reason=%s | method=%s | path=%s",
                        org_id, reason, request.method, path)
            return JSONResponse(
                status_code=503,
                content={"detail": f"{reason} in progress. Write operations are temporarily locked."},
            )
    return await call_next(request)


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """Validate JWT access token for all /api/* routes except /api/auth/*.

    Registered second so it runs first (outermost wrapper). Sets
    request.state.org_id before recalibrate_lock_middleware reads it.
    """
    path = request.url.path
    if path.startswith("/api/") and not path.startswith("/api/auth/"):
        token = ""
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
        else:
            log.warning("auth | 401 | missing Bearer token | path=%s", path)
        user_ctx = auth_service.validate_access_token(token)
        if not user_ctx:
            if token:
                log.warning("auth | 401 | invalid or expired token | path=%s", path)
            return JSONResponse(status_code=401, content={"detail": "Unauthorized"})

        # Store in request.state for route handlers (explicit access)
        request.state.user = user_ctx.email
        request.state.user_id = user_ctx.user_id

        # Resolve the ACTIVE org + role from memberships. The token only
        # identifies the user; org/role are authoritative per-request because a
        # user may belong to several orgs with a different role in each. The
        # X-Org-Context header selects which org the request acts on (admins use
        # it to act cross-org; members use it to switch between their orgs).
        requested_org = request.headers.get("X-Org-Context", "") or user_ctx.org_id
        from app.services.orgs import resolve_active_membership
        try:
            active = await resolve_active_membership(user_ctx.user_id, requested_org)
        except Exception:
            # DB hiccup — fall back to hints rather than hard-failing auth. Honor
            # the requested org (X-Org-Context) over the token's baked-in org, so
            # a transient DB error doesn't silently switch a member who has
            # selected a non-home org back onto their stale token org.
            active = {"org_id": requested_org or user_ctx.org_id or "", "role": user_ctx.role}
        org_id = active["org_id"]
        role = active["role"]
        request.state.role = role
        request.state.org_id = org_id

        # Store in ContextVar so the service layer picks it up automatically.
        from app.context import UserContext, current_user
        current_user.set(UserContext(
            user_id=user_ctx.user_id,
            org_id=org_id,
            email=user_ctx.email,
            role=role,
        ))

    return await call_next(request)


from app.routes import org as org_router

app.include_router(auth.router, prefix="/api/auth")
app.include_router(documents.router, prefix="/api/documents")
app.include_router(wiki.router, prefix="/api/wiki")
app.include_router(operations.router, prefix="/api/ops")
app.include_router(activity.router, prefix="/api/activity")
app.include_router(org_router.router, prefix="/api/orgs")
app.include_router(admin_router_module.router, prefix="/api/admin")
app.include_router(notifications_router.router, prefix="/api/notifications")

app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/health")
async def health():
    """Load-balancer health check — verifies DB connectivity."""
    from app.db import get_db
    db_status = "ok"
    try:
        async with get_db() as db:
            await db.execute(sa.text("SELECT 1"))
    except Exception:
        db_status = "fail"
    overall = "ok" if db_status == "ok" else "degraded"
    return {"status": overall, "db": db_status}


@app.get("/admin")
async def admin_dashboard():
    return FileResponse("app/static/admin.html")


@app.get("/")
async def index():
    return FileResponse("app/static/index.html")
