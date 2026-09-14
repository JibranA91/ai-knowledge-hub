"""Integration test fixtures.

Requires Docker for testcontainers PostgreSQL with pgvector, or an explicitly
configured disposable DATABASE_URL with pgvector available.
All tests in this package are skipped when testcontainers is unavailable.

Fixtures:
  postgres_url      — asyncpg URL for the throwaway Postgres container
  test_engine       — SQLAlchemy async engine pointed at the container
  patch_app_db      — autouse: redirects app.db to use the test engine
  clean_db          — autouse: truncates all tables after each test
  mock_bedrock      — autouse: patches BedrockService.converse so no AWS creds needed
  reset_recalibrate — autouse: resets in-memory recalibrate job state after each test
  auth_headers      — {"Authorization": "Bearer <token>"} for a valid test token
  client            — httpx.AsyncClient wired to the FastAPI app
"""
import os
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    async_sessionmaker,
    AsyncSession,
)

# ── Testcontainers availability guard ─────────────────────────────────────

# ── Database URL resolution ───────────────────────────────────────────────
# When DATABASE_URL is already set in the environment (GitHub Actions service
# container, local Postgres, etc.) use it directly and skip testcontainers.
# When it's absent or points to a placeholder, spin up a throwaway container.

# Use a real external DB when DATABASE_URL is explicitly set to a reachable instance
# (GitHub Actions service container, a local Postgres, etc.).
# Fall back to testcontainers when DATABASE_URL is absent or is the unit-test
# placeholder ("…@localhost/x") that the unit job sets.
_CI_DB_URL = os.environ.get("DATABASE_URL", "")
_USE_TC = not _CI_DB_URL or "localhost/x" in _CI_DB_URL

try:
    from testcontainers.postgres import PostgresContainer
    _HAS_TC = True
except Exception:
    _HAS_TC = False

# Skip the whole module only if we need testcontainers but it isn't available.
pytestmark = pytest.mark.skipif(
    _USE_TC and not _HAS_TC,
    reason="testcontainers not installed or Docker unavailable",
)


# ── PostgreSQL container (session-scoped — one per test run) ──────────────

@pytest.fixture(scope="session")
def postgres_container():
    if not _USE_TC:
        yield None
        return
    if not _HAS_TC:
        pytest.skip("testcontainers not available")
    with PostgresContainer("pgvector/pgvector:pg16") as container:
        yield container


@pytest.fixture(scope="session")
def postgres_url(postgres_container):
    if not _USE_TC:
        # Already an asyncpg URL from the environment
        url = _CI_DB_URL
        if not url.startswith("postgresql+asyncpg://"):
            url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
        return url
    raw = postgres_container.get_connection_url()
    return raw.replace("postgresql+psycopg2://", "postgresql+asyncpg://").replace(
        "postgresql://", "postgresql+asyncpg://"
    )


@pytest.fixture(scope="session")
def sync_postgres_url(postgres_container):
    if not _USE_TC:
        url = _CI_DB_URL
        return url.replace("postgresql+asyncpg://", "postgresql://", 1).replace(
            "postgresql+psycopg2://", "postgresql://", 1
        )
    raw = postgres_container.get_connection_url()
    return raw.replace("postgresql+psycopg2://", "postgresql://")


@pytest_asyncio.fixture(scope="session")
async def test_engine(postgres_url):
    engine = create_async_engine(postgres_url, echo=False, pool_pre_ping=True)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture(scope="session")
async def test_session_factory(test_engine):
    return async_sessionmaker(test_engine, expire_on_commit=False, class_=AsyncSession)


# ── Alembic migrations (once per test session) ────────────────────────────

@pytest.fixture(scope="session", autouse=True)
def run_migrations(sync_postgres_url, postgres_url):
    from alembic import command
    from alembic.config import Config

    # Point the app at the test DB for the rest of this process
    os.environ["DATABASE_URL"] = postgres_url
    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", sync_postgres_url)
    command.upgrade(cfg, "head")


# ── Redirect app.db to the test engine ───────────────────────────────────

@pytest_asyncio.fixture(autouse=True)
async def patch_app_db(test_session_factory):
    import app.db as db_module

    original = db_module.AsyncSessionLocal
    db_module.AsyncSessionLocal = test_session_factory
    yield
    db_module.AsyncSessionLocal = original


# ── Table cleanup after every test ────────────────────────────────────────
# Order matters: child tables (FK targets) must be truncated before parents.

_TABLES = [
    "usage_log",
    "rate_limit_counters",
    "auth_tokens",
    "notifications",
    "ingest_jobs",
    "chat_sessions",
    "wiki_links",
    "wiki_pages",
    "audit_log",
    "recalibrate_jobs",
    "wiki_files",
    # org_memberships FKs users/organizations/templates/workspaces — drop first.
    "org_memberships",
    # Phase 7 — order matters: users before workspaces/templates (FK targets)
    "users",
    "workspaces",
    "permission_templates",
    "organizations",
]

_DEFAULT_ORG_ID = "00000000-0000-0000-0000-000000000001"

_WIKI_FILES_DEFAULTS = [
    ("wiki/index.md", "# Wiki Index\n\n"),
    ("wiki/log.md", ""),
    ("schema/AGENTS.md", ""),
]


@pytest_asyncio.fixture(autouse=True)
async def clean_db(test_engine):
    yield
    async with test_engine.begin() as conn:
        for tbl in _TABLES:
            await conn.execute(sa.text(f"DELETE FROM {tbl}"))

        # Re-seed the default org so the auth fixture works.
        # max_tokens_per_day_org mirrors orgs.DEFAULT_ORG_MAX_TOKENS_PER_DAY so
        # integration tests match the production boot-time seed.
        await conn.execute(sa.text("""
            INSERT INTO organizations (id, name, slug, max_tokens_per_day_org)
            VALUES (:id, 'Default Organization', 'default', 10000000)
            ON CONFLICT DO NOTHING
        """), {"id": _DEFAULT_ORG_ID})

        # Re-seed wiki files for the default org
        for key, content in _WIKI_FILES_DEFAULTS:
            await conn.execute(
                sa.text("""
                    INSERT INTO wiki_files (org_id, key, content, updated_at)
                    VALUES (CAST(:org AS UUID), :k, :c, NOW())
                    ON CONFLICT (org_id, key) DO NOTHING
                """),
                {"org": _DEFAULT_ORG_ID, "k": key, "c": content},
            )

        # Re-seed global built-in permission templates (org_id IS NULL since migration 013).
        # These are deleted by the table cleanup above; tests that need
        # templates (e.g. test_update_builtin_template_forbidden) depend on
        # them being present at the start of every test.
        await conn.execute(sa.text("""
            INSERT INTO permission_templates
                (id, org_id, name, description, is_builtin,
                 can_upload, can_upload_writer_draft,
                 can_delete_files, can_download_files,
                 can_view_wiki, can_edit_wiki, can_delete_wiki_pages,
                 can_query, can_chat, can_use_writer,
                 max_uploads_per_day, max_chat_messages_per_day,
                 can_recalibrate, can_run_lint, can_manage_schema,
                 can_view_audit_log, can_view_graph, can_rebuild_graph,
                 can_manage_workspace, can_approve_ingest, can_cancel_ingest)
            VALUES
                ('00000000-0000-0000-0000-000000000010', NULL,
                 'read_only', 'View-only access to wiki and graph', true,
                 false, false, false, false, true, false, false, false, false, false, NULL, NULL,
                 false, false, false, false, true, false, false, false, false),
                ('00000000-0000-0000-0000-000000000011', NULL,
                 'contributor', 'Upload documents, query and chat with the wiki', true,
                 true, true, false, true, true, false, false, true, true, false, 20, 200,
                 false, false, false, false, true, false, false, true, true),
                ('00000000-0000-0000-0000-000000000012', NULL,
                 'power_user', 'Full access except recalibration', true,
                 true, true, true, true, true, true, true, true, true, true, NULL, NULL,
                 false, true, true, true, true, true, true, true, true)
            ON CONFLICT (name) WHERE org_id IS NULL DO NOTHING
        """))


# ── Auth token fixture ────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def default_user(patch_app_db):
    """Ensure the default admin user exists and return {id, org_id, email, role}."""
    from app.services.orgs import ensure_default_org_and_admin
    org_id, user_id = await ensure_default_org_and_admin("admin", "admin")
    return {"id": user_id, "org_id": org_id, "email": "admin", "role": "admin"}


@pytest.fixture
def auth_token(default_user):
    from app.services.auth import create_access_token
    return create_access_token(
        email=default_user["email"],
        user_id=default_user["id"],
        org_id=default_user["org_id"],
        role=default_user["role"],
    )


@pytest.fixture
def auth_headers(auth_token):
    return {"Authorization": f"Bearer {auth_token}"}


@pytest_asyncio.fixture
async def user_ctx(default_user):
    """Set the ContextVar so service functions called directly in tests have an org context."""
    from app.context import UserContext, current_user
    current_user.set(UserContext(
        user_id=default_user["id"],
        org_id=default_user["org_id"],
        email=default_user["email"],
        role=default_user["role"],
    ))


# ── Bedrock mock (autouse) ────────────────────────────────────────────────
# Prevents real AWS credential lookups in every integration test.
# Tests that need to assert on the response body can override the return_value
# via the yielded mock object.

@pytest.fixture(autouse=True)
def mock_bedrock():
    with patch(
        "app.providers.bedrock.BedrockConverseClient.converse",
        new_callable=AsyncMock,
        return_value='{"issues": [], "suggestions": [], "health_score": 100}',
    ) as mock:
        yield mock


# ── Recalibrate job state reset (autouse) ────────────────────────────────
# In-memory _jobs dict persists across tests; reset it so a running status
# from one test cannot bleed into the next (e.g. test_wf22 → test_wf25).

@pytest.fixture(autouse=True)
def reset_recalibrate():
    yield
    from app.services import recalibrate_job as rjob
    for key in list(rjob._jobs):
        rjob._jobs[key].status = "idle"


# ── Ingest queue worker cleanup (autouse) ────────────────────────────────
# The per-org write workers are module-level asyncio tasks bound to the test's
# event loop. Drain them after each test so a worker from one test can't bleed
# into the next (or get orphaned on a closed loop).

@pytest_asyncio.fixture(autouse=True)
async def reset_ingest_queue():
    yield
    from app.services import ingest_queue
    await ingest_queue.shutdown()
    # shutdown() sets _stopped=True; restore the import-time default so a later
    # test file's submit()/_kick() isn't silently no-op'd by leftover state.
    ingest_queue._stopped = False


# ── httpx test client ─────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def client(tmp_path, auth_token, monkeypatch):
    """httpx.AsyncClient wired to the FastAPI app via ASGI transport.

    Startup events are mocked to avoid real AWS/S3 calls.
    The test DB is already wired via patch_app_db (autouse).
    """
    from unittest.mock import AsyncMock

    monkeypatch.setenv("STORAGE_BACKEND", "local")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))

    import app.services.aws_auth as aws_auth_module

    monkeypatch.setattr(aws_auth_module, "start_refresh_task", AsyncMock())
    monkeypatch.setattr(aws_auth_module, "stop_refresh_task", AsyncMock())

    from app import config
    monkeypatch.setattr(config.settings, "STORAGE_BACKEND", "local")
    monkeypatch.setattr(config.settings, "DATA_DIR", str(tmp_path))

    from httpx import AsyncClient, ASGITransport
    from app.main import app

    (tmp_path / "raw").mkdir(parents=True, exist_ok=True)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac
