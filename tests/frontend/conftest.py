"""Frontend test fixtures.

Requires:
  - Docker (testcontainers Postgres)
  - Playwright (`playwright install chromium`)

Starts the FastAPI app on a random port, runs Alembic migrations against a
throwaway Postgres container, and provides a Playwright `page` fixture.

All tests in this package are skipped when Playwright is unavailable.
"""
import os
import socket
import asyncio
import threading
import pytest

try:
    from playwright.sync_api import sync_playwright
    _HAS_PW = True
except Exception:
    _HAS_PW = False

try:
    from testcontainers.postgres import PostgresContainer
    _HAS_TC = True
except Exception:
    _HAS_TC = False

_SKIP = not (_HAS_PW and _HAS_TC)
pytestmark = pytest.mark.skipif(_SKIP, reason="Playwright or testcontainers not installed")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


# ── Session-scoped server ─────────────────────────────────────────────────

@pytest.fixture(scope="session")
def live_server_url(tmp_path_factory):
    if _SKIP:
        pytest.skip("dependencies missing")

    tmp = tmp_path_factory.mktemp("data")
    port = _free_port()

    with PostgresContainer("postgres:16-alpine") as pg:
        raw = pg.get_connection_url()
        async_url = raw.replace("postgresql+psycopg2://", "postgresql+asyncpg://").replace(
            "postgresql://", "postgresql+asyncpg://"
        )
        sync_url = raw.replace("postgresql+psycopg2://", "postgresql://").replace(
            "postgresql://", "postgresql+asyncpg://", 1  # keep the first occurrence
        )
        # Run migrations
        from alembic import command
        from alembic.config import Config

        os.environ["DATABASE_URL"] = async_url
        cfg = Config("alembic.ini")
        cfg.set_main_option("sqlalchemy.url", raw.replace("postgresql+psycopg2://", "postgresql://"))
        command.upgrade(cfg, "head")

        os.environ["STORAGE_BACKEND"] = "local"
        os.environ["DATA_DIR"] = str(tmp)
        os.environ["AUTH_USERNAME"] = "admin"
        os.environ["AUTH_PASSWORD"] = "admin"

        import uvicorn
        from app.main import app

        config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
        server = uvicorn.Server(config)
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()

        import time
        import httpx
        for _ in range(30):
            try:
                httpx.get(f"http://127.0.0.1:{port}/api/auth/config", timeout=1)
                break
            except Exception:
                time.sleep(0.5)

        yield f"http://127.0.0.1:{port}"
        server.should_exit = True


# ── Playwright browser / page ─────────────────────────────────────────────

@pytest.fixture(scope="session")
def browser(live_server_url):
    with sync_playwright() as pw:
        b = pw.chromium.launch(headless=True)
        yield b
        b.close()


@pytest.fixture
def page(browser, live_server_url):
    ctx = browser.new_context(base_url=live_server_url)
    p = ctx.new_page()
    yield p
    ctx.close()


_DEFAULT_ORG_ID = "00000000-0000-0000-0000-000000000001"


@pytest.fixture
def logged_in_page(page, live_server_url):
    """Page that has already completed the login flow.

    Admin users have no org. The app's initOrgSwitcher normally picks an active
    org, but we also pin the default org into state/localStorage so every api()
    call carries the X-Org-Context header required by org-scoped endpoints.
    """
    page.goto("/")
    page.fill('input[placeholder*="sername"], input[name="username"]', "admin")
    page.fill('input[placeholder*="assword"], input[name="password"]', "admin")
    page.click('button[type="submit"], button:has-text("Login"), button:has-text("Sign in")')
    page.wait_for_selector("#app-root", state="visible")
    # Pin the active org so api() includes X-Org-Context on every call. The org
    # is read from state.activeOrgId at call time (see app.js api()).
    page.evaluate(f"""
        localStorage.setItem('active_org_id', '{_DEFAULT_ORG_ID}');
        state.activeOrgId = '{_DEFAULT_ORG_ID}';
    """)
    return page
