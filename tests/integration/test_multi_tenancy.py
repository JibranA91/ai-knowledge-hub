"""Integration tests for Phase 6 multi-tenancy — tenant data isolation.

Two separate orgs are created; each gets its own wiki pages, ingest jobs, and
chat sessions. The tests verify that queries from org A cannot see org B's data.

Requires testcontainers + Docker (skipped otherwise, same as other integration tests).
"""
import pytest
import pytest_asyncio

from app.context import UserContext, current_user
from app.services.orgs import (
    create_org,
    create_user,
    verify_user,
    DEFAULT_ORG_ID,
)
from app.services.auth import create_access_token


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def org_a(patch_app_db):
    """Create org A and return its ID."""
    org_id = await create_org("Org Alpha", "org-alpha")
    await create_user(org_id, "alice@alpha.com", "pw-alpha", role="admin")
    return org_id


@pytest_asyncio.fixture
async def org_b(patch_app_db):
    """Create org B and return its ID."""
    org_id = await create_org("Org Beta", "org-beta")
    await create_user(org_id, "bob@beta.com", "pw-beta", role="admin")
    return org_id


def _set_ctx(org_id: str, email: str = "test@test.com", role: str = "admin") -> None:
    current_user.set(UserContext(
        user_id="00000000-0000-0000-0000-000000000099",
        org_id=org_id,
        email=email,
        role=role,
    ))


# ── Wiki page isolation ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_wiki_page_isolation(org_a, org_b, patch_app_db):
    """Pages written in org A must not appear when querying from org B."""
    from app.services.wiki_db import upsert_wiki_page, get_wiki_page_content, list_wiki_paths

    _set_ctx(org_a)
    await upsert_wiki_page("concepts/alpha-page.md", "# Alpha Page\n\nContent from Alpha.")

    _set_ctx(org_b)
    content = await get_wiki_page_content("concepts/alpha-page.md")
    assert content is None, "Org B must not see Org A's wiki page"

    paths = await list_wiki_paths()
    assert "concepts/alpha-page.md" not in paths


@pytest.mark.asyncio
async def test_each_org_sees_own_pages(org_a, org_b, patch_app_db):
    from app.services.wiki_db import upsert_wiki_page, get_wiki_page_content

    _set_ctx(org_a)
    await upsert_wiki_page("concepts/shared-name.md", "# Alpha's shared-name page")

    _set_ctx(org_b)
    await upsert_wiki_page("concepts/shared-name.md", "# Beta's shared-name page")

    # Org A still sees its own version
    _set_ctx(org_a)
    content_a = await get_wiki_page_content("concepts/shared-name.md")
    assert "Alpha" in content_a

    # Org B sees its own version
    _set_ctx(org_b)
    content_b = await get_wiki_page_content("concepts/shared-name.md")
    assert "Beta" in content_b


# ── Wiki files isolation ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_wiki_files_isolation(org_a, org_b, patch_app_db):
    """index.md is per-org — changes in org A must not affect org B."""
    from app.services.wiki_db import set_wiki_file, get_wiki_file

    _set_ctx(org_a)
    await set_wiki_file("wiki/index.md", "# Alpha Index\n\n- Page A")

    _set_ctx(org_b)
    await set_wiki_file("wiki/index.md", "# Beta Index\n\n- Page B")

    _set_ctx(org_a)
    index_a = await get_wiki_file("wiki/index.md")
    assert "Alpha" in index_a

    _set_ctx(org_b)
    index_b = await get_wiki_file("wiki/index.md")
    assert "Beta" in index_b


# ── Ingest job isolation ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_ingest_job_isolation(org_a, org_b, patch_app_db):
    """Ingest jobs created in org A must not be visible from org B."""
    from app.services.jobs import enqueue, get

    _set_ctx(org_a)
    await enqueue("report.pdf")

    _set_ctx(org_b)
    job = await get("report.pdf")
    assert job is None, "Org B must not see Org A's ingest job"


# ── Chat session isolation ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_chat_session_isolation(org_a, org_b, patch_app_db):
    """A session_id created by org A cannot be retrieved by org B."""
    from app.services.chat_sessions import get_or_create, get_or_create as get_sess

    _set_ctx(org_a)
    session_a = await get_or_create(None)

    _set_ctx(org_b)
    # Attempting to load org A's session_id from org B's context should create a new session
    session_for_b = await get_or_create(session_a.session_id)
    # It must be a DIFFERENT session (new UUID generated) because the session belongs to org A
    assert session_for_b.session_id != session_a.session_id


# ── Search isolation ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_search_isolation(org_a, org_b, patch_app_db):
    from app.services.wiki_db import upsert_wiki_page, search_wiki

    _set_ctx(org_a)
    await upsert_wiki_page("concepts/finance.md", "# Finance\n\nAlpha financial data.")

    _set_ctx(org_b)
    results = await search_wiki("finance")
    paths = [r["path"] for r in results]
    assert "concepts/finance.md" not in paths, "Org B's search must not return Org A's pages"


# ── Login + JWT isolation ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_login_returns_correct_org_id(org_a, org_b, patch_app_db):
    user = await verify_user("alice@alpha.com", "pw-alpha")
    assert user is not None
    assert user["org_id"] == org_a
    assert user["role"] == "admin"

    access_token = create_access_token(
        email=user["email"],
        user_id=user["id"],
        org_id=user["org_id"],
        role=user["role"],
    )
    from app.services.auth import validate_access_token
    ctx = validate_access_token(access_token)
    assert ctx.org_id == org_a
    assert ctx.email == "alice@alpha.com"


@pytest.mark.asyncio
async def test_cross_org_login_forbidden(org_a, org_b, patch_app_db):
    """Alice's credentials must NOT authenticate as Org B's user."""
    # Verify alice can log in
    alice = await verify_user("alice@alpha.com", "pw-alpha")
    assert alice is not None

    # Verify alice cannot be found as a member of org_b
    assert alice["org_id"] == org_a
    assert alice["org_id"] != org_b


# ── Usage log isolation ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_usage_log_isolation(org_a, org_b, patch_app_db):
    from app.services.usage_log import record, get_org_summary

    await record(org_a, "claude-haiku", 100, 200, "ingest")
    await record(org_b, "claude-haiku", 50, 100, "query")

    summary_a = await get_org_summary(org_a, days=7)
    summary_b = await get_org_summary(org_b, days=7)

    total_a = sum(r["total_tokens_in"] for r in summary_a)
    total_b = sum(r["total_tokens_in"] for r in summary_b)

    assert total_a == 100
    assert total_b == 50
