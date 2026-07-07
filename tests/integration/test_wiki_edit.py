"""Integration tests for the inline AI edit endpoint (POST /api/ops/wiki/edit/stream).

The endpoint streams an AI-proposed rewrite; it never writes the page (the client
applies the result via PUT /api/wiki/{path}). Bedrock streaming is mocked.
"""
import json

import pytest
import pytest_asyncio
from unittest.mock import patch

from app.services.bedrock import BedrockService

_DEFAULT_ORG_ID = "00000000-0000-0000-0000-000000000001"

_PAGE = "---\ntitle: Widget\n---\n# Widget\n\nA widget does things.\n\n## Background\n\nOld text.\n"


@pytest_asyncio.fixture
async def member_headers(default_user):
    """A plain member (no template → read_only fallback, so no can_edit_wiki)."""
    from app.services.orgs import create_user
    from app.services.auth import create_access_token
    uid = await create_user(_DEFAULT_ORG_ID, "editmember@test.com", "password123", role="member")
    token = create_access_token(email="editmember@test.com", user_id=uid,
                                org_id=_DEFAULT_ORG_ID, role="member")
    return {"Authorization": f"Bearer {token}"}


def _fake_stream_factory(chunks):
    async def _fake(self, system_prompt, messages, max_tokens=4096, operation=""):
        for c in chunks:
            yield c
    return _fake


def _parse_sse(text):
    return [json.loads(line[len("data: "):]) for line in text.splitlines()
            if line.startswith("data: ")]


# ── validation ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_edit_section_without_heading_is_400(client, auth_headers):
    r = await client.post("/api/ops/wiki/edit/stream",
                          json={"path": "concepts/x.md", "scope": "section", "action": "improve"},
                          headers=auth_headers)
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_edit_custom_without_instruction_is_400(client, auth_headers):
    r = await client.post("/api/ops/wiki/edit/stream",
                          json={"path": "concepts/x.md", "action": "custom"},
                          headers=auth_headers)
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_edit_reconcile_without_target_is_400(client, auth_headers):
    r = await client.post("/api/ops/wiki/edit/stream",
                          json={"path": "concepts/x.md", "action": "reconcile"},
                          headers=auth_headers)
    assert r.status_code == 400


# ── permission gating ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_edit_requires_can_edit_wiki(client, member_headers):
    r = await client.post("/api/ops/wiki/edit/stream",
                          json={"path": "concepts/x.md", "action": "improve"},
                          headers=member_headers)
    assert r.status_code == 403


# ── happy path (streamed) ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_edit_page_streams_done_with_full_content(client, auth_headers, user_ctx):
    from app.services.wiki_db import upsert_wiki_page
    await upsert_wiki_page("concepts/widget.md", _PAGE)

    fake = _fake_stream_factory(["# Widget\n\n", "A widget does many things, clearly.\n"])
    with patch.object(BedrockService, "converse_stream", fake):
        r = await client.post(
            "/api/ops/wiki/edit/stream",
            json={"path": "concepts/widget.md", "scope": "page", "action": "improve"},
            headers=auth_headers,
        )
    assert r.status_code == 200
    events = _parse_sse(r.text)
    types = [e["type"] for e in events]
    assert types[0] == "meta"
    assert "chunk" in types
    assert types[-1] == "done"
    assert events[-1]["full_content"] == "# Widget\n\nA widget does many things, clearly.\n"


# ── graph stays in sync after edit/delete ────────────────────────────────────

@pytest.mark.asyncio
async def test_edit_updates_graph_links(client, auth_headers, user_ctx):
    """Applying an edit that adds a link must re-resolve the graph (wiki_links),
    not just the page row. The inline AI editor relies on this — it applies via
    this same PUT route."""
    from app.services.wiki_db import upsert_wiki_page, get_wiki_links
    from app.services.graph import get_graph, reset_graph_cache
    reset_graph_cache()
    await upsert_wiki_page("concepts/target.md", "---\ntitle: Target\n---\n# Target\n\nBody.\n")
    await upsert_wiki_page("concepts/src.md", "---\ntitle: Src\n---\n# Src\n\nNothing linked yet.\n")
    await get_graph().rebuild()
    assert "concepts/target.md" not in (await get_wiki_links("concepts/src.md"))["outgoing"]

    r = await client.put(
        "/api/wiki/concepts/src.md",
        json={"content": "---\ntitle: Src\n---\n# Src\n\nNow see [Target](target.md).\n"},
        headers=auth_headers,
    )
    assert r.status_code == 200
    assert "concepts/target.md" in (await get_wiki_links("concepts/src.md"))["outgoing"]


@pytest.mark.asyncio
async def test_delete_clears_graph_links(client, auth_headers, user_ctx):
    from app.services.wiki_db import upsert_wiki_page, get_wiki_links
    from app.services.graph import get_graph, reset_graph_cache
    reset_graph_cache()
    await upsert_wiki_page("concepts/target.md", "---\ntitle: Target\n---\n# Target\n\nBody.\n")
    await upsert_wiki_page("concepts/src.md", "---\ntitle: Src\n---\n# Src\n\nSee [Target](target.md).\n")
    await get_graph().rebuild()
    assert "concepts/src.md" in (await get_wiki_links("concepts/target.md"))["incoming"]

    r = await client.delete("/api/wiki/concepts/src.md", headers=auth_headers)
    assert r.status_code == 200
    assert "concepts/src.md" not in (await get_wiki_links("concepts/target.md"))["incoming"]


@pytest.mark.asyncio
async def test_edit_section_streams_spliced_full_content(client, auth_headers, user_ctx):
    from app.services.wiki_db import upsert_wiki_page
    await upsert_wiki_page("concepts/widget.md", _PAGE)

    fake = _fake_stream_factory(["## Background\n\nRefreshed background.\n"])
    with patch.object(BedrockService, "converse_stream", fake):
        r = await client.post(
            "/api/ops/wiki/edit/stream",
            json={"path": "concepts/widget.md", "scope": "section",
                  "action": "expand", "heading": "Background"},
            headers=auth_headers,
        )
    assert r.status_code == 200
    done = _parse_sse(r.text)[-1]
    assert done["type"] == "done"
    assert "Refreshed background." in done["full_content"]
    assert "Old text." not in done["full_content"]
    # Title + frontmatter preserved (whole-page result).
    assert done["full_content"].startswith("---\ntitle: Widget")
