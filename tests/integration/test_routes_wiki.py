"""Integration tests for /api/wiki routes.

Requires PostgreSQL via testcontainers.
"""
import pytest


# ── GET /api/wiki ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_list_wiki_empty(client, auth_headers):
    resp = await client.get("/api/wiki", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_list_wiki_returns_tree(client, auth_headers):
    # Seed a page then verify it appears in the tree
    await client.put(
        "/api/wiki/concepts/foo.md",
        json={"content": "# Foo\n\nContent."},
        headers=auth_headers,
    )
    resp = await client.get("/api/wiki", headers=auth_headers)
    data = resp.json()
    assert isinstance(data, list)
    # Should have a "concepts" directory node
    concepts_dir = next((n for n in data if n.get("name") == "concepts"), None)
    assert concepts_dir is not None
    assert concepts_dir["type"] == "dir"
    assert any(c["name"] == "foo" for c in concepts_dir["children"])


# ── GET /api/wiki/search ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_search_returns_matching_pages(client, auth_headers):
    await client.put(
        "/api/wiki/concepts/llm.md",
        json={"content": "# Large Language Models\n\nLLMs are neural networks."},
        headers=auth_headers,
    )
    resp = await client.get("/api/wiki/search?q=neural", headers=auth_headers)
    assert resp.status_code == 200
    results = resp.json()
    paths = [r["path"] for r in results]
    assert "concepts/llm.md" in paths


@pytest.mark.asyncio
async def test_search_ignores_unrelated_pages(client, auth_headers):
    await client.put(
        "/api/wiki/concepts/foo.md",
        json={"content": "# Foo\n\nAbout foo concept."},
        headers=auth_headers,
    )
    resp = await client.get(
        "/api/wiki/search?q=quantum-physics", headers=auth_headers
    )
    assert resp.status_code == 200
    results = resp.json()
    paths = [r["path"] for r in results]
    assert "concepts/foo.md" not in paths


@pytest.mark.asyncio
async def test_search_empty_query_returns_results_or_empty(client, auth_headers):
    # Empty query shouldn't 500
    resp = await client.get("/api/wiki/search?q=", headers=auth_headers)
    assert resp.status_code == 200


# ── GET /api/wiki/{path} ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_page_content(client, auth_headers):
    await client.put(
        "/api/wiki/concepts/bar.md",
        json={"content": "# Bar\n\nHello from bar."},
        headers=auth_headers,
    )
    resp = await client.get("/api/wiki/concepts/bar.md", headers=auth_headers)
    assert resp.status_code == 200
    assert "Hello from bar" in resp.text


@pytest.mark.asyncio
async def test_get_page_not_found(client, auth_headers):
    resp = await client.get("/api/wiki/does/not/exist.md", headers=auth_headers)
    assert resp.status_code == 404


# ── PUT /api/wiki/{path} ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_put_creates_page(client, auth_headers):
    resp = await client.put(
        "/api/wiki/new/page.md",
        json={"content": "# New\n\nCreated."},
        headers=auth_headers,
    )
    assert resp.status_code == 200
    assert resp.json()["updated"] is True

    read = await client.get("/api/wiki/new/page.md", headers=auth_headers)
    assert read.status_code == 200
    assert "Created" in read.text


@pytest.mark.asyncio
async def test_put_updates_existing_page(client, auth_headers):
    path = "/api/wiki/concepts/update-me.md"
    await client.put(path, json={"content": "# Old"}, headers=auth_headers)
    await client.put(path, json={"content": "# New updated"}, headers=auth_headers)

    read = await client.get(path, headers=auth_headers)
    assert "New updated" in read.text


# ── DELETE /api/wiki/{path} ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_delete_page(client, auth_headers):
    path = "/api/wiki/delete-me.md"
    await client.put(path, json={"content": "# Del"}, headers=auth_headers)

    resp = await client.delete(path, headers=auth_headers)
    assert resp.status_code == 200

    read = await client.get(path, headers=auth_headers)
    assert read.status_code == 404


@pytest.mark.asyncio
async def test_delete_nonexistent_page(client, auth_headers):
    resp = await client.delete("/api/wiki/no-such-page.md", headers=auth_headers)
    assert resp.status_code == 404
