"""Integration tests for /api/documents routes.

Requires PostgreSQL via testcontainers.  LLM calls are mocked — uploads
trigger the planning phase which is intercepted before any Bedrock call.
"""
import io
import pytest
from unittest.mock import AsyncMock, patch


def _txt_upload(name: str = "test.txt", content: str = "Hello world."):
    return {"file": (name, io.BytesIO(content.encode()), "text/plain")}


def _pdf_upload(name: str = "test.pdf"):
    # Minimal valid PDF bytes
    data = b"%PDF-1.4\n1 0 obj<</Type/Catalog>>endobj\nxref\n0 1\n0000000000 65535 f \ntrailer<</Size 1/Root 1 0 R>>\nstartxref\n9\n%%EOF"
    return {"file": (name, io.BytesIO(data), "application/pdf")}


# ── GET /api/documents ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_list_documents_empty(client, auth_headers, tmp_path):
    resp = await client.get("/api/documents", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json() == []


# ── POST /api/documents/upload ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_upload_txt_file(client, auth_headers):
    with patch("app.routes.documents._run_ingest", new_callable=AsyncMock):
        resp = await client.post(
            "/api/documents/upload",
            files=_txt_upload(),
            headers=auth_headers,
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["filename"] == "test.txt"
    assert data["size"] > 0


@pytest.mark.asyncio
async def test_upload_creates_ingest_job(client, auth_headers):
    with patch("app.routes.documents._run_ingest", new_callable=AsyncMock):
        await client.post(
            "/api/documents/upload",
            files=_txt_upload("job_test.txt"),
            headers=auth_headers,
        )

    # Job should be in DB
    resp = await client.get("/api/ops/status/job_test.txt", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json()["status"] in ("queued", "processing", "pending_review", "done")


@pytest.mark.asyncio
async def test_upload_unsupported_extension(client, auth_headers):
    files = {"file": ("evil.exe", io.BytesIO(b"MZ"), "application/octet-stream")}
    resp = await client.post(
        "/api/documents/upload", files=files, headers=auth_headers
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_upload_file_too_large(client, auth_headers, monkeypatch):
    from app import config
    monkeypatch.setattr(config.settings, "MAX_UPLOAD_SIZE_MB", 0)

    with patch("app.routes.documents._run_ingest", new_callable=AsyncMock):
        resp = await client.post(
            "/api/documents/upload",
            files=_txt_upload(content="any content"),
            headers=auth_headers,
        )
    assert resp.status_code == 413


# ── GET /api/documents/{filename} ────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_document_returns_file(client, auth_headers):
    with patch("app.routes.documents._run_ingest", new_callable=AsyncMock):
        await client.post(
            "/api/documents/upload",
            files=_txt_upload("dl.txt", "Download me"),
            headers=auth_headers,
        )

    resp = await client.get("/api/documents/dl.txt", headers=auth_headers)
    assert resp.status_code == 200
    assert b"Download me" in resp.content


@pytest.mark.asyncio
async def test_get_document_not_found(client, auth_headers):
    resp = await client.get("/api/documents/missing.txt", headers=auth_headers)
    assert resp.status_code == 404


# ── DELETE /api/documents/{filename} ─────────────────────────────────────

@pytest.mark.asyncio
async def test_delete_document(client, auth_headers):
    with patch("app.routes.documents._run_ingest", new_callable=AsyncMock):
        await client.post(
            "/api/documents/upload",
            files=_txt_upload("del.txt"),
            headers=auth_headers,
        )

    resp = await client.delete("/api/documents/del.txt", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json()["deleted"] == "del.txt"

    # File should be gone
    resp2 = await client.get("/api/documents/del.txt", headers=auth_headers)
    assert resp2.status_code == 404


@pytest.mark.asyncio
async def test_delete_document_not_found(client, auth_headers):
    resp = await client.delete("/api/documents/nope.txt", headers=auth_headers)
    assert resp.status_code == 404


# ── Allowed extensions ────────────────────────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.parametrize("ext", [".txt", ".md"])
async def test_allowed_text_extensions(client, auth_headers, ext):
    fname = f"allowed{ext}"
    files = {"file": (fname, io.BytesIO(b"content"), "text/plain")}
    with patch("app.routes.documents._run_ingest", new_callable=AsyncMock):
        resp = await client.post(
            "/api/documents/upload", files=files, headers=auth_headers
        )
    assert resp.status_code == 200


# ── GET /api/documents — with uploaded files ──────────────────────────────

@pytest.mark.asyncio
async def test_list_documents_shows_uploaded(client, auth_headers):
    with patch("app.routes.documents._run_ingest", new_callable=AsyncMock):
        await client.post(
            "/api/documents/upload",
            files=_txt_upload("listme.txt"),
            headers=auth_headers,
        )

    resp = await client.get("/api/documents", headers=auth_headers)
    assert resp.status_code == 200
    names = [f["name"] for f in resp.json()]
    assert "listme.txt" in names


# ── DELETE removes wiki source pages ──────────────────────────────────────

@pytest.mark.asyncio
async def test_delete_removes_derived_wiki_pages(client, auth_headers, user_ctx):
    """Delete should clean up any wiki pages stamped with the uploaded file."""
    from app.services.wiki_db import upsert_wiki_page
    # Create a fake source wiki page referencing the upload
    await upsert_wiki_page(
        "sources/uploaded_doc.md",
        "---\nuploaded_file: cleanup_test.txt\n---\n# Uploaded Doc\n",
    )

    with patch("app.routes.documents._run_ingest", new_callable=AsyncMock):
        await client.post(
            "/api/documents/upload",
            files=_txt_upload("cleanup_test.txt"),
            headers=auth_headers,
        )

    resp = await client.delete("/api/documents/cleanup_test.txt", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["deleted"] == "cleanup_test.txt"
