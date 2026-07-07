"""Integration tests for wiki export-completeness + import into empty orgs.

Requires PostgreSQL. Covers: AGENTS.md + format_version in the export bundle,
service-level import (pages + schema), zip-slip / bad-archive guards, and the
POST /api/ops/import route (empty-org guard, role gate, round-trip).
"""
import io
import json
import zipfile

import pytest
import pytest_asyncio

from app.context import UserContext, current_user
from app.services import orgs as orgs_svc
from app.services import wiki_import
from app.services.wiki_db import (
    upsert_wiki_page, set_wiki_file, get_wiki_file, list_wiki_paths,
)
from app.services.wiki_engine import WikiEngine


def _ctx(org_id: str, user_id: str = "00000000-0000-0000-0000-000000000000") -> None:
    current_user.set(UserContext(user_id=user_id, org_id=org_id, email="sys", role="admin"))


async def _seed(org_id: str, user_id: str) -> None:
    _ctx(org_id, user_id)
    await set_wiki_file("schema/AGENTS.md", "# Imported Schema\n\n## Page Types\nconcept\n")
    await upsert_wiki_page("concepts/x.md", "# X\n\nbody")
    await upsert_wiki_page("concepts/y.md", "# Y\n\nmore")


async def _export_current_org() -> bytes:
    zip_bytes, _, _ = await WikiEngine().build_wiki_export(include_embeddings=True)
    return zip_bytes


def _zip_with(entries: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, content in entries.items():
            zf.writestr(name, content)
    return buf.getvalue()


@pytest_asyncio.fixture
async def member_headers(default_user):
    from app.services.orgs import create_user
    from app.services.auth import create_access_token
    uid = await create_user(default_user["org_id"], "member-imp@test.com",
                            "password123", role="member")
    token = create_access_token(email="member-imp@test.com", user_id=uid,
                               org_id=default_user["org_id"], role="member")
    return {"Authorization": f"Bearer {token}"}


# ── export completeness ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_export_contains_schema_and_format_version(default_user):
    await _seed(default_user["org_id"], default_user["id"])
    zf = zipfile.ZipFile(io.BytesIO(await _export_current_org()))
    names = zf.namelist()
    assert "schema/AGENTS.md" in names
    assert "Imported Schema" in zf.read("schema/AGENTS.md").decode()
    manifest = json.loads(zf.read("manifest.json"))
    assert manifest["format_version"] == 1
    assert manifest["has_schema"] is True


# ── service-level import ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_import_bundle_copies_pages_and_schema(default_user):
    await _seed(default_user["org_id"], default_user["id"])
    bundle = await _export_current_org()

    org_b = await orgs_svc.create_org("Import Target B")
    _ctx(org_b, default_user["id"])
    result = await wiki_import.import_bundle(bundle)

    assert result["pages_imported"] == 2
    assert result["schema_imported"] is True

    paths = await list_wiki_paths()
    assert "concepts/x.md" in paths and "concepts/y.md" in paths
    assert "Imported Schema" in (await get_wiki_file("schema/AGENTS.md") or "")


@pytest.mark.asyncio
async def test_import_bad_zip_raises(default_user):
    _ctx(default_user["org_id"], default_user["id"])
    with pytest.raises(wiki_import.WikiImportError):
        await wiki_import.import_bundle(b"definitely not a zip")


@pytest.mark.asyncio
async def test_import_empty_archive_raises(default_user):
    _ctx(default_user["org_id"], default_user["id"])
    bundle = _zip_with({"manifest.json": json.dumps({"format_version": 1})})
    with pytest.raises(wiki_import.WikiImportError):
        await wiki_import.import_bundle(bundle)


@pytest.mark.asyncio
async def test_import_unsupported_format_version_raises(default_user):
    _ctx(default_user["org_id"], default_user["id"])
    bundle = _zip_with({
        "manifest.json": json.dumps({"format_version": 999}),
        "wiki/concepts/x.md": "# X",
    })
    with pytest.raises(wiki_import.WikiImportError):
        await wiki_import.import_bundle(bundle)


@pytest.mark.asyncio
async def test_import_rejects_zip_slip(default_user):
    _ctx(default_user["org_id"], default_user["id"])
    bundle = _zip_with({
        "manifest.json": json.dumps({"format_version": 1}),
        "wiki/../../evil.md": "# pwned",
    })
    with pytest.raises(wiki_import.WikiImportError):
        await wiki_import.import_bundle(bundle)


# ── route: POST /api/ops/import ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_import_api_rejects_non_empty_org(client, auth_headers, user_ctx):
    await upsert_wiki_page("concepts/exists.md", "# Exists")  # default org now non-empty
    bundle = _zip_with({"manifest.json": json.dumps({"format_version": 1}), "wiki/a.md": "# A"})
    files = {"file": ("wiki.zip", bundle, "application/zip")}
    resp = await client.post("/api/ops/import", files=files, headers=auth_headers)
    assert resp.status_code == 409


@pytest.mark.asyncio
async def test_import_api_member_forbidden(client, member_headers):
    bundle = _zip_with({"manifest.json": json.dumps({"format_version": 1}), "wiki/a.md": "# A"})
    files = {"file": ("wiki.zip", bundle, "application/zip")}
    resp = await client.post("/api/ops/import", files=files, headers=member_headers)
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_import_api_roundtrip_into_empty_org(client, auth_headers, default_user, user_ctx):
    await _seed(default_user["org_id"], default_user["id"])
    bundle = await _export_current_org()
    org_b = await orgs_svc.create_org("API Import B")

    files = {"file": ("wiki.zip", bundle, "application/zip")}
    resp = await client.post(
        "/api/ops/import", files=files,
        headers={**auth_headers, "X-Org-Context": org_b},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["pages_imported"] == 2
    assert data["schema_imported"] is True

    _ctx(org_b, default_user["id"])
    assert "concepts/x.md" in await list_wiki_paths()
    assert "Imported Schema" in (await get_wiki_file("schema/AGENTS.md") or "")
