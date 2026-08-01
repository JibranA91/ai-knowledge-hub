"""Import a wiki export bundle into the current (empty) org.

Counterpart to `WikiEngine.build_wiki_export`. Restores wiki pages + schema
(AGENTS.md) from a bundle, reusing exported embeddings when they're compatible
with this deployment (otherwise re-embedding happens on upsert when enabled).
The whole import runs inside a tracked action so it can be reverted.

The caller (route) enforces auth + the empty-org guard; this module focuses on
parsing the archive safely and writing the content.
"""
import io
import json
import posixpath
import zipfile

from app import model
from app.config import settings
from app.logger import get_logger
from app.services.graph import get_graph
from app.services.wiki_db import set_wiki_file, upsert_wiki_page
from app.services.wiki_state import begin_action

log = get_logger(__name__)

# Bundle layouts this importer understands (manifest.format_version).
_SUPPORTED_FORMAT_VERSIONS = {1}

# Derived pages are regenerated from wiki_pages/audit_log — never imported.
_DERIVED = {"index.md", "log.md"}


class WikiImportError(Exception):
    """Raised when an uploaded bundle can't be imported (bad zip, unsafe path…)."""


def _is_unsafe_rel(rel: str) -> bool:
    """True if a relative archive path tries to escape its directory (zip-slip)."""
    if not rel or rel.startswith("/") or "\\" in rel or ":" in rel:
        return True
    norm = posixpath.normpath(rel)
    return norm == ".." or norm.startswith("../") or norm.startswith("/")


def _embeddings_compatible(manifest: dict, emb_map: dict) -> bool:
    """Exported vectors are safe to reuse only when model + dimensions match."""
    if not emb_map:
        return False
    return (
        manifest.get("embedding_dimensions") == settings.EMBEDDING_DIMENSIONS
        and (manifest.get("embedding_model") or "") == model.model_id_for(model.Role.EMBEDDING)
    )


async def import_bundle(zip_bytes: bytes) -> dict:
    """Import a wiki export zip into the current org. Returns a summary dict."""
    try:
        zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except zipfile.BadZipFile:
        raise WikiImportError("Uploaded file is not a valid .zip archive")

    names = zf.namelist()

    manifest: dict = {}
    if "manifest.json" in names:
        try:
            manifest = json.loads(zf.read("manifest.json"))
        except Exception:
            manifest = {}
    fmt = manifest.get("format_version")
    if fmt is not None and fmt not in _SUPPORTED_FORMAT_VERSIONS:
        raise WikiImportError(
            f"Unsupported export format_version={fmt}. This server supports "
            f"{sorted(_SUPPORTED_FORMAT_VERSIONS)}."
        )

    emb_map: dict[str, list] = {}
    if "embeddings.json" in names:
        try:
            emb_list = json.loads(zf.read("embeddings.json"))
            emb_map = {e["path"]: e["embedding"] for e in emb_list if e.get("embedding")}
        except Exception:
            emb_map = {}
    emb_compatible = _embeddings_compatible(manifest, emb_map)

    schema_content: str | None = None
    page_entries: list[tuple[str, str]] = []
    for name in names:
        if name.endswith("/"):
            continue
        if name == "schema/AGENTS.md":
            schema_content = zf.read(name).decode("utf-8", "replace")
            continue
        if name.startswith("wiki/"):
            rel = name[len("wiki/"):]
            if not rel or rel in _DERIVED:
                continue
            if _is_unsafe_rel(rel):
                raise WikiImportError(f"Unsafe path in archive: {name}")
            page_entries.append((rel, zf.read(name).decode("utf-8", "replace")))

    if not page_entries and schema_content is None:
        raise WikiImportError("Archive contains no wiki pages or schema to import")

    pages_imported = 0
    embeddings_reused = 0
    async with begin_action("wiki_import", summary=f"import {len(page_entries)} page(s)",
                            details={"pages": len(page_entries), "has_schema": schema_content is not None}):
        if schema_content is not None:
            await set_wiki_file("schema/AGENTS.md", schema_content)
        for rel, content in page_entries:
            emb = emb_map.get(rel) if emb_compatible else None
            await upsert_wiki_page(rel, content, embedding=emb)
            pages_imported += 1
            if emb is not None:
                embeddings_reused += 1

    # Regenerate index / wiki_links / graph cache from the imported pages.
    await get_graph().rebuild()

    log.info("wiki_import | pages=%d | schema=%s | embeddings_reused=%d | compatible=%s",
             pages_imported, schema_content is not None, embeddings_reused, emb_compatible)
    return {
        "pages_imported": pages_imported,
        "schema_imported": schema_content is not None,
        "embeddings_reused": embeddings_reused,
        "embeddings_compatible": emb_compatible,
    }
