"""Phase-1 unit tests for the choke-point wrappers (§11.2).

Verifies that each of the wrapped write functions correctly emits a
`wiki_revisions` row capturing op + before/after content.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.wiki_db import (
    delete_wiki_page,
    set_wiki_file,
    upsert_wiki_page,
)
from app.services.wiki_state import (
    begin_action,
    tracked_delete,
    tracked_write_bytes,
)
from tests.fixtures.revisions import (
    _Result,
    _Row,
    make_recording_get_db,
    set_action_context,
    set_user_context,
)


# ── upsert_wiki_page ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_upsert_create_emits_create_revision():
    """New page → revision op=create, content_before=NULL."""
    # First SELECT (get_wiki_page_content for content_before): no row.
    get_db, db = make_recording_get_db([_Result(rows=[])])

    with set_user_context(), \
         set_action_context("action-123"), \
         patch("app.services.wiki_db.get_db", get_db), \
         patch("app.services.wiki_state.get_db", get_db), \
         patch("app.services.embeddings.is_enabled", return_value=False):
        await upsert_wiki_page("concepts/new.md", "# New page\n")

    revs = db.queries_matching("INSERT INTO wiki_revisions")
    assert len(revs) == 1
    params = revs[0][1]
    assert params["op"] == "create"
    assert params["content_before"] is None
    assert params["content_after"] == "# New page\n"
    assert params["target_kind"] == "page"
    assert params["target_key"] == "concepts/new.md"


@pytest.mark.asyncio
async def test_upsert_update_captures_before_content():
    """Existing page → revision op=update, both before/after populated."""
    existing = "# Old\nv1 content"
    # SELECT for content_before returns the existing row.
    get_db, db = make_recording_get_db([_Result(rows=[_Row(content=existing)])])

    with set_user_context(), \
         set_action_context("action-456"), \
         patch("app.services.wiki_db.get_db", get_db), \
         patch("app.services.wiki_state.get_db", get_db), \
         patch("app.services.embeddings.is_enabled", return_value=False):
        await upsert_wiki_page("concepts/existing.md", "# New\nv2 content")

    revs = db.queries_matching("INSERT INTO wiki_revisions")
    assert revs[0][1]["op"] == "update"
    assert revs[0][1]["content_before"] == existing
    assert revs[0][1]["content_after"] == "# New\nv2 content"


@pytest.mark.asyncio
async def test_upsert_no_action_open_skips_revision():
    """Same write outside any action should still succeed — no revision row."""
    get_db, db = make_recording_get_db([_Result(rows=[])])

    with set_user_context(), \
         patch("app.services.wiki_db.get_db", get_db), \
         patch("app.services.wiki_state.get_db", get_db), \
         patch("app.services.embeddings.is_enabled", return_value=False):
        await upsert_wiki_page("concepts/untracked.md", "body")

    assert db.queries_matching("INSERT INTO wiki_revisions") == []
    # but the page write still happened
    assert db.queries_matching("INSERT INTO wiki_pages") != []


# ── delete_wiki_page ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_delete_captures_content_before():
    existing = "# To delete\nbody"
    get_db, db = make_recording_get_db([_Result(rows=[_Row(content=existing)])])

    with set_user_context(), \
         set_action_context("action-del"), \
         patch("app.services.wiki_db.get_db", get_db), \
         patch("app.services.wiki_state.get_db", get_db):
        await delete_wiki_page("concepts/goner.md")

    revs = db.queries_matching("INSERT INTO wiki_revisions")
    assert len(revs) == 1
    params = revs[0][1]
    assert params["op"] == "delete"
    assert params["content_before"] == existing
    assert params["content_after"] is None


@pytest.mark.asyncio
async def test_delete_missing_page_is_noop():
    """Deleting a non-existent path emits no revision and no DELETE."""
    get_db, db = make_recording_get_db([_Result(rows=[])])

    with set_user_context(), \
         set_action_context("action-del"), \
         patch("app.services.wiki_db.get_db", get_db), \
         patch("app.services.wiki_state.get_db", get_db):
        await delete_wiki_page("concepts/never-existed.md")

    assert db.queries_matching("INSERT INTO wiki_revisions") == []
    assert db.queries_matching("DELETE FROM wiki_pages") == []


# ── set_wiki_file ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_set_wiki_file_tracks_schema_writes():
    existing = "# Old schema"
    get_db, db = make_recording_get_db([_Result(rows=[_Row(content=existing)])])

    with set_user_context(), \
         set_action_context("action-schema"), \
         patch("app.services.wiki_db.get_db", get_db), \
         patch("app.services.wiki_state.get_db", get_db):
        await set_wiki_file("schema/AGENTS.md", "# New schema")

    revs = db.queries_matching("INSERT INTO wiki_revisions")
    assert len(revs) == 1
    assert revs[0][1]["target_kind"] == "file"
    assert revs[0][1]["target_key"] == "schema/AGENTS.md"
    assert revs[0][1]["op"] == "update"
    assert revs[0][1]["content_before"] == existing
    assert revs[0][1]["content_after"] == "# New schema"


@pytest.mark.asyncio
async def test_set_wiki_file_skips_derived_graph_cache():
    """`wiki/.graph.json` is rebuildable; never emit revisions for it."""
    get_db, db = make_recording_get_db()

    with set_user_context(), \
         set_action_context("action-graph"), \
         patch("app.services.wiki_db.get_db", get_db), \
         patch("app.services.wiki_state.get_db", get_db):
        await set_wiki_file("wiki/.graph.json", '{"nodes": []}')

    assert db.queries_matching("INSERT INTO wiki_revisions") == []
    # The cache write itself still happens.
    assert db.queries_matching("INSERT INTO wiki_files") != []


# ── tracked_write_bytes / tracked_delete (S3) ─────────────────────────────

def _fake_s3_module(*, exists: bool = False, size: int = 0, content: bytes = b""):
    """Build a MagicMock that mimics the s3 module's surface."""
    fake = MagicMock()
    fake.exists.return_value = exists
    fake.get_object_size.return_value = size
    fake.read_bytes.return_value = content
    return fake


@pytest.mark.asyncio
async def test_tracked_write_bytes_new_key_emits_create_revision():
    get_db, db = make_recording_get_db()
    fake_s3 = _fake_s3_module(exists=False)

    with set_user_context(), \
         set_action_context("action-up"), \
         patch("app.services.wiki_state.get_db", get_db), \
         patch("app.services.wiki_state.s3", fake_s3):
        await tracked_write_bytes("org/raw/new.pdf", b"PDF DATA")

    fake_s3.write_bytes.assert_called_once_with("org/raw/new.pdf", b"PDF DATA")
    revs = db.queries_matching("INSERT INTO wiki_revisions")
    assert revs[0][1]["op"] == "create"
    assert revs[0][1]["content_before"] is None
    assert revs[0][1]["content_after"] == "org/raw/new.pdf"
    assert revs[0][1]["bytes_size_after"] == len(b"PDF DATA")


@pytest.mark.asyncio
async def test_tracked_write_bytes_overwrite_archives_prior_object():
    get_db, db = make_recording_get_db()
    fake_s3 = _fake_s3_module(exists=True, size=1024, content=b"OLD CONTENT")

    with set_user_context(), \
         set_action_context("action-overwrite"), \
         patch("app.services.wiki_state.get_db", get_db), \
         patch("app.services.wiki_state.s3", fake_s3):
        await tracked_write_bytes("org/raw/handbook.pdf", b"NEW")

    # Wrote the archive copy first, then the new bytes.
    write_calls = [c.args for c in fake_s3.write_bytes.call_args_list]
    assert write_calls[0] == ("archive/action-overwrite/org/raw/handbook.pdf", b"OLD CONTENT")
    assert write_calls[1] == ("org/raw/handbook.pdf", b"NEW")

    revs = db.queries_matching("INSERT INTO wiki_revisions")
    assert revs[0][1]["op"] == "update"
    assert revs[0][1]["content_before"] == "archive/action-overwrite/org/raw/handbook.pdf"
    assert revs[0][1]["bytes_size_before"] == 1024


@pytest.mark.asyncio
async def test_tracked_delete_archives_then_deletes():
    get_db, db = make_recording_get_db()
    fake_s3 = _fake_s3_module(exists=True, size=42, content=b"BYE")

    with set_user_context(), \
         set_action_context("action-del"), \
         patch("app.services.wiki_state.get_db", get_db), \
         patch("app.services.wiki_state.s3", fake_s3):
        await tracked_delete("org/raw/old.pdf")

    fake_s3.write_bytes.assert_called_once_with("archive/action-del/org/raw/old.pdf", b"BYE")
    fake_s3.delete.assert_called_once_with("org/raw/old.pdf")
    revs = db.queries_matching("INSERT INTO wiki_revisions")
    assert revs[0][1]["op"] == "delete"
    assert revs[0][1]["content_before"] == "archive/action-del/org/raw/old.pdf"
    assert revs[0][1]["content_after"] is None


@pytest.mark.asyncio
async def test_tracked_delete_missing_key_is_noop():
    get_db, db = make_recording_get_db()
    fake_s3 = _fake_s3_module(exists=False)

    with set_user_context(), \
         set_action_context("action-x"), \
         patch("app.services.wiki_state.get_db", get_db), \
         patch("app.services.wiki_state.s3", fake_s3):
        await tracked_delete("org/raw/nothing.pdf")

    fake_s3.delete.assert_not_called()
    fake_s3.write_bytes.assert_not_called()
    assert db.queries_matching("INSERT INTO wiki_revisions") == []


@pytest.mark.asyncio
async def test_tracked_write_bytes_no_action_skips_archive():
    """Untracked writes (no action open) overwrite without archiving."""
    get_db, db = make_recording_get_db()
    fake_s3 = _fake_s3_module(exists=True, size=10, content=b"OLD")

    with set_user_context(), \
         patch("app.services.wiki_state.get_db", get_db), \
         patch("app.services.wiki_state.s3", fake_s3):
        await tracked_write_bytes("org/raw/x.pdf", b"NEW")

    # Only the live write happened — no archive copy because no action_id.
    fake_s3.write_bytes.assert_called_once_with("org/raw/x.pdf", b"NEW")
    assert db.queries_matching("INSERT INTO wiki_revisions") == []
