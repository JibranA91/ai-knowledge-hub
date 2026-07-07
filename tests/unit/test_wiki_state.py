"""Phase-1 unit tests for `app.services.wiki_state` — action lifecycle +
ContextVar propagation. See `.plans/CHANGE_TRACKING_AND_REVERT.md` §11.1.
"""
import asyncio

import pytest
from unittest.mock import patch

from app.context import current_action
from app.services.wiki_state import begin_action, record_revision, tracked_action
from tests.fixtures.revisions import make_recording_get_db, set_user_context


# ── begin_action lifecycle ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_begin_action_inserts_running_row():
    get_db, db = make_recording_get_db()
    with set_user_context(), patch("app.services.wiki_state.get_db", get_db):
        async with begin_action("manual_edit", summary="edit page X") as action_id:
            assert action_id is not None
            assert current_action.get() == action_id

    inserts = db.queries_matching("INSERT INTO wiki_actions")
    assert len(inserts) == 1
    _, params = inserts[0]
    assert params["action_type"] == "manual_edit"
    assert params["summary"] == "edit page X"
    assert params["org_id"] == "org-1"
    assert params["user_id"] == "user-1"


@pytest.mark.asyncio
async def test_begin_action_clean_exit_marks_done():
    get_db, db = make_recording_get_db()
    with set_user_context(), patch("app.services.wiki_state.get_db", get_db):
        async with begin_action("manual_edit"):
            pass

    finalizes = db.queries_matching("UPDATE wiki_actions")
    assert len(finalizes) == 1
    _, params = finalizes[0]
    assert params["status"] == "done"


@pytest.mark.asyncio
async def test_begin_action_exception_marks_error_and_reraises():
    get_db, db = make_recording_get_db()

    class Boom(RuntimeError): ...

    with set_user_context(), patch("app.services.wiki_state.get_db", get_db):
        with pytest.raises(Boom):
            async with begin_action("manual_edit"):
                raise Boom("nope")

    finalizes = db.queries_matching("UPDATE wiki_actions")
    assert finalizes[0][1]["status"] == "error"


@pytest.mark.asyncio
async def test_action_id_propagates_via_contextvar():
    get_db, _ = make_recording_get_db()
    captured: list[str | None] = []

    with set_user_context(), patch("app.services.wiki_state.get_db", get_db):
        async with begin_action("manual_edit") as action_id:
            captured.append(current_action.get())
        captured.append(current_action.get())  # after exit

    assert captured[0] is not None
    assert captured[0] == action_id
    assert captured[1] is None


@pytest.mark.asyncio
async def test_nested_begin_action_is_noop():
    """The inner block reuses the outer action_id; no second row inserted."""
    get_db, db = make_recording_get_db()
    with set_user_context(), patch("app.services.wiki_state.get_db", get_db):
        async with begin_action("outer") as outer_id:
            async with begin_action("inner") as inner_id:
                assert inner_id == outer_id

    inserts = db.queries_matching("INSERT INTO wiki_actions")
    assert len(inserts) == 1


@pytest.mark.asyncio
async def test_action_propagates_to_asyncio_create_task():
    """ContextVars propagate to tasks created inside the block — critical for
    recalibrate / ingest background work that runs via `asyncio.create_task`.
    """
    get_db, _ = make_recording_get_db()
    captured: list[str | None] = []

    async def background_work():
        captured.append(current_action.get())

    with set_user_context(), patch("app.services.wiki_state.get_db", get_db):
        async with begin_action("recalibrate") as action_id:
            task = asyncio.create_task(background_work())
            await task

    assert captured[0] == action_id


@pytest.mark.asyncio
async def test_no_org_context_skips_tracking():
    """No org → action ID is None and no row is written. Matches the contract
    used by startup seeding and internal migrations."""
    get_db, db = make_recording_get_db()
    with patch("app.services.wiki_state.get_db", get_db):
        async with begin_action("manual_edit") as action_id:
            assert action_id is None

    assert db.queries_matching("INSERT INTO wiki_actions") == []


# ── tracked_action decorator ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_tracked_action_decorator_opens_and_closes():
    get_db, db = make_recording_get_db()

    @tracked_action("manual_edit", summary_fn=lambda path: f"edit {path}")
    async def handler(path: str):
        return f"wrote {path}"

    with set_user_context(), patch("app.services.wiki_state.get_db", get_db):
        result = await handler("concepts/foo.md")

    assert result == "wrote concepts/foo.md"
    inserts = db.queries_matching("INSERT INTO wiki_actions")
    assert inserts[0][1]["summary"] == "edit concepts/foo.md"
    finalizes = db.queries_matching("UPDATE wiki_actions")
    assert finalizes[0][1]["status"] == "done"


@pytest.mark.asyncio
async def test_tracked_action_without_summary_fn():
    get_db, db = make_recording_get_db()

    @tracked_action("manual_delete")
    async def handler():
        return "ok"

    with set_user_context(), patch("app.services.wiki_state.get_db", get_db):
        await handler()

    assert db.queries_matching("INSERT INTO wiki_actions")[0][1]["summary"] == ""


# ── record_revision ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_record_revision_skipped_when_no_action_open():
    """Wrapped writes called outside any action should succeed silently (no
    revision row). This is what keeps existing untracked code paths working.
    """
    get_db, db = make_recording_get_db()
    with set_user_context(), patch("app.services.wiki_state.get_db", get_db):
        await record_revision(target_kind="page", target_key="x.md", op="create",
                              content_before=None, content_after="hello")

    assert db.queries_matching("INSERT INTO wiki_revisions") == []


@pytest.mark.asyncio
async def test_record_revision_emits_row_when_action_open():
    get_db, db = make_recording_get_db()
    with set_user_context(), patch("app.services.wiki_state.get_db", get_db):
        async with begin_action("manual_edit"):
            await record_revision(
                target_kind="page",
                target_key="concepts/foo.md",
                op="update",
                content_before="v1",
                content_after="v2",
            )

    inserts = db.queries_matching("INSERT INTO wiki_revisions")
    assert len(inserts) == 1
    _, params = inserts[0]
    assert params["target_kind"] == "page"
    assert params["target_key"] == "concepts/foo.md"
    assert params["op"] == "update"
    assert params["content_before"] == "v1"
    assert params["content_after"] == "v2"
