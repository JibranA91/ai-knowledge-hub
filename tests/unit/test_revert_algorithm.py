"""Pure-Python tests for the revert algorithm (§11.4).

These cover the revert engine without touching Postgres or S3 — both are
mocked. The algorithm under test is the coalescing revert in
`wiki_state.revert_action`: it loads every revision in the window in one
query (ASC by id) and applies a single inverse op per distinct object — the
inverse of that object's *oldest* revision, whose `content_before` is the
pre-window state. Intermediate edits are never replayed.

The `_SequencedDB` below returns one scripted result per `execute()` call, in
order. The call sequence for a successful `revert_action` is:
  1. GET_ACTION             → the target row
  2. INSERT wiki_actions    → (begin_action opens the revert action)
  3. UPDATE revert_of_id
  4. ACTIONS_SINCE          → actions in the window (DESC, with status)
  5. REVISIONS_SINCE        → every revision in the window (ASC by id)
  6. MARK_REVERTED (bulk)
  7. SET_REVERTED_ACTIONS   → record undone actions on the revert's own row
  8. GET_REVERTED_ACTIONS_FOR → only when the window contains a revert action
                                (status reconciliation / redo cleanup)
  9. FINALIZE_ACTION        → (begin_action closes the revert action)

(Builders use the `_window` helper, which omits step 8 unless a revert action
is in the window.)
"""
import datetime as dt
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.wiki_state import RevertError, revert_action
from tests.fixtures.revisions import (
    _Result,
    _Row,
    set_user_context,
)


def _action_row(*, id: str, action_type: str = "manual_edit", status: str = "done",
                summary: str = "", started_at=None):
    return _Row(
        id=id,
        action_type=action_type,
        status=status,
        summary=summary,
        started_at=started_at or dt.datetime(2026, 5, 1, 12, 0, 0),
        org_id="org-1",
    )


def _rev(*, id: int, target_kind: str, target_key: str, op: str,
         content_before=None, content_after=None):
    return _Row(
        id=id,
        target_kind=target_kind,
        target_key=target_key,
        op=op,
        content_before=content_before,
        content_after=content_after,
    )


class _SequencedDB:
    """Async DB session whose `execute()` returns pre-scripted results in order.

    Each call records (sql, params) and pops the next result from the queue.
    """

    def __init__(self, results: list) -> None:
        self.results = list(results)
        self.calls: list = []

    async def execute(self, statement, params=None):
        sql = str(getattr(statement, "text", statement))
        self.calls.append((sql, params or {}))
        return self.results.pop(0) if self.results else _Result()


def _make_sequenced_get_db(results: list):
    from contextlib import asynccontextmanager
    db = _SequencedDB(results)

    @asynccontextmanager
    async def _get_db():
        yield db

    return _get_db, db


# ── Guard clauses ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_revert_missing_action_raises():
    get_db, _ = _make_sequenced_get_db([_Result(rows=[])])  # GET_ACTION → no row
    with set_user_context(), \
         patch("app.services.wiki_state.get_db", get_db):
        with pytest.raises(RevertError, match="not found"):
            await revert_action("nonexistent-id")


@pytest.mark.asyncio
async def test_revert_of_revert_is_allowed_and_redoes():
    """Reverting a revert is a redo, not an error. A revert records its own
    inverse ops as revisions; undoing those restores each object to its exact
    pre-revert state. Here the revert had rolled x.md from v2 back to v1, so
    reverting the revert puts x.md back to v2.
    """
    target = _action_row(id="r1", action_type="revert",
                         summary="revert manual_edit: edit x")
    # r1 originally reverted action n1 (which had been 'done'); reverting r1
    # must flip n1's stale 'reverted' status back to 'done'.
    results = _window(
        target,
        _Row(id="r1", action_type="revert", started_at=target.started_at),
        revisions=[_rev(id=70, target_kind="page", target_key="x.md", op="update",
                        content_before="v2", content_after="v1")],
        reverted_actions_rows=[_Row(details={"reverted_actions": [{"id": "n1", "status": "done"}]})],
    )
    get_db, db = _make_sequenced_get_db(results)

    upsert_calls: list[tuple[str, str]] = []

    async def fake_upsert(path: str, content: str, ingested_from: str = ""):
        upsert_calls.append((path, content))

    with set_user_context(), \
         patch("app.services.wiki_state.get_db", get_db), \
         patch("app.services.wiki_db.upsert_wiki_page", AsyncMock(side_effect=fake_upsert)), \
         patch("app.services.wiki_state._rebuild_graph_for", AsyncMock()), \
         patch("app.services.wiki_db.append_audit_log", AsyncMock()):
        result = await revert_action("r1")

    assert upsert_calls == [("x.md", "v2")]      # redo: x.md back to v2
    assert result["reverted_action_ids"] == ["r1"]

    # n1's status is reconciled back to 'done' (it's live again).
    restore_calls = [(s, p) for s, p in db.calls if "status = :status" in s]
    assert restore_calls, "expected a status-reconciliation UPDATE"
    _, params = restore_calls[0]
    assert params["status"] == "done"
    assert params["ids"] == ["n1"]


@pytest.mark.asyncio
async def test_revert_of_already_reverted_rejected():
    get_db, _ = _make_sequenced_get_db([
        _Result(rows=[_action_row(id="abc", status="reverted")]),
    ])
    with set_user_context(), \
         patch("app.services.wiki_state.get_db", get_db):
        with pytest.raises(RevertError, match="already"):
            await revert_action("abc")


@pytest.mark.asyncio
async def test_revert_of_running_action_rejected():
    get_db, _ = _make_sequenced_get_db([
        _Result(rows=[_action_row(id="abc", status="running")]),
    ])
    with set_user_context(), \
         patch("app.services.wiki_state.get_db", get_db):
        with pytest.raises(RevertError, match="still running"):
            await revert_action("abc")


# ── audit_log emission ────────────────────────────────────────────────────

def _window(target, *action_rows, revisions, reverted_actions_rows=None):
    """Build the result script for a successful revert (see module docstring).

    `action_rows` are the rows ACTIONS_SINCE returns (DESC). `revisions` is the
    flat, id-ASC list every revision in the window — what REVISIONS_SINCE
    returns now that we load them in one query rather than per action.

    Status reconciliation issues an extra SELECT (`GET_REVERTED_ACTIONS_FOR`)
    only when the window contains a `revert` action; `reverted_actions_rows`
    supplies that result. ACTIONS_SINCE rows now carry `status`, so any row
    missing one is defaulted to 'done' here.
    """
    for a in action_rows:
        if not hasattr(a, "status"):
            a.status = "done"

    results = [
        _Result(rows=[target]),            # 1. GET_ACTION
        _Result(),                         # 2. INSERT revert action
        _Result(),                         # 3. UPDATE revert_of_id
        _Result(rows=list(action_rows)),   # 4. ACTIONS_SINCE
        _Result(rows=list(revisions)),     # 5. REVISIONS_SINCE
        _Result(),                         # 6. MARK_REVERTED (bulk)
        _Result(),                         # 7. SET_REVERTED_ACTIONS (record on revert row)
    ]
    if any(getattr(a, "action_type", None) == "revert" for a in action_rows):
        results.append(_Result(rows=reverted_actions_rows or []))  # 8. GET_REVERTED_ACTIONS_FOR
    results.append(_Result())              # FINALIZE_ACTION
    return results


# ── Coalescing helper ───────────────────────────────────────────────────────

def test_coalesce_keeps_only_oldest_revision_per_object():
    """Many edits to the same object collapse to its single oldest revision."""
    from app.services.wiki_state import _coalesce_revisions
    revs = [
        _rev(id=10, target_kind="page", target_key="a.md", op="create",
             content_before=None, content_after="a1"),
        _rev(id=11, target_kind="page", target_key="b.md", op="update",
             content_before="b0", content_after="b1"),
        _rev(id=12, target_kind="page", target_key="a.md", op="update",
             content_before="a1", content_after="a2"),
        _rev(id=13, target_kind="page", target_key="a.md", op="update",
             content_before="a2", content_after="a3"),
    ]
    coalesced = _coalesce_revisions(revs)
    by_key = {(r.target_kind, r.target_key): r for r in coalesced}
    assert len(coalesced) == 2
    assert by_key[("page", "a.md")].id == 10   # oldest a.md revision wins
    assert by_key[("page", "b.md")].id == 11


# ── audit_log emission ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_revert_writes_audit_log_entry():
    """A successful revert must append a `revert` audit_log row attributed to
    the user, so the operation is trackable alongside everything else.
    """
    target = _action_row(id="a1", action_type="manual_edit", summary="edit x.md")
    results = _window(
        target,
        _Row(id="a1", action_type="manual_edit", started_at=target.started_at),
        revisions=[_rev(id=10, target_kind="page", target_key="x.md", op="update",
                        content_before="v1", content_after="v2")],
    )
    get_db, _ = _make_sequenced_get_db(results)
    audit_mock = AsyncMock()

    with set_user_context(), \
         patch("app.services.wiki_state.get_db", get_db), \
         patch("app.services.wiki_db.upsert_wiki_page", AsyncMock()), \
         patch("app.services.wiki_state._rebuild_graph_for", AsyncMock()), \
         patch("app.services.wiki_db.append_audit_log", audit_mock):
        await revert_action("a1")

    audit_mock.assert_called_once()
    kwargs = audit_mock.call_args.kwargs
    assert kwargs["operation"] == "revert"
    assert "a1" in kwargs["raw_text"]
    assert kwargs["details"]["target_action_id"] == "a1"
    assert kwargs["details"]["reverted_action_ids"] == ["a1"]
    assert kwargs["details"]["pages_touched"] == ["x.md"]


# ── Page restore ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_revert_create_op_deletes_page():
    """Action created a page → revert deletes it."""
    target = _action_row(id="a1", action_type="manual_edit")
    results = _window(
        target,
        _Row(id="a1", action_type="manual_edit", started_at=target.started_at),
        revisions=[_rev(id=10, target_kind="page", target_key="x.md", op="create",
                        content_before=None, content_after="hi")],
    )
    get_db, _ = _make_sequenced_get_db(results)

    delete_calls: list[str] = []

    async def fake_delete(path: str):
        delete_calls.append(path)

    with set_user_context(), \
         patch("app.services.wiki_state.get_db", get_db), \
         patch("app.services.wiki_db.delete_wiki_page", AsyncMock(side_effect=fake_delete)), \
         patch("app.services.wiki_state._rebuild_graph_for", AsyncMock()), \
         patch("app.services.wiki_db.append_audit_log", AsyncMock()):
        result = await revert_action("a1")

    assert delete_calls == ["x.md"]
    assert result["pages_touched"] == ["x.md"]
    assert result["reverted_action_ids"] == ["a1"]


@pytest.mark.asyncio
async def test_revert_update_op_restores_prior_content():
    """Action updated a page → revert writes content_before back."""
    target = _action_row(id="a1")
    results = _window(
        target,
        _Row(id="a1", action_type="manual_edit", started_at=target.started_at),
        revisions=[_rev(id=20, target_kind="page", target_key="x.md", op="update",
                        content_before="v1", content_after="v2")],
    )
    get_db, _ = _make_sequenced_get_db(results)

    upsert_calls: list[tuple[str, str]] = []

    async def fake_upsert(path: str, content: str, ingested_from: str = ""):
        upsert_calls.append((path, content))

    with set_user_context(), \
         patch("app.services.wiki_state.get_db", get_db), \
         patch("app.services.wiki_db.upsert_wiki_page", AsyncMock(side_effect=fake_upsert)), \
         patch("app.services.wiki_state._rebuild_graph_for", AsyncMock()), \
         patch("app.services.wiki_db.append_audit_log", AsyncMock()):
        await revert_action("a1")

    assert upsert_calls == [("x.md", "v1")]


@pytest.mark.asyncio
async def test_revert_delete_op_recreates_page():
    """Action deleted a page → revert recreates with content_before."""
    target = _action_row(id="a1", action_type="manual_delete")
    results = _window(
        target,
        _Row(id="a1", action_type="manual_delete", started_at=target.started_at),
        revisions=[_rev(id=30, target_kind="page", target_key="x.md", op="delete",
                        content_before="gone but back", content_after=None)],
    )
    get_db, _ = _make_sequenced_get_db(results)

    upsert_calls: list[tuple[str, str]] = []

    async def fake_upsert(path: str, content: str, ingested_from: str = ""):
        upsert_calls.append((path, content))

    with set_user_context(), \
         patch("app.services.wiki_state.get_db", get_db), \
         patch("app.services.wiki_db.upsert_wiki_page", AsyncMock(side_effect=fake_upsert)), \
         patch("app.services.wiki_state._rebuild_graph_for", AsyncMock()), \
         patch("app.services.wiki_db.append_audit_log", AsyncMock()):
        await revert_action("a1")

    assert upsert_calls == [("x.md", "gone but back")]


# ── Multi-action revert ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_revert_discards_later_actions_too():
    """Reverting A2 also discards A3 and A4. State returns to the snapshot
    right after A1.

    p.md was edited in all three actions (v1→v2 in a2, v2→v3 in a3, v3→v4 in
    a4). Coalescing means we DON'T replay each step — we restore p.md once, to
    the `content_before` of its oldest revision in the window (v1). One write,
    not three, regardless of how many edits happened.
    """
    a2 = _action_row(id="a2", started_at=dt.datetime(2026, 5, 2))
    a3 = _Row(id="a3", action_type="manual_edit", started_at=dt.datetime(2026, 5, 3))
    a4 = _Row(id="a4", action_type="manual_edit", started_at=dt.datetime(2026, 5, 4))

    results = _window(
        a2,
        a4, a3, _Row(id="a2", action_type="manual_edit", started_at=a2.started_at),
        # REVISIONS_SINCE returns the whole window ASC by id: a2's edit is oldest.
        revisions=[
            _rev(id=20, target_kind="page", target_key="p.md", op="update",
                 content_before="v1", content_after="v2"),
            _rev(id=30, target_kind="page", target_key="p.md", op="update",
                 content_before="v2", content_after="v3"),
            _rev(id=40, target_kind="page", target_key="p.md", op="update",
                 content_before="v3", content_after="v4"),
        ],
    )
    get_db, db = _make_sequenced_get_db(results)

    upsert_calls: list[tuple[str, str]] = []

    async def fake_upsert(path: str, content: str, ingested_from: str = ""):
        upsert_calls.append((path, content))

    with set_user_context(), \
         patch("app.services.wiki_state.get_db", get_db), \
         patch("app.services.wiki_db.upsert_wiki_page", AsyncMock(side_effect=fake_upsert)), \
         patch("app.services.wiki_state._rebuild_graph_for", AsyncMock()), \
         patch("app.services.wiki_db.append_audit_log", AsyncMock()):
        result = await revert_action("a2")

    # Coalesced: a single restore back to v1 — not v3, v2, v1 one at a time.
    assert upsert_calls == [("p.md", "v1")]
    assert result["reverted_action_ids"] == ["a4", "a3", "a2"]


# ── S3 raw restore ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_revert_s3_create_deletes_live_object():
    target = _action_row(id="a1", action_type="document_upload")
    results = _window(
        target,
        _Row(id="a1", action_type="document_upload", started_at=target.started_at),
        revisions=[_rev(id=50, target_kind="s3_raw", target_key="org/raw/file.pdf",
                        op="create", content_before=None,
                        content_after="org/raw/file.pdf")],
    )
    get_db, _ = _make_sequenced_get_db(results)

    fake_s3 = MagicMock()

    with set_user_context(), \
         patch("app.services.wiki_state.get_db", get_db), \
         patch("app.services.wiki_state.s3", fake_s3), \
         patch("app.services.wiki_state._rebuild_graph_for", AsyncMock()), \
         patch("app.services.wiki_db.append_audit_log", AsyncMock()):
        await revert_action("a1")

    fake_s3.delete.assert_called_once_with("org/raw/file.pdf")


@pytest.mark.asyncio
async def test_revert_s3_delete_restores_from_archive():
    target = _action_row(id="a1", action_type="document_delete")
    results = _window(
        target,
        _Row(id="a1", action_type="document_delete", started_at=target.started_at),
        revisions=[_rev(id=60, target_kind="s3_raw", target_key="org/raw/file.pdf",
                        op="delete", content_before="archive/a1/org/raw/file.pdf",
                        content_after=None)],
    )
    get_db, _ = _make_sequenced_get_db(results)

    fake_s3 = MagicMock()
    fake_s3.exists.return_value = True
    fake_s3.read_bytes.return_value = b"OLD PDF"

    with set_user_context(), \
         patch("app.services.wiki_state.get_db", get_db), \
         patch("app.services.wiki_state.s3", fake_s3), \
         patch("app.services.wiki_state._rebuild_graph_for", AsyncMock()), \
         patch("app.services.wiki_db.append_audit_log", AsyncMock()):
        await revert_action("a1")

    fake_s3.read_bytes.assert_called_once_with("archive/a1/org/raw/file.pdf")
    fake_s3.write_bytes.assert_called_once_with("org/raw/file.pdf", b"OLD PDF")


# ── Coalescing across ops on one object ────────────────────────────────────

@pytest.mark.asyncio
async def test_create_then_update_in_window_nets_to_delete():
    """An object created then updated within the window didn't exist before
    the window, so the net revert is a single delete — the intermediate update
    is never replayed. The oldest revision (the create) decides the inverse.
    """
    target = _action_row(id="a1")
    results = _window(
        target,
        _Row(id="a1", action_type="manual_edit", started_at=target.started_at),
        # ASC by id: create (oldest) then update.
        revisions=[
            _rev(id=100, target_kind="page", target_key="p.md", op="create",
                 content_before=None, content_after="step1"),
            _rev(id=101, target_kind="page", target_key="p.md", op="update",
                 content_before="step1", content_after="step2"),
        ],
    )
    get_db, _ = _make_sequenced_get_db(results)

    upsert_calls: list[str] = []
    delete_calls: list[str] = []

    async def fake_upsert(path, content, ingested_from=""):
        upsert_calls.append(content)

    async def fake_delete(path):
        delete_calls.append(path)

    with set_user_context(), \
         patch("app.services.wiki_state.get_db", get_db), \
         patch("app.services.wiki_db.upsert_wiki_page", AsyncMock(side_effect=fake_upsert)), \
         patch("app.services.wiki_db.delete_wiki_page", AsyncMock(side_effect=fake_delete)), \
         patch("app.services.wiki_state._rebuild_graph_for", AsyncMock()), \
         patch("app.services.wiki_db.append_audit_log", AsyncMock()):
        await revert_action("a1")

    # Single net op: delete. No content is rewritten.
    assert upsert_calls == []
    assert delete_calls == ["p.md"]
