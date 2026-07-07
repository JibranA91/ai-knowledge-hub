"""Integration tests for the recalibration fingerprint store (idempotency).

Exercises the wiki_db get_recalibration_state / record_recalibration round-trip
that lets master recalibration skip already-reviewed, unchanged pages and
measure staleness from the last review. Each fingerprint is a (content_sha,
context_sha) pair. Requires PostgreSQL (migrations 026 + 027).
"""
import pytest

from app.context import UserContext, current_user
from app.services.wiki_db import get_recalibration_state, record_recalibration


def _set_ctx(user_id: str, org_id: str) -> None:
    current_user.set(UserContext(user_id=user_id, org_id=org_id, email="admin", role="admin"))


@pytest.mark.asyncio
async def test_record_and_read_roundtrip(client, default_user):
    _set_ctx(default_user["id"], default_user["org_id"])
    await record_recalibration([
        ("concepts/fp-a.md", "csa", "xsa"),
        ("concepts/fp-b.md", "csb", "xsb"),
    ])
    state = await get_recalibration_state()
    assert state["concepts/fp-a.md"]["content_sha"] == "csa"
    assert state["concepts/fp-a.md"]["context_sha"] == "xsa"
    assert state["concepts/fp-b.md"]["content_sha"] == "csb"
    assert state["concepts/fp-a.md"]["reviewed_at"] is not None


@pytest.mark.asyncio
async def test_upsert_overwrites_hashes_and_bumps_reviewed_at(client, default_user):
    _set_ctx(default_user["id"], default_user["org_id"])
    await record_recalibration([("concepts/fp-c.md", "cs-old", "xs-old")], prune=False)
    first = (await get_recalibration_state())["concepts/fp-c.md"]
    await record_recalibration([("concepts/fp-c.md", "cs-new", "xs-new")], prune=False)
    second = (await get_recalibration_state())["concepts/fp-c.md"]
    assert second["content_sha"] == "cs-new"
    assert second["context_sha"] == "xs-new"
    assert second["reviewed_at"] >= first["reviewed_at"]


@pytest.mark.asyncio
async def test_prune_drops_absent_paths(client, default_user):
    _set_ctx(default_user["id"], default_user["org_id"])
    await record_recalibration([
        ("concepts/fp-keep.md", "s1", "x1"),
        ("concepts/fp-gone.md", "s2", "x2"),
    ])
    # A later full run no longer includes fp-gone.md → it's pruned to the corpus.
    await record_recalibration([("concepts/fp-keep.md", "s1", "x1")], prune=True)
    state = await get_recalibration_state()
    assert "concepts/fp-keep.md" in state
    assert "concepts/fp-gone.md" not in state


@pytest.mark.asyncio
async def test_no_prune_keeps_other_paths(client, default_user):
    _set_ctx(default_user["id"], default_user["org_id"])
    await record_recalibration([
        ("concepts/fp-x.md", "sx", "xx"),
        ("concepts/fp-y.md", "sy", "xy"),
    ])
    # A targeted run touches only fp-x.md; fp-y.md must survive (prune=False).
    await record_recalibration([("concepts/fp-x.md", "sx2", "xx2")], prune=False)
    state = await get_recalibration_state()
    assert state["concepts/fp-x.md"]["content_sha"] == "sx2"
    assert "concepts/fp-y.md" in state
