"""Unit tests for the notification fan-out hub (no DB).

We bypass the asyncpg LISTEN connection by presetting a fake `_conn`, then
exercise subscribe / dispatch / unsubscribe directly.
"""
import pytest

from app.services.notif_stream import _Hub, CHANNEL


class _FakeConn:
    def is_closed(self):
        return False


@pytest.mark.asyncio
async def test_dispatch_wakes_subscriber():
    h = _Hub(); h._conn = _FakeConn()
    q = await h.subscribe("u1")
    h._on_notify(None, 0, CHANNEL, "u1")
    assert q.get_nowait() == "u1"


@pytest.mark.asyncio
async def test_no_cross_user_delivery():
    h = _Hub(); h._conn = _FakeConn()
    q = await h.subscribe("u1")
    h._on_notify(None, 0, CHANNEL, "u2")
    assert q.empty()


@pytest.mark.asyncio
async def test_multiple_subscribers_same_user_all_woken():
    h = _Hub(); h._conn = _FakeConn()
    q1 = await h.subscribe("u1")
    q2 = await h.subscribe("u1")
    h._on_notify(None, 0, CHANNEL, "u1")
    assert q1.get_nowait() == "u1"
    assert q2.get_nowait() == "u1"


@pytest.mark.asyncio
async def test_unsubscribe_removes_and_cleans_up():
    h = _Hub(); h._conn = _FakeConn()
    q = await h.subscribe("u1")
    h.unsubscribe("u1", q)
    h._on_notify(None, 0, CHANNEL, "u1")
    assert q.empty()
    assert "u1" not in h._subs


@pytest.mark.asyncio
async def test_dispatch_to_unknown_user_is_noop():
    h = _Hub(); h._conn = _FakeConn()
    h._on_notify(None, 0, CHANNEL, "nobody")  # must not raise


# ── B9: self-healing listener reconnect ────────────────────────────────────

class _FakeAsyncpgConn:
    def __init__(self):
        self._closed = False
    def is_closed(self):
        return self._closed
    async def add_listener(self, channel, cb):
        pass
    def add_termination_listener(self, cb):
        pass
    async def close(self):
        self._closed = True


@pytest.mark.asyncio
async def test_ensure_listener_reconnects_after_drop(monkeypatch):
    """B9: a dropped LISTEN connection is re-established by _ensure_listener (the
    mechanism the background heartbeat drives), without needing a new subscribe."""
    import app.services.notif_stream as ns
    conns = []

    async def fake_connect(_dsn):
        c = _FakeAsyncpgConn(); conns.append(c); return c

    monkeypatch.setattr(ns.asyncpg, "connect", fake_connect)
    h = _Hub()
    await h._ensure_listener()
    assert len(conns) == 1
    conns[0]._closed = True              # simulate a connection drop
    await h._ensure_listener()
    assert len(conns) == 2               # reconnected on a fresh connection


@pytest.mark.asyncio
async def test_start_launches_heartbeat_and_shutdown_cancels():
    """B9: start() runs a background heartbeat; shutdown() cancels it cleanly."""
    h = _Hub()
    h.start()
    assert h._heartbeat is not None and not h._heartbeat.done()
    await h.shutdown()
    assert h._heartbeat is None


def test_on_terminated_only_forgets_current_connection():
    """A stale termination event must not clobber a freshly reconnected handle."""
    h = _Hub()
    cur = _FakeConn(); h._conn = cur
    h._on_terminated(object())           # stale conn → ignored
    assert h._conn is cur
    h._on_terminated(cur)                # current conn → forgotten
    assert h._conn is None
