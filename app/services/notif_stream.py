"""Real-time notification fan-out via Postgres LISTEN/NOTIFY.

A single process-wide LISTEN connection receives NOTIFY signals (payload =
recipient user_id) and wakes any SSE streams subscribed for that user, which
then re-fetch their unread notifications.

Replica-safe: NOTIFY broadcasts to every replica's LISTEN connection, so a
notification created on one replica reaches the user's stream wherever it's
connected. The signal carries only the user_id — streams pull the actual rows
from the DB, so payload size and consistency are never a concern.
"""
import asyncio
import os

import asyncpg

from app.config import settings
from app.logger import get_logger

log = get_logger(__name__)

CHANNEL = "wiki_notifications"

# How often the background heartbeat re-checks the LISTEN connection and
# reconnects it if it has dropped. Without this, a dropped listener only
# re-armed on the next subscribe(), so already-connected streams silently fell
# back to their 25s backstop poll forever after a single connection blip.
_HEARTBEAT_SECS = 20


def _dsn() -> str:
    """asyncpg DSN derived from the app's DATABASE_URL (drop the SQLAlchemy driver)."""
    url = os.environ.get("DATABASE_URL") or settings.DATABASE_URL
    return (
        url.replace("postgresql+asyncpg://", "postgresql://")
           .replace("postgresql+psycopg://", "postgresql://")
    )


class _Hub:
    def __init__(self) -> None:
        self._subs: dict[str, set[asyncio.Queue]] = {}
        self._conn: asyncpg.Connection | None = None
        self._lock = asyncio.Lock()
        self._heartbeat: asyncio.Task | None = None

    async def _ensure_listener(self) -> None:
        # Reconnect if never connected or the listener connection has dropped.
        if self._conn is not None and not self._conn.is_closed():
            return
        async with self._lock:
            if self._conn is not None and not self._conn.is_closed():
                return
            conn = await asyncpg.connect(_dsn())
            await conn.add_listener(CHANNEL, self._on_notify)
            # Drop our handle promptly when the connection dies so the heartbeat
            # (and the next subscribe) reconnect instead of using a dead socket.
            try:
                conn.add_termination_listener(self._on_terminated)
            except Exception:
                pass
            self._conn = conn
            log.info("notif_stream | LISTEN %s established", CHANNEL)

    def _on_terminated(self, conn) -> None:
        """asyncpg termination callback — forget a dead connection (only if it's
        still the current one, so a stale event can't clobber a fresh reconnect)."""
        if conn is self._conn:
            self._conn = None
            log.warning("notif_stream | LISTEN connection terminated; will reconnect")

    def start(self) -> None:
        """Launch the background heartbeat that keeps the listener connected."""
        if self._heartbeat is None or self._heartbeat.done():
            self._heartbeat = asyncio.create_task(self._heartbeat_loop())

    async def _heartbeat_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(_HEARTBEAT_SECS)
                if self._subs:  # only hold a listener while someone is subscribed
                    await self._ensure_listener()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.warning("notif_stream | heartbeat reconnect failed; will retry", exc_info=True)

    def _on_notify(self, _conn, _pid, _channel, payload: str) -> None:
        """asyncpg callback — wake every stream subscribed for this user."""
        for q in list(self._subs.get(payload, ())):
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                pass  # stream is behind; its next backstop fetch will catch up

    async def subscribe(self, user_id: str) -> asyncio.Queue:
        await self._ensure_listener()
        q: asyncio.Queue = asyncio.Queue(maxsize=64)
        self._subs.setdefault(user_id, set()).add(q)
        return q

    def unsubscribe(self, user_id: str, q: asyncio.Queue) -> None:
        subs = self._subs.get(user_id)
        if subs:
            subs.discard(q)
            if not subs:
                self._subs.pop(user_id, None)

    async def shutdown(self) -> None:
        if self._heartbeat is not None:
            self._heartbeat.cancel()
            try:
                await self._heartbeat
            except (asyncio.CancelledError, Exception):
                pass
            self._heartbeat = None
        conn, self._conn = self._conn, None
        self._subs.clear()
        if conn is not None:
            try:
                await conn.close()
            except Exception:
                pass


_hub = _Hub()


def hub() -> _Hub:
    return _hub
