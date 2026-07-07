"""Integration tests for the real-time notification SSE stream.

Exercises the endpoint's init backlog and the full Postgres LISTEN/NOTIFY push
round-trip. Requires PostgreSQL.
"""
import asyncio
import json

import pytest
import pytest_asyncio


@pytest_asyncio.fixture(autouse=True)
async def _reset_hub():
    # Close the LISTEN connection after each test so the next test opens a fresh
    # one bound to its own event loop (avoids asyncpg cross-loop reuse errors).
    yield
    from app.services import notif_stream
    await notif_stream.hub().shutdown()


def _data(sse_chunk: str) -> dict:
    return json.loads(sse_chunk.split("data: ", 1)[1])


@pytest.mark.asyncio
async def test_event_stream_emits_init_backlog(default_user):
    from app.routes.notifications import notification_event_stream
    gen = notification_event_stream(default_user["id"])
    try:
        init = await asyncio.wait_for(gen.__anext__(), timeout=5)
        evt = _data(init)
        assert evt["type"] == "init"
        assert isinstance(evt["items"], list)
    finally:
        await gen.aclose()


@pytest.mark.asyncio
async def test_event_stream_pushes_new_notification(default_user):
    """Full LISTEN/NOTIFY round-trip: creating a notification wakes the stream."""
    from app.routes.notifications import notification_event_stream
    from app.services import notif_svc

    gen = notification_event_stream(default_user["id"])
    try:
        init = await asyncio.wait_for(gen.__anext__(), timeout=5)
        assert _data(init)["type"] == "init"

        # Drive the next event in the background so the generator is parked on
        # the queue when the NOTIFY fires.
        nxt = asyncio.create_task(gen.__anext__())
        await asyncio.sleep(0.1)
        await notif_svc.create(
            user_id=default_user["id"], org_id=default_user["org_id"],
            type="ingest_done", title="Done", body="ok", link="pushed.txt",
        )
        evt = _data(await asyncio.wait_for(nxt, timeout=10))
        assert evt["type"] == "new"
        assert any(i["link"] == "pushed.txt" for i in evt["items"])
    finally:
        await gen.aclose()


@pytest.mark.asyncio
async def test_stream_endpoint_requires_auth(client):
    resp = await client.get("/api/notifications/stream")
    assert resp.status_code == 401
