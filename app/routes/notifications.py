import asyncio
import json

from fastapi import APIRouter, HTTPException, Request

from app.logger import get_logger
from app.services import notif_svc
from app.sse import sse_response

log = get_logger(__name__)
router = APIRouter(tags=["notifications"])

# Idle gap between keepalive comments; also a backstop re-check in case a NOTIFY
# was missed (e.g. the listener connection blipped).
_KEEPALIVE_SECS = 25


def _user_id(request: Request) -> str:
    return getattr(request.state, "user_id", "") or ""


@router.get("")
async def list_notifications(request: Request):
    user_id = _user_id(request)
    items = await notif_svc.get_unread(user_id)
    return items


async def notification_event_stream(user_id: str):
    """SSE generator: an `init` backlog event, then `new` events as they arrive.

    Woken instantly by Postgres NOTIFY (via the hub), with a periodic backstop
    re-check + comment keepalive that also detects client disconnects.
    """
    from app.services import notif_stream
    queue = await notif_stream.hub().subscribe(user_id)
    seen: set[str] = set()
    try:
        items = await notif_svc.get_unread(user_id)
        seen.update(n["id"] for n in items)
        yield f"data: {json.dumps({'type': 'init', 'items': items})}\n\n"
        while True:
            try:
                await asyncio.wait_for(queue.get(), timeout=_KEEPALIVE_SECS)
            except asyncio.TimeoutError:
                pass  # fall through to a backstop re-check + keepalive
            fresh = await notif_svc.get_unread(user_id)
            new = [n for n in fresh if n["id"] not in seen]
            seen.update(n["id"] for n in new)
            if new:
                yield f"data: {json.dumps({'type': 'new', 'items': new})}\n\n"
            else:
                yield ": keepalive\n\n"
    finally:
        notif_stream.hub().unsubscribe(user_id, queue)


@router.get("/stream")
async def stream_notifications(request: Request):
    """Server-Sent Events stream of the user's notifications (real-time).

    Auth is the normal Bearer header (the client uses fetch, not EventSource).
    """
    user_id = _user_id(request)
    if not user_id:
        raise HTTPException(401, "Unauthorized")
    return sse_response(notification_event_stream(user_id))


@router.post("/read-all")
async def mark_all_read(request: Request):
    await notif_svc.mark_all_read(_user_id(request))
    return {"ok": True}


@router.post("/{notif_id}/read")
async def mark_read(notif_id: str, request: Request):
    await notif_svc.mark_read(notif_id, _user_id(request))
    return {"ok": True}


@router.post("/read-by-link")
async def mark_read_by_link(request: Request):
    body = await request.json()
    link = body.get("link", "")
    await notif_svc.mark_read_by_link(_user_id(request), link)
    return {"ok": True}
