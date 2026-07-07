"""Shared helper for Server-Sent Events (SSE) responses."""
from fastapi.responses import StreamingResponse


def sse_response(generator) -> StreamingResponse:
    """Wrap an async generator as an SSE response with the standard media type
    and anti-buffering headers (X-Accel-Buffering disables proxy buffering so
    events flush to the client live). Single source for the header config the
    chat / writer / edit / notification streams all use.
    """
    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
