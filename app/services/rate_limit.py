"""Postgres-backed per-user rate limiter.

Uses INSERT … ON CONFLICT DO UPDATE to atomically increment a per-minute counter.
No Redis required — Postgres handles the atomicity.

Usage in routes:
    from fastapi import Depends
    from app.services.rate_limit import rate_limit_upload

    @router.post("/upload", dependencies=[Depends(rate_limit_upload)])
    async def upload(...): ...
"""
from datetime import datetime, timezone

import sqlalchemy as sa
from fastapi import HTTPException, Request

from app.config import settings
from app.db import get_db

_UPSERT = sa.text("""
    INSERT INTO rate_limit_counters (identifier, endpoint, window_start, count)
    VALUES (:identifier, :endpoint, :window_start, 1)
    ON CONFLICT (identifier, endpoint, window_start) DO UPDATE
        SET count = rate_limit_counters.count + 1
    RETURNING count
""")


def _window_start() -> datetime:
    now = datetime.now(timezone.utc)
    return now.replace(second=0, microsecond=0)


def _get_user(request: Request) -> str:
    return getattr(request.state, "user", "anonymous")


async def check_rate_limit(identifier: str, endpoint: str, limit: int) -> None:
    """Raise 429 if identifier has exceeded limit requests in the current minute."""
    async with get_db() as db:
        result = await db.execute(_UPSERT, {
            "identifier": identifier,
            "endpoint": endpoint,
            "window_start": _window_start(),
        })
        count = result.scalar()
    if count > limit:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded — max {limit} requests/min for {endpoint}.",
        )


async def rate_limit_upload(request: Request) -> None:
    await check_rate_limit(_get_user(request), "upload", settings.RATE_LIMIT_UPLOAD_PER_MINUTE)


async def rate_limit_query(request: Request) -> None:
    await check_rate_limit(_get_user(request), "query", settings.RATE_LIMIT_QUERY_PER_MINUTE)


async def rate_limit_chat(request: Request) -> None:
    await check_rate_limit(_get_user(request), "chat", settings.RATE_LIMIT_CHAT_PER_MINUTE)


async def rate_limit_recalibrate(request: Request) -> None:
    await check_rate_limit(_get_user(request), "recalibrate", settings.RATE_LIMIT_RECALIBRATE_PER_MINUTE)
