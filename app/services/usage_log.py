"""Bedrock usage tracking — one row per LLM call, scoped to org + user."""
import sqlalchemy as sa

from app.db import get_db
from app.logger import get_logger

log = get_logger(__name__)

_INSERT = sa.text("""
    INSERT INTO usage_log (org_id, user_id, model_id, tokens_in, tokens_out, operation)
    VALUES (CAST(:org_id AS UUID), CAST(:user_id AS UUID), :model_id, :tokens_in, :tokens_out, :operation)
""")

_ORG_SUMMARY = sa.text("""
    SELECT
        model_id,
        operation,
        SUM(tokens_in)  AS total_tokens_in,
        SUM(tokens_out) AS total_tokens_out,
        COUNT(*)        AS calls
    FROM usage_log
    WHERE org_id = CAST(:org_id AS UUID)
      AND created_at >= NOW() - make_interval(days => :days)
    GROUP BY model_id, operation
    ORDER BY total_tokens_out DESC
""")


async def record(org_id: str, model_id: str, tokens_in: int,
                 tokens_out: int, operation: str, user_id: str = "") -> None:
    """Insert a usage row. Silently swallows errors so LLM calls never fail due to logging."""
    try:
        async with get_db() as db:
            await db.execute(_INSERT, {
                "org_id": org_id,
                "user_id": user_id or "00000000-0000-0000-0000-000000000000",
                "model_id": model_id,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "operation": operation,
            })
    except Exception as exc:
        log.warning("usage_log.record failed (non-fatal): %s", exc)


async def get_org_summary(org_id: str, days: int = 30) -> list[dict]:
    """Return aggregated usage for the past N days."""
    async with get_db() as db:
        result = await db.execute(_ORG_SUMMARY, {"org_id": org_id, "days": days})
        rows = result.fetchall()
    return [
        {
            "model_id": r.model_id,
            "operation": r.operation,
            "total_tokens_in": r.total_tokens_in,
            "total_tokens_out": r.total_tokens_out,
            "calls": r.calls,
        }
        for r in rows
    ]
