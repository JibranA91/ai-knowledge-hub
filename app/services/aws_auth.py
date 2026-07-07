"""
Minimal AWS credential helper with optional IAM role assumption + auto-refresh.

If ASSUMED_ROLE_ARN is set in .env, credentials are obtained via STS AssumeRole
and automatically refreshed before they expire.  Otherwise, static credentials
(or the default boto3 credential chain) are used as-is.

Usage:
    from app.services.aws_auth import get_credentials

    creds = get_credentials()   # → dict or None
    # Pass to boto3.client(..., **creds) or ChatBedrockConverse(..., **creds)
"""

import asyncio
import logging
from contextlib import suppress
from datetime import datetime, timezone
from typing import Optional

import boto3
from botocore.exceptions import ClientError

from app.config import settings

log = logging.getLogger(__name__)

# ── Credential cache ───────────────────────────────────────────────────────
_credentials: Optional[dict] = None          # {AccessKeyId, SecretAccessKey, SessionToken, Expiration}
_expiration: Optional[datetime] = None
_refresh_task: Optional[asyncio.Task] = None
_BUFFER_SECONDS = 300                        # refresh 5 min before expiry


def _is_expired() -> bool:
    if not _credentials or not _expiration:
        return True
    return (_expiration - datetime.now(timezone.utc)).total_seconds() < _BUFFER_SECONDS


def _assume_role(force: bool = False) -> Optional[dict]:
    """Call STS AssumeRole and update the cache.  Returns None if no ARN configured."""
    global _credentials, _expiration

    if not settings.ASSUMED_ROLE_ARN:
        return None

    if not force and not _is_expired():
        return _credentials

    log.info("Assuming IAM role: %s", settings.ASSUMED_ROLE_ARN)

    # Build the STS client — bootstrap with static creds if provided
    sts_kwargs: dict = {"region_name": settings.AWS_REGION}
    if settings.AWS_ACCESS_KEY_ID:
        sts_kwargs["aws_access_key_id"] = settings.AWS_ACCESS_KEY_ID
        sts_kwargs["aws_secret_access_key"] = settings.AWS_SECRET_ACCESS_KEY

    try:
        sts = boto3.client("sts", **sts_kwargs)
        resp = sts.assume_role(
            RoleArn=settings.ASSUMED_ROLE_ARN,
            RoleSessionName=settings.ASSUMED_ROLE_SESSION_NAME,
            DurationSeconds=settings.ASSUMED_ROLE_DURATION,
        )
        _credentials = resp["Credentials"]
        _expiration = _credentials["Expiration"]
        log.info("Role assumed successfully, expires at %s", _expiration)
        return _credentials
    except ClientError as e:
        log.error("STS AssumeRole failed: %s", e)
        raise


def get_credentials() -> dict:
    """
    Return a dict suitable for passing to boto3.client(**creds) or
    ChatBedrockConverse(**creds).  Keys: aws_access_key_id,
    aws_secret_access_key, aws_session_token (may be None), region_name.

    If no role ARN is configured, static .env credentials are used (or empty
    strings, which lets boto3 fall back to its default credential chain).
    """
    creds = _assume_role()

    if creds:
        return {
            "aws_access_key_id": creds["AccessKeyId"],
            "aws_secret_access_key": creds["SecretAccessKey"],
            "aws_session_token": creds["SessionToken"],
            "region_name": settings.AWS_REGION,
        }

    # No role assumption — use static creds from .env (may be empty → default chain)
    result: dict = {"region_name": settings.AWS_REGION}
    if settings.AWS_ACCESS_KEY_ID:
        result["aws_access_key_id"] = settings.AWS_ACCESS_KEY_ID
        result["aws_secret_access_key"] = settings.AWS_SECRET_ACCESS_KEY
    return result


# ── Background refresh task ────────────────────────────────────────────────

async def _refresh_loop():
    """Refresh credentials every 50 minutes (well before the 60-min default expiry)."""
    while True:
        try:
            await asyncio.sleep(50 * 60)
            log.info("Scheduled credential refresh starting…")
            await asyncio.to_thread(_assume_role, True)
            log.info("Scheduled credential refresh complete")
        except asyncio.CancelledError:
            log.info("Credential refresh task cancelled")
            raise
        except Exception as e:
            log.error("Credential refresh failed: %s", e)
            await asyncio.sleep(60)  # retry after 1 min


async def start_refresh_task():
    global _refresh_task
    if not settings.ASSUMED_ROLE_ARN:
        return  # nothing to refresh
    if _refresh_task is None:
        # Do an initial assume at startup
        await asyncio.to_thread(_assume_role)
        _refresh_task = asyncio.create_task(_refresh_loop())
        log.info("AWS credential refresh task started")


async def stop_refresh_task():
    global _refresh_task
    if _refresh_task:
        _refresh_task.cancel()
        with suppress(asyncio.CancelledError):
            await _refresh_task
        _refresh_task = None
        log.info("AWS credential refresh task stopped")
