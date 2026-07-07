"""JWT auth — short-lived access tokens (JWT, 15 min) + long-lived refresh tokens (opaque, DB, 7 days).

Phase 6: access tokens now carry org_id + user_id + role claims.
Credentials are looked up from the users table (no more AUTH_USERNAME/AUTH_PASSWORD comparison).
The single-tenant admin is seeded from config during startup.
"""
import hashlib
import secrets
from datetime import datetime, timedelta, timezone

import sqlalchemy as sa
from jose import JWTError, jwt

from app.config import settings
from app.context import UserContext
from app.db import get_db

_INSERT_REFRESH = sa.text("""
    INSERT INTO auth_tokens (token_hash, token_type, username, org_id, user_id, expires_at)
    VALUES (:token_hash, 'refresh', :email, CAST(:org_id AS UUID), CAST(:user_id AS UUID),
            NOW() + INTERVAL '7 days')
    ON CONFLICT (token_hash) DO NOTHING
""")

_SELECT_REFRESH = sa.text("""
    SELECT username, org_id, user_id FROM auth_tokens
    WHERE token_hash = :token_hash
      AND token_type = 'refresh'
      AND expires_at > NOW()
""")

_DELETE           = sa.text("DELETE FROM auth_tokens WHERE token_hash = :token_hash")
_DELETE_BY_USER   = sa.text("DELETE FROM auth_tokens WHERE user_id = CAST(:user_id AS UUID)")
_PURGE            = sa.text("DELETE FROM auth_tokens WHERE expires_at <= NOW()")


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def create_access_token(email: str, user_id: str = "", org_id: str = "",
                        role: str = "member") -> str:
    """Create a short-lived JWT access token carrying org_id + user_id + role."""
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.JWT_ACCESS_TTL_MINUTES)
    payload = {
        "sub": user_id or email,   # user_id UUID when available, email as fallback
        "email": email,
        "org_id": org_id,
        "role": role,
        "exp": expire,
        "type": "access",
        "jti": secrets.token_hex(8),
    }
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)


async def create_refresh_token(email: str, user_id: str = "", org_id: str = "") -> str:
    """Create a long-lived opaque refresh token and persist its hash in the DB."""
    token = secrets.token_hex(32)
    async with get_db() as db:
        await db.execute(_INSERT_REFRESH, {
            "token_hash": _hash(token),
            "email": email,
            "org_id": org_id or "00000000-0000-0000-0000-000000000001",
            "user_id": user_id or "00000000-0000-0000-0000-000000000000",
        })
    return token


def validate_access_token(token: str) -> UserContext | None:
    """Validate a JWT access token. Returns UserContext on success, None on failure."""
    if not token:
        return None
    try:
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM])
        if payload.get("type") != "access":
            return None
        email = payload.get("email") or payload.get("sub", "")
        user_id = payload.get("sub", "")
        org_id = payload.get("org_id") or ""   # normalize None → ""
        role = payload.get("role", "member")
        return UserContext(
            user_id=user_id,
            org_id=org_id,
            email=email,
            role=role,
        )
    except JWTError:
        return None


async def validate_refresh_token(token: str) -> dict | None:
    """Validate an opaque refresh token. Returns {email, org_id, user_id} or None."""
    if not token:
        return None
    async with get_db() as db:
        result = await db.execute(_SELECT_REFRESH, {"token_hash": _hash(token)})
        row = result.fetchone()
    if not row:
        return None
    return {
        "email": row.username,
        "org_id": str(row.org_id) if row.org_id else "",
        "user_id": str(row.user_id) if row.user_id else "",
    }


async def revoke_token(token: str) -> None:
    async with get_db() as db:
        await db.execute(_DELETE, {"token_hash": _hash(token)})


async def revoke_all_tokens_for_user(user_id: str) -> None:
    """Revoke all refresh tokens for a user — forces re-login on next request."""
    async with get_db() as db:
        await db.execute(_DELETE_BY_USER, {"user_id": user_id})


async def purge_expired() -> None:
    async with get_db() as db:
        await db.execute(_PURGE)
