"""Unit tests for Phase 6 auth — JWT carries org_id + user_id + role."""
import pytest
from app.context import UserContext


def _make_token(email="alice@test.com", user_id="uid-1", org_id="org-1", role="admin"):
    from app.services.auth import create_access_token
    return create_access_token(email=email, user_id=user_id, org_id=org_id, role=role)


def test_validate_access_token_returns_user_context():
    from app.services.auth import validate_access_token
    token = _make_token()
    ctx = validate_access_token(token)
    assert isinstance(ctx, UserContext)
    assert ctx.email == "alice@test.com"
    assert ctx.org_id == "org-1"
    assert ctx.role == "admin"


def test_validate_access_token_bad_returns_none():
    from app.services.auth import validate_access_token
    assert validate_access_token("garbage") is None
    assert validate_access_token("") is None


def test_validate_access_token_wrong_type():
    """A refresh-type JWT should not validate as an access token."""
    from jose import jwt
    from app.config import settings
    import secrets
    from datetime import datetime, timedelta, timezone
    expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    payload = {"sub": "user", "type": "refresh", "exp": expire, "jti": secrets.token_hex(8)}
    bad_token = jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)
    from app.services.auth import validate_access_token
    assert validate_access_token(bad_token) is None


def test_org_id_in_jwt_payload():
    """Decoded JWT should contain org_id and role claims."""
    from jose import jwt
    from app.config import settings
    token = _make_token(org_id="my-org-uuid", role="member")
    payload = jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM])
    assert payload["org_id"] == "my-org-uuid"
    assert payload["role"] == "member"
    assert payload["email"] == "alice@test.com"


def test_context_var_isolation():
    """UserContext set in one coroutine must not leak to another."""
    from app.context import current_user
    ctx_a = UserContext(user_id="u1", org_id="org-a", email="a@x.com", role="admin")
    ctx_b = UserContext(user_id="u2", org_id="org-b", email="b@x.com", role="member")

    token_a = current_user.set(ctx_a)
    assert current_user.get() == ctx_a

    token_b = current_user.set(ctx_b)
    assert current_user.get() == ctx_b

    # Reset in reverse order
    current_user.reset(token_b)
    assert current_user.get() == ctx_a

    current_user.reset(token_a)
    assert current_user.get() is None
