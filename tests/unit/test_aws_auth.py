"""Unit tests for app/services/aws_auth.py.

STS calls are mocked — no AWS account required.
Module-level cache state is reset before each test via the autouse fixture.
"""
from contextlib import suppress
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def reset_aws_auth_state():
    """Wipe the module-level credential cache before and after every test."""
    import app.services.aws_auth as m
    m._credentials = None
    m._expiration = None
    m._refresh_task = None
    yield
    m._credentials = None
    m._expiration = None
    m._refresh_task = None


# ── _is_expired ───────────────────────────────────────────────────────────

def test_is_expired_when_no_credentials():
    import app.services.aws_auth as m
    assert m._is_expired() is True


def test_is_expired_when_expiry_is_past():
    import app.services.aws_auth as m
    m._credentials = {"AccessKeyId": "k"}
    m._expiration = datetime.now(timezone.utc) - timedelta(hours=1)
    assert m._is_expired() is True


def test_is_expired_within_buffer():
    import app.services.aws_auth as m
    m._credentials = {"AccessKeyId": "k"}
    # Within the 300s buffer
    m._expiration = datetime.now(timezone.utc) + timedelta(seconds=200)
    assert m._is_expired() is True


def test_not_expired_when_fresh():
    import app.services.aws_auth as m
    m._credentials = {"AccessKeyId": "k"}
    m._expiration = datetime.now(timezone.utc) + timedelta(hours=1)
    assert m._is_expired() is False


# ── _assume_role ──────────────────────────────────────────────────────────

def test_assume_role_no_arn_returns_none():
    from app.services.aws_auth import _assume_role
    with patch("app.services.aws_auth.settings") as s:
        s.ASSUMED_ROLE_ARN = ""
        result = _assume_role()
    assert result is None


def test_assume_role_calls_sts():
    import app.services.aws_auth as m
    expiry = datetime.now(timezone.utc) + timedelta(hours=1)
    mock_sts = MagicMock()
    mock_sts.assume_role.return_value = {
        "Credentials": {
            "AccessKeyId": "ASIA_KEY",
            "SecretAccessKey": "SECRET",
            "SessionToken": "TOKEN",
            "Expiration": expiry,
        }
    }
    with patch("app.services.aws_auth.settings") as s, \
         patch("boto3.client", return_value=mock_sts):
        s.ASSUMED_ROLE_ARN = "arn:aws:iam::123456789:role/TestRole"
        s.ASSUMED_ROLE_SESSION_NAME = "TestSession"
        s.ASSUMED_ROLE_DURATION = 3600
        s.AWS_REGION = "us-east-1"
        s.AWS_ACCESS_KEY_ID = ""
        creds = m._assume_role(force=True)

    assert creds["AccessKeyId"] == "ASIA_KEY"
    assert m._credentials is not None
    assert m._expiration == expiry


def test_assume_role_cached_when_not_expired():
    """If credentials are fresh, _assume_role should not call STS again."""
    import app.services.aws_auth as m
    expiry = datetime.now(timezone.utc) + timedelta(hours=1)
    m._credentials = {
        "AccessKeyId": "CACHED_KEY",
        "SecretAccessKey": "CACHED_SECRET",
        "SessionToken": "CACHED_TOKEN",
        "Expiration": expiry,
    }
    m._expiration = expiry

    with patch("app.services.aws_auth.settings") as s, \
         patch("boto3.client") as mock_boto:
        s.ASSUMED_ROLE_ARN = "arn:aws:iam::123:role/Role"
        result = m._assume_role(force=False)

    mock_boto.assert_not_called()
    assert result["AccessKeyId"] == "CACHED_KEY"


# ── get_credentials ───────────────────────────────────────────────────────

def test_get_credentials_no_role_returns_static():
    with patch("app.services.aws_auth.settings") as s:
        s.ASSUMED_ROLE_ARN = ""
        s.AWS_REGION = "eu-west-1"
        s.AWS_ACCESS_KEY_ID = "STATIC_KEY"
        s.AWS_SECRET_ACCESS_KEY = "STATIC_SECRET"
        from app.services.aws_auth import get_credentials
        creds = get_credentials()

    assert creds["aws_access_key_id"] == "STATIC_KEY"
    assert creds["aws_secret_access_key"] == "STATIC_SECRET"
    assert creds["region_name"] == "eu-west-1"
    assert "aws_session_token" not in creds


def test_get_credentials_no_role_empty_key_omits_key_fields():
    """When no static key is set, the returned dict should not include key fields."""
    with patch("app.services.aws_auth.settings") as s:
        s.ASSUMED_ROLE_ARN = ""
        s.AWS_REGION = "us-east-1"
        s.AWS_ACCESS_KEY_ID = ""
        s.AWS_SECRET_ACCESS_KEY = ""
        from app.services.aws_auth import get_credentials
        creds = get_credentials()

    assert "aws_access_key_id" not in creds
    assert creds["region_name"] == "us-east-1"


def test_get_credentials_with_role_returns_assumed():
    import app.services.aws_auth as m
    expiry = datetime.now(timezone.utc) + timedelta(hours=1)
    mock_sts = MagicMock()
    mock_sts.assume_role.return_value = {
        "Credentials": {
            "AccessKeyId": "ASSUMED_KEY",
            "SecretAccessKey": "ASSUMED_SECRET",
            "SessionToken": "ASSUMED_TOKEN",
            "Expiration": expiry,
        }
    }
    with patch("app.services.aws_auth.settings") as s, \
         patch("boto3.client", return_value=mock_sts):
        s.ASSUMED_ROLE_ARN = "arn:aws:iam::123:role/Role"
        s.ASSUMED_ROLE_SESSION_NAME = "Sess"
        s.ASSUMED_ROLE_DURATION = 3600
        s.AWS_REGION = "us-east-1"
        s.AWS_ACCESS_KEY_ID = ""
        creds = m.get_credentials()

    assert creds["aws_access_key_id"] == "ASSUMED_KEY"
    assert creds["aws_session_token"] == "ASSUMED_TOKEN"


# ── start_refresh_task / stop_refresh_task ────────────────────────────────

@pytest.mark.asyncio
async def test_start_refresh_task_no_op_without_arn():
    import app.services.aws_auth as m
    with patch("app.services.aws_auth.settings") as s:
        s.ASSUMED_ROLE_ARN = ""
        await m.start_refresh_task()
    assert m._refresh_task is None


@pytest.mark.asyncio
async def test_stop_refresh_task_no_op_when_none():
    import app.services.aws_auth as m
    m._refresh_task = None
    await m.stop_refresh_task()  # should not raise


def test_assume_role_with_static_creds_passes_to_sts():
    """When AWS_ACCESS_KEY_ID is set, it is forwarded to the STS client."""
    import app.services.aws_auth as m
    from datetime import timedelta
    expiry = datetime.now(timezone.utc) + timedelta(hours=1)
    mock_sts = MagicMock()
    mock_sts.assume_role.return_value = {
        "Credentials": {
            "AccessKeyId": "KEY",
            "SecretAccessKey": "SECRET",
            "SessionToken": "TOKEN",
            "Expiration": expiry,
        }
    }
    with patch("app.services.aws_auth.settings") as s, \
         patch("boto3.client", return_value=mock_sts) as mock_boto:
        s.ASSUMED_ROLE_ARN = "arn:aws:iam::123:role/Role"
        s.ASSUMED_ROLE_SESSION_NAME = "Sess"
        s.ASSUMED_ROLE_DURATION = 3600
        s.AWS_REGION = "us-east-1"
        s.AWS_ACCESS_KEY_ID = "STATIC_KEY"
        s.AWS_SECRET_ACCESS_KEY = "STATIC_SECRET"
        m._assume_role(force=True)

    call_kwargs = mock_boto.call_args[1]
    assert call_kwargs.get("aws_access_key_id") == "STATIC_KEY"


def test_assume_role_sts_error_propagates():
    """STS ClientError should propagate from _assume_role."""
    from botocore.exceptions import ClientError
    mock_sts = MagicMock()
    mock_sts.assume_role.side_effect = ClientError(
        {"Error": {"Code": "AccessDenied", "Message": "Forbidden"}}, "AssumeRole"
    )
    with patch("app.services.aws_auth.settings") as s, \
         patch("boto3.client", return_value=mock_sts):
        s.ASSUMED_ROLE_ARN = "arn:aws:iam::123:role/Role"
        s.ASSUMED_ROLE_SESSION_NAME = "Sess"
        s.ASSUMED_ROLE_DURATION = 3600
        s.AWS_REGION = "us-east-1"
        s.AWS_ACCESS_KEY_ID = ""
        with pytest.raises(ClientError):
            from app.services.aws_auth import _assume_role
            _assume_role(force=True)


@pytest.mark.asyncio
async def test_start_refresh_task_creates_task_with_arn():
    import app.services.aws_auth as m
    expiry = datetime.now(timezone.utc) + timedelta(hours=1)
    fake_creds = {
        "AccessKeyId": "K",
        "SecretAccessKey": "S",
        "SessionToken": "T",
        "Expiration": expiry,
    }
    with patch("app.services.aws_auth.settings") as s, \
         patch("asyncio.to_thread") as mock_thread:
        s.ASSUMED_ROLE_ARN = "arn:aws:iam::123:role/Role"
        mock_thread.return_value = fake_creds
        await m.start_refresh_task()

    assert m._refresh_task is not None
    m._refresh_task.cancel()


@pytest.mark.asyncio
async def test_stop_refresh_task_cancels_and_clears():
    import asyncio as aio
    import app.services.aws_auth as m

    async def _dummy():
        try:
            await aio.sleep(9999)
        except aio.CancelledError:
            raise

    m._refresh_task = aio.create_task(_dummy())
    await m.stop_refresh_task()
    assert m._refresh_task is None


@pytest.mark.asyncio
async def test_refresh_loop_handles_cancellation():
    """_refresh_loop should exit cleanly when cancelled."""
    import asyncio as aio
    import app.services.aws_auth as m

    with patch("asyncio.sleep", new_callable=AsyncMock), \
         patch("asyncio.to_thread", new_callable=AsyncMock):
        task = aio.create_task(m._refresh_loop())
        # Give it one iteration then cancel
        await aio.sleep(0)
        task.cancel()
        with suppress(aio.CancelledError):
            await task
    # Should have exited without raising (CancelledError is re-raised inside)
    assert task.done()


@pytest.mark.asyncio
async def test_refresh_loop_continues_on_exception():
    """_refresh_loop should keep running after a non-CancelledError exception."""
    import asyncio as aio
    import app.services.aws_auth as m

    call_count = 0

    async def _mock_sleep(secs):
        nonlocal call_count
        call_count += 1
        if call_count >= 3:
            raise aio.CancelledError()

    async def _mock_to_thread(fn, *args):
        raise RuntimeError("Credential refresh failed")

    with patch("asyncio.sleep", side_effect=_mock_sleep), \
         patch("asyncio.to_thread", side_effect=_mock_to_thread):
        with suppress(aio.CancelledError):
            await m._refresh_loop()

    assert call_count >= 2  # retried after the error
