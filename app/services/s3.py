"""File storage abstraction — local filesystem or AWS S3.

STORAGE_BACKEND=local  (default): all reads/writes go to DATA_DIR/<key>
STORAGE_BACKEND=s3               : all reads/writes go to S3 bucket DATA_BUCKET
"""
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

from app.config import settings
from app.logger import get_logger

log = get_logger(__name__)

_client = None


# ── Local filesystem helpers ───────────────────────────────────────────────

def _local(key: str) -> Path:
    return Path(settings.DATA_DIR) / key


# ── S3 client management ───────────────────────────────────────────────────

def _make_client():
    from app.services.aws_auth import get_credentials
    creds = get_credentials()
    kwargs: dict = {"region_name": settings.AWS_REGION}
    if settings.AWS_ENDPOINT_URL:
        kwargs["endpoint_url"] = settings.AWS_ENDPOINT_URL
    for k in ("aws_access_key_id", "aws_secret_access_key", "aws_session_token"):
        if creds.get(k):
            kwargs[k] = creds[k]
    return boto3.client("s3", **kwargs)


def get_s3():
    global _client
    if _client is None:
        _client = _make_client()
    return _client


def refresh_client() -> None:
    """Called by aws_auth when credentials rotate."""
    global _client
    _client = None


# ── Public API — same interface regardless of backend ─────────────────────

def ensure_bucket() -> None:
    """Ensure storage is ready: creates local directories or verifies the S3 bucket."""
    if settings.STORAGE_BACKEND == "local":
        base = Path(settings.DATA_DIR)
        for d in ("raw", "wiki", "wiki/sources", "wiki/concepts", "wiki/entities",
                  "wiki/queries", "schema"):
            (base / d).mkdir(parents=True, exist_ok=True)
        log.info("Local storage ready | data_dir=%s", settings.DATA_DIR)
        return
    s3 = get_s3()
    try:
        s3.head_bucket(Bucket=settings.DATA_BUCKET)
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code in ("404", "NoSuchBucket", "403"):
            try:
                if settings.AWS_REGION == "us-east-1":
                    s3.create_bucket(Bucket=settings.DATA_BUCKET)
                else:
                    s3.create_bucket(
                        Bucket=settings.DATA_BUCKET,
                        CreateBucketConfiguration={"LocationConstraint": settings.AWS_REGION},
                    )
                log.info("S3 bucket created: %s", settings.DATA_BUCKET)
            except ClientError as err:
                log.warning("Could not create S3 bucket %s: %s", settings.DATA_BUCKET, err)


def read(key: str) -> str:
    """Return content as UTF-8 string, or '' if not found."""
    if settings.STORAGE_BACKEND == "local":
        p = _local(key)
        return p.read_text(encoding="utf-8") if p.exists() else ""
    try:
        obj = get_s3().get_object(Bucket=settings.DATA_BUCKET, Key=key)
        return obj["Body"].read().decode("utf-8")
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404", "NoSuchBucket"):
            return ""
        raise


def read_bytes(key: str) -> bytes:
    """Return raw bytes, or b'' if not found."""
    if settings.STORAGE_BACKEND == "local":
        p = _local(key)
        return p.read_bytes() if p.exists() else b""
    try:
        obj = get_s3().get_object(Bucket=settings.DATA_BUCKET, Key=key)
        return obj["Body"].read()
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404", "NoSuchBucket"):
            return b""
        raise


def write(key: str, content: str) -> None:
    if settings.STORAGE_BACKEND == "local":
        p = _local(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return
    get_s3().put_object(Bucket=settings.DATA_BUCKET, Key=key, Body=content.encode("utf-8"))


def write_bytes(key: str, data: bytes) -> None:
    if settings.STORAGE_BACKEND == "local":
        p = _local(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        return
    get_s3().put_object(Bucket=settings.DATA_BUCKET, Key=key, Body=data)


def delete(key: str) -> None:
    if settings.STORAGE_BACKEND == "local":
        try:
            _local(key).unlink()
        except FileNotFoundError:
            pass
        return
    try:
        get_s3().delete_object(Bucket=settings.DATA_BUCKET, Key=key)
    except ClientError:
        pass


def exists(key: str) -> bool:
    if settings.STORAGE_BACKEND == "local":
        return _local(key).exists()
    try:
        get_s3().head_object(Bucket=settings.DATA_BUCKET, Key=key)
        return True
    except ClientError:
        return False


def list_keys(prefix: str) -> list[str]:
    """List all keys under a prefix."""
    if settings.STORAGE_BACKEND == "local":
        base = Path(settings.DATA_DIR)
        prefix_path = base / prefix.rstrip("/")
        if not prefix_path.exists():
            return []
        return sorted(
            p.relative_to(base).as_posix()
            for p in prefix_path.rglob("*")
            if p.is_file()
        )
    s3 = get_s3()
    keys: list[str] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=settings.DATA_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys


def get_object_size(key: str) -> int:
    if settings.STORAGE_BACKEND == "local":
        p = _local(key)
        return p.stat().st_size if p.exists() else 0
    try:
        obj = get_s3().head_object(Bucket=settings.DATA_BUCKET, Key=key)
        return obj["ContentLength"]
    except ClientError:
        return 0


def get_object_mtime(key: str) -> float:
    if settings.STORAGE_BACKEND == "local":
        p = _local(key)
        return p.stat().st_mtime if p.exists() else 0.0
    try:
        obj = get_s3().head_object(Bucket=settings.DATA_BUCKET, Key=key)
        return obj["LastModified"].timestamp()
    except ClientError:
        return 0.0


def presigned_url(key: str, expiry: int = 3600) -> str:
    return get_s3().generate_presigned_url(
        "get_object",
        Params={"Bucket": settings.DATA_BUCKET, "Key": key},
        ExpiresIn=expiry,
    )


def org_prefix(key: str) -> str:
    """Prepend the current request's org_id to an S3 key for tenant isolation.

    Example: org_prefix("raw/doc.pdf") → "<org_uuid>/raw/doc.pdf"
    Falls back to the bare key when no org context is set (startup, tests).
    """
    try:
        from app.context import get_org_id
        org_id = get_org_id()
        return f"{org_id}/{key}"
    except RuntimeError:
        return key
