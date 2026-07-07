"""Unit tests for app/services/s3.py.

Tests both the local-filesystem backend (using tmp_path) and the S3 backend
(mocked via unittest.mock).  All tests isolate settings via monkeypatch.
"""
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path


# ── Helpers ────────────────────────────────────────────────────────────────

def _patch_local(monkeypatch, tmp_path):
    """Point s3 module at tmp_path via the settings singleton."""
    from app import config
    monkeypatch.setattr(config.settings, "STORAGE_BACKEND", "local")
    monkeypatch.setattr(config.settings, "DATA_DIR", str(tmp_path))
    # Invalidate any cached boto3 client
    import app.services.s3 as s3_mod
    s3_mod._client = None


def _patch_s3(monkeypatch, mock_client):
    """Point s3 module at a mocked S3 client."""
    from app import config
    monkeypatch.setattr(config.settings, "STORAGE_BACKEND", "s3")
    monkeypatch.setattr(config.settings, "DATA_BUCKET", "test-bucket")
    import app.services.s3 as s3_mod
    s3_mod._client = mock_client


# ── Local backend ─────────────────────────────────────────────────────────

def test_local_read_returns_content(monkeypatch, tmp_path):
    _patch_local(monkeypatch, tmp_path)
    (tmp_path / "wiki").mkdir()
    (tmp_path / "wiki" / "page.md").write_text("Hello", encoding="utf-8")

    from app.services.s3 import read
    assert read("wiki/page.md") == "Hello"


def test_local_read_missing_returns_empty(monkeypatch, tmp_path):
    _patch_local(monkeypatch, tmp_path)
    from app.services.s3 import read
    assert read("wiki/missing.md") == ""


def test_local_read_bytes(monkeypatch, tmp_path):
    _patch_local(monkeypatch, tmp_path)
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "file.bin").write_bytes(b"\x00\x01\x02")

    from app.services.s3 import read_bytes
    assert read_bytes("raw/file.bin") == b"\x00\x01\x02"


def test_local_read_bytes_missing_returns_empty(monkeypatch, tmp_path):
    _patch_local(monkeypatch, tmp_path)
    from app.services.s3 import read_bytes
    assert read_bytes("raw/missing.bin") == b""


def test_local_write_creates_file(monkeypatch, tmp_path):
    _patch_local(monkeypatch, tmp_path)
    from app.services.s3 import write
    write("wiki/new.md", "Content")
    assert (tmp_path / "wiki" / "new.md").read_text() == "Content"


def test_local_write_bytes_creates_file(monkeypatch, tmp_path):
    _patch_local(monkeypatch, tmp_path)
    from app.services.s3 import write_bytes
    write_bytes("raw/data.bin", b"\xDE\xAD\xBE\xEF")
    assert (tmp_path / "raw" / "data.bin").read_bytes() == b"\xDE\xAD\xBE\xEF"


def test_local_delete_removes_file(monkeypatch, tmp_path):
    _patch_local(monkeypatch, tmp_path)
    p = tmp_path / "raw" / "del.txt"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("bye")

    from app.services.s3 import delete, exists
    assert exists("raw/del.txt")
    delete("raw/del.txt")
    assert not exists("raw/del.txt")


def test_local_delete_missing_does_not_raise(monkeypatch, tmp_path):
    _patch_local(monkeypatch, tmp_path)
    from app.services.s3 import delete
    delete("raw/nonexistent.txt")  # should not raise


def test_local_exists_true(monkeypatch, tmp_path):
    _patch_local(monkeypatch, tmp_path)
    p = tmp_path / "wiki" / "exists.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("content")
    from app.services.s3 import exists
    assert exists("wiki/exists.md") is True


def test_local_exists_false(monkeypatch, tmp_path):
    _patch_local(monkeypatch, tmp_path)
    from app.services.s3 import exists
    assert exists("wiki/nope.md") is False


def test_local_list_keys(monkeypatch, tmp_path):
    _patch_local(monkeypatch, tmp_path)
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "a.txt").write_text("a")
    (raw / "b.txt").write_text("b")

    from app.services.s3 import list_keys
    keys = list_keys("raw/")
    assert "raw/a.txt" in keys
    assert "raw/b.txt" in keys


def test_local_list_keys_empty_dir(monkeypatch, tmp_path):
    _patch_local(monkeypatch, tmp_path)
    from app.services.s3 import list_keys
    assert list_keys("raw/") == []


def test_local_get_object_size(monkeypatch, tmp_path):
    _patch_local(monkeypatch, tmp_path)
    p = tmp_path / "raw" / "sized.txt"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("hello")

    from app.services.s3 import get_object_size
    assert get_object_size("raw/sized.txt") == 5


def test_local_get_object_size_missing(monkeypatch, tmp_path):
    _patch_local(monkeypatch, tmp_path)
    from app.services.s3 import get_object_size
    assert get_object_size("raw/nope.txt") == 0


def test_local_ensure_bucket_creates_dirs(monkeypatch, tmp_path):
    _patch_local(monkeypatch, tmp_path)
    from app.services.s3 import ensure_bucket
    ensure_bucket()
    assert (tmp_path / "raw").is_dir()
    assert (tmp_path / "wiki").is_dir()
    assert (tmp_path / "schema").is_dir()


# ── S3 backend ────────────────────────────────────────────────────────────

def _make_s3_client(content: bytes = b"S3 content"):
    mock = MagicMock()
    mock.get_object.return_value = {"Body": MagicMock(read=MagicMock(return_value=content))}
    mock.head_object.return_value = {"ContentLength": len(content)}
    return mock


def test_s3_read(monkeypatch):
    mock_client = _make_s3_client(b"S3 data")
    _patch_s3(monkeypatch, mock_client)

    from app.services.s3 import read
    result = read("wiki/page.md")
    assert result == "S3 data"
    mock_client.get_object.assert_called_once_with(Bucket="test-bucket", Key="wiki/page.md")


def test_s3_read_missing(monkeypatch):
    from botocore.exceptions import ClientError
    mock_client = MagicMock()
    mock_client.get_object.side_effect = ClientError(
        {"Error": {"Code": "NoSuchKey", "Message": "Not found"}}, "GetObject"
    )
    _patch_s3(monkeypatch, mock_client)

    from app.services.s3 import read
    assert read("wiki/missing.md") == ""


def test_s3_write(monkeypatch):
    mock_client = MagicMock()
    _patch_s3(monkeypatch, mock_client)

    from app.services.s3 import write
    write("wiki/new.md", "Content here")
    mock_client.put_object.assert_called_once_with(
        Bucket="test-bucket", Key="wiki/new.md", Body=b"Content here"
    )


def test_s3_delete(monkeypatch):
    mock_client = MagicMock()
    _patch_s3(monkeypatch, mock_client)

    from app.services.s3 import delete
    delete("raw/file.txt")
    mock_client.delete_object.assert_called_once_with(Bucket="test-bucket", Key="raw/file.txt")


def test_s3_exists_true(monkeypatch):
    mock_client = MagicMock()
    mock_client.head_object.return_value = {"ContentLength": 10}
    _patch_s3(monkeypatch, mock_client)

    from app.services.s3 import exists
    assert exists("raw/file.txt") is True


def test_s3_exists_false(monkeypatch):
    from botocore.exceptions import ClientError
    mock_client = MagicMock()
    mock_client.head_object.side_effect = ClientError(
        {"Error": {"Code": "404", "Message": "Not found"}}, "HeadObject"
    )
    _patch_s3(monkeypatch, mock_client)

    from app.services.s3 import exists
    assert exists("raw/missing.txt") is False


def test_s3_list_keys(monkeypatch):
    mock_client = MagicMock()
    mock_paginator = MagicMock()
    mock_paginator.paginate.return_value = [
        {"Contents": [{"Key": "raw/a.txt"}, {"Key": "raw/b.txt"}]}
    ]
    mock_client.get_paginator.return_value = mock_paginator
    _patch_s3(monkeypatch, mock_client)

    from app.services.s3 import list_keys
    keys = list_keys("raw/")
    assert "raw/a.txt" in keys
    assert "raw/b.txt" in keys


def test_refresh_client_resets(monkeypatch):
    mock_client = MagicMock()
    _patch_s3(monkeypatch, mock_client)

    import app.services.s3 as s3_mod
    from app.services.s3 import refresh_client
    refresh_client()
    assert s3_mod._client is None
