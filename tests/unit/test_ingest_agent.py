"""Unit tests for app/services/ingest_agent.py.

Tests only the pure-Python helpers: _parse_json, _split_chunks, _extract_text,
_base_state.  The LangGraph graph execution is not tested here (requires a real
LLM / heavy mocking of the entire graph).
"""
import io
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ── _parse_json ───────────────────────────────────────────────────────────

def test_parse_json_clean():
    from app.utils import parse_llm_json
    assert parse_llm_json('{"key": 1}') == {"key": 1}


def test_parse_json_code_block():
    from app.utils import parse_llm_json
    assert parse_llm_json('```json\n{"x": true}\n```') == {"x": True}


def test_parse_json_embedded():
    from app.utils import parse_llm_json
    assert parse_llm_json('Here: {"a": 2} — done.') == {"a": 2}


def test_parse_json_raises_on_none():
    from app.utils import parse_llm_json
    with pytest.raises(ValueError):
        parse_llm_json("no json here at all")


# ── _split_chunks ─────────────────────────────────────────────────────────

def test_split_chunks_short_text_returns_single():
    from app.services.ingest_agent import IngestAgent
    result = IngestAgent._split_chunks("short text", chunk_size=100, overlap=10)
    assert result == ["short text"]


def test_split_chunks_long_text_splits():
    from app.services.ingest_agent import IngestAgent
    text = "a" * 200
    result = IngestAgent._split_chunks(text, chunk_size=100, overlap=10)
    assert len(result) > 1


def test_split_chunks_overlap_produces_multiple_chunks():
    from app.services.ingest_agent import IngestAgent
    # Use text with frequent newlines so rfind always advances past overlap
    text = ("word word word word word\n") * 20   # 500 chars, newline every 25
    chunks = IngestAgent._split_chunks(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    # All chunks should be non-empty
    for chunk in chunks:
        assert len(chunk) > 0


def test_split_chunks_exact_size():
    from app.services.ingest_agent import IngestAgent
    text = "x" * 100
    result = IngestAgent._split_chunks(text, chunk_size=100, overlap=10)
    assert result == [text]


# ── _read_raw / _parse_raw ────────────────────────────────────────────────

def test_extract_text_txt_file(monkeypatch, tmp_path):
    from app import config
    monkeypatch.setattr(config.settings, "STORAGE_BACKEND", "local")
    monkeypatch.setattr(config.settings, "DATA_DIR", str(tmp_path))
    import app.services.s3 as s3_mod
    s3_mod._client = None

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "doc.txt").write_bytes(b"Hello world!")

    from app.services.ingest_agent import IngestAgent
    agent = IngestAgent.__new__(IngestAgent)
    raw, suffix = agent._read_raw("doc.txt")
    result = IngestAgent._parse_raw(raw, suffix)
    assert result == "Hello world!"


def test_extract_text_md_file(monkeypatch, tmp_path):
    from app import config
    monkeypatch.setattr(config.settings, "STORAGE_BACKEND", "local")
    monkeypatch.setattr(config.settings, "DATA_DIR", str(tmp_path))
    import app.services.s3 as s3_mod
    s3_mod._client = None

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "doc.md").write_bytes(b"# Title\n\nContent.")

    from app.services.ingest_agent import IngestAgent
    agent = IngestAgent.__new__(IngestAgent)
    raw, suffix = agent._read_raw("doc.md")
    result = IngestAgent._parse_raw(raw, suffix)
    assert "Title" in result


def test_extract_text_missing_file_raises(monkeypatch, tmp_path):
    from app import config
    monkeypatch.setattr(config.settings, "STORAGE_BACKEND", "local")
    monkeypatch.setattr(config.settings, "DATA_DIR", str(tmp_path))
    import app.services.s3 as s3_mod
    s3_mod._client = None

    (tmp_path / "raw").mkdir()

    from app.services.ingest_agent import IngestAgent
    agent = IngestAgent.__new__(IngestAgent)
    with pytest.raises(FileNotFoundError):
        agent._read_raw("missing.txt")


def test_extract_text_unknown_extension_falls_back(monkeypatch, tmp_path):
    from app import config
    monkeypatch.setattr(config.settings, "STORAGE_BACKEND", "local")
    monkeypatch.setattr(config.settings, "DATA_DIR", str(tmp_path))
    import app.services.s3 as s3_mod
    s3_mod._client = None

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "doc.xyz").write_bytes(b"Unknown format text")

    from app.services.ingest_agent import IngestAgent
    agent = IngestAgent.__new__(IngestAgent)
    raw, suffix = agent._read_raw("doc.xyz")
    result = IngestAgent._parse_raw(raw, suffix)
    assert "Unknown format text" in result


# ── _base_state ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_base_state_returns_correct_keys():
    from app.services.ingest_agent import IngestAgent
    agent = IngestAgent.__new__(IngestAgent)

    with patch("app.services.wiki_db.get_wiki_file", new_callable=AsyncMock, return_value="schema content"), \
         patch("app.services.wiki_db.get_compact_index", new_callable=AsyncMock, return_value="concepts/a.md — A"), \
         patch("app.services.wiki_db.semantic_search_wiki", new_callable=AsyncMock, return_value=[]):
        state = await agent._base_state("doc.txt", "full text", "chunk text", 0, 1, 0.0)

    assert state["filename"] == "doc.txt"
    assert state["doc_text"] == "full text"
    assert state["chunk_text"] == "chunk text"
    assert state["chunk_index"] == 0
    assert state["total_chunks"] == 1
    assert state["plan"] == []
    assert state["conflicts"] == []
    assert state["pages_created"] == []
    assert state["pages_updated"] == []


@pytest.mark.asyncio
async def test_base_state_uses_default_index_when_empty():
    from app.services.ingest_agent import IngestAgent
    agent = IngestAgent.__new__(IngestAgent)

    with patch("app.services.wiki_db.get_wiki_file", new_callable=AsyncMock, return_value=None), \
         patch("app.services.wiki_db.get_compact_index", new_callable=AsyncMock, return_value="No pages yet."), \
         patch("app.services.wiki_db.semantic_search_wiki", new_callable=AsyncMock, return_value=[]):
        state = await agent._base_state("doc.txt", "text", "chunk", 0, 1, 0.0)

    assert "No pages yet" in state["index"]
