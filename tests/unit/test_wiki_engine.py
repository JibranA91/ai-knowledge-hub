"""Unit tests for app/services/wiki_engine.py.

All LLM calls (BedrockService.converse) and DB calls are mocked.
"""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def _make_engine():
    """Create a WikiEngine instance without triggering __init__ side-effects."""
    from app.services.wiki_engine import WikiEngine
    engine = WikiEngine.__new__(WikiEngine)
    engine.query_bedrock = AsyncMock()
    engine._ingest_agent = MagicMock()
    return engine


# ── Internal helper methods ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_schema_returns_empty_when_none():
    engine = _make_engine()
    with patch("app.services.wiki_db.get_wiki_file", new_callable=AsyncMock, return_value=None):
        result = await engine._schema()
    assert result == ""


@pytest.mark.asyncio
async def test_compact_index_returns_placeholder_when_empty():
    engine = _make_engine()
    with patch("app.services.wiki_db.get_compact_index", new_callable=AsyncMock, return_value="No pages yet."):
        result = await engine._compact_index()
    assert "No pages yet" in result


@pytest.mark.asyncio
async def test_read_page_returns_empty_when_missing():
    engine = _make_engine()
    with patch("app.services.wiki_db.get_wiki_page_content", new_callable=AsyncMock, return_value=None):
        result = await engine._read_page("concepts/missing.md")
    assert result == ""


@pytest.mark.asyncio
async def test_write_page_calls_upsert():
    engine = _make_engine()
    with patch("app.services.wiki_db.upsert_wiki_page", new_callable=AsyncMock) as mock_upsert:
        await engine._write_page("concepts/foo.md", "# Content")
    mock_upsert.assert_awaited_once_with("concepts/foo.md", "# Content")


@pytest.mark.asyncio
async def test_append_log_calls_audit_log():
    engine = _make_engine()
    with patch("app.services.wiki_db.append_audit_log", new_callable=AsyncMock) as mock_audit:
        await engine._append_log("## [2026-01-01] lint | score=90")
    mock_audit.assert_awaited_once()


# ── ingest / plan / execute_ingest ────────────────────────────────────────

@pytest.mark.asyncio
async def test_ingest_delegates_to_agent():
    engine = _make_engine()
    engine._ingest_agent.run = AsyncMock(return_value={"pages_created": ["concepts/a.md"], "pages_updated": []})
    mock_graph = MagicMock()
    mock_graph.update_pages = AsyncMock()
    with patch.object(engine, "_graph", return_value=mock_graph):
        result = await engine.ingest("doc.txt")
    engine._ingest_agent.run.assert_awaited_once_with("doc.txt")
    mock_graph.update_pages.assert_awaited_once()


@pytest.mark.asyncio
async def test_ingest_skips_graph_update_when_no_touched():
    engine = _make_engine()
    engine._ingest_agent.run = AsyncMock(return_value={"pages_created": [], "pages_updated": []})
    mock_graph = MagicMock()
    mock_graph.update_pages = AsyncMock()
    with patch.object(engine, "_graph", return_value=mock_graph):
        await engine.ingest("doc.txt")
    mock_graph.update_pages.assert_not_awaited()


@pytest.mark.asyncio
async def test_plan_delegates_to_agent():
    engine = _make_engine()
    engine._ingest_agent.plan = AsyncMock(return_value={"plan": [], "conflicts": []})
    result = await engine.plan("doc.txt")
    engine._ingest_agent.plan.assert_awaited_once_with("doc.txt")
    assert result["plan"] == []


@pytest.mark.asyncio
async def test_execute_ingest_updates_graph():
    engine = _make_engine()
    engine._ingest_agent.execute = AsyncMock(return_value={"pages_created": ["a.md"], "pages_updated": []})
    mock_graph = MagicMock()
    mock_graph.update_pages = AsyncMock()
    with patch.object(engine, "_graph", return_value=mock_graph):
        result = await engine.execute_ingest("doc.txt", [], [], "log entry")
    mock_graph.update_pages.assert_awaited_once()


@pytest.mark.asyncio
async def test_graph_dict_delegates_to_graph():
    engine = _make_engine()
    mock_graph = MagicMock()
    mock_graph.ensure_loaded = AsyncMock(return_value=mock_graph)
    mock_graph.as_dict.return_value = {"nodes": [], "edges": []}
    with patch.object(engine, "_graph", return_value=mock_graph):
        result = await engine.graph_dict()
    assert result == {"nodes": [], "edges": []}


# ── plan_chat ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_plan_chat_returns_updated_plan():
    engine = _make_engine()
    response = json.dumps({
        "reply": "Updated plan.",
        "updated_plan": [{"path": "concepts/foo.md", "action": "create", "brief": "foo"}],
        "index_additions": ["- [Foo](concepts/foo.md)"],
        "log_entry": "## [2026] ingest | added foo",
    })
    engine.query_bedrock.converse = AsyncMock(return_value=response)

    with patch("app.services.wiki_db.get_wiki_file", new_callable=AsyncMock, return_value=""):
        result = await engine.plan_chat(
            filename="doc.txt",
            message="Add a foo concept.",
            current_plan=[],
            history=[],
            doc_text="Some content.",
        )

    assert result["reply"] == "Updated plan."
    assert len(result["updated_plan"]) == 1


@pytest.mark.asyncio
async def test_plan_chat_handles_non_json_response():
    engine = _make_engine()
    engine.query_bedrock.converse = AsyncMock(return_value="I cannot help.")

    with patch("app.services.wiki_db.get_wiki_file", new_callable=AsyncMock, return_value=""):
        result = await engine.plan_chat(
            filename="doc.txt",
            message="?",
            current_plan=[],
            history=[],
            doc_text="text",
        )

    assert result["reply"] == "I cannot help."
    assert result["updated_plan"] is None


@pytest.mark.asyncio
async def test_plan_chat_includes_conflicts_in_prompt():
    engine = _make_engine()
    captured = []

    async def _spy(system, msgs, **kw):
        captured.append(system)
        return json.dumps({"reply": "ok", "updated_plan": None, "index_additions": None, "log_entry": None})

    engine.query_bedrock.converse.side_effect = _spy

    conflicts = [{"path": "concepts/a.md", "existing_claim": "X", "new_claim": "Y", "resolution": "surface"}]
    with patch("app.services.wiki_db.get_wiki_file", new_callable=AsyncMock, return_value=""):
        await engine.plan_chat("doc.txt", "msg", [], [], "doc", conflicts=conflicts)

    assert captured and "Detected Conflicts" in captured[0]


# ── query ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_query_returns_answer_and_sources():
    engine = _make_engine()
    engine.query_bedrock.converse = AsyncMock(return_value="42 is the answer.")

    with patch.object(engine, "_find_relevant_pages", new_callable=AsyncMock, return_value=["concepts/a.md"]), \
         patch.object(engine, "_read_page", new_callable=AsyncMock, return_value="# A\n\nContent."), \
         patch.object(engine, "_schema", new_callable=AsyncMock, return_value=""):
        result = await engine.query("What is 42?")

    assert result["answer"] == "42 is the answer."
    assert "concepts/a.md" in result["sources"]
    assert result["saved_to"] is None


@pytest.mark.asyncio
async def test_query_saves_to_wiki_when_requested():
    engine = _make_engine()
    engine.query_bedrock.converse = AsyncMock(return_value="The answer.")

    with patch.object(engine, "_find_relevant_pages", new_callable=AsyncMock, return_value=[]), \
         patch.object(engine, "_schema", new_callable=AsyncMock, return_value=""), \
         patch.object(engine, "_write_page", new_callable=AsyncMock) as mock_write, \
         patch.object(engine, "_append_log", new_callable=AsyncMock):
        result = await engine.query("What is the answer?", save_to_wiki=True)

    mock_write.assert_awaited_once()
    assert result["saved_to"] is not None
    assert result["saved_to"].startswith("queries/")


# ── chat ──────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_chat_creates_session_and_returns_answer():
    from app.services.chat_sessions import ChatSession
    engine = _make_engine()
    engine.query_bedrock.converse = AsyncMock(return_value="Hello!")

    session = ChatSession()
    with patch("app.services.chat_sessions.get_or_create", new_callable=AsyncMock, return_value=session), \
         patch("app.services.chat_sessions.save", new_callable=AsyncMock), \
         patch.object(engine, "_find_relevant_pages", new_callable=AsyncMock, return_value=[]), \
         patch.object(engine, "_schema", new_callable=AsyncMock, return_value=""), \
         patch.object(engine, "_maybe_summarize", new_callable=AsyncMock):
        result = await engine.chat(None, "Hi!")

    assert result["answer"] == "Hello!"
    assert result["session_id"] == session.session_id


@pytest.mark.asyncio
async def test_chat_appends_messages_to_session():
    from app.services.chat_sessions import ChatSession
    engine = _make_engine()
    engine.query_bedrock.converse = AsyncMock(return_value="World!")

    session = ChatSession()
    with patch("app.services.chat_sessions.get_or_create", new_callable=AsyncMock, return_value=session), \
         patch("app.services.chat_sessions.save", new_callable=AsyncMock), \
         patch.object(engine, "_find_relevant_pages", new_callable=AsyncMock, return_value=[]), \
         patch.object(engine, "_schema", new_callable=AsyncMock, return_value=""), \
         patch.object(engine, "_maybe_summarize", new_callable=AsyncMock):
        await engine.chat(session.session_id, "Hello!")

    assert len(session.messages) == 2
    assert session.messages[0]["role"] == "user"
    assert session.messages[1]["role"] == "assistant"


@pytest.mark.asyncio
async def test_chat_saves_to_wiki_when_requested():
    from app.services.chat_sessions import ChatSession
    engine = _make_engine()
    engine.query_bedrock.converse = AsyncMock(return_value="answer")

    session = ChatSession()
    with patch("app.services.chat_sessions.get_or_create", new_callable=AsyncMock, return_value=session), \
         patch("app.services.chat_sessions.save", new_callable=AsyncMock), \
         patch.object(engine, "_find_relevant_pages", new_callable=AsyncMock, return_value=[]), \
         patch.object(engine, "_schema", new_callable=AsyncMock, return_value=""), \
         patch.object(engine, "_maybe_summarize", new_callable=AsyncMock), \
         patch.object(engine, "_write_page", new_callable=AsyncMock) as mock_write, \
         patch.object(engine, "_append_log", new_callable=AsyncMock):
        result = await engine.chat(None, "What is X?", save_to_wiki=True)

    mock_write.assert_awaited_once()
    assert result["saved_to"] is not None


# ── chat_stream ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_chat_stream_yields_sse_events():
    from app.services.chat_sessions import ChatSession

    async def _fake_stream(*args, **kwargs):
        yield "Hello"
        yield " world"

    engine = _make_engine()
    engine.query_bedrock.converse_stream = _fake_stream

    session = ChatSession()
    with patch("app.services.chat_sessions.get_or_create", new_callable=AsyncMock, return_value=session), \
         patch("app.services.chat_sessions.save", new_callable=AsyncMock), \
         patch.object(engine, "_find_relevant_pages", new_callable=AsyncMock, return_value=[]), \
         patch.object(engine, "_maybe_summarize", new_callable=AsyncMock):
        events = []
        async for event in engine.chat_stream(None, "Hi"):
            events.append(event)

    event_types = [json.loads(e[len("data: "):])["type"] for e in events]
    assert "meta" in event_types
    assert "chunk" in event_types
    assert "done" in event_types


# ── edit_stream (inline AI editing) ────────────────────────────────────────

_EDIT_DOC = "---\ntitle: Demo\n---\n# Demo\n\nIntro.\n\n## Background\n\nOld background.\n\n## Details\n\nStuff.\n"


def _edit_engine(stream_chunks):
    """A WikiEngine whose edit model streams *stream_chunks*, with grounding stubbed."""
    engine = _make_engine()

    async def _fake_stream(*args, **kwargs):
        for c in stream_chunks:
            yield c

    engine.edit_bedrock = AsyncMock()
    engine.edit_bedrock.converse_stream = _fake_stream
    engine._find_relevant_pages = AsyncMock(return_value=[])
    engine._schema = AsyncMock(return_value="")
    engine._compact_index = AsyncMock(return_value="")
    return engine


def _events(raw_events):
    return [json.loads(e[len("data: "):]) for e in raw_events]


@pytest.mark.asyncio
async def test_edit_stream_page_returns_full_content():
    engine = _edit_engine(["# Demo\n\n", "Better intro.\n"])
    with patch.object(engine, "_read_page", new_callable=AsyncMock, return_value=_EDIT_DOC):
        evts = _events([e async for e in engine.edit_stream("concepts/d.md", "page", "improve")])

    types = [e["type"] for e in evts]
    assert types[0] == "meta"
    assert "chunk" in types
    assert types[-1] == "done"
    done = evts[-1]
    assert done["full_content"] == "# Demo\n\nBetter intro.\n"
    assert done["scope"] == "page"


@pytest.mark.asyncio
async def test_edit_stream_section_splices_back_into_page():
    engine = _edit_engine(["## Background\n\nFresh background.\n"])
    with patch.object(engine, "_read_page", new_callable=AsyncMock, return_value=_EDIT_DOC):
        evts = _events([e async for e in engine.edit_stream(
            "concepts/d.md", "section", "expand", heading="Background")])

    done = evts[-1]
    assert done["type"] == "done"
    # New section spliced in; siblings preserved; frontmatter/title intact.
    assert "Fresh background." in done["full_content"]
    assert "Old background." not in done["full_content"]
    assert "## Details" in done["full_content"]
    assert done["full_content"].startswith("---\ntitle: Demo")


@pytest.mark.asyncio
async def test_edit_stream_strips_code_fence():
    engine = _edit_engine(["```markdown\n# Demo\n\nClean.\n```"])
    with patch.object(engine, "_read_page", new_callable=AsyncMock, return_value=_EDIT_DOC):
        evts = _events([e async for e in engine.edit_stream("concepts/d.md", "page", "improve")])
    assert evts[-1]["full_content"] == "# Demo\n\nClean.\n"


@pytest.mark.asyncio
async def test_edit_stream_errors_on_missing_page():
    engine = _edit_engine(["unused"])
    with patch.object(engine, "_read_page", new_callable=AsyncMock, return_value=""):
        evts = _events([e async for e in engine.edit_stream("missing.md", "page", "improve")])
    assert len(evts) == 1
    assert evts[0]["type"] == "error"
    assert "not found" in evts[0]["message"].lower()


@pytest.mark.asyncio
async def test_edit_stream_errors_on_unknown_section():
    engine = _edit_engine(["unused"])
    with patch.object(engine, "_read_page", new_callable=AsyncMock, return_value=_EDIT_DOC):
        evts = _events([e async for e in engine.edit_stream(
            "concepts/d.md", "section", "improve", heading="Nonexistent")])
    assert evts[-1]["type"] == "error"
    assert not any(e["type"] == "done" for e in evts)


@pytest.mark.asyncio
async def test_edit_stream_prompt_includes_link_catalog():
    """The cross-linking catalog (as [[page-name]] tokens) is injected into the
    system prompt so the agent can link concepts it mentions."""
    captured = {}

    async def capturing_stream(system_prompt, messages, **kw):
        captured["sys"] = system_prompt
        yield "# Demo\n\nBody.\n"

    engine = _make_engine()
    engine.edit_bedrock = AsyncMock()
    engine.edit_bedrock.converse_stream = capturing_stream
    engine._find_relevant_pages = AsyncMock(return_value=[])
    engine._schema = AsyncMock(return_value="")
    engine._compact_index = AsyncMock(return_value=(
        "concepts/widget.md — Widget\nconcepts/gadget.md — Gadget\nindex.md — Index"
    ))

    with patch.object(engine, "_read_page", new_callable=AsyncMock, return_value=_EDIT_DOC):
        _ = [e async for e in engine.edit_stream("concepts/d.md", "page", "improve")]

    sys = captured["sys"]
    assert "CROSS-LINKING" in sys
    assert "[[widget]]" in sys and "[[gadget]]" in sys   # stems, ready to use
    assert "[[index]]" not in sys                         # index.md excluded


@pytest.mark.asyncio
async def test_edit_stream_reconcile_includes_other_page_in_sources():
    engine = _edit_engine(["# Demo\n\nReconciled.\n"])

    async def _read(path):
        return "# Other\n\nAuthoritative." if path == "concepts/other.md" else _EDIT_DOC

    with patch.object(engine, "_read_page", new=AsyncMock(side_effect=_read)):
        evts = _events([e async for e in engine.edit_stream(
            "concepts/d.md", "page", "reconcile", reconcile_with="concepts/other.md")])

    meta = evts[0]
    assert meta["type"] == "meta"
    assert "concepts/other.md" in meta["sources"]


# ── _parse_json ───────────────────────────────────────────────────────────

def test_parse_json_clean():
    from app.utils import parse_llm_json
    assert parse_llm_json('{"key": "value"}') == {"key": "value"}


def test_parse_json_code_block():
    from app.utils import parse_llm_json
    text = '```json\n{"pages": [1, 2]}\n```'
    assert parse_llm_json(text) == {"pages": [1, 2]}


def test_parse_json_code_block_no_lang():
    from app.utils import parse_llm_json
    text = '```\n{"x": true}\n```'
    assert parse_llm_json(text) == {"x": True}


def test_parse_json_embedded_in_prose():
    from app.utils import parse_llm_json
    text = 'Sure! Here is the plan: {"a": 1} — done.'
    assert parse_llm_json(text) == {"a": 1}


def test_parse_json_raises_on_no_json():
    from app.utils import parse_llm_json
    with pytest.raises(ValueError):
        parse_llm_json("no JSON here at all")


# ── _find_relevant_pages ──────────────────────────────────────────────────
# The function now calls wiki_db.find_relevant_pages first; LLM is only used
# as a fallback when DB returns nothing. Tests that exercise the LLM path must
# patch wiki_db.find_relevant_pages to return [].

@pytest.mark.asyncio
async def test_find_relevant_pages_uses_db_when_available():
    """DB results short-circuit the LLM call."""
    engine = _make_engine()
    mock_graph = MagicMock()
    mock_graph.neighbors.return_value = []

    with patch("app.services.wiki_db.find_relevant_pages", new_callable=AsyncMock,
               return_value=["concepts/foo.md", "concepts/bar.md"]), \
         patch.object(engine, "_graph", return_value=mock_graph):
        pages = await engine._find_relevant_pages("what is foo?")

    assert "concepts/foo.md" in pages
    assert "concepts/bar.md" in pages
    engine.query_bedrock.converse.assert_not_called()


@pytest.mark.asyncio
async def test_find_relevant_pages_db_results_get_graph_neighbours():
    """Graph neighbours are appended even when DB provides the primary results."""
    engine = _make_engine()
    mock_graph = MagicMock()
    mock_graph.neighbors.return_value = ["neighbour.md"]

    with patch("app.services.wiki_db.find_relevant_pages", new_callable=AsyncMock,
               return_value=["a.md"]), \
         patch.object(engine, "_graph", return_value=mock_graph):
        pages = await engine._find_relevant_pages("question")

    assert "a.md" in pages
    assert "neighbour.md" in pages


@pytest.mark.asyncio
async def test_find_relevant_pages_falls_back_to_llm_when_db_empty():
    """When DB returns nothing, the LLM index scan is used."""
    engine = _make_engine()
    engine.query_bedrock.converse.return_value = (
        '{"relevant_pages": ["concepts/foo.md", "concepts/bar.md"]}'
    )
    mock_graph = MagicMock()
    mock_graph.neighbors.return_value = []

    with patch("app.services.wiki_db.find_relevant_pages", new_callable=AsyncMock, return_value=[]), \
         patch.object(engine, "_compact_index", new_callable=AsyncMock, return_value="concepts/foo.md — Foo"), \
         patch("app.services.wiki_db.wiki_page_exists", new_callable=AsyncMock, return_value=True), \
         patch.object(engine, "_graph", return_value=mock_graph):
        pages = await engine._find_relevant_pages("what is foo?")

    assert "concepts/foo.md" in pages
    assert "concepts/bar.md" in pages


@pytest.mark.asyncio
async def test_find_relevant_pages_llm_excludes_index_and_log():
    """LLM fallback excludes index.md and log.md from results."""
    engine = _make_engine()
    engine.query_bedrock.converse.return_value = (
        '{"relevant_pages": ["index.md", "log.md", "concepts/real.md"]}'
    )
    mock_graph = MagicMock()
    mock_graph.neighbors.return_value = []

    with patch("app.services.wiki_db.find_relevant_pages", new_callable=AsyncMock, return_value=[]), \
         patch.object(engine, "_compact_index", new_callable=AsyncMock, return_value="concepts/foo.md — Foo"), \
         patch("app.services.wiki_db.wiki_page_exists", new_callable=AsyncMock, return_value=True), \
         patch.object(engine, "_graph", return_value=mock_graph):
        pages = await engine._find_relevant_pages("question")

    assert "index.md" not in pages
    assert "log.md" not in pages
    assert "concepts/real.md" in pages


@pytest.mark.asyncio
async def test_find_relevant_pages_llm_excludes_nonexistent():
    """LLM fallback skips paths that don't exist in the DB."""
    engine = _make_engine()
    engine.query_bedrock.converse.return_value = (
        '{"relevant_pages": ["missing.md"]}'
    )
    mock_graph = MagicMock()
    mock_graph.neighbors.return_value = []

    with patch("app.services.wiki_db.find_relevant_pages", new_callable=AsyncMock, return_value=[]), \
         patch.object(engine, "_compact_index", new_callable=AsyncMock, return_value="concepts/foo.md — Foo"), \
         patch("app.services.wiki_db.wiki_page_exists", new_callable=AsyncMock, return_value=False), \
         patch.object(engine, "_graph", return_value=mock_graph):
        pages = await engine._find_relevant_pages("question")

    assert "missing.md" not in pages


@pytest.mark.asyncio
async def test_find_relevant_pages_bad_llm_response():
    """Non-JSON LLM response returns empty list rather than raising."""
    engine = _make_engine()
    engine.query_bedrock.converse.return_value = "Sorry, I cannot help with that."
    mock_graph = MagicMock()
    mock_graph.neighbors.return_value = []

    with patch("app.services.wiki_db.find_relevant_pages", new_callable=AsyncMock, return_value=[]), \
         patch.object(engine, "_compact_index", new_callable=AsyncMock, return_value="concepts/foo.md — Foo"), \
         patch("app.services.wiki_db.wiki_page_exists", new_callable=AsyncMock, return_value=True), \
         patch.object(engine, "_graph", return_value=mock_graph):
        pages = await engine._find_relevant_pages("question")

    assert pages == []


# ── _maybe_summarize ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_maybe_summarize_triggers_after_threshold():
    from app.services.chat_sessions import ChatSession, SUMMARIZE_AFTER, KEEP_RECENT
    engine = _make_engine()
    engine.query_bedrock.converse.return_value = "This is the summary."

    session = ChatSession()
    session.messages = [
        {"role": "user", "text": f"msg {i}", "sources": []}
        for i in range(SUMMARIZE_AFTER + 1)
    ]
    await engine._maybe_summarize(session)

    assert session.summary == "This is the summary."
    assert len(session.messages) == KEEP_RECENT


@pytest.mark.asyncio
async def test_maybe_summarize_no_trigger_under_threshold():
    from app.services.chat_sessions import ChatSession, SUMMARIZE_AFTER
    engine = _make_engine()
    engine.query_bedrock.converse.return_value = "Should not be called."

    session = ChatSession()
    session.messages = [
        {"role": "user", "text": f"msg {i}", "sources": []}
        for i in range(SUMMARIZE_AFTER - 1)
    ]
    await engine._maybe_summarize(session)

    # LLM should NOT have been called
    engine.query_bedrock.converse.assert_not_called()
    assert session.summary == ""


@pytest.mark.asyncio
async def test_maybe_summarize_prepends_existing_summary():
    from app.services.chat_sessions import ChatSession, SUMMARIZE_AFTER
    engine = _make_engine()
    engine.query_bedrock.converse.return_value = "New summary."

    session = ChatSession()
    session.summary = "Old summary."
    session.messages = [
        {"role": "user", "text": f"msg {i}", "sources": []}
        for i in range(SUMMARIZE_AFTER + 1)
    ]
    await engine._maybe_summarize(session)

    # The prompt sent to LLM should include the prior summary
    call_args = engine.query_bedrock.converse.call_args
    prompt_text = call_args[0][1][0]["content"][0]["text"]
    assert "Old summary." in prompt_text


# ── lint ──────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
def _mock_graph(adj=None):
    """A _graph() stand-in whose ensure_loaded() yields an object with ._adj."""
    loaded = MagicMock()
    loaded._adj = adj or {}
    holder = MagicMock()
    holder.ensure_loaded = AsyncMock(return_value=loaded)
    return holder


async def test_lint_returns_structured_result():
    engine = _make_engine()
    # The LLM supplies only the qualitative issues/suggestions now; the score is
    # computed deterministically and the LLM's health_score (if any) is ignored.
    engine.query_bedrock.converse.return_value = json.dumps({
        "issues": [{"type": "cross_ref", "description": "weak linking", "affected_pages": ["concepts/real.md"]}],
        "suggestions": ["Link related concepts."],
        "health_score": 72,
    })

    with patch("app.services.wiki_db.list_wiki_pages_with_content", new_callable=AsyncMock) as mock_pages, \
         patch("app.services.wiki_db.get_recalibration_state", new_callable=AsyncMock, return_value={}), \
         patch.object(engine, "_graph", return_value=_mock_graph()), \
         patch.object(engine, "_compact_index", new_callable=AsyncMock, return_value="concepts/foo.md — Foo"), \
         patch.object(engine, "_append_log", new_callable=AsyncMock):
        mock_pages.return_value = [("concepts/real.md", "Content")]
        result = await engine.lint()

    # Score is computed (deterministic), NOT the LLM's 72.
    assert isinstance(result["health_score"], int)
    assert 0 <= result["health_score"] <= 100
    assert "breakdown" in result and "total_pages" in result
    # Qualitative issues/suggestions still flow through from the LLM.
    assert len(result["issues"]) == 1
    assert result["issues"][0]["type"] == "cross_ref"
    assert len(result["suggestions"]) == 1


@pytest.mark.asyncio
async def test_lint_handles_malformed_llm_response():
    engine = _make_engine()
    engine.query_bedrock.converse.return_value = "Not valid JSON at all."

    with patch("app.services.wiki_db.list_wiki_pages_with_content", new_callable=AsyncMock) as mock_pages, \
         patch("app.services.wiki_db.get_recalibration_state", new_callable=AsyncMock, return_value={}), \
         patch.object(engine, "_graph", return_value=_mock_graph()), \
         patch.object(engine, "_compact_index", new_callable=AsyncMock, return_value="concepts/foo.md — Foo"), \
         patch.object(engine, "_append_log", new_callable=AsyncMock):
        mock_pages.return_value = []
        result = await engine.lint()

    # Malformed audit is swallowed; the deterministic score is still returned.
    assert result["health_score"] == 100  # empty wiki
    assert result["issues"] == []
    assert result["suggestions"] == []


# ── qualitative-audit excerpt (no false "truncated page" reports) ────────────

def test_audit_excerpt_short_page_unchanged():
    from app.services.wiki_engine import _audit_excerpt
    s = "# Short\n\n" + "word " * 50
    assert _audit_excerpt(s) == s


def test_audit_excerpt_long_page_marked_not_silently_cut():
    """A page longer than the cap is capped BUT carries an explicit marker, so the
    audit LLM won't mistake the excerpt's end for a truncated page (the 3000-char
    cut previously flagged complete pages as 'incomplete / ends mid-sentence')."""
    from app.services.wiki_engine import _audit_excerpt, _AUDIT_PAGE_CHARS
    long = "x" * (_AUDIT_PAGE_CHARS + 1000)
    out = _audit_excerpt(long)
    assert out.startswith("x" * _AUDIT_PAGE_CHARS)
    assert "excerpt truncated for length" in out
    # Cap is well above the old 3000 so typical concept pages go in full.
    assert _AUDIT_PAGE_CHARS > 3000


@pytest.mark.asyncio
async def test_build_wiki_export_returns_tuple():
    """build_wiki_export returns (bytes, page_count, has_embeddings)."""
    engine = _make_engine()
    pages = [
        {"path": "concepts/foo.md", "content": "# Foo\n\nContent.", "embedding": None},
        {"path": "sources/doc.md", "content": "# Doc\n\nSource.", "embedding": None},
    ]
    with patch("app.services.wiki_db.list_wiki_pages_for_export", new_callable=AsyncMock, return_value=pages), \
         patch("app.services.wiki_db.get_wiki_file", new_callable=AsyncMock, return_value="{}"):
        result = await engine.build_wiki_export(include_embeddings=True)

    zip_bytes, page_count, has_embeddings = result
    assert isinstance(zip_bytes, bytes)
    assert page_count == 2
    assert has_embeddings is False


@pytest.mark.asyncio
async def test_build_wiki_export_zip_structure():
    """ZIP contains wiki pages, graph.json, manifest.json, retriever.py, README.md."""
    import io, zipfile
    engine = _make_engine()
    pages = [{"path": "concepts/foo.md", "content": "# Foo", "embedding": None}]

    with patch("app.services.wiki_db.list_wiki_pages_for_export", new_callable=AsyncMock, return_value=pages), \
         patch("app.services.wiki_db.get_wiki_file", new_callable=AsyncMock, return_value="{}"):
        zip_bytes, _, _ = await engine.build_wiki_export(include_embeddings=False)

    names = zipfile.ZipFile(io.BytesIO(zip_bytes)).namelist()
    assert "wiki/concepts/foo.md" in names
    assert "graph.json" in names
    assert "manifest.json" in names
    assert "retriever.py" in names
    assert "README.md" in names
    assert "embeddings.json" not in names


@pytest.mark.asyncio
async def test_build_wiki_export_manifest_fields():
    """manifest.json contains all required fields with correct values."""
    import io, zipfile
    engine = _make_engine()
    pages = [{"path": "index.md", "content": "# Index", "embedding": None}]

    with patch("app.services.wiki_db.list_wiki_pages_for_export", new_callable=AsyncMock, return_value=pages), \
         patch("app.services.wiki_db.get_wiki_file", new_callable=AsyncMock, return_value="{}"):
        zip_bytes, _, _ = await engine.build_wiki_export(include_embeddings=False)

    manifest = json.loads(zipfile.ZipFile(io.BytesIO(zip_bytes)).read("manifest.json"))
    assert "exported_at" in manifest
    assert manifest["page_count"] == 1
    assert manifest["has_embeddings"] is False
    assert "embedding_dimensions" in manifest
    assert "embedding_model" in manifest


@pytest.mark.asyncio
async def test_build_wiki_export_embeddings_included_when_present():
    """embeddings.json is written and has_embeddings=True when pages have embeddings."""
    import io, zipfile
    engine = _make_engine()
    pages = [
        {"path": "concepts/foo.md", "content": "# Foo", "embedding": [0.1, 0.2, 0.3]},
        {"path": "concepts/bar.md", "content": "# Bar", "embedding": None},
    ]

    with patch("app.services.wiki_db.list_wiki_pages_for_export", new_callable=AsyncMock, return_value=pages), \
         patch("app.services.wiki_db.get_wiki_file", new_callable=AsyncMock, return_value="{}"):
        zip_bytes, page_count, has_embeddings = await engine.build_wiki_export(include_embeddings=True)

    assert has_embeddings is True
    assert page_count == 2
    zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    assert "embeddings.json" in zf.namelist()
    emb_list = json.loads(zf.read("embeddings.json"))
    assert len(emb_list) == 1
    assert emb_list[0]["path"] == "concepts/foo.md"
    assert emb_list[0]["embedding"] == [0.1, 0.2, 0.3]


@pytest.mark.asyncio
async def test_build_wiki_export_no_embeddings_json_when_none_present():
    """embeddings.json is absent when all pages have embedding=None."""
    import io, zipfile
    engine = _make_engine()
    pages = [{"path": "concepts/foo.md", "content": "# Foo", "embedding": None}]

    with patch("app.services.wiki_db.list_wiki_pages_for_export", new_callable=AsyncMock, return_value=pages), \
         patch("app.services.wiki_db.get_wiki_file", new_callable=AsyncMock, return_value="{}"):
        zip_bytes, _, has_embeddings = await engine.build_wiki_export(include_embeddings=True)

    assert has_embeddings is False
    assert "embeddings.json" not in zipfile.ZipFile(io.BytesIO(zip_bytes)).namelist()


@pytest.mark.asyncio
async def test_build_wiki_export_retriever_is_valid_python():
    """retriever.py in the ZIP is syntactically valid Python."""
    import io, zipfile, ast
    engine = _make_engine()
    pages = [{"path": "index.md", "content": "# Index", "embedding": None}]

    with patch("app.services.wiki_db.list_wiki_pages_for_export", new_callable=AsyncMock, return_value=pages), \
         patch("app.services.wiki_db.get_wiki_file", new_callable=AsyncMock, return_value="{}"):
        zip_bytes, _, _ = await engine.build_wiki_export(include_embeddings=False)

    src = zipfile.ZipFile(io.BytesIO(zip_bytes)).read("retriever.py").decode("utf-8")
    ast.parse(src)  # raises SyntaxError if invalid


@pytest.mark.asyncio
async def test_lint_skips_query_pages():
    """Pages under queries/ directory are not sent to the LLM."""
    engine = _make_engine()
    engine.query_bedrock.converse.return_value = json.dumps({
        "issues": [], "suggestions": [], "health_score": 100
    })

    captured_text = []

    async def _fake_converse(system, msgs, **kw):
        captured_text.append(msgs[0]["content"][0]["text"])
        return '{"issues": [], "suggestions": [], "health_score": 100}'

    engine.query_bedrock.converse.side_effect = _fake_converse

    pages = [
        ("concepts/real.md", "Real content"),
        ("queries/q1.md", "Query result — should be excluded"),
    ]

    with patch("app.services.wiki_db.list_wiki_pages_with_content", new_callable=AsyncMock) as mock_pages, \
         patch("app.services.wiki_db.get_recalibration_state", new_callable=AsyncMock, return_value={}), \
         patch.object(engine, "_compact_index", new_callable=AsyncMock, return_value="concepts/foo.md — Foo"), \
         patch.object(engine, "_append_log", new_callable=AsyncMock):
        mock_pages.return_value = pages
        await engine.lint()

    assert captured_text
    assert "queries/q1.md" not in captured_text[0]
