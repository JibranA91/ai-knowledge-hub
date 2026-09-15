"""Provider-to-writer persistence through real SDK decoding and a disposable DB."""
import json
from unittest.mock import AsyncMock

import httpx
import openai as sdk
import pytest
import sqlalchemy as sa

from app import model
from app.config import Settings
from app.db import get_db
from app.providers import openai as adapter
from app.services import chat_sessions
from app.services.wiki_engine import WikiEngine


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal", ["completed", "incomplete", None])
async def test_writer_only_saves_completed_response(user_ctx, default_user, monkeypatch, terminal):
    configured = Settings(_env_file=None, LLM_PROVIDER="openai", LLM_API_KEY="test-key",
                          MODEL_DEFAULT="gpt41mini", MODEL_EMBEDDING="")
    monkeypatch.setattr(model, "settings", configured)
    monkeypatch.setattr(adapter, "settings", configured)
    monkeypatch.setattr(WikiEngine, "_find_relevant_pages", AsyncMock(return_value=[]))
    text = "[DRAFT_START]# Synthetic draft\n\nSaved body.[DRAFT_END][DRAFT_READY]"
    events = [{"type": "response.output_text.delta", "sequence_number": 0,
               "item_id": "msg_test", "output_index": 0, "content_index": 0, "delta": text}]
    if terminal:
        events.append({"type": "response." + terminal, "sequence_number": 1, "response": {
            "id": "resp_test", "object": "response", "created_at": 1, "model": "gpt-4.1-mini-2025-04-14",
            "status": terminal, "output": [], "usage": {"input_tokens": 5, "output_tokens": 8,
                "total_tokens": 13, "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0}}}})
    wire = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()

    def client(**kwargs):
        return sdk.AsyncOpenAI(**kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, headers={"content-type": "text/event-stream"}, content=wire))))

    monkeypatch.setattr(adapter, "AsyncOpenAI", client)
    received = [json.loads(event.strip()[6:]) async for event in WikiEngine().writer_chat_stream(None, "write a draft")]
    sid = next(e["session_id"] for e in received if e["type"] == "meta")
    draft = await chat_sessions.get_or_create(sid)
    if terminal == "completed":
        assert received[-1]["type"] == "done"
        assert "Saved body." in draft.draft_content
    else:
        assert received[-1]["type"] == "error"
        assert not draft.draft_content and not draft.messages
    async with get_db() as db:
        rows = (await db.execute(sa.text("SELECT org_id,user_id,model_id,tokens_in,tokens_out FROM usage_log"))).fetchall()
    # An explicit incomplete response still reports billable usage; an abrupt
    # disconnect has no authoritative totals. Neither permits saving a draft.
    assert len(rows) == int(terminal is not None)
    if rows:
        assert str(rows[0].org_id) == default_user["org_id"]
        assert str(rows[0].user_id) == default_user["id"]
        assert rows[0].model_id == "gpt-4.1-mini-2025-04-14"
        assert (rows[0].tokens_in, rows[0].tokens_out) == (5, 8)
