"""Direct Anthropic writer persistence and org-scoped accounting, with no live calls."""
import json
from unittest.mock import AsyncMock

import anthropic as sdk
import httpx2
import pytest
import sqlalchemy as sa

from app import model
from app.config import Settings
from app.db import get_db
from app.providers import anthropic as adapter
from app.services import chat_sessions
from app.services.wiki_engine import WikiEngine


@pytest.mark.asyncio
@pytest.mark.parametrize("complete", [True, False])
async def test_anthropic_writer_persists_only_completed_stream(user_ctx, default_user, monkeypatch, complete):
    configured = Settings(_env_file=None, LLM_PROVIDER="anthropic", LLM_API_KEY="test-key",
                          MODEL_DEFAULT="haiku45", MODEL_EMBEDDING="")
    monkeypatch.setattr(model, "settings", configured)
    monkeypatch.setattr(adapter, "settings", configured)
    monkeypatch.setattr(WikiEngine, "_find_relevant_pages", AsyncMock(return_value=[]))
    text = "[DRAFT_START]# Synthetic draft\n\nSaved body.[DRAFT_END][DRAFT_READY]"
    events = [
        {"type": "message_start", "message": {"id": "msg_test", "type": "message", "role": "assistant",
            "model": "claude-haiku-4-5-20251001", "content": [], "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": 5, "output_tokens": 0}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}},
    ]
    if complete:
        events += [{"type": "content_block_stop", "index": 0},
                   {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": 8}}, {"type": "message_stop"}]
    wire = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()

    def client(**kwargs):
        return sdk.AsyncAnthropic(**kwargs, http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(
            lambda request: httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=wire))))

    monkeypatch.setattr(adapter, "AsyncAnthropic", client)
    engine = WikiEngine()
    received = [json.loads(event.strip()[6:]) async for event in engine.writer_chat_stream(None, "write a draft")]
    sid = next(e["session_id"] for e in received if e["type"] == "meta")
    draft = await chat_sessions.get_or_create(sid)
    if complete:
        assert received[-1]["type"] == "done"
        assert "Saved body." in draft.draft_content
    else:
        assert received[-1]["type"] == "error"
        assert not draft.draft_content and not draft.messages
    async with get_db() as db:
        rows = (await db.execute(sa.text("SELECT org_id,user_id,model_id,tokens_in,tokens_out FROM usage_log"))).fetchall()
    assert len(rows) == int(complete)
    if rows:
        assert str(rows[0].org_id) == default_user["org_id"]
        assert str(rows[0].user_id) == default_user["id"]
        assert rows[0].model_id == "claude-haiku-4-5-20251001"
        assert (rows[0].tokens_in, rows[0].tokens_out) == (5, 8)
