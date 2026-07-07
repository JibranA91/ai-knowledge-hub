"""End-to-end regression for the writer 'Draft is empty' Save & Ingest failure.

When the model streams a draft but omits the closing [DRAFT_END] (truncation /
token limit), the body was previously dropped server-side — so the preview
showed a draft but the session's draft_content stayed empty and Save & Ingest
failed with "Draft is empty". These tests drive the real writer_chat_stream with
a mocked Bedrock stream and assert the draft is persisted. Requires PostgreSQL.
"""
import json

import pytest
from unittest.mock import patch

from app.services import chat_sessions
from app.services.bedrock import BedrockService
from app.services.wiki_engine import WikiEngine


def _sse_events(raw: list[str]) -> list[dict]:
    out = []
    for r in raw:
        line = r.strip()
        if line.startswith("data: "):
            out.append(json.loads(line[len("data: "):]))
    return out


async def _run_writer_turn(message: str, streamed: str) -> str:
    """Drive one writer turn with a mocked Bedrock stream; return the session_id."""
    async def fake_stream(self, system_prompt, messages, **kw):
        yield streamed

    engine = WikiEngine()
    raw: list[str] = []
    with patch.object(BedrockService, "converse_stream", fake_stream):
        async for ev in engine.writer_chat_stream(None, message):
            raw.append(ev)
    return next(e["session_id"] for e in _sse_events(raw) if e.get("type") == "meta")


@pytest.mark.asyncio
async def test_unterminated_draft_is_persisted_for_ingest(client, user_ctx, default_user):
    """A draft streamed WITHOUT [DRAFT_END] still lands in draft_content, so the
    ingest empty-check passes (the reported Save & Ingest bug)."""
    body = "---\ntitle: Widget\n---\n# Widget\n\nBody text, no closing marker."
    sid = await _run_writer_turn("write a widget page", "[DRAFT_START]\n" + body)

    draft = await chat_sessions.get_draft(sid)
    assert draft is not None
    assert draft["draft_content"].strip() != ""          # not empty → ingest won't 400
    assert "Body text, no closing marker." in draft["draft_content"]


@pytest.mark.asyncio
async def test_terminated_draft_still_persists(client, user_ctx, default_user):
    """A normally-terminated draft is unaffected by the unterminated-draft fix."""
    sid = await _run_writer_turn(
        "write it", "[DRAFT_START]\n# Title\n\nFull body.\n[DRAFT_END] all done"
    )
    draft = await chat_sessions.get_draft(sid)
    assert "Full body." in draft["draft_content"]
