"""Unit tests for the writer SSE stream parser (`app.services.writer_stream`).

The parser must:
  - emit plain chat text as `chunk` events
  - emit full-draft bodies as `draft_chunk` then a `draft_done`
  - emit section patches as `section_chunk` (with heading) then `section_done`
  - tolerate markers split arbitrarily across chunk boundaries
  - not false-trigger on bare `[SECTION_END]` or `[DRAFT_END]` without a preceding start
"""
from app.services.writer_stream import WriterStreamParser


def _drive(parser: WriterStreamParser, chunks: list[str]) -> list[dict]:
    events: list[dict] = []
    for c in chunks:
        events.extend(parser.feed(c))
    events.extend(parser.flush())
    return events


# ── Plain chat ─────────────────────────────────────────────────────────────

def test_plain_chat_single_chunk():
    p = WriterStreamParser()
    evts = _drive(p, ["Hello, how can I help?"])
    assert evts == [{"type": "chunk", "text": "Hello, how can I help?"}]
    assert p.draft_content == ""


def test_plain_chat_many_chunks():
    p = WriterStreamParser()
    evts = _drive(p, ["Hi ", "there ", "friend"])
    text = "".join(e["text"] for e in evts if e["type"] == "chunk")
    assert text == "Hi there friend"


# ── Full draft ─────────────────────────────────────────────────────────────

def test_full_draft_single_chunk():
    p = WriterStreamParser()
    evts = _drive(p, ["prefix [DRAFT_START]Body of the draft.[DRAFT_END] tail"])
    types = [e["type"] for e in evts]
    assert types == ["chunk", "draft_chunk", "draft_done", "chunk"]
    assert evts[0]["text"] == "prefix "
    assert evts[1]["text"] == "Body of the draft."
    assert evts[3]["text"] == " tail"
    assert p.draft_content == "Body of the draft."


def test_full_draft_multi_chunk():
    p = WriterStreamParser()
    evts = _drive(p, ["[DRAFT_START]Hello ", "world", "[DRAFT_END]"])
    drafts = [e["text"] for e in evts if e["type"] == "draft_chunk"]
    assert "".join(drafts) == "Hello world"
    assert any(e["type"] == "draft_done" for e in evts)
    assert p.draft_content == "Hello world"


def test_marker_split_across_two_chunks():
    """`[DRAFT_` arrives separately from `START]`."""
    p = WriterStreamParser()
    evts = _drive(p, ["pre[DRAFT_", "START]body[DRAFT_END]post"])
    types = [e["type"] for e in evts]
    assert "draft_chunk" in types
    assert "draft_done" in types
    chat_texts = [e["text"] for e in evts if e["type"] == "chunk"]
    assert "".join(chat_texts) == "prepost"
    assert p.draft_content == "body"


def test_marker_split_across_three_chunks():
    """[DR / AFT_ST / ART] arrive in three pieces."""
    p = WriterStreamParser()
    evts = _drive(p, ["[DR", "AFT_ST", "ART]content[DRAFT_END]"])
    drafts = [e["text"] for e in evts if e["type"] == "draft_chunk"]
    assert "".join(drafts) == "content"
    assert p.draft_content == "content"


def test_draft_strips_leading_newline_after_start_marker():
    """`[DRAFT_START]\\n---\\n...` → draft body must begin with `---`, not `\\n---`.

    Frontend frontmatter-strip regex anchors at `^---`, so a leading newline
    would prevent stripping and render the YAML block as `<hr>` + literal text.
    """
    p = WriterStreamParser()
    chunks = ["[DRAFT_START]\n---\ntitle: x\n---\n\n# Title\nbody\n[DRAFT_END]"]
    _drive(p, chunks)
    assert p.draft_content.startswith("---\n"), repr(p.draft_content[:30])


def test_empty_draft_body():
    p = WriterStreamParser()
    evts = _drive(p, ["[DRAFT_START][DRAFT_END]"])
    types = [e["type"] for e in evts]
    assert types == ["draft_done"]
    assert p.draft_content == ""


# ── Section patches ────────────────────────────────────────────────────────

def test_section_patch_single_chunk():
    p = WriterStreamParser()
    evts = _drive(p, ["[SECTION_START:Overview]\n## Overview\nNew text.[SECTION_END]"])
    types = [e["type"] for e in evts]
    assert "section_chunk" in types
    assert "section_done" in types
    section_events = [e for e in evts if e["type"] == "section_chunk"]
    assert all(e["heading"] == "Overview" for e in section_events)
    body = "".join(e["text"] for e in section_events)
    assert body == "## Overview\nNew text."
    assert p.section_patches == [("Overview", "## Overview\nNew text.")]


def test_section_patch_header_split():
    """The `]` closing the SECTION_START header arrives in a later chunk."""
    p = WriterStreamParser()
    evts = _drive(p, ["[SECTION_START:Plan", "ning]\nbody[SECTION_END]"])
    section_events = [e for e in evts if e["type"] == "section_chunk"]
    assert section_events, "expected at least one section_chunk"
    assert all(e["heading"] == "Planning" for e in section_events)
    assert p.section_patches == [("Planning", "body")]


# ── False-trigger resilience ───────────────────────────────────────────────

def test_lone_end_marker_is_plain_text():
    """`[DRAFT_END]` without a preceding `[DRAFT_START]` stays as chat."""
    p = WriterStreamParser()
    evts = _drive(p, ["Look at this: [DRAFT_END] (sample)"])
    chat = "".join(e["text"] for e in evts if e["type"] == "chunk")
    assert chat == "Look at this: [DRAFT_END] (sample)"
    assert p.draft_content == ""
    assert not p.section_patches


def test_lone_section_end_is_plain_text():
    p = WriterStreamParser()
    evts = _drive(p, ["Trailing [SECTION_END] is harmless"])
    chat = "".join(e["text"] for e in evts if e["type"] == "chunk")
    assert chat == "Trailing [SECTION_END] is harmless"


def test_text_starting_with_bracket_but_not_marker():
    p = WriterStreamParser()
    evts = _drive(p, ["[note: this is a normal bracketed phrase]"])
    chat = "".join(e["text"] for e in evts if e["type"] == "chunk")
    assert chat == "[note: this is a normal bracketed phrase]"


# ── Mixed flow ─────────────────────────────────────────────────────────────

def test_chat_then_draft_then_chat():
    p = WriterStreamParser()
    chunks = [
        "Sure, here's a draft:\n",
        "[DRAFT_START]\n# Title\n",
        "Body...\n[DRAFT_END]\n",
        "Let me know if you'd like changes.",
    ]
    evts = _drive(p, chunks)
    types = [e["type"] for e in evts]
    assert types.count("chunk") >= 2
    assert types.count("draft_chunk") >= 1
    assert types.count("draft_done") == 1
    # The optional newline directly after `[DRAFT_START]` is stripped so the
    # draft body begins flush at column 0 (see frontmatter-strip discussion).
    assert p.draft_content == "# Title\nBody...\n"


def test_two_section_patches_in_one_stream():
    p = WriterStreamParser()
    evts = _drive(p, [
        "Updating two sections:\n",
        "[SECTION_START:A]\n## A\nnew A[SECTION_END]\n",
        "[SECTION_START:B]\n## B\nnew B[SECTION_END]",
    ])
    headings = [h for h, _ in p.section_patches]
    bodies = [b for _, b in p.section_patches]
    assert headings == ["A", "B"]
    assert bodies == ["## A\nnew A", "## B\nnew B"]


# ── [DRAFT_READY] marker ───────────────────────────────────────────────────

def test_draft_ready_alone_emits_event_and_sets_flag():
    p = WriterStreamParser()
    evts = _drive(p, ["Looks complete. [DRAFT_READY]"])
    types = [e["type"] for e in evts]
    assert "draft_ready" in types
    assert p.draft_ready_seen is True


def test_draft_ready_after_full_draft_in_same_stream():
    p = WriterStreamParser()
    evts = _drive(p, [
        "Here is the draft:\n[DRAFT_START]\n# Title\nBody\n[DRAFT_END]\n",
        "I'm satisfied. [DRAFT_READY]\n",
    ])
    types = [e["type"] for e in evts]
    # draft body parsed, then ready event emitted
    assert "draft_chunk" in types
    assert "draft_done" in types
    assert "draft_ready" in types
    # draft_done must come before draft_ready in event order
    assert types.index("draft_done") < types.index("draft_ready")
    assert p.draft_ready_seen is True
    assert p.draft_content.strip().endswith("Body")


def test_draft_ready_split_across_chunks():
    """The marker must survive being split mid-token, like other markers."""
    p = WriterStreamParser()
    evts = _drive(p, ["Looks good. [DRAFT_", "READY] All done."])
    types = [e["type"] for e in evts]
    assert "draft_ready" in types
    assert p.draft_ready_seen is True
    # Plain text after the marker should still arrive as chat
    tail = "".join(e["text"] for e in evts if e["type"] == "chunk" and "All done" in e.get("text", ""))
    assert "All done" in tail


def test_draft_ready_idempotent_across_multiple_emissions():
    """If the model emits the marker twice in one stream, only one event fires."""
    p = WriterStreamParser()
    evts = _drive(p, ["[DRAFT_READY] First. [DRAFT_READY] Second."])
    ready_events = [e for e in evts if e["type"] == "draft_ready"]
    assert len(ready_events) == 1
    assert p.draft_ready_seen is True


def test_no_ready_marker_leaves_flag_false():
    p = WriterStreamParser()
    _drive(p, ["[DRAFT_START]\n# Title\nBody\n[DRAFT_END]"])
    assert p.draft_ready_seen is False


def test_draft_ready_inside_draft_body_does_not_fire():
    """The marker is a CHAT-state token. Inside [DRAFT_START]...[DRAFT_END] it
    is treated as literal draft text — otherwise a section body that mentions
    the literal string would falsely mark the doc ready."""
    p = WriterStreamParser()
    evts = _drive(p, ["[DRAFT_START]\nMentioning [DRAFT_READY] in body\n[DRAFT_END]"])
    types = [e["type"] for e in evts]
    assert "draft_ready" not in types
    assert p.draft_ready_seen is False
    assert "[DRAFT_READY]" in p.draft_content


# ── Unterminated blocks (model omitted/truncated the closing marker) ─────────

def test_unterminated_draft_is_recovered_on_flush():
    """Regression: a model that streams a draft but omits/truncates [DRAFT_END]
    (common when the response hits the token limit) must still produce
    draft_content. Otherwise the preview shows a draft but the server stores
    none → "Draft is empty" on Save & Ingest."""
    p = WriterStreamParser()
    evts = _drive(p, ["[DRAFT_START]\n# Title\n\nBody text here."])
    assert p.draft_content == "# Title\n\nBody text here."
    assert any(e["type"] == "draft_chunk" for e in evts)
    assert any(e["type"] == "draft_done" for e in evts)


def test_unterminated_draft_split_across_chunks_recovered():
    p = WriterStreamParser()
    _drive(p, ["[DRAFT_START]\n# T", "itle\n\nBody", " continues"])
    assert p.draft_content == "# Title\n\nBody continues"


def test_unterminated_draft_drops_truncated_end_marker_fragment():
    """A truncated closing-marker fragment must not be kept as draft body."""
    p = WriterStreamParser()
    _drive(p, ["[DRAFT_START]\n# Title\n\nBody.\n[DRAFT_EN"])
    assert "[DRAFT" not in p.draft_content
    assert p.draft_content.rstrip() == "# Title\n\nBody."


def test_unterminated_section_is_recovered_on_flush():
    p = WriterStreamParser()
    _drive(p, ["[SECTION_START:Notes]\n## Notes\n\nPatched body."])
    assert p.section_patches == [("Notes", "## Notes\n\nPatched body.")]


def test_terminated_draft_still_works_after_flush_change():
    """A normally-terminated draft is unaffected by the flush recovery."""
    p = WriterStreamParser()
    _drive(p, ["[DRAFT_START]\n# T\n\nB\n[DRAFT_END] thanks!"])
    assert p.draft_content == "# T\n\nB\n"
