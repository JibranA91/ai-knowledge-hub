"""Streaming parser for Writer Mode Bedrock output.

The writer LLM may embed markers in its response:

    [DRAFT_START] ... [DRAFT_END]
    [SECTION_START:Exact Heading] ... [SECTION_END]

This parser consumes raw text chunks (which may split markers across chunk
boundaries) and yields structured events for the SSE stream:

    {"type": "chunk",          "text": "..."}            # plain chat text
    {"type": "draft_chunk",    "text": "..."}            # full-draft body
    {"type": "draft_done"}                               # emitted on [DRAFT_END]
    {"type": "section_chunk",  "text": "...",
                               "heading": "..."}         # section patch body
    {"type": "section_done",   "heading": "..."}         # emitted on [SECTION_END]

Usage:
    parser = WriterStreamParser()
    for chunk in upstream:
        for evt in parser.feed(chunk):
            yield evt
    for evt in parser.flush():
        yield evt

`parser.draft_content` holds the full draft body once a [DRAFT_END] has been
seen. `parser.section_patches` is a list of (heading, body) pairs for each
completed section patch.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

DRAFT_START = "[DRAFT_START]"
DRAFT_END = "[DRAFT_END]"
DRAFT_READY = "[DRAFT_READY]"
SECTION_START_PREFIX = "[SECTION_START:"
SECTION_END = "[SECTION_END]"

# Longest marker prefix we might need to hold back at the end of the buffer
# while waiting for the rest of a marker to arrive in the next chunk.
_MARKER_MAX_LEN = max(len(DRAFT_START), len(DRAFT_END), len(DRAFT_READY),
                      len(SECTION_START_PREFIX), len(SECTION_END))


@dataclass
class WriterStreamParser:
    """Incremental marker-aware stream parser. Single-pass; not thread-safe."""

    state: str = "CHAT"                       # CHAT | DRAFT | SECTION
    buffer: str = ""                          # unconsumed input
    section_heading: str | None = None
    draft_content: str = ""                   # populated on [DRAFT_END]
    section_patches: list[tuple[str, str]] = field(default_factory=list)
    draft_ready_seen: bool = False            # True once [DRAFT_READY] appeared
    _draft_buffer: str = ""
    _section_buffer: str = ""

    def feed(self, chunk: str) -> Iterator[dict]:
        """Process *chunk* and yield any events that became complete."""
        if not chunk:
            return
        self.buffer += chunk
        while True:
            before_state = self.state
            before_len = len(self.buffer)
            for evt in self._step():
                yield evt
            progressed = self.state != before_state or len(self.buffer) != before_len
            if not progressed:
                break

    def flush(self) -> Iterator[dict]:
        """End-of-stream: finalize whatever is still open.

        The held-back tail (≤ marker length, withheld in case it began a marker)
        is drained into the active body, unless it's an incomplete marker
        fragment (e.g. a truncated ``[DRAFT_EN``), which is dropped.

        Crucially, an unterminated DRAFT or SECTION is COMMITTED rather than
        dropped: models frequently omit or truncate the closing ``[DRAFT_END]``
        (especially when a long draft hits the token limit). The body was already
        streamed to the client, so dropping it left the server with an empty
        draft while the preview showed content — surfacing as "Draft is empty"
        on Save & Ingest. Committing it keeps server and preview in sync.
        """
        tail = self.buffer
        self.buffer = ""

        if self.state == "DRAFT":
            # Drop only an INCOMPLETE closing-marker fragment (e.g. "[DRAFT_EN");
            # real trailing content is kept.
            if tail and not _is_partial_marker(tail):
                self._draft_buffer += tail
                yield {"type": "draft_chunk", "text": tail}
            if self._draft_buffer:
                self.draft_content = self._draft_buffer
                self.state = "CHAT"
                yield {"type": "draft_done"}
            return

        if self.state == "SECTION":
            if tail and not _is_partial_marker(tail):
                self._section_buffer += tail
                yield {"type": "section_chunk", "text": tail, "heading": self.section_heading}
            if self._section_buffer:
                self.section_patches.append((self.section_heading or "", self._section_buffer))
                heading = self.section_heading
                self.section_heading = None
                self.state = "CHAT"
                yield {"type": "section_done", "heading": heading}
            return

        # CHAT state — anything still held back is plain text (even a lone '[' or
        # a complete-but-contextless marker like a bare [DRAFT_END]).
        if tail:
            yield {"type": "chunk", "text": tail}

    # ── Internal: one progress step per call ─────────────────────────────

    def _step(self) -> Iterator[dict]:
        if self.state == "CHAT":
            yield from self._step_chat()
        elif self.state == "DRAFT":
            yield from self._step_draft()
        elif self.state == "SECTION":
            yield from self._step_section()

    def _step_chat(self) -> Iterator[dict]:
        # Find earliest of DRAFT_START, DRAFT_READY, or SECTION_START_PREFIX
        # (DRAFT_READY must come before DRAFT_START in lookup since one is a
        # prefix-overlap risk with the other — but they differ on the 7th char
        # ('R' vs '_END]') so find() handles them independently.)
        d = self.buffer.find(DRAFT_START)
        r = self.buffer.find(DRAFT_READY)
        s = self.buffer.find(SECTION_START_PREFIX)
        candidates = [(i, kind) for i, kind in
                      [(d, "draft"), (r, "ready"), (s, "section")] if i != -1]
        if candidates:
            idx, kind = min(candidates)
            # Emit any plain text before the marker
            if idx > 0:
                yield {"type": "chunk", "text": self.buffer[:idx]}
            if kind == "draft":
                self.buffer = self.buffer[idx + len(DRAFT_START):]
                # Strip an optional leading newline so the draft body starts
                # at column 0 — important so the YAML frontmatter regex in
                # the frontend matches and gets stripped before rendering.
                if self.buffer.startswith("\n"):
                    self.buffer = self.buffer[1:]
                self.state = "DRAFT"
                self._draft_buffer = ""
            elif kind == "ready":
                # Standalone signal: agent declares the current draft ready.
                # Idempotent — emit only once per turn even if the model
                # repeats the marker.
                self.buffer = self.buffer[idx + len(DRAFT_READY):]
                if self.buffer.startswith("\n"):
                    self.buffer = self.buffer[1:]
                if not self.draft_ready_seen:
                    self.draft_ready_seen = True
                    yield {"type": "draft_ready"}
            else:
                # Need the closing ']' on the SECTION_START header line
                rest = self.buffer[idx + len(SECTION_START_PREFIX):]
                close = rest.find("]")
                if close == -1:
                    # Header line not yet complete — emit nothing further, wait
                    self.buffer = self.buffer[idx:]  # drop everything before marker
                    return
                heading = rest[:close].strip()
                self.section_heading = heading
                self.buffer = rest[close + 1:]
                # Strip optional leading newline so the section body starts cleanly
                if self.buffer.startswith("\n"):
                    self.buffer = self.buffer[1:]
                self.state = "SECTION"
                self._section_buffer = ""
            return

        # No full marker found — emit safe prefix, keep the trailing tail
        safe = _safe_emit_len(self.buffer)
        if safe > 0:
            yield {"type": "chunk", "text": self.buffer[:safe]}
            self.buffer = self.buffer[safe:]

    def _step_draft(self) -> Iterator[dict]:
        end_idx = self.buffer.find(DRAFT_END)
        if end_idx != -1:
            if end_idx > 0:
                body = self.buffer[:end_idx]
                self._draft_buffer += body
                yield {"type": "draft_chunk", "text": body}
            self.buffer = self.buffer[end_idx + len(DRAFT_END):]
            self.draft_content = self._draft_buffer
            self.state = "CHAT"
            yield {"type": "draft_done"}
            return
        safe = _safe_emit_len(self.buffer)
        if safe > 0:
            body = self.buffer[:safe]
            self._draft_buffer += body
            yield {"type": "draft_chunk", "text": body}
            self.buffer = self.buffer[safe:]

    def _step_section(self) -> Iterator[dict]:
        end_idx = self.buffer.find(SECTION_END)
        if end_idx != -1:
            if end_idx > 0:
                body = self.buffer[:end_idx]
                self._section_buffer += body
                yield {"type": "section_chunk", "text": body, "heading": self.section_heading}
            self.buffer = self.buffer[end_idx + len(SECTION_END):]
            self.section_patches.append((self.section_heading or "", self._section_buffer))
            heading = self.section_heading
            self.section_heading = None
            self.state = "CHAT"
            yield {"type": "section_done", "heading": heading}
            return
        safe = _safe_emit_len(self.buffer)
        if safe > 0:
            body = self.buffer[:safe]
            self._section_buffer += body
            yield {"type": "section_chunk", "text": body, "heading": self.section_heading}
            self.buffer = self.buffer[safe:]


def _is_partial_marker(tail: str) -> bool:
    """True if *tail* is a PROPER prefix of a marker — i.e. an incomplete marker
    the stream cut off (e.g. "[DRAFT_EN"). A complete marker (equal to one) is
    treated as literal content, not dropped."""
    return any(m != tail and m.startswith(tail) for m in
               (DRAFT_START, DRAFT_END, DRAFT_READY, SECTION_START_PREFIX, SECTION_END))


def _safe_emit_len(buf: str) -> int:
    """Number of chars at the start of *buf* that cannot begin a marker.

    A marker always starts with '['. So if the last '[' in *buf* is in the
    final _MARKER_MAX_LEN chars, hold from there. Otherwise emit everything.
    """
    if not buf:
        return 0
    tail = buf[-_MARKER_MAX_LEN:]
    last_bracket = tail.rfind("[")
    if last_bracket == -1:
        return len(buf)
    return len(buf) - len(tail) + last_bracket
