"""Pure helpers for locating and rewriting markdown sections by ATX heading.

Extracted from chat_sessions so the document-writer's section patching and the
inline AI editor share one well-tested implementation. No DB, no I/O — just
string surgery, which makes these trivial to unit-test.
"""
import re

# A markdown ATX heading: 1-6 leading '#', whitespace, then the heading text.
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


def list_headings(content: str) -> list[tuple[int, str]]:
    """Return ``[(level, title)]`` for every ATX heading, in document order."""
    return [(len(m.group(1)), m.group(2).strip()) for m in _HEADING_RE.finditer(content or "")]


def find_section_bounds(content: str, heading: str) -> tuple[list[tuple[int, int, int]], str | None]:
    """Locate occurrences of *heading*. Each tuple is ``(start, body_end, level)``.

    ``start`` is the offset of the heading line's ``#``. ``body_end`` is the
    offset of the next heading at the same or higher level (or ``len(content)``
    if none). ``level`` is the number of ``#`` characters on the heading line.

    Returns ``(matches, error)``: error is set (and matches may be empty or
    ambiguous) when the heading is not found exactly once.
    """
    target = heading.strip()
    headings: list[tuple[int, int, str]] = []  # (start, level, title)
    for m in _HEADING_RE.finditer(content):
        headings.append((m.start(), len(m.group(1)), m.group(2).strip()))

    matches: list[tuple[int, int, int]] = []
    for i, (start, level, title) in enumerate(headings):
        if title != target:
            continue
        body_end = len(content)
        for j in range(i + 1, len(headings)):
            nstart, nlevel, _ = headings[j]
            if nlevel <= level:
                body_end = nstart
                break
        matches.append((start, body_end, level))

    if not matches:
        return [], "heading not found"
    if len(matches) > 1:
        return matches, f"heading is ambiguous ({len(matches)} matches)"
    return matches, None


def extract_section(content: str, heading: str) -> tuple[str | None, str | None]:
    """Return ``(section_text, error)`` — the heading line plus its body."""
    matches, err = find_section_bounds(content, heading)
    if err:
        return None, err
    start, body_end, _ = matches[0]
    return content[start:body_end], None


def replace_section(content: str, heading: str, new_section: str) -> tuple[str | None, str | None]:
    """Replace the section under *heading* with *new_section*.

    Returns ``(new_content, error)``. On ambiguity / not-found, ``new_content``
    is None and the caller should leave the document untouched.
    """
    matches, err = find_section_bounds(content, heading)
    if err:
        return None, err
    start, body_end, _ = matches[0]
    # Preserve a single trailing newline boundary so the next heading keeps its
    # blank-line separation.
    replacement = new_section.rstrip() + "\n"
    return content[:start] + replacement + content[body_end:], None
