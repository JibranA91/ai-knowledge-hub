"""Unit tests for app.utils frontmatter helpers (page_type, page_type_or_infer,
infer_type_from_path, stamp_frontmatter_field).

These helpers are load-bearing across five sites in the codebase
(ingest, documents-delete, recalibrate skip + date preservation, lint), so
breakage here cascades.
"""
from app.utils import (
    PAGE_TYPES,
    infer_type_from_path,
    page_type,
    page_type_or_infer,
    stamp_frontmatter_field,
)


_PAGE_WITH_TYPE = """---
title: "Foo"
type: source_summary
---

# Foo
"""

_PAGE_NO_TYPE = """---
title: "Foo"
tags: [a]
---

# Foo
"""

_PAGE_NO_FRONTMATTER = "# Just a heading\n\nBody."

_PAGE_QUOTED_TYPE = """---
type: "entity"
---

# X
"""


def test_page_types_vocabulary_is_exactly_five():
    assert PAGE_TYPES == {"source_summary", "concept", "entity", "rca", "query_result"}


# ── page_type ─────────────────────────────────────────────────────────────

def test_page_type_reads_value():
    assert page_type(_PAGE_WITH_TYPE) == "source_summary"


def test_page_type_handles_quoted_value():
    assert page_type(_PAGE_QUOTED_TYPE) == "entity"


def test_page_type_returns_none_when_field_missing():
    assert page_type(_PAGE_NO_TYPE) is None


def test_page_type_returns_none_without_frontmatter():
    assert page_type(_PAGE_NO_FRONTMATTER) is None


def test_page_type_returns_none_for_empty_content():
    assert page_type("") is None


# ── infer_type_from_path ──────────────────────────────────────────────────

def test_infer_from_path_each_known_prefix():
    assert infer_type_from_path("sources/foo.md") == "source_summary"
    assert infer_type_from_path("concepts/foo.md") == "concept"
    assert infer_type_from_path("entities/foo.md") == "entity"
    assert infer_type_from_path("rca/foo.md") == "rca"
    assert infer_type_from_path("queries/foo.md") == "query_result"


def test_infer_from_path_unknown_returns_none():
    assert infer_type_from_path("misc/foo.md") is None
    assert infer_type_from_path("foo.md") is None
    assert infer_type_from_path("") is None


# ── page_type_or_infer ────────────────────────────────────────────────────

def test_type_or_infer_prefers_frontmatter():
    # Frontmatter says source_summary, but path is under concepts/. Frontmatter wins.
    content = _PAGE_WITH_TYPE  # type: source_summary
    assert page_type_or_infer("concepts/foo.md", content) == "source_summary"


def test_type_or_infer_falls_back_to_path_when_no_frontmatter_type():
    assert page_type_or_infer("sources/foo.md", _PAGE_NO_TYPE) == "source_summary"
    assert page_type_or_infer("queries/foo.md", _PAGE_NO_FRONTMATTER) == "query_result"


def test_type_or_infer_returns_none_when_neither_signal():
    assert page_type_or_infer("misc/foo.md", _PAGE_NO_TYPE) is None


# ── stamp_frontmatter_field ───────────────────────────────────────────────

def test_stamp_injects_field_when_missing():
    out = stamp_frontmatter_field(_PAGE_NO_TYPE, "type", "concept")
    assert page_type(out) == "concept"
    # Existing fields must be preserved
    assert "title:" in out
    assert "tags:" in out


def test_stamp_replaces_existing_field():
    out = stamp_frontmatter_field(_PAGE_WITH_TYPE, "type", "entity")
    assert page_type(out) == "entity"
    # Original value must be gone
    assert "source_summary" not in out


def test_stamp_is_idempotent():
    once  = stamp_frontmatter_field(_PAGE_NO_TYPE, "type", "concept")
    twice = stamp_frontmatter_field(once,           "type", "concept")
    assert once == twice


def test_stamp_returns_unchanged_when_no_frontmatter_block():
    # If a page has no frontmatter the helper is a no-op — caller must ensure
    # frontmatter exists. We don't want to invent a block on its behalf.
    out = stamp_frontmatter_field(_PAGE_NO_FRONTMATTER, "type", "concept")
    assert out == _PAGE_NO_FRONTMATTER


def test_stamp_handles_crlf_frontmatter():
    crlf = "---\r\ntitle: X\r\n---\r\n\r\n# X\r\n"
    out = stamp_frontmatter_field(crlf, "type", "rca")
    assert "type: rca" in out


def test_stamp_only_modifies_first_frontmatter_line_not_horizontal_rule():
    # Pages often contain a `---` horizontal rule in the body. Stamping must
    # not inject into the body separator.
    content = "---\ntitle: X\n---\n\n# Heading\n\nBody.\n\n---\n\nFooter.\n"
    out = stamp_frontmatter_field(content, "type", "concept")
    # The body's `---` separator must still be present exactly once after the
    # frontmatter closer. (Frontmatter open `---`, frontmatter close `---`,
    # body `---` = 3 occurrences.)
    assert out.count("---") == 3
    assert page_type(out) == "concept"


# ── R1: shared parse_frontmatter / resolve_title / frontmatter_end ──────────

from app.utils import parse_frontmatter, resolve_title, frontmatter_end


def test_parse_frontmatter_returns_dict():
    fm = parse_frontmatter("---\ntitle: Foo\ntags: [a, b]\n---\n# Foo\n")
    assert fm["title"] == "Foo"
    assert fm["tags"] == ["a", "b"]


def test_parse_frontmatter_missing_or_malformed_returns_empty():
    assert parse_frontmatter("# No frontmatter\n") == {}
    assert parse_frontmatter("") == {}
    # A non-mapping YAML document (e.g. a list) is not valid frontmatter → {}.
    assert parse_frontmatter("---\n- just\n- a\n- list\n---\n") == {}


def test_resolve_title_prefers_frontmatter_then_h1_then_stem():
    assert resolve_title("---\ntitle: From FM\n---\n# H1 Title\n", "concepts/x.md") == "From FM"
    assert resolve_title("# H1 Title\n\nbody", "concepts/x.md") == "H1 Title"
    assert resolve_title("no heading here", "concepts/my-page.md") == "My Page"


def test_resolve_title_empty_frontmatter_title_falls_through_to_h1():
    assert resolve_title('---\ntitle: ""\n---\n# Real Title\n', "concepts/x.md") == "Real Title"


def test_frontmatter_end_points_past_the_block():
    content = "---\ntitle: Foo\n---\nBODY"
    assert content[frontmatter_end(content):] == "BODY"
    assert frontmatter_end("# no fm\n") == 0


# ── R6: shared unconditional fence stripper ─────────────────────────────────

from app.utils import strip_fence_unconditional


def test_strip_fence_unconditional():
    assert strip_fence_unconditional("```markdown\n# Hi\n\nbody\n```") == "# Hi\n\nbody"
    assert strip_fence_unconditional("```\nplain\n```") == "plain"
    assert strip_fence_unconditional('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert strip_fence_unconditional("no fence here") == "no fence here"
    assert strip_fence_unconditional("") == ""
