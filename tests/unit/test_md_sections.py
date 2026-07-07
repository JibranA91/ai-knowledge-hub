"""Unit tests for the pure markdown-section helpers (no DB, no loop)."""
from app.services.md_sections import (
    extract_section,
    find_section_bounds,
    list_headings,
    replace_section,
)

DOC = """---
title: Demo
---
# Title

Intro paragraph.

## Background

Some background text.
More background.

## Details

### Sub detail

Nested content.

## Conclusion

The end.
"""


def test_list_headings_returns_levels_and_titles_in_order():
    assert list_headings(DOC) == [
        (1, "Title"),
        (2, "Background"),
        (2, "Details"),
        (3, "Sub detail"),
        (2, "Conclusion"),
    ]


def test_list_headings_empty_for_no_headings():
    assert list_headings("just text, no headings") == []
    assert list_headings("") == []


def test_extract_section_returns_heading_line_and_body():
    section, err = extract_section(DOC, "Background")
    assert err is None
    assert section.startswith("## Background")
    assert "Some background text." in section
    assert "More background." in section
    # Stops at the next same-level heading.
    assert "## Details" not in section


def test_extract_section_includes_nested_subsections():
    section, err = extract_section(DOC, "Details")
    assert err is None
    assert "### Sub detail" in section
    assert "Nested content." in section
    # Stops at the next level-2 heading.
    assert "## Conclusion" not in section


def test_extract_section_not_found():
    section, err = extract_section(DOC, "Nope")
    assert section is None
    assert err == "heading not found"


def test_replace_section_swaps_only_that_section():
    new, err = replace_section(DOC, "Background", "## Background\n\nRewritten.")
    assert err is None
    assert "Rewritten." in new
    assert "Some background text." not in new
    # Other sections untouched.
    assert "## Details" in new
    assert "The end." in new
    # The replaced section still bounded by the following heading.
    assert new.index("Rewritten.") < new.index("## Details")


def test_replace_section_preserves_surrounding_structure():
    new, _ = replace_section(DOC, "Conclusion", "## Conclusion\n\nNew ending.")
    assert new.rstrip().endswith("New ending.")
    assert "# Title" in new
    assert "## Background" in new


def test_ambiguous_heading_is_rejected():
    dup = "## Notes\n\nA\n\n## Notes\n\nB\n"
    matches, err = find_section_bounds(dup, "Notes")
    assert "ambiguous" in err
    assert len(matches) == 2
    # replace/extract refuse to act on an ambiguous heading.
    assert replace_section(dup, "Notes", "## Notes\n\nC")[0] is None
    assert extract_section(dup, "Notes")[0] is None
