"""Shared utility functions used across service modules."""
import json
import re

try:
    import yaml as _yaml
    _HAVE_YAML = True
except Exception:  # PyYAML is optional — frontmatter parsing degrades to {}
    _HAVE_YAML = False


def strip_outer_wrapper_fence(text: str) -> tuple[str, bool]:
    """Strip a whole-document ```markdown ... ``` wrapper from a stored page.

    Conservative: only strips when the enclosed content begins with YAML
    frontmatter (`---`) or a markdown heading (`#`). Pages that legitimately
    start with a code block (e.g. a tutorial page) are left untouched.

    Returns (cleaned_text, was_modified).
    """
    if not text:
        return text, False
    s = text.lstrip()
    fence_open = re.match(r"^```[A-Za-z0-9_-]*[ \t]*\n", s)
    if not fence_open:
        return text, False
    body = s[fence_open.end():]
    first_chars = body.lstrip()[:5]
    if not (first_chars.startswith("---") or first_chars.startswith("#")):
        return text, False
    body = re.sub(r"\n?```[ \t]*$", "", body)
    return body.strip(), True


_FENCE_OPEN_RE  = re.compile(r"^```[A-Za-z0-9_-]*[ \t]*\r?\n?")
_FENCE_CLOSE_RE = re.compile(r"\r?\n?```[ \t]*$")


def strip_fence_unconditional(text: str) -> str:
    """Strip a leading ```lang fence and a trailing ``` from LLM output,
    unconditionally (vs strip_outer_wrapper_fence, which only strips when the
    body looks like a real page). For places that always expect plain output
    and never a legitimate leading code block (the inline editor, recalibration
    page rewrites)."""
    t = (text or "").strip()
    t = _FENCE_OPEN_RE.sub("", t, count=1)
    t = _FENCE_CLOSE_RE.sub("", t)
    return t.strip()


def parse_llm_json(text: str) -> dict:
    """Parse JSON from LLM output, stripping markdown code fences if present.

    Falls back to regex extraction when the model wraps the JSON in prose.
    Raises ValueError if no valid JSON object can be found.
    """
    cleaned = re.sub(r"^```(?:json)?\n?", "", text.strip())
    cleaned = re.sub(r"\n?```$", "", cleaned.strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if m:
            return json.loads(m.group())
        raise ValueError(f"Could not parse JSON from LLM response: {text[:400]}")


# ── Frontmatter helpers ────────────────────────────────────────────────────

# The five page types the server stamps and recognises. The LLM picks one per
# page in its plan; sites that need server-managed behaviour (provenance
# stamping, deletion cleanup, recalibration skip) key off these values rather
# than path prefixes — so orgs can rename folders freely without breaking it.
PAGE_TYPES = {"source_summary", "concept", "entity", "rca", "query_result"}

# Path-prefix → page type. Used as the legacy fallback when frontmatter is
# missing (pre-migration pages) and by the Alembic backfill migration to
# infer the type for existing rows.
_TYPE_BY_PREFIX = (
    ("sources/", "source_summary"),
    ("concepts/", "concept"),
    ("entities/", "entity"),
    ("rca/",      "rca"),
    ("queries/",  "query_result"),
)

_FRONTMATTER_BLOCK_RE = re.compile(r'^---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|$)', re.DOTALL)
_TYPE_LINE_RE        = re.compile(r'(?m)^type:\s*["\']?(.+?)["\']?\s*$')
# First level-1 heading — used for title resolution (NOT section parsing, which
# matches any level; see app.services.md_sections).
_H1_RE               = re.compile(r'^#\s+(.+)', re.MULTILINE)


def parse_frontmatter(content: str) -> dict:
    """Parse a page's leading YAML frontmatter block into a dict.

    Returns {} when there's no block, it's malformed, the value isn't a mapping,
    or PyYAML is unavailable. Shared by the graph and the page store so
    frontmatter handling can't drift between them.
    """
    m = _FRONTMATTER_BLOCK_RE.match(content or "")
    if not m or not _HAVE_YAML:
        return {}
    try:
        data = _yaml.safe_load(m.group(1))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def frontmatter_end(content: str) -> int:
    """Offset just past the frontmatter block (0 if none) — where the body starts."""
    m = _FRONTMATTER_BLOCK_RE.match(content or "")
    return m.end() if m else 0


def resolve_title(content: str, rel_path: str) -> str:
    """Page title: frontmatter `title:`, else the first H1, else the humanized
    filename stem. The single source of truth for both the graph and the store."""
    t = parse_frontmatter(content).get("title")
    if isinstance(t, str) and t.strip():
        return t
    h1 = _H1_RE.search(content or "")
    if h1:
        return h1.group(1).strip()
    stem = rel_path.split("/")[-1].rsplit(".", 1)[0]
    return stem.replace("-", " ").title()


def page_type(content: str) -> str | None:
    """Return the `type:` value from a page's YAML frontmatter, or None."""
    fm = _FRONTMATTER_BLOCK_RE.match(content or "")
    if not fm:
        return None
    m = _TYPE_LINE_RE.search(fm.group(1))
    return m.group(1).strip() if m else None


def page_type_or_infer(path: str, content: str) -> str | None:
    """Return frontmatter type if present; else infer from path prefix.

    The fallback lets unmigrated legacy pages (no `type:` field yet) still
    get correct server-side behaviour.
    """
    t = page_type(content)
    if t:
        return t
    for prefix, inferred in _TYPE_BY_PREFIX:
        if path.startswith(prefix):
            return inferred
    return None


def infer_type_from_path(path: str) -> str | None:
    """Look up the page type from a path prefix alone. Returns None if no folder matches."""
    for prefix, inferred in _TYPE_BY_PREFIX:
        if path.startswith(prefix):
            return inferred
    return None


def stamp_frontmatter_field(content: str, field: str, value: str) -> str:
    """Ensure `field: value` is present in the frontmatter block.

    Replaces an existing `field:` line if one exists; otherwise injects the
    line just inside the opening `---`. If no frontmatter block exists,
    returns content unchanged (the caller should ensure pages have one).
    """
    fm = _FRONTMATTER_BLOCK_RE.match(content or "")
    if not fm:
        return content
    field_re = re.compile(rf'(?m)^{re.escape(field)}:.*$')
    block = fm.group(1)
    if field_re.search(block):
        new_block = field_re.sub(f"{field}: {value}", block, count=1)
        return content[:fm.start(1)] + new_block + content[fm.end(1):]
    # Inject right after the opening `---\n`
    return re.sub(r"(?m)^(---[ \t]*\r?\n)", rf"\1{field}: {value}\n", content, count=1)
