"""
LangGraph ingestion agent — split into plan and execute phases.

Full graph (used by run() / re-ingest):
  START → planner ⇄ tools → parse_plan → write_pages → finalize → END

Plan graph (used by plan() — stops after planning for user review):
  START → planner ⇄ tools → parse_plan → END

Write graph (used by execute() — runs after user approves):
  START → write_pages → finalize → END
"""

import asyncio
import json
import operator
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from app.config import settings
from app.logger import get_logger

# Lazy semaphores — created on first use so they bind to the running event loop
_write_page_semaphore: asyncio.Semaphore | None = None


def _get_write_semaphore() -> asyncio.Semaphore:
    global _write_page_semaphore
    if _write_page_semaphore is None:
        _write_page_semaphore = asyncio.Semaphore(settings.WRITE_PAGE_CONCURRENCY)
    return _write_page_semaphore
from app.services.bedrock import make_chat_llm
from app.utils import (
    PAGE_TYPES,
    infer_type_from_path,
    page_type_or_infer,
    parse_llm_json,
    stamp_frontmatter_field,
    strip_outer_wrapper_fence,
)

log = get_logger(__name__)


class IngestCancelledError(Exception):
    """Raised when an in-progress ingest write is cooperatively cancelled.

    A plain Exception (not BaseException) so it doesn't get confused with task
    cancellation: it propagates up through the write graph and is handled by
    _execute_ingest, which marks the job 'cancelled' WITHOUT killing the per-org
    queue worker (which would happen if we reused asyncio.CancelledError).
    """


def _TODAY() -> str:
    from zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo(settings.APP_TIMEZONE)).strftime("%Y-%m-%d")


_MAX_TOOL_ROUNDS = 3


# ── Writer output sanitizer ────────────────────────────────────────────────

# Conversational preambles writer LLMs sometimes prepend before the actual page.
_WRITER_PREAMBLE_PATTERNS = [
    re.compile(r"^here(?:'s| is)\b.*?(?:\n|$)", re.IGNORECASE),
    re.compile(r"^sure[!,]?\s*(?:here\b.*?)?(?:\n|$)", re.IGNORECASE),
    re.compile(r"^certainly[!,]?.*?(?:\n|$)", re.IGNORECASE),
    re.compile(r"^below is\b.*?(?:\n|$)", re.IGNORECASE),
    re.compile(r"^the (?:wiki )?(?:markdown )?page (?:is|follows|begins).*?(?:\n|$)", re.IGNORECASE),
]


def _sanitize_writer_output(text: str) -> str:
    """Strip wrapping code fences and chatty preambles from raw writer output.

    Some writer models (notably Llama variants) wrap the whole page in a
    ```markdown ... ``` fence and/or prefix it with a conversational line.
    Both break frontend rendering because the inner content is parsed as code
    or the closing fence gets misinterpreted by an inner code block.
    """
    s = text.lstrip()

    # Strip conversational preamble lines (a few iterations in case multiple stack).
    for _ in range(3):
        for pat in _WRITER_PREAMBLE_PATTERNS:
            m = pat.match(s)
            if m:
                s = s[m.end():].lstrip()
                break
        else:
            break

    s = s.strip()

    # Strip an outer wrapping code fence, but ONLY when the inside looks like a
    # real wiki page (frontmatter or heading) — shared with the page store so a
    # tutorial page's legitimate leading code block is never stripped.
    s, _ = strip_outer_wrapper_fence(s)
    return s


# ── State ──────────────────────────────────────────────────────────────────

class IngestState(TypedDict):
    filename: str
    doc_text: str
    chunk_text: str
    chunk_index: int
    total_chunks: int
    schema: str
    index: str           # compact "path — title" list for all pages
    relevant_pages: list[dict]  # [{path, title, content, score}] from semantic search
    messages: Annotated[list, add_messages]
    plan: list[dict]
    conflicts: list[dict]
    log_entry: str
    pages_created: list[str]
    pages_updated: list[str]
    errors: Annotated[list, operator.add]
    user_notes: str
    _t_start: float


# ── Helpers ────────────────────────────────────────────────────────────────

# ── Postgres advisory lock helpers ─────────────────────────────────────────

async def _advisory_lock(db_session, path: str) -> None:
    """Acquire a transaction-scoped advisory lock keyed on path. Released on commit."""
    import sqlalchemy as sa
    await db_session.execute(
        sa.text("SELECT pg_advisory_xact_lock(1, hashtext(:path))"),
        {"path": path},
    )


# ── Plan post-processing ───────────────────────────────────────────────────

def _process_parsed_plan(data: dict, filename: str) -> dict:
    """Apply type inference and conflict annotation to a freshly parsed plan dict.

    Shared by the primary parse path and the post-failure retry path so both
    routes apply identical normalisation.
    """
    pages = data.get("pages", [])
    log.info(
        "Plan parsed | file=%s | pages=%d (%s)",
        filename, len(pages),
        ", ".join(p["path"] for p in pages),
    )
    for page in pages:
        t = (page.get("type") or "").strip()
        if t not in PAGE_TYPES:
            inferred = infer_type_from_path(page["path"])
            if inferred:
                log.info(
                    "Plan | page=%s missing/invalid type=%r — inferred %r from path",
                    page["path"], t, inferred,
                )
                page["type"] = inferred
            else:
                log.warning(
                    "Plan | page=%s has no recognised type and no folder match — leaving untyped",
                    page["path"],
                )
                page["type"] = None

    conflicts = data.get("conflicts", [])
    if conflicts:
        log.info("Conflicts detected | file=%s | count=%d", filename, len(conflicts))
        conflict_by_path = {c["path"]: c for c in conflicts}
        for page in pages:
            if page["path"] in conflict_by_path:
                c = conflict_by_path[page["path"]]
                page["brief"] += (
                    f"\n\nCONFLICT DETECTED — resolution={c['resolution']}: "
                    f"existing says '{c['existing_claim']}'; "
                    f"new document says '{c['new_claim']}'. "
                    "If resolution=surface, add a '## Conflicting Sources' section."
                )
    return {
        "plan": pages,
        "conflicts": conflicts,
        "log_entry": data.get("log_entry", ""),
    }


# ── Shared node factories ──────────────────────────────────────────────────

def _make_planner_nodes():
    """Returns (read_existing_page tool, planner_node, parse_plan_node, route_planner)."""

    @tool
    async def read_existing_page(path: str) -> str:
        """Read the current markdown content of a wiki page before updating it.

        Use this ONLY for wiki content pages in directories like sources/, concepts/,
        entities/, rca/, queries/ — e.g. 'concepts/my-topic.md'.
        Do NOT call this for index.md or log.md; their current content is already
        provided to you in the system prompt above.
        """
        _SPECIAL = {"index.md", "wiki/index.md", "log.md", "wiki/log.md"}
        if path in _SPECIAL:
            log.warning("Tool: read_existing_page | path=%s | blocked — use system-prompt index instead", path)
            return (
                f"[Do not call read_existing_page for '{path}'. "
                "The current wiki index is already provided in your system prompt. "
                "Use it directly instead of calling this tool.]"
            )
        from app.services.wiki_db import get_wiki_page_content
        content = await get_wiki_page_content(path)
        log.info("Tool: read_existing_page | path=%s | found=%s", path, bool(content))
        return content if content else f"[Page not found: {path}]"

    planner_llm = make_chat_llm(settings.BEDROCK_INGEST_MODEL_ID, max_tokens=4096, operation="ingest_plan").bind_tools([read_existing_page])

    async def planner_node(state: IngestState) -> dict:
        round_num = sum(1 for m in state["messages"] if hasattr(m, "tool_calls") and m.tool_calls) + 1
        schema_chars = len(state.get("schema") or "")
        log.info(
            "Planner | file=%s | round=%d | schema_chars=%d | schema_present=%s",
            state["filename"], round_num, schema_chars, schema_chars > 0,
        )

        # Format semantically relevant pages for the planner.
        # Truncate content to avoid blowing the context budget — the planner
        # can call read_existing_page for the full text of any page it needs.
        _PAGE_PREVIEW = 800
        relevant = state.get("relevant_pages") or []
        if relevant:
            relevant_section = "\n\n".join(
                f"=== {p['path']} (similarity: {p['score']:.2f}) ===\n"
                + p["content"][:_PAGE_PREVIEW]
                + ("…\n[truncated — call read_existing_page for full content]" if len(p["content"]) > _PAGE_PREVIEW else "")
                for p in relevant
            )
        else:
            relevant_section = "None — this appears to be a new topic area."

        # Cap the compact index to avoid large wikis flooding the prompt.
        _INDEX_MAX_LINES = 150
        index_lines = (state["index"] or "").splitlines()
        if len(index_lines) > _INDEX_MAX_LINES:
            index_display = "\n".join(index_lines[:_INDEX_MAX_LINES]) + f"\n… ({len(index_lines) - _INDEX_MAX_LINES} more pages)"
        else:
            index_display = state["index"]

        system_prompt = f"""You are a wiki maintainer. Analyse the source document and plan which wiki pages to create or update.

Wiki Schema (MANDATORY — this is the single source of truth for directory structure, page types, naming conventions, and formats):
{state['schema']}

All existing wiki pages (path — title, for create vs update routing):
{index_display}

Semantically similar existing pages — full content (study these carefully for conflicts and overlaps):
{relevant_section}

Use the read_existing_page tool if you need to inspect a page that is NOT listed above.
When you have gathered everything you need, respond with ONLY a JSON object (no other text):
{{
  "pages": [
    {{"path": "dir/slug.md", "action": "create", "type": "source_summary | concept | entity | rca | query_result", "brief": "Detailed description of what this page should contain"}},
    {{"path": "dir/name.md", "action": "update", "type": "source_summary | concept | entity | rca | query_result", "brief": "Exact changes to make and why"}}
  ],
  "conflicts": [
    {{"path": "dir/name.md", "existing_claim": "Short quote of what the existing page says", "new_claim": "What the new document says instead", "resolution": "surface | new_wins | existing_wins"}}
  ],
  "log_entry": "## [{_TODAY()}] ingest | Document Title\\nSummary of updates."
}}

Rules:
- The Wiki Schema above is the authoritative reference — derive all directory names, page types, naming conventions, and checklist steps from it.
- `type` must be one of: source_summary, concept, entity, rca, query_result — pick the one that matches the page's role. The server uses this to stamp provenance and gate cleanup; folder names are advisory.
- Prefer FEWER, BROADER pages over many narrow ones; aim for 4-6 pages per ingest.
- Be specific in each brief — the writer only sees the brief and the source document.
- For each conflict: surface = preserve both claims, new_wins = overwrite, existing_wins = keep old."""
        
        log.debug("Planner system prompt | file=%s | prompt=%s", state["filename"], system_prompt.replace("\n", "\\n"))

        system = SystemMessage(content=system_prompt)

        if not state["messages"]:
            chunk_info = (
                f" [chunk {state['chunk_index']+1}/{state['total_chunks']}]"
                if state["total_chunks"] > 1 else ""
            )
            human = HumanMessage(
                content=f"Plan the ingest for{chunk_info}:\n\nFilename: {state['filename']}\n\n{state['chunk_text']}"
            )
            response = await planner_llm.ainvoke([system, human])
            return {"messages": [human, response]}
        else:
            response = await planner_llm.ainvoke([system] + state["messages"])
            return {"messages": [response]}

    async def parse_plan_node(state: IngestState) -> dict:
        last = state["messages"][-1]
        content = last.content if isinstance(last.content, str) else ""
        has_tool_calls = bool(getattr(last, "tool_calls", None))

        # Primary parse attempt — skip if the last message was a tool call
        # with no text body (the max-tool-rounds short-circuit case).
        if content.strip() and not has_tool_calls:
            try:
                return _process_parsed_plan(parse_llm_json(content), state["filename"])
            except Exception as e:
                initial_error = f"{e}"
        elif has_tool_calls:
            initial_error = "planner hit max tool rounds without producing JSON"
        else:
            initial_error = "planner returned an empty message"

        # Retry once: ask the planner to stop calling tools and emit ONLY JSON.
        # We reuse the same llm (so its tool grammar stays valid for Bedrock),
        # but the recovery prompt explicitly forbids further tool use.
        log.warning(
            "Plan parsing failed — retrying with JSON-only follow-up | file=%s | initial=%s",
            state["filename"], initial_error,
        )
        try:
            retry_system = SystemMessage(content=(
                "You previously analyzed a document to plan wiki updates. "
                "Now produce your final answer. Do NOT call any tools. "
                "Respond with ONLY the JSON object — no markdown fences, no commentary.\n\n"
                f"Wiki Schema (for directory/naming conventions):\n{state.get('schema', '')}\n\n"
                "JSON format:\n"
                '{"pages": [{"path": "dir/slug.md", "action": "create|update", '
                '"type": "source_summary|concept|entity|rca|query_result", "brief": "..."}], '
                '"conflicts": [...], "log_entry": "..."}\n'
                'If you have nothing useful to plan, return '
                '{"pages": [], "conflicts": [], "log_entry": ""}.'
            ))
            recovery_msg = HumanMessage(content=(
                "Stop calling tools. Emit the final JSON plan now. "
                "JSON only — no other text."
            ))
            retry_resp = await planner_llm.ainvoke(
                [retry_system] + state["messages"] + [recovery_msg]
            )
            retry_text = retry_resp.content if isinstance(retry_resp.content, str) else ""
            data = parse_llm_json(retry_text)
            log.info(
                "Plan parsed on retry | file=%s | pages=%d",
                state["filename"], len(data.get("pages", [])),
            )
            return _process_parsed_plan(data, state["filename"])
        except Exception as e2:
            log.error(
                "Plan parsing failed after retry | file=%s | initial=%s | retry=%s",
                state["filename"], initial_error, e2,
            )
            return {
                "plan": [],
                "conflicts": [],
                "errors": [f"Plan parsing failed: {initial_error}; retry also failed: {e2}"],
            }

    def route_planner(state: IngestState) -> str:
        last = state["messages"][-1]
        tool_rounds = sum(1 for m in state["messages"] if hasattr(m, "tool_calls") and m.tool_calls)
        if tool_rounds >= _MAX_TOOL_ROUNDS:
            log.warning("Max tool rounds reached | file=%s | forcing parse", state["filename"])
            return "parse_plan"
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return "parse_plan"

    return read_existing_page, planner_node, parse_plan_node, route_planner


def _make_writer_nodes():
    """Returns (write_pages_node, finalize_node)."""

    writer_llm = make_chat_llm(settings.BEDROCK_INGEST_WRITER_MODEL_ID, max_tokens=8192, operation="ingest_write")
    _MAX_CONTINUATIONS = 2

    async def _invoke_with_continuation(messages: list) -> str:
        response = await writer_llm.ainvoke(messages)
        parts = [response.content if isinstance(response.content, str) else ""]
        for _ in range(_MAX_CONTINUATIONS):
            stop_reason = (
                response.response_metadata.get("stopReason")
                or response.response_metadata.get("stop_reason")
                or ""
            )
            if stop_reason != "max_tokens":
                break
            log.warning("Ingest writer | truncated (max_tokens), requesting continuation")
            continuation_messages = list(messages) + [
                AIMessage(content=parts[-1]),
                HumanMessage(content="Continue exactly where you left off. Do not repeat any content already written."),
            ]
            response = await writer_llm.ainvoke(continuation_messages)
            parts.append(response.content if isinstance(response.content, str) else "")
        return "".join(parts)

    async def write_pages_node(state: IngestState) -> dict:
        schema_chars = len(state.get("schema") or "")
        log.info(
            "Writer | file=%s | pages=%d | schema_chars=%d | schema_present=%s",
            state["filename"], len(state["plan"]), schema_chars, schema_chars > 0,
        )
        notes_addendum = (
            "\n\n⚠ MANDATORY REVIEWER INSTRUCTIONS — treat these as hard requirements:\n\n"
            f"{state['user_notes']}" if state.get("user_notes") else ""
        )

        async def write_one(entry: dict) -> tuple[str, str, str]:
            existing = ""
            if entry["action"] == "update":
                from app.services.wiki_db import get_wiki_page_content
                existing = await get_wiki_page_content(entry["path"]) or ""

            system_msg = SystemMessage(
                content=(
                    f"You write wiki pages. "
                    f"The Wiki Schema below is the authoritative reference for page formats, "
                    f"frontmatter fields, directory conventions, cross-linking syntax, and conflict sections — follow it exactly.\n\n"
                    f"Wiki Schema:\n{state['schema']}\n\n"
                    "Return ONLY raw markdown — no JSON, no code fences.\n"
                    "If the task brief contains 'CONFLICT DETECTED', apply the conflict resolution format defined in the schema."
                    + notes_addendum
                )
            )
            human_msg = HumanMessage(content=(
                f"Write wiki page: {entry['path']}\n\nTask: {entry['brief']}\n\n"
                + (f"Existing content to update:\n{existing}\n\n" if existing else "This is a new page.\n\n")
                + f"Existing pages (for cross-linking — use path as wikilink target):\n{state['index']}\n\n"
                + f"---\nSource document ({state['filename']}):\n\n{state['chunk_text']}"
            ))
            response = await _invoke_with_continuation([system_msg, human_msg])
            response = _sanitize_writer_output(response)

            # Ensure H1 heading after frontmatter
            fm_match = re.search(r'^---[ \t]*\n[\s\S]*?\n---[ \t]*(\n|$)', response)
            if fm_match:
                after_fm = response[fm_match.end():]
                if not re.match(r'\s*#[^#]', after_fm):
                    title_m = re.search(r'(?m)^title:\s*["\']?(.+?)["\']?\s*$', response[:fm_match.end()])
                    title = title_m.group(1).strip() if title_m else entry["path"].split("/")[-1].replace("-", " ").title()
                    response = response[:fm_match.end()] + f"\n# {title}\n\n" + after_fm.lstrip('\n')

            # Stamp the page type from the plan into frontmatter, so server-side
            # behaviour (provenance, deletion cleanup, recalibration skip) can key
            # off `type:` rather than path prefix.
            page_type_value = entry.get("type")
            if page_type_value:
                response = stamp_frontmatter_field(response, "type", page_type_value)

            # Stamp source pages with ingestion metadata
            effective_type = page_type_value or page_type_or_infer(entry["path"], response)
            if effective_type == "source_summary":
                today = _TODAY()
                if entry["action"] == "update" and existing:
                    fm = re.search(r"(?m)^date_ingested:\s*(.+)$", existing)
                    orig_fm = fm.group(1).strip() if fm else today
                    body = re.search(r"(?m)^\*\*Date ingested\*\*:\s*(.+)$", existing)
                    orig_body = body.group(1).strip() if body else today
                else:
                    orig_fm = orig_body = today
                response = re.sub(r"(?m)^date_ingested:.*$", f"date_ingested: {orig_fm}", response)
                response = re.sub(r"(?m)^\*\*Date ingested\*\*:.*$", f"**Date ingested**: {orig_body}", response)
                if "uploaded_file:" not in response:
                    response = re.sub(
                        r"(?m)^(---[ \t]*\n)",
                        f"\\1uploaded_file: {state['filename']}\n",
                        response, count=1,
                    )

            from app.services.wiki_db import upsert_wiki_page
            await upsert_wiki_page(entry["path"], response, ingested_from=state["filename"])
            log.info("Page written | path=%s | action=%s", entry["path"], entry["action"])
            return entry["path"], entry["action"]

        sem = _get_write_semaphore()

        async def write_one_bounded(e: dict):
            async with sem:
                # Cooperative cancel: re-check the flag between pages so an
                # admin/user cancel aborts a long write. Pages already written
                # remain recoverable via History/revert.
                from app.services import jobs as _job_store
                if await _job_store.is_cancel_requested(state["filename"]):
                    raise IngestCancelledError()
                return await write_one(e)

        results = await asyncio.gather(
            *[write_one_bounded(e) for e in state["plan"]], return_exceptions=True
        )

        created, updated, errors, cancelled = [], [], [], False
        for r in results:
            if isinstance(r, IngestCancelledError):
                cancelled = True
                continue
            if isinstance(r, Exception):
                log.error("Page write failed | error=%s", r)
                errors.append(str(r))
                continue
            path, action = r
            (created if action == "create" else updated).append(path)

        if cancelled:
            log.info("Writer | cancel requested mid-write | file=%s | written=%d",
                     state["filename"], len(created) + len(updated))
            raise IngestCancelledError()

        return {"pages_created": created, "pages_updated": updated, "errors": errors}

    async def finalize_node(state: IngestState) -> dict:
        from app.services.wiki_db import append_audit_log

        if state.get("log_entry"):
            log_entry = re.sub(
                r"(?m)^## \[\d{4}-\d{2}-\d{2}\]",
                f"## [{_TODAY()}]",
                state["log_entry"],
            )
            await append_audit_log("ingest", log_entry)

        elapsed = time.perf_counter() - state.get("_t_start", time.perf_counter())
        log.info(
            "Ingest finalised | file=%s | created=%d | updated=%d | errors=%d | elapsed=%.1fs",
            state["filename"],
            len(state.get("pages_created", [])),
            len(state.get("pages_updated", [])),
            len(state.get("errors", [])),
            elapsed,
        )
        return {}

    return write_pages_node, finalize_node


# ── Graph builders ─────────────────────────────────────────────────────────

def build_ingest_graph():
    read_existing_page, planner_node, parse_plan_node, route_planner = _make_planner_nodes()
    write_pages_node, finalize_node = _make_writer_nodes()

    def route_after_parse(state: IngestState) -> str:
        return "write_pages" if state.get("plan") else END

    builder = StateGraph(IngestState)
    builder.add_node("planner", planner_node)
    builder.add_node("tools", ToolNode([read_existing_page]))
    builder.add_node("parse_plan", parse_plan_node)
    builder.add_node("write_pages", write_pages_node)
    builder.add_node("finalize", finalize_node)

    builder.add_edge(START, "planner")
    builder.add_conditional_edges("planner", route_planner, {"tools": "tools", "parse_plan": "parse_plan"})
    builder.add_edge("tools", "planner")
    builder.add_conditional_edges("parse_plan", route_after_parse, {"write_pages": "write_pages", END: END})
    builder.add_edge("write_pages", "finalize")
    builder.add_edge("finalize", END)
    return builder.compile()


def build_plan_graph():
    read_existing_page, planner_node, parse_plan_node, route_planner = _make_planner_nodes()

    builder = StateGraph(IngestState)
    builder.add_node("planner", planner_node)
    builder.add_node("tools", ToolNode([read_existing_page]))
    builder.add_node("parse_plan", parse_plan_node)

    builder.add_edge(START, "planner")
    builder.add_conditional_edges("planner", route_planner, {"tools": "tools", "parse_plan": "parse_plan"})
    builder.add_edge("tools", "planner")
    builder.add_edge("parse_plan", END)
    return builder.compile()


def build_write_graph():
    write_pages_node, finalize_node = _make_writer_nodes()

    builder = StateGraph(IngestState)
    builder.add_node("write_pages", write_pages_node)
    builder.add_node("finalize", finalize_node)
    builder.add_edge(START, "write_pages")
    builder.add_edge("write_pages", "finalize")
    builder.add_edge("finalize", END)
    return builder.compile()


# ── Public class ───────────────────────────────────────────────────────────

class IngestAgent:
    def __init__(self):
        self._graph = build_ingest_graph()
        self._plan_graph = build_plan_graph()
        self._write_graph = build_write_graph()

    def _read_raw(self, filename: str) -> tuple[bytes, str]:
        """Read raw bytes from storage. Runs in the async context so the org ContextVar is live."""
        from app.services import s3
        suffix = Path(filename).suffix.lower()
        key = s3.org_prefix(f"raw/{filename}")
        raw = s3.read_bytes(key)
        if not raw:
            raise FileNotFoundError(
                f"File not found in storage (key: {key}). "
                "Check that the file was uploaded successfully."
            )
        return raw, suffix

    @staticmethod
    def _parse_raw(raw: bytes, suffix: str) -> str:
        """Parse raw bytes to text. CPU-intensive for PDF/DOCX — runs in thread pool."""
        if suffix in (".txt", ".md"):
            return raw.decode("utf-8", errors="replace")
        if suffix == ".pdf":
            import io
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(raw))
            return "\n".join(p.extract_text() or "" for p in reader.pages)
        if suffix in (".docx", ".doc"):
            import io
            from docx import Document
            return "\n".join(p.text for p in Document(io.BytesIO(raw)).paragraphs)
        return raw.decode("utf-8", errors="replace")

    _CHUNK_SIZE = 28_000
    _CHUNK_OVERLAP = 1_500

    @staticmethod
    def _split_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
        if len(text) <= chunk_size:
            return [text]
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            if end < len(text):
                nl = text.rfind("\n", start, end)
                if nl > start:
                    end = nl
            chunks.append(text[start:end])
            start = end - overlap
        return chunks

    @staticmethod
    def _assess_complexity(text: str) -> str | None:
        """Returns a human-readable rejection reason, or None if safe to ingest."""
        total_chars = len(text)
        max_chars = settings.MAX_INGEST_CHARS
        if total_chars > max_chars:
            effective_stride = IngestAgent._CHUNK_SIZE - IngestAgent._CHUNK_OVERLAP
            estimated_chunks = max(1, (total_chars - IngestAgent._CHUNK_OVERLAP + effective_stride - 1) // effective_stride)
            return (
                f"File is too large to ingest safely ({total_chars:,} characters, "
                f"~{estimated_chunks} LLM planning passes required). "
                f"Maximum is {max_chars:,} characters. "
                "Split the file into smaller sections and upload each separately."
            )

        html_cells = len(re.findall(r"<t[dh][\s>/]", text, re.IGNORECASE))
        max_cells = settings.MAX_HTML_TABLE_CELLS
        if html_cells > max_cells:
            return (
                f"File contains {html_cells:,} HTML table cells (<td>/<th>), "
                f"exceeding the limit of {max_cells:,}. "
                "Dense HTML tables exhaust LLM context. "
                "Convert tables to markdown format, split the file, or convert file to PDF."
            )

        return None

    async def _base_state(self, filename: str, doc_text: str, chunk_text: str,
                    chunk_index: int, total_chunks: int, t_start: float) -> dict:
        from app.services.wiki_db import get_wiki_file, get_compact_index, semantic_search_wiki
        from app.services import embeddings

        schema = await get_wiki_file("schema/AGENTS.md") or ""
        if schema:
            sections = [l.strip() for l in schema.splitlines() if l.startswith("## ")]
            log.info(
                "AGENTS.md loaded | file=%s | chars=%d | sections=%s",
                filename, len(schema), sections,
            )
        else:
            log.warning("AGENTS.md is EMPTY — agents will not follow directory conventions | file=%s", filename)

        # Compact index: path — title per line, O(n*50chars) vs full index.md prose
        compact_index = await get_compact_index()

        # Semantic search: embed the chunk and find the most similar existing pages.
        # Falls back gracefully to an empty list when embeddings are disabled.
        relevant_pages: list[dict] = []
        if embeddings.is_enabled():
            query_vec = await embeddings.embed_text(chunk_text)
            if query_vec is not None:
                relevant_pages = await semantic_search_wiki(query_vec, top_k=5)
                log.info(
                    "Semantic pre-fetch | file=%s | relevant_pages=%d | top_score=%.3f",
                    filename,
                    len(relevant_pages),
                    relevant_pages[0]["score"] if relevant_pages else 0.0,
                )
            else:
                log.warning("Semantic pre-fetch skipped — embed_text returned None | file=%s", filename)
        else:
            log.info("Semantic pre-fetch skipped — embeddings disabled | file=%s", filename)

        return {
            "filename": filename,
            "doc_text": doc_text,
            "chunk_text": chunk_text,
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "schema": schema,
            "index": compact_index,
            "relevant_pages": relevant_pages,
            "messages": [],
            "plan": [],
            "conflicts": [],
            "log_entry": "",
            "pages_created": [],
            "pages_updated": [],
            "errors": [],
            "user_notes": "",
            "_t_start": t_start,
        }

    async def run(self, filename: str) -> dict:
        t_start = time.perf_counter()
        log.info("IngestAgent.run | file=%s", filename)

        raw, suffix = self._read_raw(filename)
        loop = asyncio.get_event_loop()
        doc_text = await loop.run_in_executor(None, self._parse_raw, raw, suffix)

        rejection = self._assess_complexity(doc_text)
        if rejection:
            log.warning("IngestAgent.run | rejected | file=%s | reason=%s", filename, rejection)
            raise ValueError(rejection)

        chunks = self._split_chunks(doc_text, self._CHUNK_SIZE, self._CHUNK_OVERLAP)
        log.info("IngestAgent | file=%s | chunks=%d | total_chars=%d", filename, len(chunks), len(doc_text))

        all_created, all_updated, all_errors = [], [], []
        last_log_entry = ""

        for i, chunk in enumerate(chunks):
            log.info("IngestAgent | chunk %d/%d | file=%s", i + 1, len(chunks), filename)
            state = await self._base_state(filename, doc_text, chunk, i, len(chunks), t_start)
            # Refresh compact index from DB for each subsequent chunk
            if i > 0:
                from app.services.wiki_db import get_compact_index
                state["index"] = await get_compact_index()

            result = await self._graph.ainvoke(state)
            all_created.extend(result.get("pages_created", []))
            all_updated.extend(result.get("pages_updated", []))
            all_errors.extend(result.get("errors", []))
            if result.get("log_entry"):
                last_log_entry = result["log_entry"]

        return {"pages_created": all_created, "pages_updated": all_updated,
                "log_entry": last_log_entry, "errors": all_errors}

    async def plan(self, filename: str) -> dict:
        t_start = time.perf_counter()
        log.info("IngestAgent.plan | file=%s", filename)

        raw, suffix = self._read_raw(filename)
        loop = asyncio.get_event_loop()
        doc_text = await loop.run_in_executor(None, self._parse_raw, raw, suffix)

        rejection = self._assess_complexity(doc_text)
        if rejection:
            log.warning("IngestAgent.plan | rejected | file=%s | reason=%s", filename, rejection)
            raise ValueError(rejection)

        chunks = self._split_chunks(doc_text, self._CHUNK_SIZE, self._CHUNK_OVERLAP)

        all_pages: dict[str, dict] = {}
        all_conflicts: list[dict] = []
        all_errors: list[str] = []
        last_log_entry = ""

        for i, chunk in enumerate(chunks):
            state = await self._base_state(filename, doc_text, chunk, i, len(chunks), t_start)
            result = await self._plan_graph.ainvoke(state)
            for entry in result.get("plan", []):
                all_pages[entry["path"]] = entry
            seen_conflict_paths = {c["path"] for c in all_conflicts}
            for c in result.get("conflicts", []):
                if c["path"] not in seen_conflict_paths:
                    all_conflicts.append(c)
                    seen_conflict_paths.add(c["path"])
            all_errors.extend(result.get("errors", []))
            if result.get("log_entry"):
                last_log_entry = result["log_entry"]

        combined_plan = list(all_pages.values())
        log.info("IngestAgent.plan complete | file=%s | pages=%d | conflicts=%d | errors=%d | elapsed=%.1fs",
                 filename, len(combined_plan), len(all_conflicts), len(all_errors), time.perf_counter() - t_start)
        return {
            "plan": combined_plan,
            "conflicts": all_conflicts,
            "log_entry": last_log_entry,
            "doc_text": doc_text,
            "errors": all_errors,
        }

    async def execute(self, filename: str, plan: list[dict],
                      log_entry: str, user_notes: str = "", doc_text: str = "") -> dict:
        t_start = time.perf_counter()
        log.info("IngestAgent.execute | file=%s | pages=%d", filename, len(plan))

        chunk_text = doc_text[:self._CHUNK_SIZE] if doc_text else ""
        state = await self._base_state(filename, doc_text, chunk_text, 0, 1, t_start)
        state.update({
            "plan": plan,
            "log_entry": log_entry,
            "user_notes": user_notes,
        })

        result = await self._write_graph.ainvoke(state)
        return {
            "pages_created": result.get("pages_created", []),
            "pages_updated": result.get("pages_updated", []),
            "log_entry": log_entry,
            "errors": result.get("errors", []),
        }
