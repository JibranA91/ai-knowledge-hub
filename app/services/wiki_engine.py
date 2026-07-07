import io
import json
import re
import time
import zipfile
from datetime import datetime, UTC

from app.config import settings
from app.logger import get_logger
from app.services import md_sections
from app.services.bedrock import BedrockService
from app.services.export_templates import EXPORT_README_TEMPLATE
from app.services.ingest_agent import IngestAgent
from app.utils import page_type_or_infer, parse_llm_json, strip_fence_unconditional

log = get_logger(__name__)

# Per-page character budget for the qualitative lint audit. The audit LLM sees an
# excerpt of each page (not the full body) to keep the prompt bounded. Without an
# explicit marker it would mistake an excerpt's cutoff for the page ending and
# wrongly report long pages as "incomplete / truncated" (see _audit_excerpt).
_AUDIT_PAGE_CHARS = 8000


def _audit_excerpt(content: str, cap: int = _AUDIT_PAGE_CHARS) -> str:
    """Excerpt a page for the qualitative audit with an explicit truncation marker,
    so the LLM never mistakes an excerpt cutoff for a genuinely truncated page."""
    if len(content) <= cap:
        return content
    return content[:cap] + "\n\n[... excerpt truncated for length; the page continues beyond this point ...]"


_RETRIEVER_TEMPLATE = '''\
"""Self-contained wiki retriever — no LLM required.

Usage:
    python retriever.py "your question"
    python retriever.py "your question" --top-k 10 --graph-hops 2
    python retriever.py "your question" --query-vector vec.json --cosine-weight 0.8

vec.json must contain a JSON array of floats (e.g. produced by a Bedrock embedding call).
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Optional

try:
    import numpy as _np
    _HAVE_NUMPY = True
except ImportError:
    _HAVE_NUMPY = False

_wiki_cache: dict | None = None


def load_wiki(export_dir: str = ".") -> dict:
    """Load wiki pages, graph, and embeddings from an export directory."""
    global _wiki_cache
    if _wiki_cache is not None:
        return _wiki_cache

    root = Path(export_dir)
    pages: list[dict] = []
    for md_file in sorted((root / "wiki").glob("**/*.md")):
        rel = str(md_file.relative_to(root / "wiki"))
        content = md_file.read_text(encoding="utf-8", errors="replace")
        title = _extract_title(rel, content)
        pages.append({"path": rel, "title": title, "content": content})

    graph: dict = {}
    graph_file = root / "graph.json"
    if graph_file.exists():
        raw = json.loads(graph_file.read_text(encoding="utf-8"))
        graph = raw.get("adj", raw)

    embeddings: dict[str, list[float]] = {}
    emb_file = root / "embeddings.json"
    if emb_file.exists():
        for item in json.loads(emb_file.read_text(encoding="utf-8")):
            if item.get("embedding"):
                embeddings[item["path"]] = item["embedding"]

    _wiki_cache = {"pages": pages, "graph": graph, "embeddings": embeddings}
    return _wiki_cache


def _extract_title(path: str, content: str) -> str:
    m = re.search(r"^#\\s+(.+)", content, re.MULTILINE)
    if m:
        return m.group(1).strip()
    stem = path.split("/")[-1].rsplit(".", 1)[0]
    return stem.replace("-", " ").title()


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _bm25_score(pages: list[dict], query_terms: list[str]) -> list[float]:
    """BM25 relevance scores for each page."""
    k1, b = 1.5, 0.75
    avg_len = sum(len(_tokenize(p["content"])) for p in pages) / max(len(pages), 1)

    df: dict[str, int] = {}
    tf_lists: list[dict[str, int]] = []
    for p in pages:
        tokens = _tokenize(p["content"] + " " + p["title"])
        tf: dict[str, int] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        tf_lists.append(tf)
        for t in set(tokens):
            df[t] = df.get(t, 0) + 1

    n = len(pages)
    scores: list[float] = []
    for i, p in enumerate(pages):
        tf = tf_lists[i]
        doc_len = sum(tf.values())
        score = 0.0
        for term in query_terms:
            if term not in tf:
                continue
            idf = math.log((n - df.get(term, 0) + 0.5) / (df.get(term, 0) + 0.5) + 1)
            tf_norm = (tf[term] * (k1 + 1)) / (tf[term] + k1 * (1 - b + b * doc_len / max(avg_len, 1)))
            score += idf * tf_norm
        scores.append(score)
    return scores


def _cosine_score(
    query_vector: list[float],
    pages: list[dict],
    embeddings: dict[str, list[float]],
) -> list[float]:
    """Cosine similarity scores — returns 0.0 for pages with no embedding."""
    if not _HAVE_NUMPY:
        return [0.0] * len(pages)

    qv = _np.array(query_vector, dtype=float)
    qv_norm = _np.linalg.norm(qv)
    if qv_norm == 0:
        return [0.0] * len(pages)
    qv = qv / qv_norm

    scores: list[float] = []
    for p in pages:
        emb = embeddings.get(p["path"])
        if emb is None:
            scores.append(0.0)
        else:
            dv = _np.array(emb, dtype=float)
            dv_norm = _np.linalg.norm(dv)
            scores.append(float(_np.dot(qv, dv / dv_norm)) if dv_norm > 0 else 0.0)
    return scores


def find_relevant(
    question: str,
    query_vector: Optional[list[float]] = None,
    top_k: int = 5,
    bm25_weight: float = 0.4,
    cosine_weight: float = 0.6,
    export_dir: str = ".",
) -> list[dict]:
    """Return top_k pages most relevant to question, sorted by score."""
    wiki = load_wiki(export_dir)
    pages = wiki["pages"]
    if not pages:
        return []

    query_terms = _tokenize(question)
    bm25 = _bm25_score(pages, query_terms)

    if query_vector is not None and _HAVE_NUMPY:
        cosine = _cosine_score(query_vector, pages, wiki["embeddings"])
        total = bm25_weight + cosine_weight
        bw, cw = bm25_weight / total, cosine_weight / total
        max_bm25 = max(bm25) or 1.0
        scores = [bw * (b / max_bm25) + cw * c for b, c in zip(bm25, cosine)]
    else:
        max_bm25 = max(bm25) or 1.0
        scores = [b / max_bm25 for b in bm25]

    ranked = sorted(
        [
            {"path": p["path"], "title": p["title"], "content": p["content"], "score": s}
            for p, s in zip(pages, scores)
            if s > 0
        ],
        key=lambda x: x["score"],
        reverse=True,
    )
    return ranked[:top_k]


def expand_with_graph(
    paths: list[str],
    hops: int = 1,
    export_dir: str = ".",
) -> list[str]:
    """Add graph neighbours of paths (up to hops away)."""
    wiki = load_wiki(export_dir)
    graph = wiki["graph"]
    result = list(paths)
    current = set(paths)
    for _ in range(hops):
        neighbors: set[str] = set()
        for p in list(current):
            for neighbor in graph.get(p, []):
                if neighbor not in result:
                    neighbors.add(neighbor)
        result.extend(neighbors)
        current = neighbors
    return result


def retrieve(
    question: str,
    query_vector: Optional[list[float]] = None,
    top_k: int = 5,
    graph_hops: int = 1,
    bm25_weight: float = 0.4,
    cosine_weight: float = 0.6,
    export_dir: str = ".",
) -> list[dict]:
    """Main entrypoint. Returns [{path, title, content, score}] sorted by relevance."""
    results = find_relevant(question, query_vector, top_k, bm25_weight, cosine_weight, export_dir)
    if graph_hops > 0 and results:
        expanded = expand_with_graph([r["path"] for r in results], graph_hops, export_dir)
        wiki = load_wiki(export_dir)
        pages_map = {p["path"]: p for p in wiki["pages"]}
        extra = [
            {"path": p, "title": pages_map[p]["title"], "content": pages_map[p]["content"], "score": 0.0}
            for p in expanded
            if p not in {r["path"] for r in results} and p in pages_map
        ]
        results = results + extra
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Retrieve relevant wiki pages for a question.")
    parser.add_argument("question", help="The question to answer")
    parser.add_argument("--query-vector", metavar="FILE",
                        help="JSON file containing a pre-computed query embedding (list of floats)")
    parser.add_argument("--top-k", type=int, default=5, metavar="N")
    parser.add_argument("--graph-hops", type=int, default=1, metavar="N")
    parser.add_argument("--bm25-weight", type=float, default=0.4)
    parser.add_argument("--cosine-weight", type=float, default=0.6)
    parser.add_argument("--export-dir", default=".", help="Path to the unzipped export directory")
    args = parser.parse_args()

    query_vec = None
    if args.query_vector:
        query_vec = json.loads(Path(args.query_vector).read_text(encoding="utf-8"))

    results = retrieve(
        args.question,
        query_vector=query_vec,
        top_k=args.top_k,
        graph_hops=args.graph_hops,
        bm25_weight=args.bm25_weight,
        cosine_weight=args.cosine_weight,
        export_dir=args.export_dir,
    )

    for i, r in enumerate(results, 1):
        print(f"\\n{\'=\' * 60}")
        print(f"[{i}] {r[\'path\']} (score: {r[\'score\']:.4f})")
        print(f"Title: {r[\'title\']}")
        print(f"{\'=\' * 60}")
        print(r["content"][:500])
        if len(r["content"]) > 500:
            print("... [truncated]")
'''

def _TODAY() -> str:
    from zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo(settings.APP_TIMEZONE)).strftime("%Y-%m-%d")


def _render_query_page(question: str, answer: str, sources: list[str]) -> str:
    """Render a saved query/chat page with `type: query_result` frontmatter."""
    safe_question = question.replace('"', "'")
    sources_text = ", ".join(sources) if sources else "none"
    return (
        "---\n"
        f'title: "{safe_question[:120]}"\n'
        "type: query_result\n"
        f"date: {_TODAY()}\n"
        "---\n\n"
        f"# Q: {question}\n\n"
        f"*{_TODAY()}*\n\n"
        f"{answer}\n\n"
        "---\n"
        f"*Sources: {sources_text}*\n"
    )


class WikiEngine:
    def __init__(self):
        self.query_bedrock = BedrockService(settings.BEDROCK_QUERY_MODEL_ID)
        # Dedicated client for the conversational AI Writer agent. By
        # default this uses the same model as query, but it can be pointed at
        # a stronger model (e.g. Sonnet) via BEDROCK_DRAFT_AGENT_MODEL_ID.
        self.draft_agent_bedrock = BedrockService(settings.BEDROCK_DRAFT_AGENT_MODEL_ID)
        # Inline AI editor — rewrites a whole page or one section on demand.
        self.edit_bedrock = BedrockService(settings.BEDROCK_EDIT_MODEL_ID)
        self._ingest_agent = IngestAgent()

    # ── Internal helpers ───────────────────────────────────────────────────

    async def _schema(self) -> str:
        from app.services.wiki_db import get_wiki_file
        return await get_wiki_file("schema/AGENTS.md") or ""

    async def _compact_index(self) -> str:
        from app.services.wiki_db import get_compact_index
        return await get_compact_index()

    async def _read_page(self, rel_path: str) -> str:
        from app.services.wiki_db import get_wiki_page_content
        return await get_wiki_page_content(rel_path) or ""

    async def _write_page(self, rel_path: str, content: str) -> None:
        from app.services.wiki_db import upsert_wiki_page
        await upsert_wiki_page(rel_path, content)

    async def _append_log(self, entry: str) -> None:
        from app.services.wiki_db import append_audit_log
        await append_audit_log("log", entry)

    # ── Public operations ──────────────────────────────────────────────────

    def _graph(self):
        from app.services.graph import get_graph
        return get_graph()

    async def ingest(self, filename: str) -> dict:
        result = await self._ingest_agent.run(filename)
        touched = result.get("pages_created", []) + result.get("pages_updated", [])
        if touched:
            await self._graph().update_pages(touched)
        return result

    async def plan(self, filename: str) -> dict:
        return await self._ingest_agent.plan(filename)

    async def plan_chat(self, filename: str, message: str, current_plan: list[dict],
                        history: list[dict], doc_text: str, conflicts: list[dict] | None = None) -> dict:
        import json
        schema = await self._schema()
        plan_json = json.dumps(current_plan, indent=2)

        conflicts_section = ""
        if conflicts:
            conflict_lines = ["## Detected Conflicts (require user resolution before writing)"]
            for c in conflicts:
                conflict_lines.append(
                    f"- **`{c['path']}`**: existing wiki says \"{c['existing_claim']}\" "
                    f"but new document says \"{c['new_claim']}\" "
                    f"(planner default resolution: {c.get('resolution', 'surface')})"
                )
            conflict_lines.append(
                "\nFor each conflict, if the user has not yet told you which version is correct, "
                "you MUST ask them before finalising the plan."
            )
            conflicts_section = "\n\n" + "\n".join(conflict_lines)

        system_prompt = f"""You are an AI wiki planner helping a user review and refine an ingest plan.

When you need to update the plan, respond ONLY with a JSON object:
{{
  "reply": "Explanation of what you changed",
  "updated_plan": [{{"path": "...", "action": "create|update", "brief": "..."}}],
  "log_entry": "## [{_TODAY()}] ingest | ..." or null
}}

When only answering a question (no plan change), respond ONLY with:
{{
  "reply": "Your message",
  "updated_plan": null,
  "log_entry": null
}}

Always return valid JSON. No prose outside the JSON object.

Wiki Schema:
{schema}

Current Plan ({len(current_plan)} pages):
{plan_json}{conflicts_section}

Document filename: {filename}
Document excerpt:
{doc_text[:3000]}"""

        api_msgs = [
            {"role": h["role"], "content": [{"text": h["content"] or ""}]}
            for h in history
        ]
        api_msgs.append({"role": "user", "content": [{"text": message}]})

        raw = await self.query_bedrock.converse(system_prompt, api_msgs, max_tokens=4096, operation="plan_chat")
        try:
            data = parse_llm_json(raw)
            return {
                "reply": data.get("reply") or raw,
                "updated_plan": data.get("updated_plan"),
                "log_entry": data.get("log_entry"),
            }
        except Exception:
            return {"reply": raw or "", "updated_plan": None, "log_entry": None}

    async def execute_ingest(self, filename: str, plan: list[dict],
                             log_entry: str, user_notes: str = "", doc_text: str = "") -> dict:
        result = await self._ingest_agent.execute(filename, plan, log_entry, user_notes, doc_text)
        touched = result.get("pages_created", []) + result.get("pages_updated", [])
        if touched:
            await self._graph().update_pages(touched)
        return result

    async def graph_dict(self) -> dict:
        graph = await self._graph().ensure_loaded()
        return graph.as_dict()

    async def _find_relevant_pages(self, question: str) -> list[str]:
        """Return up to 8 page paths relevant to *question*.

        Strategy (in priority order):
        1. Hybrid BM25 + vector search via DB when embedding is configured.
        2. BM25-only DB search as a fast intermediate fallback.
        3. LLM-driven index scan when BM25 also returns nothing (original behaviour).

        In all cases, 1-hop graph neighbours of the selected pages are appended
        (up to 3 extra pages) to widen context.
        """
        from app.services import embeddings as _emb
        from app.services.wiki_db import find_relevant_pages as db_find, wiki_page_exists

        # ── DB-based search (embedding or BM25) ────────────────────────────
        db_paths = await db_find(question, limit=8)
        if db_paths:
            extra = self._graph().neighbors(db_paths, hops=1)
            added = [p for p in extra[:3] if p not in db_paths]
            db_paths.extend(added)
            if added:
                log.debug("_find_relevant_pages | graph_neighbors_added=%s", added)
            return db_paths

        # ── LLM fallback (no DB results — empty wiki or no keyword match) ──
        log.info("_find_relevant_pages | db_returned_nothing | falling back to LLM index scan | q=%r", question[:60])
        find_prompt = """Given a question and a wiki index, return ONLY JSON:
{"relevant_pages": ["path/to/page.md"]}
Include at most 8 pages. Only include paths that appear in the index."""
        find_msgs = [{"role": "user", "content": [
            {"text": f"Question: {question}\n\nWiki pages (path — title):\n{await self._compact_index()}"}
        ]}]
        raw = await self.query_bedrock.converse(find_prompt, find_msgs, max_tokens=512, operation="find_pages")
        try:
            relevant = parse_llm_json(raw).get("relevant_pages", [])
        except Exception as e:
            log.warning("_find_relevant_pages | LLM JSON parse failed: %s | raw=%r", e, raw[:200])
            relevant = []
        _EXCLUDED = {"index.md", "log.md"}
        valid: list[str] = []
        for p in relevant:
            if isinstance(p, str) and p.endswith(".md") and p not in _EXCLUDED:
                if await wiki_page_exists(p):
                    valid.append(p)
        extra = self._graph().neighbors(valid, hops=1)
        added = [p for p in extra[:3] if p not in valid]
        valid.extend(added)
        log.info("_find_relevant_pages | llm_fallback | pages=%s", valid)
        return valid

    async def query(self, question: str, save_to_wiki: bool = False) -> dict:
        t = time.perf_counter()
        log.info("Query started | question=%r", question[:80])

        relevant = await self._find_relevant_pages(question)
        log.info("Query | relevant_pages=%s", relevant)

        page_contents = []
        for p in relevant:
            content = await self._read_page(p)
            if content:
                page_contents.append(f"**{p}**\n{content}")
        pages_text = "\n\n---\n\n".join(page_contents) or "No relevant pages found in the wiki yet."

        schema = await self._schema()
        answer_prompt = f"""Answer questions using the wiki as your source. Cite page names.
{f'Schema: {schema}' if schema else ''}"""

        answer_msgs = [{"role": "user", "content": [
            {"text": f"Question: {question}\n\nRelevant pages:\n{pages_text}"}
        ]}]
        answer = await self.query_bedrock.converse(answer_prompt, answer_msgs, max_tokens=4096, operation="query")

        saved_to = None
        if save_to_wiki:
            from app.services.wiki_state import begin_action
            slug = re.sub(r"[^a-z0-9]+", "-", question.lower())[:50].strip("-")
            saved_to = f"queries/{slug}.md"
            async with begin_action("saved_query", summary=question[:120]):
                await self._write_page(saved_to, _render_query_page(question, answer, relevant))
            await self._append_log(f"## [{_TODAY()}] query | {question[:80]}\nSaved to {saved_to}")

        log.info("Query complete | elapsed=%.1fs | saved_to=%s", time.perf_counter() - t, saved_to)
        return {"answer": answer, "sources": relevant, "saved_to": saved_to}

    async def _maybe_summarize(self, session) -> None:
        from app.services import chat_sessions
        if len(session.messages) <= chat_sessions.SUMMARIZE_AFTER:
            return
        log.info("Chat | summarizing | session=%s | messages=%d", session.session_id, len(session.messages))
        msgs_text = "\n".join(f"{m['role'].upper()}: {m['text'][:400]}" for m in session.messages)
        prefix = f"Previous summary:\n{session.summary}\n\n" if session.summary else ""
        session.summary = await self.query_bedrock.converse(
            "Summarize this conversation in 3-5 sentences, capturing key topics and conclusions.",
            [{"role": "user", "content": [{"text": f"{prefix}Conversation:\n{msgs_text}"}]}],
            max_tokens=512,
            operation="summarize",
        )
        session.messages = session.messages[-chat_sessions.KEEP_RECENT:]

    async def chat(self, session_id: str | None, message: str, save_to_wiki: bool = False) -> dict:
        from app.services import chat_sessions
        t = time.perf_counter()
        session = await chat_sessions.get_or_create(session_id)
        log.info("Chat | session=%s | message=%r", session.session_id, message[:80])

        await self._maybe_summarize(session)

        relevant = await self._find_relevant_pages(message)
        page_contents = []
        for p in relevant:
            content = await self._read_page(p)
            if content:
                page_contents.append(f"**{p}**\n{content}")
        pages_text = "\n\n---\n\n".join(page_contents) or "No relevant pages found in the wiki yet."

        system_prompt = "You are a concise assistant for a company knowledge base. " \
        "Answer in 2-4 sentences unless a longer answer is clearly needed. " \
        "Use plain prose — avoid bullet points unless listing 3+ distinct items. " \
        "Cite wiki page names inline when referencing information. " \
        "Do not repeat or echo the user's question as a heading or title in your response." \
        "If you don't know the answer, say you don't know — do not try to fabricate an answer. " \
        "If the question is unclear, ask for clarification rather than making assumptions. " \
        "If there is an information gap, ask the user to upload a document that you can ingest and learn from."

        if session.summary:
            system_prompt += f"\n\nPrevious conversation summary:\n{session.summary}"

        api_msgs = [
            {"role": m["role"], "content": [{"text": m["text"]}]}
            for m in session.messages
        ]
        api_msgs.append({"role": "user", "content": [
            {"text": f"{message}\n\n---\nRelevant wiki pages:\n{pages_text}"}
        ]})

        answer = await self.query_bedrock.converse(system_prompt, api_msgs, max_tokens=4096, operation="chat")

        session.messages.append({"role": "user", "text": message, "sources": []})
        session.messages.append({"role": "assistant", "text": answer, "sources": relevant})
        await chat_sessions.save(session)

        saved_to = None
        if save_to_wiki:
            from app.services.wiki_state import begin_action
            slug = re.sub(r"[^a-z0-9]+", "-", message.lower())[:50].strip("-")
            saved_to = f"queries/{slug}.md"
            async with begin_action("saved_query", summary=message[:120]):
                await self._write_page(saved_to, _render_query_page(message, answer, relevant))
            await self._append_log(f"## [{_TODAY()}] chat | {message[:80]}\nSaved to {saved_to}")

        log.info("Chat complete | session=%s | elapsed=%.1fs", session.session_id, time.perf_counter() - t)
        return {"session_id": session.session_id, "answer": answer, "sources": relevant, "saved_to": saved_to}

    async def chat_stream(self, session_id: str | None, message: str, save_to_wiki: bool = False):
        import json as _json
        from app.services import chat_sessions
        t = time.perf_counter()
        session = await chat_sessions.get_or_create(session_id)
        log.info("Chat (stream) | session=%s | message=%r", session.session_id, message[:80])

        await self._maybe_summarize(session)

        relevant = await self._find_relevant_pages(message)
        page_contents = []
        for p in relevant:
            content = await self._read_page(p)
            if content:
                page_contents.append(f"**{p}**\n{content}")
        pages_text = "\n\n---\n\n".join(page_contents) or "No relevant pages found in the wiki yet."

        system_prompt = "You are a concise assistant for a company knowledge base. Answer in 2-4 sentences unless a longer answer is clearly needed. Use plain prose. Cite wiki page names inline when referencing information. Do not repeat or echo the user's question as a heading or title in your response."
        if session.summary:
            system_prompt += f"\n\nPrevious conversation summary:\n{session.summary}"

        api_msgs = [
            {"role": m["role"], "content": [{"text": m["text"]}]}
            for m in session.messages
        ]
        api_msgs.append({"role": "user", "content": [
            {"text": f"{message}\n\n---\nRelevant wiki pages:\n{pages_text}"}
        ]})

        yield f"data: {_json.dumps({'type': 'meta', 'session_id': session.session_id, 'sources': relevant})}\n\n"

        answer_parts: list[str] = []
        async for chunk in self.query_bedrock.converse_stream(system_prompt, api_msgs, max_tokens=4096, operation="chat_stream"):
            answer_parts.append(chunk)
            yield f"data: {_json.dumps({'type': 'chunk', 'text': chunk})}\n\n"

        answer = "".join(answer_parts)

        session.messages.append({"role": "user", "text": message, "sources": []})
        session.messages.append({"role": "assistant", "text": answer, "sources": relevant})
        await chat_sessions.save(session)

        saved_to = None
        if save_to_wiki:
            from app.services.wiki_state import begin_action
            slug = re.sub(r"[^a-z0-9]+", "-", message.lower())[:50].strip("-")
            saved_to = f"queries/{slug}.md"
            async with begin_action("saved_query", summary=message[:120]):
                await self._write_page(saved_to, _render_query_page(message, answer, relevant))
            await self._append_log(f"## [{_TODAY()}] chat | {message[:80]}\nSaved to {saved_to}")

        log.info("Chat (stream) complete | session=%s | elapsed=%.1fs", session.session_id, time.perf_counter() - t)
        yield f"data: {_json.dumps({'type': 'done', 'saved_to': saved_to})}\n\n"

    async def lint(self) -> dict:
        """Wiki health check.

        The headline `health_score` is **deterministic** — computed by
        `wiki_health` from the same five structural signals recalibrate's triage
        uses (stale, orphan, stub, broken_links, duplicate_candidate). Because
        the score is derived from exactly what recalibration repairs, running a
        recalibration can only raise (or hold) it. The LLM is used only for the
        *qualitative* audit (contradictions, cross-ref gaps, suggestions) and runs
        at temperature 0 for stability; it no longer decides the score.
        """
        from app.services.wiki_db import list_wiki_pages_with_content
        from app.services import wiki_health
        t = time.perf_counter()
        log.info("Lint started")

        # Same page universe recalibrate triages: skip index/log and user-saved
        # query pages, so the score reflects the corpus recalibration acts on.
        full_pages: dict[str, str] = {}
        for rel, content in await list_wiki_pages_with_content():
            if rel in ("index.md", "log.md"):
                continue
            if page_type_or_infer(rel, content) == "query_result":
                continue
            if content:
                full_pages[rel] = content

        # ── Deterministic structural score (no LLM) ─────────────────────────
        # Feed the same recalibration review times triage uses, so the score and
        # triage never disagree on staleness (a reviewed page is fresh in both).
        from app.services.wiki_db import get_recalibration_state
        recal_state = await get_recalibration_state()
        reviewed_at = {p: info["reviewed_at"] for p, info in recal_state.items()}
        graph = await self._graph().ensure_loaded()
        health = wiki_health.compute_health(full_pages, graph._adj, reviewed_at=reviewed_at)
        # Content gaps: concepts linked from several pages but with no page yet.
        link_gaps = wiki_health.classify_link_gaps(full_pages)
        article_candidates = link_gaps["auto_create"]
        log.info(
            "Lint | pages=%d | score=%d | flagged=%d | breakdown=%s | gaps=%d",
            health["total_pages"], health["health_score"], health["flagged_pages"],
            health["breakdown"], len(article_candidates),
        )

        # ── Qualitative audit (LLM, temperature 0) ──────────────────────────
        pages_text = "\n\n---\n\n".join(f"**{k}**\n{_audit_excerpt(v)}" for k, v in full_pages.items())
        system_prompt = """Audit a wiki for QUALITATIVE health issues a human reviewer would care about — \
contradictions between pages, missing pages that should exist, and weak cross-referencing. \
Page bodies may be shown as truncated EXCERPTS (marked "[... excerpt truncated for length ...]"). \
Do NOT report a page as incomplete, truncated, or "ending mid-sentence": you are viewing an \
excerpt, not the full page. \
Do NOT score the wiki. Return ONLY JSON:
{
  "issues": [{"type": "contradiction|missing_page|stale|cross_ref", "description": "...", "affected_pages": []}],
  "suggestions": ["..."]
}"""
        msgs = [{"role": "user", "content": [
            {"text": f"Wiki pages (path — title):\n{await self._compact_index()}\n\nPage content:\n{pages_text}"}
        ]}]
        issues: list = []
        suggestions: list = []
        try:
            raw = await self.query_bedrock.converse(
                system_prompt, msgs, max_tokens=4096, operation="lint", temperature=0,
            )
            parsed = parse_llm_json(raw)
            issues = parsed.get("issues") or []
            suggestions = parsed.get("suggestions") or []
        except Exception as e:
            log.warning("Lint | qualitative audit failed (deterministic score still returned): %s", e)

        result = {
            "health_score": health["health_score"],
            "breakdown": health["breakdown"],
            "total_pages": health["total_pages"],
            "flagged_pages": health["flagged_pages"],
            "issues": issues,
            "suggestions": suggestions,
            "article_candidates": article_candidates,
            "link_gaps": link_gaps,
        }

        await self._append_log(
            f"## [{_TODAY()}] lint | Health check\n"
            f"Score: {health['health_score']}/100, "
            f"Flagged: {health['flagged_pages']}/{health['total_pages']}, "
            f"Issues: {len(issues)}"
        )
        log.info(
            "Lint complete | score=%d | issues=%d | elapsed=%.1fs",
            health["health_score"], len(issues), time.perf_counter() - t,
        )
        return result

    async def build_wiki_export(self, include_embeddings: bool) -> tuple[bytes, int, bool]:
        from app.services.wiki_db import list_wiki_pages_for_export, get_wiki_file

        pages = await list_wiki_pages_for_export(include_embeddings)
        graph_raw = await get_wiki_file("wiki/.graph.json") or "{}"
        schema_md = await get_wiki_file("schema/AGENTS.md") or ""

        has_embeddings = include_embeddings and any(p["embedding"] for p in pages)
        manifest = {
            # Bump when the bundle layout changes in a way importers must gate on.
            "format_version": 1,
            "exported_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "page_count": len(pages),
            "has_schema": bool(schema_md),
            "has_embeddings": has_embeddings,
            "embedding_dimensions": settings.EMBEDDING_DIMENSIONS,
            "embedding_model": settings.BEDROCK_EMBEDDING_MODEL_ID,
        }

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for page in pages:
                zf.writestr(f"wiki/{page['path']}", page["content"])
            # The schema (AGENTS.md) governs structure + future ingests, so it
            # must travel with the bundle for an import to reproduce the wiki.
            if schema_md:
                zf.writestr("schema/AGENTS.md", schema_md)
            zf.writestr("graph.json", graph_raw)
            zf.writestr("manifest.json", json.dumps(manifest, indent=2))
            zf.writestr("retriever.py", _RETRIEVER_TEMPLATE)
            if has_embeddings:
                embeddings_section = (
                    f"This export **includes** pre-computed vector embeddings.\n\n"
                    f"- **Model**: `{settings.BEDROCK_EMBEDDING_MODEL_ID}`\n"
                    f"- **Dimensions**: `{settings.EMBEDDING_DIMENSIONS}`\n\n"
                    f"To use cosine/hybrid retrieval you must generate query vectors with the "
                    f"same model at the same dimension count."
                )
            else:
                embeddings_section = (
                    "This export was created **without** embeddings. "
                    "Only BM25 keyword retrieval is available (`retriever.py` works out of the box).\n\n"
                    "Re-export with *Include embeddings* enabled to unlock hybrid retrieval."
                )
            readme = EXPORT_README_TEMPLATE.format(
                embeddings_section=embeddings_section,
                embedding_model=settings.BEDROCK_EMBEDDING_MODEL_ID or "N/A",
                embedding_dimensions=settings.EMBEDDING_DIMENSIONS or "N/A",
            )
            zf.writestr("README.md", readme)
            if has_embeddings:
                emb_list = [
                    {"path": p["path"], "embedding": p["embedding"]}
                    for p in pages
                    if p["embedding"]
                ]
                zf.writestr("embeddings.json", json.dumps(emb_list))

        data = buf.getvalue()
        log.info("build_wiki_export | pages=%d | has_embeddings=%s | size=%d",
                 len(pages), has_embeddings, len(data))
        return data, len(pages), has_embeddings

    # ── Writer Mode ────────────────────────────────────────────────────────

    async def writer_chat_stream(self, session_id: str | None, message: str):
        """Streaming chat for the document-writer agent.

        Yields SSE-formatted strings (`data: {...}\\n\\n`). The agent embeds
        `[DRAFT_START]...[DRAFT_END]` or `[SECTION_START:Heading]...[SECTION_END]`
        markers in its reply; the embedded `WriterStreamParser` splits these
        into separate event types so the frontend can render the live draft
        in a side panel while keeping chat text in the chat bubble.
        """
        import json as _json
        from app.services import chat_sessions
        from app.services.wiki_db import append_audit_log
        from app.services.writer_stream import WriterStreamParser

        t = time.perf_counter()
        session = await chat_sessions.get_or_create(session_id, mode="writer")
        log.info("Writer chat (stream) | session=%s | message=%r",
                 session.session_id, message[:80])

        await self._maybe_summarize(session)

        relevant = await self._find_relevant_pages(message)
        page_contents = []
        for p in relevant:
            content = await self._read_page(p)
            if content:
                page_contents.append(f"**{p}**\n{content}")
        pages_text = "\n\n---\n\n".join(page_contents) or "No relevant pages found in the wiki yet."

        schema = await self._schema()
        system_prompt = _writer_system_prompt(schema, pages_text, session.draft_content)
        if session.summary:
            system_prompt += f"\n\nPrevious conversation summary:\n{session.summary}"

        api_msgs = [
            {"role": m["role"], "content": [{"text": m["text"]}]}
            for m in session.messages
        ]
        api_msgs.append({"role": "user", "content": [{"text": message}]})

        yield f"data: {_json.dumps({'type': 'meta', 'session_id': session.session_id, 'sources': relevant})}\n\n"

        parser = WriterStreamParser()
        full_assistant_text: list[str] = []
        section_errors: list[str] = []

        async for chunk in self.draft_agent_bedrock.converse_stream(
            system_prompt, api_msgs, max_tokens=4096, operation="writer_chat"
        ):
            full_assistant_text.append(chunk)
            for evt in parser.feed(chunk):
                # Filter out internal `_done` events here; we'll handle them after stream
                if evt["type"] in ("draft_done", "section_done"):
                    continue
                yield f"data: {_json.dumps(evt)}\n\n"
        for evt in parser.flush():
            yield f"data: {_json.dumps(evt)}\n\n"

        # ── Post-stream: persist draft + apply section patches ──────────────
        has_draft = False
        section_heading_for_done: str | None = None

        if parser.draft_content:
            session.draft_content = parser.draft_content
            has_draft = True
            await append_audit_log(
                "writer_draft_written",
                f"session={session.session_id}",
                {"session_id": session.session_id, "chars": len(parser.draft_content)},
            )

        for heading, body in parser.section_patches:
            new_content, err = await chat_sessions.patch_draft_section(
                session.session_id, heading, body
            )
            if err:
                section_errors.append(f"{heading}: {err}")
                yield (
                    f"data: {_json.dumps({'type': 'error', 'retry': True, 'heading': heading, 'message': f'Section patch failed for [{heading}]: {err}. Please retry with a more specific heading or emit a full [DRAFT_START] rewrite.'})}\n\n"
                )
                continue
            session.draft_content = new_content or session.draft_content
            section_heading_for_done = heading
            await append_audit_log(
                "writer_section_patched",
                f"heading={heading} | session={session.session_id}",
                {"session_id": session.session_id, "heading": heading},
            )

        # ── Resolve final draft_ready state for this turn ──────────────────
        # Rule: a turn that emits new content (full draft or section patch)
        # resets readiness to whatever the agent declared in THIS turn. A turn
        # with no content change but a [DRAFT_READY] marker flips it on. A
        # turn with neither leaves the prior readiness state untouched.
        content_changed = has_draft or any(
            heading == section_heading_for_done
            for heading, _ in parser.section_patches
        ) or bool(parser.section_patches)
        if content_changed:
            session.draft_ready = parser.draft_ready_seen
        elif parser.draft_ready_seen:
            session.draft_ready = True

        # ── Append turn to history and save ────────────────────────────────
        assistant_text = "".join(full_assistant_text)
        session.messages.append({"role": "user", "text": message, "sources": []})
        session.messages.append({"role": "assistant", "text": assistant_text, "sources": relevant})
        await chat_sessions.save(session)

        log.info(
            "Writer chat (stream) complete | session=%s | elapsed=%.1fs | draft=%s | sections=%d | ready=%s | errs=%d",
            session.session_id, time.perf_counter() - t, has_draft,
            len(parser.section_patches), session.draft_ready, len(section_errors),
        )
        yield (
            f"data: {_json.dumps({'type': 'done', 'has_draft': has_draft, 'section_heading': section_heading_for_done, 'draft_ready': session.draft_ready, 'errors': section_errors or None})}\n\n"
        )


    async def edit_stream(
        self,
        path: str,
        scope: str,
        action: str,
        heading: str | None = None,
        instruction: str = "",
        reconcile_with: str | None = None,
    ):
        """Stream an AI-proposed rewrite of a wiki page (or one section).

        Yields SSE-formatted strings (`data: {...}\\n\\n`). Read-only: it never
        writes the page. The client diffs the proposal against the live page and,
        if accepted, applies it via the normal PUT /api/wiki/{path} (which tracks
        a revision and stays revertible). Events:
          meta   {sources}            — grounding pages used
          chunk  {text}               — streamed proposed markdown
          error  {message}            — page/section/model problem; terminal
          done   {full_content, ...}  — the full proposed page after splicing
        """
        import json as _json

        current = await self._read_page(path)
        if not current.strip():
            yield f"data: {_json.dumps({'type': 'error', 'message': f'Page not found or empty: {path}'})}\n\n"
            return

        # Resolve the target text the model rewrites: the whole page, or one section.
        if scope == "section":
            if not heading:
                yield f"data: {_json.dumps({'type': 'error', 'message': 'A section heading is required for a section edit.'})}\n\n"
                return
            target, err = md_sections.extract_section(current, heading)
            if err:
                yield f"data: {_json.dumps({'type': 'error', 'message': f'Section [{heading}]: {err}'})}\n\n"
                return
        else:
            scope = "page"
            target = current

        # Grounding: the reconcile target (if any) plus a few related pages so the
        # model can keep facts/terminology consistent and avoid contradictions.
        sources: list[str] = []
        context_blocks: list[str] = []
        if reconcile_with:
            other = await self._read_page(reconcile_with)
            if other:
                sources.append(reconcile_with)
                context_blocks.append(f"**{reconcile_with}** (reconcile against this)\n{other}")
        for p in await self._find_relevant_pages(f"{path}\n{instruction}"):
            if p == path or p == reconcile_with or len(sources) >= 4:
                continue
            body = await self._read_page(p)
            if body:
                sources.append(p)
                context_blocks.append(f"**{p}**\n{body}")
        pages_text = "\n\n---\n\n".join(context_blocks) or "No related pages."

        schema = await self._schema()
        # Catalog of linkable pages (as ready-to-use [[page-name]] tokens) so the
        # agent can cross-link any concept/entity/source it mentions that already
        # has a page — matching the wiki's [[page-name]] convention. The graph
        # resolves these on apply (see _refresh_graph_after_write).
        link_catalog = _link_catalog(await self._compact_index(), exclude=path)
        system_prompt = _edit_system_prompt(
            schema, pages_text, scope, action, instruction, heading, link_catalog
        )
        user_msg = (
            f"Rewrite the following {'section' if scope == 'section' else 'page'} "
            f"per the instructions. Output ONLY the rewritten markdown.\n\n"
            f"```markdown\n{target}\n```"
        )

        yield f"data: {_json.dumps({'type': 'meta', 'sources': sources, 'scope': scope, 'heading': heading})}\n\n"

        collected: list[str] = []
        try:
            async for chunk in self.edit_bedrock.converse_stream(
                system_prompt,
                [{"role": "user", "content": [{"text": user_msg}]}],
                max_tokens=4096,
                operation="wiki_edit",
            ):
                collected.append(chunk)
                yield f"data: {_json.dumps({'type': 'chunk', 'text': chunk})}\n\n"
        except Exception as exc:  # noqa: BLE001 — surface model/transport errors to the client
            log.exception("edit_stream | model error | path=%s", path)
            yield f"data: {_json.dumps({'type': 'error', 'message': f'Generation failed: {exc}'})}\n\n"
            return

        proposed = strip_fence_unconditional("".join(collected))
        if not proposed:
            yield f"data: {_json.dumps({'type': 'error', 'message': 'The model returned an empty rewrite.'})}\n\n"
            return

        if scope == "section":
            full_content, err = md_sections.replace_section(current, heading, proposed)
            if err:
                yield f"data: {_json.dumps({'type': 'error', 'message': f'Could not splice section back: {err}'})}\n\n"
                return
        else:
            full_content = proposed if proposed.endswith("\n") else proposed + "\n"

        yield (
            f"data: {_json.dumps({'type': 'done', 'full_content': full_content, 'scope': scope, 'heading': heading})}\n\n"
        )


def _link_catalog(compact_index: str, exclude: str = "", limit: int = 200) -> str:
    """Turn a 'path — title' compact index into ready-to-use `[[page-name]]`
    entries for the editor's cross-linking instruction. page-name is the
    filename stem (no dir, no .md), which is what the graph resolves wikilinks
    against. Skips index/log and the page being edited; caps length for tokens.
    """
    lines = []
    for raw in (compact_index or "").splitlines():
        path, _, title = raw.partition(" — ")
        path = path.strip()
        if not path or path == exclude or path in ("index.md", "log.md"):
            continue
        stem = path.rsplit("/", 1)[-1]
        if stem.endswith(".md"):
            stem = stem[:-3]
        lines.append(f"- [[{stem}]] — {title.strip()}" if title.strip() else f"- [[{stem}]]")
        if len(lines) >= limit:
            break
    return "\n".join(lines)


_EDIT_ACTIONS = {
    "expand": "Expand the content with more relevant detail, depth, and concrete specifics, "
              "while staying accurate and strictly on-topic. Do not pad with filler.",
    "summarize": "Make the content more concise: remove redundancy and tighten wording while "
                 "preserving every key fact and the original structure.",
    "improve": "Improve clarity, structure, grammar, and correctness without changing the "
               "meaning. Fix awkward phrasing, broken markdown, and obvious errors.",
    "reconcile": "Reconcile this content with the referenced page(s) below: resolve "
                 "contradictions, align terminology and facts, and prefer the most accurate, "
                 "internally consistent version. Do not invent facts not supported by the sources.",
    "custom": "Apply the user's instruction below precisely.",
}


def _edit_system_prompt(
    schema: str, pages_text: str, scope: str, action: str, instruction: str,
    heading: str | None, link_catalog: str = "",
) -> str:
    task = _EDIT_ACTIONS.get(action, _EDIT_ACTIONS["custom"])
    if scope == "section":
        scope_rules = (
            f"You are editing ONLY the section titled '{heading}'. Return the rewritten "
            f"section STARTING WITH its heading line (e.g. `## {heading}`). Do NOT include "
            f"other sections, the page title, or YAML frontmatter."
        )
    else:
        scope_rules = (
            "You are editing the WHOLE page. Preserve the YAML frontmatter block "
            "(--- ... ---) at the top, updating fields only if the instruction requires it. "
            "Return the complete page markdown."
        )
    extra = f"\n\nAdditional user instruction (highest priority):\n{instruction.strip()}" if instruction.strip() else ""
    # Cross-linking block: only when there are other pages to link to.
    linking = ""
    if link_catalog.strip():
        linking = (
            "\n\nCROSS-LINKING (important):\n"
            "- When your rewrite mentions a concept, entity, or source that has its OWN page "
            "in the catalog below, link it inline with `[[page-name]]` (the wiki's link syntax "
            "— the name in double brackets, no `.md`, exactly as listed).\n"
            "- Only link pages that genuinely match what the text refers to. NEVER invent a "
            "page name that is not in the catalog, and don't force irrelevant links.\n"
            "- Preserve links already present in the source unless they no longer apply.\n"
            "- Do not link the page to itself.\n\n"
            f"LINKABLE PAGES (use the exact [[page-name]] token shown):\n{link_catalog}"
        )
    return (
        "You are an expert technical editor for a company knowledge base. You rewrite "
        "existing wiki markdown on demand.\n\n"
        f"TASK: {task}{extra}\n\n"
        f"SCOPE: {scope_rules}\n\n"
        "OUTPUT RULES (follow exactly):\n"
        "- Output ONLY the rewritten markdown — no preamble, no explanation, no commentary.\n"
        "- Do NOT wrap your output in a ``` code fence.\n"
        "- Keep the same general structure and heading levels unless the instruction says otherwise.\n"
        "- Stay faithful to the source material; never fabricate facts.\n"
        "- Follow the wiki schema below.\n"
        f"{linking}\n\n"
        f"WIKI SCHEMA:\n{schema}\n\n"
        f"RELATED WIKI PAGES (for grounding and consistency):\n\n{pages_text}"
    )


def _writer_system_prompt(schema: str, pages_text: str, current_draft: str) -> str:
    """Build the writer-mode system prompt.

    The agent is given the wiki schema, relevant existing pages, and the
    current draft so it can decide whether to ask follow-ups, do a full
    rewrite, or patch one section.
    """
    draft_block = (
        f"\n\nCURRENT DRAFT (this is what is in the user's preview right now — "
        f"reference it when deciding between a full rewrite and a section patch):\n"
        f"```markdown\n{current_draft}\n```"
        if current_draft else
        "\n\nCURRENT DRAFT: (empty — the user has not received a draft yet)"
    )
    return (
        "You are a document-writing agent for a company knowledge base. "
        "Your job is to help the user author a single, well-structured markdown "
        "page through conversation. You operate in three phases:\n\n"
        "PHASE 1 — GATHER: Ask focused questions one at a time to collect the "
        "information you need (purpose, audience, scope, key facts, related "
        "wiki pages, contradictions with what already exists). Continue until "
        "you are confident you can write a useful page.\n\n"
        "PHASE 2 — WRITE: Once you have enough information, output a brief "
        "chat reply (≤2 sentences) and then the document wrapped in markers:\n\n"
        "    [DRAFT_START]\n"
        "    ---\n"
        "    title: ...\n"
        "    type: concept | source_summary | entity | rca | query_result\n"
        "    tags: [...]\n"
        "    related: [...]\n"
        "    ---\n"
        "    # Title\n"
        "    ...full markdown body...\n"
        "    [DRAFT_END]\n\n"
        "For revisions affecting one section only, emit just that section:\n\n"
        "    [SECTION_START:Exact Heading Text]\n"
        "    ## Exact Heading Text\n"
        "    ...replacement content...\n"
        "    [SECTION_END]\n\n"
        "Rules for markers (READ CAREFULLY — wrong marker choice causes errors):\n"
        "- Each `[DRAFT_START]...[DRAFT_END]` replaces the ENTIRE draft. Use this "
        "for the FIRST draft and for any change that alters structure.\n"
        "- Each `[SECTION_START:Heading]...[SECTION_END]` replaces ONLY that section.\n"
        "- **If the CURRENT DRAFT block below is `(empty — the user has not "
        "received a draft yet)`, you MUST use `[DRAFT_START]`. Do NOT emit "
        "`[SECTION_START:...]` when there is no existing draft — the patch will "
        "fail because there is no heading to find.**\n"
        "- The heading you put after `SECTION_START:` must EXACTLY match an "
        "existing heading in the CURRENT DRAFT (case sensitive, exclude the "
        "leading `#` characters and any trailing spaces). Look at the current "
        "draft block carefully before choosing the heading.\n"
        "- If the same heading appears more than once in the draft, do a full "
        "`[DRAFT_START]` rewrite instead — ambiguous section patches will be rejected.\n"
        "- Do NOT use markers if you are only asking a question.\n\n"
        "PHASE 3 — REVIEW: After writing, check the draft against the wiki "
        "knowledge below. Ask clarifying follow-up questions as needed.\n\n"
        "CRITICAL — readiness signal:\n"
        "- The 'Save & Ingest' button stays DISABLED until you emit the literal "
        "token `[DRAFT_READY]` on its own line. This is the only way the user "
        "can save the draft.\n"
        "- Emit `[DRAFT_READY]` ONLY when you are confident the draft is "
        "complete, accurate, and free of outstanding follow-ups. Tell the "
        "user it is ready and they can click 'Save & Ingest'.\n"
        "- DO NOT emit `[DRAFT_READY]` if you still have a clarifying question, "
        "if the user just asked for a change you haven't applied yet, or if "
        "the draft is still empty.\n"
        "- ANY new `[DRAFT_START]...[DRAFT_END]` or `[SECTION_START:...]...[SECTION_END]` "
        "you emit later automatically clears the readiness flag — you must "
        "re-emit `[DRAFT_READY]` in a subsequent turn once the revised draft "
        "is again complete.\n"
        "- Do NOT emit `[DRAFT_READY]` inside the body of a draft or section — "
        "it must appear in your chat reply, outside any marker block.\n\n"
        "Follow the wiki schema strictly:\n\n"
        f"{schema}\n\n"
        "Relevant existing wiki pages (for grounding and to avoid duplication):\n\n"
        f"{pages_text}"
        f"{draft_block}"
    )
