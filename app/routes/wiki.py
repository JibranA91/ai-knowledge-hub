from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from app.logger import get_logger
from app.services.permissions import require_permission
from app.services.wiki_db import (
    list_wiki_pages_paginated,
    list_wiki_paths,
    search_wiki as db_search_wiki,
    upsert_wiki_page,
    delete_wiki_page as db_delete_wiki_page,
    get_wiki_page_content,
    wiki_page_exists,
    get_wiki_links,
)
from app.services.wiki_state import tracked_action

log = get_logger(__name__)
router = APIRouter(tags=["wiki"])


async def _refresh_graph_after_write(path: str) -> None:
    """Keep the knowledge graph + wiki_links table in sync after a page write.

    upsert_wiki_page / delete_wiki_page only touch wiki_pages — they don't
    re-resolve links. Manual edits, AI edits, and deletes all funnel through the
    routes below, so refreshing here keeps every edit path graph-consistent
    (only the ingest pipeline did this before). ensure_loaded() first so the
    incremental update doesn't persist an empty graph over the real one. Best
    effort: the page write already succeeded, and a full rebuild can recover, so
    a graph hiccup must not fail the request.
    """
    try:
        from app.services.graph import get_graph
        graph = await get_graph().ensure_loaded()
        await graph.update_pages([path])
    except Exception as e:
        log.warning("graph refresh after write failed | path=%s | %s", path, e)


def _build_tree(paths: list[str]) -> list:
    """Convert a flat list of posix paths into a nested tree structure."""
    tree: dict = {}
    for path in sorted(paths):
        parts = path.split("/")
        node = tree
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = None  # leaf marker

    def _to_list(d: dict, prefix: str = "") -> list:
        result = []
        for name, children in sorted(d.items()):
            full_path = f"{prefix}{name}" if prefix else name
            if children is None:
                result.append({"type": "file", "name": name.rsplit(".", 1)[0], "path": full_path})
            else:
                sub = _to_list(children, f"{full_path}/")
                if sub:
                    result.append({"type": "dir", "name": name, "path": full_path, "children": sub})
        return result

    return _to_list(tree)


@router.get("", dependencies=[Depends(require_permission("can_view_wiki"))])
async def list_wiki(
    page: int = Query(default=0, ge=0, description="Page number (1-indexed). 0 = return full tree (legacy)."),
    limit: int = Query(default=50, ge=1, le=200, description="Items per page."),
    tag: Optional[str] = Query(default=None, description="Filter by tag."),
):
    """List wiki pages.

    - `page=0` (default): returns the legacy nested tree structure for the sidebar.
    - `page>=1`: returns a paginated flat list with metadata (title, tags, summary).
    """
    if page == 0:
        paths = await list_wiki_paths()
        return _build_tree(paths)
    return await list_wiki_pages_paginated(page=page, limit=limit, tag=tag)


@router.get("/search", dependencies=[Depends(require_permission("can_view_wiki"))])
async def search_wiki(q: str):
    return await db_search_wiki(q)


@router.get("/{path:path}", dependencies=[Depends(require_permission("can_view_wiki"))])
async def get_wiki_page(path: str):
    content = await get_wiki_page_content(path)
    if content is None:
        raise HTTPException(404, f"Page not found: {path}")
    return PlainTextResponse(content)


class PageUpdate(BaseModel):
    content: str


@router.put("/{path:path}", dependencies=[Depends(require_permission("can_edit_wiki"))])
@tracked_action("manual_edit", summary_fn=lambda path, update: f"edit {path}")
async def update_wiki_page(path: str, update: PageUpdate):
    await upsert_wiki_page(path, update.content)
    await _refresh_graph_after_write(path)
    return {"path": path, "updated": True}


@router.delete("/{path:path}", dependencies=[Depends(require_permission("can_delete_wiki_pages"))])
@tracked_action("manual_delete", summary_fn=lambda path: f"delete {path}")
async def delete_wiki_page_route(path: str):
    if not await wiki_page_exists(path):
        raise HTTPException(404, "Page not found")
    await db_delete_wiki_page(path)
    await _refresh_graph_after_write(path)  # drops the node + clears its links
    return {"deleted": path}


@router.get("/{path:path}/links")
async def get_page_links(path: str):
    """Return the outgoing and incoming wiki links for a page (from DB table)."""
    if not await wiki_page_exists(path):
        raise HTTPException(404, f"Page not found: {path}")
    return await get_wiki_links(path)
