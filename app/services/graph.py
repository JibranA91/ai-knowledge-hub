"""
WikiGraph — bidirectional link graph over wiki pages.

Parses [[wikilinks]], [text](path.md) markdown links, and YAML frontmatter
`related:` fields. Cached in the wiki_files DB table (key: wiki/.graph.json).
"""

import json
import posixpath
import re
from pathlib import Path  # used in _rebuild_stem_index for filename stems

from app.utils import parse_frontmatter, resolve_title

from app.logger import get_logger

log = get_logger(__name__)

_WIKILINK_RE    = re.compile(r'\[\[([^\]|#]+?)(?:[|#][^\]]*)?\]\]')
_MDLINK_RE      = re.compile(r'\[[^\]]*\]\(([^)#?\s]+\.md)\)')
_GRAPH_DB_KEY = "wiki/.graph.json"


def _norm(s: str) -> str:
    """Lowercase, collapse separators for fuzzy slug matching."""
    return re.sub(r'[-_\s]+', '-', s.lower().strip())


def _resolve(base_dir: str, target: str) -> str:
    joined = posixpath.join(base_dir, target) if base_dir else target
    return posixpath.normpath(joined)


class WikiGraph:
    def __init__(self):
        self._adj: dict[str, set[str]] = {}
        self._meta: dict[str, dict] = {}
        self._stem_index: dict[str, str] = {}
        self._loaded = False

    async def ensure_loaded(self) -> "WikiGraph":
        """Load graph from DB on first use. Idempotent."""
        if not self._loaded:
            log.debug("WikiGraph.ensure_loaded | loading from DB")
            await self._load()
            self._loaded = True
            log.debug("WikiGraph.ensure_loaded | loaded | pages=%d", len(self._meta))
        else:
            log.debug("WikiGraph.ensure_loaded | cache_hit | pages=%d", len(self._meta))
        return self

    async def _save(self) -> None:
        try:
            from app.services.wiki_db import set_wiki_file
            data = {
                'adj':  {k: sorted(v) for k, v in self._adj.items()},
                'meta': self._meta,
            }
            await set_wiki_file(_GRAPH_DB_KEY, json.dumps(data, indent=2))
        except Exception as e:
            log.warning('WikiGraph: save failed: %s', e)

    async def _load(self) -> None:
        try:
            from app.services.wiki_db import get_wiki_file
            raw = await get_wiki_file(_GRAPH_DB_KEY)
            if raw:
                data = json.loads(raw)
                self._adj  = {k: set(v) for k, v in data.get('adj', {}).items()}
                self._meta = data.get('meta', {})
                self._rebuild_stem_index()
                log.info('WikiGraph loaded | pages=%d edges=%d',
                         len(self._meta), sum(len(v) for v in self._adj.values()))
                return
        except Exception as e:
            log.warning('WikiGraph: load failed, starting empty: %s', e)
        self._adj  = {}
        self._meta = {}
        self._stem_index = {}

    def _rebuild_stem_index(self) -> None:
        self._stem_index = {_norm(Path(p).stem): p for p in self._meta}

    def _extract_meta(self, rel: str, content: str) -> dict:
        fm = parse_frontmatter(content)
        return {
            'title':    resolve_title(content, rel),
            'tags':     list(fm.get('tags', []) or []),
            'entities': list(fm.get('entities', []) or []),
            '_fm': fm,
        }

    def _extract_links(self, page_rel: str, content: str, fm: dict) -> set[str]:
        links: set[str] = set()
        base = posixpath.dirname(page_rel)

        for m in _MDLINK_RE.finditer(content):
            target = m.group(1).strip()
            if target.startswith(('http://', 'https://')):
                continue
            links.add(_resolve(base, target))

        for rel_path in (fm.get('related', []) or []):
            if isinstance(rel_path, str):
                links.add(_resolve(base, rel_path))

        for m in _WIKILINK_RE.finditer(content):
            slug   = _norm(m.group(1).strip())
            target = self._stem_index.get(slug)
            if target and target != page_rel:
                links.add(target)

        return {lnk for lnk in links if lnk in self._meta}

    async def rebuild(self) -> None:
        """Full graph rebuild from wiki_pages DB table. Two-pass to resolve wikilinks."""
        from app.services.wiki_db import list_wiki_pages_with_content, replace_wiki_links
        self._meta = {}
        self._adj  = {}
        raw: dict[str, tuple[str, dict]] = {}

        for path, content in await list_wiki_pages_with_content():
            if path in ('index.md', 'log.md') or path.startswith('.'):
                continue
            try:
                meta = self._extract_meta(path, content)
                fm   = meta.pop('_fm')
                self._meta[path] = meta
                raw[path] = (content, fm)
            except Exception as e:
                log.warning('WikiGraph.rebuild: skipping %s: %s', path, e)

        self._rebuild_stem_index()

        for path, (content, fm) in raw.items():
            self._adj[path] = self._extract_links(path, content, fm)

        await self._save()

        # Persist links to wiki_links DB table for SQL graph queries.
        for path, targets in self._adj.items():
            try:
                await replace_wiki_links(path, list(targets))
            except Exception as e:
                log.warning('WikiGraph.rebuild: wiki_links update failed for %s: %s', path, e)

        edge_count = sum(len(v) for v in self._adj.values())
        log.info('WikiGraph rebuilt | pages=%d edges=%d', len(self._meta), edge_count)

    async def update_pages(self, paths: list[str]) -> None:
        """Re-index specific pages after an ingest write. Two-pass for consistency."""
        from app.services.wiki_db import get_wiki_page_content, replace_wiki_links
        # Incremental update assumes the full graph is already in memory:
        # _extract_links only keeps targets present in self._meta, and _save()
        # persists the whole adjacency. If the per-org cache is cold (cross-replica
        # ingest failover, or a revert), skipping this would drop every pre-existing
        # edge. ensure_loaded() is idempotent and cheap on a warm cache.
        await self.ensure_loaded()
        raw: dict[str, tuple[str, dict]] = {}

        for rel in paths:
            content = await get_wiki_page_content(rel)
            if content is None:
                self._adj.pop(rel, None)
                self._meta.pop(rel, None)
                try:
                    await replace_wiki_links(rel, [])
                except Exception:
                    pass
                continue
            try:
                meta = self._extract_meta(rel, content)
                fm   = meta.pop('_fm')
                self._meta[rel] = meta
                raw[rel] = (content, fm)
            except Exception as e:
                log.warning('WikiGraph.update_pages: skipping %s: %s', rel, e)

        self._rebuild_stem_index()

        for rel, (content, fm) in raw.items():
            self._adj[rel] = self._extract_links(rel, content, fm)

        await self._save()

        # Persist updated links to the wiki_links DB table.
        for rel, (content, fm) in raw.items():
            try:
                await replace_wiki_links(rel, list(self._adj.get(rel, set())))
            except Exception as e:
                log.warning('WikiGraph.update_pages: wiki_links update failed for %s: %s', rel, e)

    def get_clusters(self) -> dict:
        """Compute Louvain community clusters and return cluster-level graph data.

        Each node carries a ``cluster`` int. The ``clusters`` list gives aggregate
        info (size, sample pages) so the frontend can render a high-level view
        before the user drills down.
        """
        try:
            import networkx as nx
            G = nx.Graph()
            for src, targets in self._adj.items():
                if src not in self._meta:
                    continue
                for tgt in targets:
                    if tgt in self._meta:
                        G.add_edge(src, tgt)

            # Add isolated nodes (no edges) as singletons.
            for path in self._meta:
                if path not in G:
                    G.add_node(path)

            if G.number_of_nodes() == 0:
                log.info("WikiGraph.get_clusters | empty graph")
                return {"clusters": [], "nodes": []}

            log.debug("WikiGraph.get_clusters | nodes=%d edges=%d", G.number_of_nodes(), G.number_of_edges())
            communities = list(nx.community.louvain_communities(G, seed=42))
            cluster_map: dict[str, int] = {}
            for i, community in enumerate(communities):
                for node in community:
                    cluster_map[node] = i

            nodes = [
                {
                    "id": path,
                    "title": meta.get("title", path),
                    "tags": meta.get("tags", []),
                    "cluster": cluster_map.get(path, -1),
                    "degree": len(self._adj.get(path, set())),
                }
                for path, meta in self._meta.items()
            ]

            in_deg: dict[str, int] = {}
            for targets in self._adj.values():
                for t in targets:
                    in_deg[t] = in_deg.get(t, 0) + 1

            clusters = []
            for i, community in enumerate(communities):
                sample = sorted(
                    community,
                    key=lambda p: in_deg.get(p, 0) + len(self._adj.get(p, set())),
                    reverse=True,
                )[:5]
                clusters.append({"id": i, "size": len(community), "hub_pages": sample})

            log.info("WikiGraph.get_clusters | clusters=%d nodes=%d", len(clusters), len(nodes))
            return {"clusters": clusters, "nodes": nodes}
        except Exception as e:
            log.warning("WikiGraph.get_clusters failed: %s", e)
            return {"clusters": [], "nodes": [{"id": p, "title": m.get("title", p),
                                               "tags": m.get("tags", []), "cluster": -1,
                                               "degree": len(self._adj.get(p, set()))}
                                              for p, m in self._meta.items()]}

    def neighbors(self, paths: list[str], hops: int = 1) -> list[str]:
        visited  = set(paths)
        frontier = set(paths)
        for _ in range(hops):
            nxt: set[str] = set()
            for p in frontier:
                for t in self._adj.get(p, set()):
                    if t not in visited:
                        nxt.add(t)
                for src, targets in self._adj.items():
                    if p in targets and src not in visited:
                        nxt.add(src)
            visited  |= nxt
            frontier  = nxt
        return [p for p in visited if p not in set(paths)]

    def as_dict(self) -> dict:
        in_deg: dict[str, int] = {}
        for targets in self._adj.values():
            for t in targets:
                in_deg[t] = in_deg.get(t, 0) + 1

        nodes = []
        for path, meta in self._meta.items():
            out = len([t for t in self._adj.get(path, set()) if t in self._meta])
            nodes.append({
                'id':       path,
                'title':    meta.get('title', path),
                'tags':     meta.get('tags', []),
                'entities': meta.get('entities', []),
                'degree':   in_deg.get(path, 0) + out,
            })

        seen: set[tuple] = set()
        edges = []
        for src, targets in self._adj.items():
            if src not in self._meta:
                continue
            for tgt in targets:
                if tgt in self._meta:
                    key = (min(src, tgt), max(src, tgt))
                    if key not in seen:
                        seen.add(key)
                        edges.append({'source': src, 'target': tgt})

        return {'nodes': nodes, 'edges': edges}

    def generate_html(self) -> str:
        theme_assets = (
            '<link rel="stylesheet" href="/static/prism.css">'
            '<script src="/static/graph-theme.js"></script>'
        )
        try:
            from pyvis.network import Network  # type: ignore
        except Exception as _import_err:
            return (
                theme_assets + '<body style="background:var(--bg);color:var(--danger);font-family:sans-serif;'
                'display:flex;align-items:center;justify-content:center;height:100vh;margin:0;padding:24px">'
                f'Import error: {_import_err}</body>'
            )

        data = self.as_dict()
        if not data['nodes']:
            return (
                '<html><head>' + theme_assets + '</head><body style="background:var(--bg);color:var(--text-muted);font-family:system-ui,sans-serif;'
                'display:flex;align-items:center;justify-content:center;height:100vh;margin:0">'
                'No wiki pages indexed yet.</body></html>'
            )

        max_deg = max((n['degree'] for n in data['nodes']), default=0) or 1

        _DIR = {
            'sources':  {'bg': '#0C2340', 'border': '#3D9BE9', 'highlight_bg': '#1A4A7A'},
            'concepts': {'bg': '#0A2818', 'border': '#2ECC71', 'highlight_bg': '#1A5030'},
            'queries':  {'bg': '#210F38', 'border': '#9B59B6', 'highlight_bg': '#3D1F5E'},
        }
        _DEFAULT = {'bg': '#111E2E', 'border': '#4A6FA5', 'highlight_bg': '#1E3A5F'}

        net = Network(
            height='100%', width='100%',
            bgcolor='#060E1A', font_color='#C2D4E8',
            cdn_resources='in_line',
            directed=False,
        )

        for node in data['nodes']:
            theme = _DIR.get(node['id'].split('/')[0], _DEFAULT)
            size  = 16 + round((node['degree'] / max_deg) * 36)
            label = node['title'] if len(node['title']) <= 26 else node['title'][:24] + '…'

            tooltip_parts = [node['title'], node['id']]
            if node['tags']:
                tooltip_parts.append('Tags: ' + ', '.join(node['tags'][:5]))
            tooltip_parts.append(
                f"{node['degree']} connection{'s' if node['degree'] != 1 else ''}"
                ' · click to open'
            )

            net.add_node(
                node['id'],
                label=label,
                title='\n'.join(tooltip_parts),
                color={
                    'background': theme['bg'],
                    'border':     theme['border'],
                    'highlight':  {'background': theme['highlight_bg'], 'border': '#FFFFFF'},
                    'hover':      {'background': theme['highlight_bg'], 'border': theme['border']},
                },
                size=size,
                font={'color': '#C2D4E8', 'size': 11, 'face': 'Inter, sans-serif',
                      'strokeWidth': 3, 'strokeColor': '#060E1A'},
                borderWidth=1.8,
                borderWidthSelected=3,
                shadow={'enabled': True, 'color': 'rgba(0,0,0,0.5)', 'size': 8, 'x': 0, 'y': 3},
            )

        for edge in data['edges']:
            src_dir = edge['source'].split('/')[0]
            accent  = _DIR.get(src_dir, _DEFAULT)['border']
            net.add_edge(
                edge['source'], edge['target'],
                color={'color': '#1A2F48', 'highlight': accent, 'hover': accent},
                width=1.5,
                smooth={'type': 'dynamic'},
            )

        net.set_options("""{
  "nodes": { "scaling": {"min": 16, "max": 52} },
  "edges": { "selectionWidth": 2.5, "hoverWidth": 2 },
  "physics": {
    "enabled": true,
    "solver": "forceAtlas2Based",
    "forceAtlas2Based": {
      "gravitationalConstant": -55,
      "centralGravity": 0.008,
      "springLength": 120,
      "springConstant": 0.09,
      "damping": 0.5,
      "avoidOverlap": 0.6
    },
    "stabilization": {"enabled": true, "iterations": 250, "fit": true},
    "maxVelocity": 60,
    "minVelocity": 0.5
  },
  "interaction": {
    "hover": true,
    "tooltipDelay": 120,
    "hideEdgesOnDrag": false,
    "multiselect": false,
    "navigationButtons": false,
    "keyboard": {"enabled": false}
  }
}""")

        html = net.generate_html(notebook=False).replace('<head>', '<head>' + theme_assets)

        inject = """
<style>
  html, body {
    margin: 0 !important; padding: 0 !important;
    width: 100vw !important; height: 100vh !important;
    overflow: hidden !important; background: var(--bg) !important;
  }
  #mynetwork {
    position: fixed !important; top: 0 !important; left: 0 !important;
    width: 100vw !important; height: 100vh !important;
    border: none !important; outline: none !important;
    background-color: var(--bg) !important;
    margin: 0 !important; padding: 0 !important; float: none !important;
  }
  div.vis-tooltip {
    background: var(--surface) !important; border: 1px solid var(--border) !important;
    color: var(--text) !important; border-radius: 10px !important;
    padding: 10px 14px !important;
    font-family: Inter, 'Segoe UI', sans-serif !important;
    font-size: 12px !important; line-height: 1.7 !important;
    max-width: 260px !important; box-shadow: 0 8px 24px rgba(0,0,0,0.7) !important;
    white-space: pre-line !important; pointer-events: none !important;
  }
</style>
<script>
  (function poll() {
    if (typeof network !== 'undefined') {
      if (window.applyPrismGraphTheme) window.applyPrismGraphTheme();
      function fitWhenReady() {
        var mn = document.getElementById('mynetwork');
        if (mn && mn.offsetWidth > 100) {
          network.redraw();
          network.fit({ animation: { duration: 600, easingFunction: 'easeInOutQuad' } });
        } else { setTimeout(fitWhenReady, 60); }
      }
      network.on('stabilizationIterationsDone', function() {
        network.setOptions({ physics: { enabled: false } });
        fitWhenReady();
      });
      window.addEventListener('resize', function() { network.redraw(); network.fit(); });
      network.on('click', function(params) {
        if (params.nodes && params.nodes.length)
          window.parent.postMessage({type: 'graphNavigate', path: params.nodes[0]}, '*');
      });
      network.on('hoverNode', function() {
        document.getElementById('mynetwork').style.cursor = 'pointer';
      });
      network.on('blurNode', function() {
        document.getElementById('mynetwork').style.cursor = 'default';
      });
    } else { setTimeout(poll, 40); }
  })();
</script>
"""
        return html.replace('</body>', inject + '\n</body>')


# ── Module-level singleton ─────────────────────────────────────────────────

# One in-memory graph per org. The graph holds org-scoped pages/links, so a
# single process-global instance would serve whichever org last loaded/rebuilt
# it to every other org (cross-tenant leak). Key by the active org instead —
# mirrors the per-org pattern used by recalibrate_job.
_instances: dict[str, WikiGraph] = {}


def _graph_key() -> str:
    """Active org id, or "" for no-context (startup/background) operations."""
    from app.context import current_user
    ctx = current_user.get(None)
    return ctx.org_id if (ctx and ctx.org_id) else ""


def get_graph() -> WikiGraph:
    key = _graph_key()
    g = _instances.get(key)
    if g is None:
        g = WikiGraph()
        _instances[key] = g
    return g


def reset_graph_cache() -> None:
    """Drop all cached per-org graph instances. Used by tests."""
    _instances.clear()
