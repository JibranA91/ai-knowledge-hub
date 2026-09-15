'use strict';

// ── State ──────────────────────────────────────────────────────────────────
const state = {
  activePage: null,
  activeSource: null,
  queryCollapsed: false,
  pollingJobs: {},
  chatSessionId: localStorage.getItem('chat_session_id') || null,
  pageHistory: [],   // stack of previously visited page paths
  authToken: localStorage.getItem('auth_token') || null,
  refreshToken: localStorage.getItem('refresh_token') || null,
  streamEnabled: true,     // overwritten by /api/auth/config on boot
  maxUploadMB: 50,         // overwritten by /api/auth/config on boot
  embeddingEnabled: false, // overwritten by /api/auth/config on boot
  userRole: localStorage.getItem('user_role') || null,           // role IN THE ACTIVE ORG: "admin" | "supervisor" | "member"
  // The org this session is acting on, sent as X-Org-Context on every request.
  // Admins pick any org; members pick one of the orgs they belong to.
  activeOrgId: localStorage.getItem('active_org_id') || localStorage.getItem('admin_org_id') || null,
  memberships: JSON.parse(localStorage.getItem('memberships') || '[]'), // [{org_id, org_name, role}]
  userEmail: localStorage.getItem('user_email') || null,
  userOrgName: localStorage.getItem('user_org_name') || null,
  perms: JSON.parse(localStorage.getItem('user_perms') || '{}'), // effective permission flags
  // Writer mode
  writerMode: false,
  writerSessionId: null,
  writerDraft: '',
  writerDraftReady: false, // agent has emitted [DRAFT_READY] for the current draft
  writerFilename: '',
  // filename → writer session_id, for drafts to delete once ingest succeeds
  writerIngestSessions: {},
};

// ── In-flight stream tracking ──────────────────────────────────────────────
// Active AbortControllers for SSE streams (chat, writer chat). Aborting these
// on signout closes the underlying fetch, which propagates a CancelledError
// to the server-side generator and ends the Bedrock streaming call.
const _activeStreamControllers = new Set();
function abortAllActiveStreams() {
  for (const c of _activeStreamControllers) { try { c.abort(); } catch {} }
  _activeStreamControllers.clear();
}

// ── API helpers ────────────────────────────────────────────────────────────
let _isRefreshing = false;

async function tryRefreshToken() {
  if (!state.refreshToken || _isRefreshing) return false;
  _isRefreshing = true;
  try {
    const res = await fetch('/api/auth/refresh', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ refresh_token: state.refreshToken }),
    });
    if (!res.ok) return false;
    const data = await res.json();
    state.authToken = data.access_token;
    localStorage.setItem('auth_token', data.access_token);
    return true;
  } catch {
    return false;
  } finally {
    _isRefreshing = false;
  }
}

async function api(method, path, body, _skipRefresh = false) {
  const opts = { method, headers: {} };
  if (state.authToken) opts.headers['Authorization'] = `Bearer ${state.authToken}`;
  if (state.activeOrgId) opts.headers['X-Org-Context'] = state.activeOrgId;
  if (body) { opts.headers['Content-Type'] = 'application/json'; opts.body = JSON.stringify(body); }
  const res = await fetch(path, opts);
  if (res.status === 401) {
    if (!_skipRefresh) {
      const refreshed = await tryRefreshToken();
      if (refreshed) return api(method, path, body, true);
    }
    showLoginScreen();
    throw new Error('Session expired. Please sign in again.');
  }
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    const msg = err.detail || res.statusText;
    if (res.status === 429) {
      // Title comes from the backend message itself, so upload/chat/query
      // quotas don't get mislabeled as "Token limit reached".
      notifAddSystem({ type: 'system', title: 'Limit reached', body: msg, status: 'error' });
      throw Object.assign(new Error(msg), { isQuota: true });
    }
    throw new Error(msg);
  }
  const ct = res.headers.get('content-type') || '';
  return ct.includes('application/json') ? res.json() : res.text();
}

// fetch() for raw streaming endpoints (SSE) that transparently refreshes the
// access token once on a 401 and retries, so a token expiring mid-session
// doesn't bounce the user to login when a valid refresh token exists. Mutates
// opts.headers.Authorization on a successful refresh. Returns the Response
// (still 401 if the refresh itself failed — caller decides what to do then).
async function streamFetch(url, opts) {
  let res = await fetch(url, opts);
  if (res.status === 401 && await tryRefreshToken()) {
    opts.headers = { ...opts.headers, Authorization: `Bearer ${state.authToken}` };
    res = await fetch(url, opts);
  }
  return res;
}

// ── Markdown ───────────────────────────────────────────────────────────────
function renderMd(text) {
  // Strip YAML frontmatter before rendering
  const stripped = text.replace(/^---[ \t]*\r?\n[\s\S]*?\r?\n---[ \t]*(\r?\n|$)/, '');
  // Convert [[Page Name]] and [[path/to/page.md|Display Text]] wiki-links to clickable spans
  const processed = stripped.replace(/\[\[([^\]]+)\]\]/g, (_, inner) => {
    const pipeIdx = inner.indexOf('|');
    let query, display;
    if (pipeIdx !== -1) {
      // [[path/to/page.md|Display Text]] — use display text, search by filename stem
      const path = inner.slice(0, pipeIdx).trim();
      display = inner.slice(pipeIdx + 1).trim();
      // Use the stem (filename without extension) as the search query
      const stem = path.split('/').pop().replace(/\.md$/i, '');
      query = stem || display;
    } else {
      query = inner.trim();
      display = inner.trim();
    }
    return `<span class="wiki-link" data-query="${escHtml(query)}">${escHtml(display)}</span>`;
  });
  // Wiki and AI-generated content is untrusted, so the rendered HTML must be
  // sanitized (marked does NOT sanitize) — otherwise raw <script>/onerror/
  // javascript: payloads would execute on view. Sanitize marked's output with
  // DOMPurify (keeps our wiki-link spans: class + data-* are allowed by default).
  // If either library failed to load (offline), fall back to escaped plaintext
  // rather than emitting unsanitized HTML.
  if (typeof marked === 'undefined' || typeof DOMPurify === 'undefined') {
    return '<pre>' + escHtml(processed) + '</pre>';
  }
  const html = typeof renderMarkdownWithMath === 'function'
    ? renderMarkdownWithMath(processed) : marked.parse(processed);
  return DOMPurify.sanitize(html);
}

function escHtml(text) {
  // Escape quotes too (not just & < >): escHtml is used inside double-quoted
  // attributes such as data-query="${escHtml(...)}" in renderMd, where an
  // unescaped " would break out of the attribute. Matches admin.js's escHtml.
  return String(text)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

// ── Toast system ───────────────────────────────────────────────────────────
function showToast(message, type = 'info') {
  const container = document.getElementById('toast-container');
  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;

  const icons = {
    success: `<svg viewBox="0 0 16 16" fill="none"><path d="M3 8l3.5 3.5L13 5" stroke="#0BA860" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>`,
    error:   `<svg viewBox="0 0 16 16" fill="none"><path d="M4 4l8 8M12 4l-8 8" stroke="#E03131" stroke-width="1.8" stroke-linecap="round"/></svg>`,
    info:    `<svg viewBox="0 0 16 16" fill="none"><circle cx="8" cy="8" r="6" stroke="#6AADFF" stroke-width="1.5"/><path d="M8 7v4M8 5v.5" stroke="#6AADFF" stroke-width="1.5" stroke-linecap="round"/></svg>`,
  };

  toast.innerHTML = `
    ${icons[type] || icons.info}
    <span class="toast-msg">${escHtml(message)}</span>
    <button class="toast-close" aria-label="Dismiss">&times;</button>
  `;
  toast.querySelector('.toast-close').addEventListener('click', e => { e.stopPropagation(); dismissToast(toast); });
  container.appendChild(toast);
  setTimeout(() => dismissToast(toast), 5000);
}

function dismissToast(toast) {
  if (!toast.isConnected) return;
  toast.style.animation = 'toast-out 0.25s var(--ease) forwards';
  setTimeout(() => toast.remove(), 280);
}

// ── SVG icon snippets ──────────────────────────────────────────────────────
const FILE_ICON = `<svg viewBox="0 0 16 16" fill="none"><path d="M3 2h7l3 3v9H3V2z" stroke="currentColor" stroke-width="1.3"/><path d="M10 2v3h3" stroke="currentColor" stroke-width="1.3"/></svg>`;

function sourceFileIcon(filename) {
  const ext = (filename.split('.').pop() || '').toLowerCase();
  const extMap = { pdf: 'ext-pdf', docx: 'ext-docx', doc: 'ext-doc', md: 'ext-md', txt: 'ext-txt' };
  const cls = extMap[ext] || '';
  return `<div class="src-icon ${cls}">${ext.slice(0, 3)}</div>`;
}

// ── Sidebar: Wiki tree ─────────────────────────────────────────────────────
// The welcome screen's "Import Wiki" affordance only makes sense for
// admins/supervisors AND only when the org's wiki is empty — import refuses to
// merge into a populated wiki. Driven from loadWikiTree (the one place that
// knows the page count), so it stays correct on boot, org switch, and after any
// wiki change (import, first ingest).
function setWelcomeImportVisible(isEmpty) {
  const btn = document.getElementById('btn-welcome-import');
  if (!btn) return;
  const canImport = state.userRole === 'admin' || state.userRole === 'supervisor';
  btn.classList.toggle('hidden', !(canImport && isEmpty));
}

async function loadWikiTree() {
  const tree = document.getElementById('wiki-tree');
  tree.innerHTML = '<div class="loading-row"><span class="spinner"></span> Loading…</div>';
  try {
    const pages = await api('GET', '/api/wiki');
    tree.innerHTML = '';
    setWelcomeImportVisible(!pages.length);
    if (!pages.length) {
      tree.innerHTML = '<div class="loading-row" style="opacity:0.6">No wiki pages yet.</div>';
      return;
    }
    renderTree(pages, tree);
  } catch (e) {
    setWelcomeImportVisible(false);  // unknown state → don't offer a destructive import
    tree.innerHTML = `<div class="loading-row" style="color:#FF8080">${escHtml(e.message)}</div>`;
  }
}

function renderTree(nodes, container, depth = 0) {
  for (const node of nodes) {
    if (node.type === 'dir') {
      const label = document.createElement('div');
      label.className = 'tree-section';
      label.textContent = node.name;
      container.appendChild(label);
      renderTree(node.children, container, depth + 1);
    } else {
      const item = document.createElement('div');
      const depthClass = depth > 0 ? ` depth-${Math.min(depth, 2)}` : '';
      item.className = 'tree-item' + depthClass + (state.activePage === node.path ? ' active' : '');
      item.dataset.path = node.path;
      item.innerHTML = `${FILE_ICON}<span class="label">${escHtml(node.name)}</span>`;
      item.addEventListener('click', () => loadWikiPage(node.path));
      container.appendChild(item);
    }
  }
}

async function loadWikiPage(path, pushHistory = true) {
  if (pushHistory && state.activePage && state.activePage !== path) {
    state.pageHistory.push(state.activePage);
  }
  state.activePage = path;
  state.activeSource = null;
  setActiveTreeItem('wiki-tree', path);
  showPageView();

  const parts = path.split('/');
  const bc = document.getElementById('page-breadcrumb');
  bc.innerHTML = parts
    .map((p, i) => i === parts.length - 1
      ? `<strong>${escHtml(p)}</strong>`
      : `${escHtml(p)} <span style="opacity:0.4">/</span>`)
    .join(' ');

  // Back button
  const backBtn = document.getElementById('btn-page-back');
  if (backBtn) backBtn.style.display = state.pageHistory.length ? 'flex' : 'none';

  document.getElementById('page-actions').innerHTML = '';

  const meta = document.getElementById('page-meta');
  if (meta) { meta.classList.add('hidden'); meta.innerHTML = ''; }

  const content = document.getElementById('page-content');
  content.innerHTML = '<div class="content-loading"><span class="spinner spinner-dark"></span> Loading…</div>';

  try {
    const text = await api('GET', `/api/wiki/${path}`);
    state.currentPageRaw = text;             // kept for AI-edit diffing + heading parsing
    content.innerHTML = renderMd(text);
    renderPageActions(path);                 // adds "AI Edit" when the user may edit
    loadPageMeta(path);  // fire-and-forget provenance line
  } catch (e) {
    content.innerHTML = `<p style="color:var(--danger);padding:28px">${escHtml(e.message)}</p>`;
  }
}

// Whether the current user may edit wiki content (admins/supervisors always;
// members need the can_edit_wiki flag).
function canEditWiki() {
  return state.userRole === 'admin' || state.userRole === 'supervisor'
    || !!state.perms.can_edit_wiki;
}

// Populate the page toolbar's action area. Currently just the AI-edit affordance,
// shown only to users who can edit (so it can actually be applied).
function renderPageActions(path) {
  const actions = document.getElementById('page-actions');
  if (!actions) return;
  actions.innerHTML = '';
  if (!canEditWiki()) return;
  const btn = document.createElement('button');
  btn.className = 'btn btn-secondary btn-sm';
  btn.id = 'btn-ai-edit';
  btn.title = 'Rewrite this page or a section with AI';
  btn.innerHTML = '<img src="/static/img/ai-edit.png" class="ai-op-icon" alt="" aria-hidden="true"> AI Edit';
  btn.addEventListener('click', () => openAiEdit(path));
  actions.appendChild(btn);
}

// Render the "last edited by … · N contributors" line under the page toolbar.
// Best-effort: stays hidden when the page predates change-tracking or the call
// fails, so it never blocks reading the page.
async function loadPageMeta(path) {
  const meta = document.getElementById('page-meta');
  if (!meta) return;
  try {
    const a = await api('GET', `/api/activity/page?path=${encodeURIComponent(path)}`);
    if (state.activePage !== path) return;      // user navigated away mid-fetch
    if (!a || !a.change_count) return;          // nothing tracked — keep hidden
    const who = a.last_edited_by || 'system';
    const when = a.last_edited_at ? relTime(a.last_edited_at) : '';
    const nContrib = (a.contributors || []).length;
    const contribTitle = (a.contributors || [])
      .map(c => `${c.email} (${c.count})`).join(', ');
    const contribChip = nContrib > 1
      ? `<span class="page-meta-contrib" title="${escHtml(contribTitle)}">`
        + `<svg viewBox="0 0 16 16" width="12" height="12" fill="none" stroke="currentColor" stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round">`
        + `<circle cx="6" cy="5.4" r="2.2"/><path d="M2.4 12.6c0-2 1.6-3.3 3.6-3.3s3.6 1.3 3.6 3.3"/>`
        + `<path d="M10.4 3.9a2.1 2.1 0 0 1 0 3.9M11.2 12.6c0-1.5-.6-2.6-1.5-3.2"/></svg>`
        + `${nContrib} contributors</span>`
      : '';
    const historyBtn = `<button class="page-meta-history" data-path="${escHtml(path)}" title="View this page's full change history">`
      + `<svg viewBox="0 0 16 16" width="13" height="13" fill="none" stroke="currentColor" stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round">`
      + `<circle cx="8" cy="8" r="6"/><path d="M8 4.7V8l2.3 1.4"/></svg>`
      + `History</button>`;
    meta.innerHTML =
      avatarBadge(who)
      + `<span class="page-meta-text">Last edited by <strong>${escHtml(who)}</strong>`
      + (when ? `<span class="page-meta-sep">·</span>${escHtml(when)}` : '')
      + `</span>`
      + `<span class="page-meta-actions">${contribChip}${historyBtn}</span>`;
    meta.classList.remove('hidden');
  } catch { /* non-fatal — leave hidden */ }
}

// ── Activity: relative time + recent changes feed ───────────────────────────

// Small stroke icons keyed by name, drawn inside a 16-viewbox <svg>.
const ACTIVITY_ICONS = {
  pencil:  'M2.5 13h11M3.5 10.3l6.2-6.2 2 2-6.2 6.2H3.5v-2z',
  trash:   'M3 4.5h10M6.5 4.5V3h3v1.5M4.5 4.5l.6 8h5.8l.6-8',
  upload:  'M8 10.5V3.5M5 6l3-3 3 3M3.5 12.5h9',
  refresh: 'M12.6 7a4.6 4.6 0 1 0-1.1 3.3M12.6 3.6V7H9.2',
  undo:    'M5.7 6.5H2.6V3.4M2.9 6.7A4.7 4.7 0 1 1 4 11.4',
  bookmark:'M4 3.2h8v9.6l-4-2.8-4 2.8V3.2z',
  schema:  'M3 3.2h10v9.6H3zM3 6.6h10M6.6 6.6v6.2',
};

// action_type → { label, icon, tone (solid colour), bg (tint) } for the feed.
const ACTIVITY_TYPE_META = {
  manual_edit:          { label: 'edited',             icon: 'pencil',   tone: 'var(--info)',    bg: 'var(--info-bg)' },
  manual_delete:        { label: 'deleted',            icon: 'trash',    tone: 'var(--danger)',  bg: 'var(--danger-bg)' },
  upload_ingest:        { label: 'ingested',           icon: 'upload',   tone: 'var(--success)', bg: 'var(--success-bg)' },
  writer_ingest:        { label: 'wrote',              icon: 'pencil',   tone: 'var(--primary)', bg: 'var(--primary-muted)' },
  document_upload:      { label: 'uploaded',           icon: 'upload',   tone: 'var(--success)', bg: 'var(--success-bg)' },
  document_delete:      { label: 'removed',            icon: 'trash',    tone: 'var(--danger)',  bg: 'var(--danger-bg)' },
  saved_query:          { label: 'saved a Q&A',        icon: 'bookmark', tone: 'var(--info)',    bg: 'var(--info-bg)' },
  recalibrate:          { label: 'recalibrated',       icon: 'refresh',  tone: '#9B6DFF',        bg: 'rgba(155,109,255,0.15)' },
  targeted_recalibrate: { label: 'recalibrated',       icon: 'refresh',  tone: '#9B6DFF',        bg: 'rgba(155,109,255,0.15)' },
  schema_update:        { label: 'updated the schema', icon: 'schema',   tone: 'var(--warning)', bg: 'var(--warning-bg)' },
  revert:               { label: 'reverted',           icon: 'undo',     tone: 'var(--warning)', bg: 'var(--warning-bg)' },
};
const ACTIVITY_FALLBACK = { label: 'changed', icon: 'pencil', tone: 'var(--text-muted)', bg: 'var(--hover)' };

function activityIcon(name) {
  const d = ACTIVITY_ICONS[name] || ACTIVITY_ICONS.pencil;
  return `<svg viewBox="0 0 16 16" width="15" height="15" fill="none" stroke="currentColor" `
    + `stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"><path d="${d}"/></svg>`;
}

// Up to two initials from an email/name, e.g. "jane.doe@x" → "JD", "system" → "SY".
function userInitials(who) {
  const local = String(who || 'system').trim().split('@')[0];
  const parts = local.split(/[.\-_+\s]+/).filter(Boolean);
  if (parts.length >= 2) return (parts[0][0] + parts[1][0]).toUpperCase();
  return (local.slice(0, 2) || 'SY').toUpperCase();
}

// Deterministic hue [0,360) so each user gets a stable avatar colour.
function userHue(who) {
  const s = String(who || 'system');
  let h = 0;
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) % 360;
  return h;
}

function avatarBadge(who) {
  return `<span class="av" style="--av-hue:${userHue(who)}">${escHtml(userInitials(who))}</span>`;
}

// Compact relative time, e.g. "just now", "5m ago", "3h ago", "2d ago",
// falling back to a locale date for anything older than a week.
function relTime(iso) {
  const then = new Date(iso).getTime();
  if (isNaN(then)) return '';
  const secs = Math.round((Date.now() - then) / 1000);
  if (secs < 45) return 'just now';
  const mins = Math.round(secs / 60);
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.round(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.round(hrs / 24);
  if (days < 7) return `${days}d ago`;
  return new Date(iso).toLocaleDateString();
}

async function openRecentChanges() {
  openModal('modal-activity');
  const list = document.getElementById('activity-list');
  list.innerHTML = '<div class="content-loading"><span class="spinner spinner-dark"></span> Loading…</div>';
  try {
    const data = await api('GET', '/api/activity/recent?limit=40');
    const changes = data.changes || [];
    if (!changes.length) {
      list.innerHTML = '<p style="color:var(--text-muted);padding:20px;text-align:center">No wiki changes recorded yet.</p>';
      return;
    }
    list.innerHTML = '<div class="activity-feed">' + changes.map(c => {
      const m = ACTIVITY_TYPE_META[c.action_type] || ACTIVITY_FALLBACK;
      const who = escHtml(c.user_email || 'system');
      const when = c.when ? escHtml(relTime(c.when)) : '';
      const pages = (c.pages || []).map(p =>
        `<a href="#" class="activity-page-link" data-path="${escHtml(p)}" title="${escHtml(p)}">${escHtml(p)}</a>`
      ).join('');
      const more = c.page_count > (c.pages || []).length
        ? `<span class="activity-more">+${c.page_count - c.pages.length} more</span>` : '';
      const summary = c.summary ? `<div class="activity-summary">${escHtml(c.summary)}</div>` : '';
      const pageRow = (pages || more) ? `<div class="activity-pages">${pages}${more}</div>` : '';
      return `<div class="activity-row" style="--tone:${m.tone};--tone-bg:${m.bg}">
        <span class="activity-icon">${activityIcon(m.icon)}</span>
        <div class="activity-body">
          <div class="activity-head"><strong>${who}</strong> ${escHtml(m.label)}<span class="activity-dot">·</span><span class="activity-when">${when}</span></div>
          ${summary}
          ${pageRow}
        </div>
      </div>`;
    }).join('') + '</div>';
  } catch (e) {
    list.innerHTML = `<p style="color:var(--danger);padding:20px">Failed to load: ${escHtml(e.message)}</p>`;
  }
}

// Clicking an affected-page link opens that page and closes the modal.
document.addEventListener('click', e => {
  const link = e.target.closest('.activity-page-link');
  if (!link) return;
  e.preventDefault();
  closeModal('modal-activity');
  loadWikiPage(link.dataset.path);
});

// ── Per-page change history ─────────────────────────────────────────────────

// Per-revision verb: op gives the precise event ("created"/"deleted"),
// otherwise fall back to the action-type label ("edited", "recalibrated", …).
function pageHistoryVerb(change, meta) {
  if (change.op === 'create') return 'created';
  if (change.op === 'delete') return 'deleted';
  return meta.label;
}

// Line-level diff (LCS) of two texts → [{type:'context'|'added'|'removed', text}].
function computeLineDiff(oldText, newText) {
  const a = (oldText || '').split('\n');
  const b = (newText || '').split('\n');
  const m = a.length, n = b.length;
  const lcs = Array.from({ length: m + 1 }, () => new Array(n + 1).fill(0));
  for (let i = m - 1; i >= 0; i--)
    for (let j = n - 1; j >= 0; j--)
      lcs[i][j] = a[i] === b[j] ? 1 + lcs[i + 1][j + 1] : Math.max(lcs[i + 1][j], lcs[i][j + 1]);
  const out = [];
  let i = 0, j = 0;
  while (i < m || j < n) {
    if (i < m && j < n && a[i] === b[j]) { out.push({ type: 'context', text: a[i] }); i++; j++; }
    else if (j < n && (i >= m || lcs[i][j + 1] >= lcs[i + 1][j])) { out.push({ type: 'added', text: b[j] }); j++; }
    else { out.push({ type: 'removed', text: a[i] }); i++; }
  }
  return out;
}

// Render a diff to HTML, collapsing unchanged runs to "N lines hidden" hunks.
function renderLineDiffHtml(before, after) {
  const diff = computeLineDiff(before, after);
  const changed = new Set();
  diff.forEach((l, i) => { if (l.type !== 'context') changed.add(i); });
  if (!changed.size) return '<div class="diff-line context">  (no content changes)</div>';
  const PAD = 3, visible = new Set();
  changed.forEach(i => { for (let d = -PAD; d <= PAD; d++) { const j = i + d; if (j >= 0 && j < diff.length) visible.add(j); } });
  let html = '', last = -1;
  diff.forEach((line, i) => {
    if (!visible.has(i)) return;
    if (last !== -1 && i > last + 1)
      html += `<div class="diff-line hunk">@@ … ${i - last - 1} lines hidden …</div>`;
    const prefix = line.type === 'added' ? '+ ' : line.type === 'removed' ? '- ' : '  ';
    html += `<div class="diff-line ${line.type}">${escHtml(prefix + line.text)}</div>`;
    last = i;
  });
  return html;
}

async function openPageHistory(path) {
  openModal('modal-page-history');
  document.getElementById('page-history-path').textContent = path;
  const list = document.getElementById('page-history-list');
  list.innerHTML = '<div class="content-loading"><span class="spinner spinner-dark"></span> Loading…</div>';
  try {
    const data = await api('GET', `/api/activity/page/history?path=${encodeURIComponent(path)}`);
    const changes = data.changes || [];
    if (!changes.length) {
      list.innerHTML = '<p style="color:var(--text-muted);padding:20px;text-align:center">No tracked changes for this page yet.</p>';
      return;
    }
    list.innerHTML = '<div class="activity-feed">' + changes.map(c => {
      const m = ACTIVITY_TYPE_META[c.action_type] || ACTIVITY_FALLBACK;
      const who = escHtml(c.user_email || 'system');
      const when = c.when ? escHtml(relTime(c.when)) : '';
      const diffBtn = c.id != null
        ? `<button class="rev-diff-toggle" data-rev-id="${escHtml(String(c.id))}">View diff</button>`
        : '';
      return `<div class="activity-row" style="--tone:${m.tone};--tone-bg:${m.bg}">
        <span class="activity-icon">${activityIcon(m.icon)}</span>
        <div class="activity-body">
          <div class="activity-head activity-head--diff">
            <span><strong>${who}</strong> ${escHtml(pageHistoryVerb(c, m))}<span class="activity-dot">·</span><span class="activity-when">${when}</span></span>
            ${diffBtn}
          </div>
          <div class="rev-diff hidden" data-loaded="0"></div>
        </div>
      </div>`;
    }).join('') + '</div>';
  } catch (e) {
    list.innerHTML = `<p style="color:var(--danger);padding:20px">Failed to load history: ${escHtml(e.message)}</p>`;
  }
}

// The byline's "History" button opens the per-page timeline.
document.addEventListener('click', e => {
  const btn = e.target.closest('.page-meta-history');
  if (!btn) return;
  e.preventDefault();
  openPageHistory(btn.dataset.path);
});

// "View diff" toggles an inline before/after diff for a revision (fetched once).
document.addEventListener('click', async e => {
  const btn = e.target.closest('.rev-diff-toggle');
  if (!btn) return;
  const box = btn.closest('.activity-body').querySelector('.rev-diff');
  if (box.dataset.loaded === '1') {
    btn.textContent = box.classList.toggle('hidden') ? 'View diff' : 'Hide diff';
    return;
  }
  btn.disabled = true;
  btn.textContent = 'Loading…';
  try {
    const rev = await api('GET', `/api/activity/revision/${encodeURIComponent(btn.dataset.revId)}`);
    box.innerHTML = renderLineDiffHtml(rev.content_before, rev.content_after);
  } catch (err) {
    box.innerHTML = `<div class="diff-line removed">Failed to load diff: ${escHtml(err.message)}</div>`;
  }
  box.dataset.loaded = '1';
  box.classList.remove('hidden');
  btn.disabled = false;
  btn.textContent = 'Hide diff';
});

// ── Inline AI editing ───────────────────────────────────────────────────────
// "AI Edit" on a page streams an AI-proposed rewrite (whole page or one section)
// from /api/ops/wiki/edit/stream, shows a live preview + diff, and applies the
// accepted result via the normal PUT /api/wiki/{path} (tracked + revertible).

let _aiEdit = null;   // { path, proposed }

function _parseHeadings(md) {
  const out = [];
  const re = /^(#{1,6})\s+(.+?)\s*$/gm;
  let m;
  while ((m = re.exec(md || ''))) out.push({ level: m[1].length, title: m[2].trim() });
  return out;
}

function _flattenWikiPages(nodes, acc = []) {
  for (const n of nodes || []) {
    if (n.type === 'dir') _flattenWikiPages(n.children, acc);
    else if (n.path) acc.push(n.path);
  }
  return acc;
}

function _aiEditUpdateInstructionLabel() {
  const action = document.getElementById('ai-edit-action').value;
  document.getElementById('ai-edit-instruction-label').textContent =
    action === 'custom' ? 'Instruction (required)' : 'Additional instructions (optional)';
}

// The model sometimes wraps its whole rewrite in a ```markdown fence. Strip a
// leading/trailing fence so the live preview renders markdown, not a raw code
// block. (The server already strips it from the applied content; the final
// preview is re-rendered from that clean full_content on the 'done' event.)
function _stripPreviewFence(t) {
  return (t || '')
    .replace(/^\s*```[a-zA-Z0-9]*[ \t]*\r?\n/, '')
    .replace(/\r?\n?```[ \t]*$/, '');
}

async function openAiEdit(path) {
  _aiEdit = { path, proposed: null };
  document.getElementById('ai-edit-path').textContent = path;

  document.getElementById('ai-edit-scope').value = 'page';
  document.getElementById('ai-edit-action').value = 'improve';
  document.getElementById('ai-edit-instruction').value = '';
  document.getElementById('ai-edit-section-wrap').classList.add('hidden');
  document.getElementById('ai-edit-reconcile-wrap').classList.add('hidden');
  document.getElementById('ai-edit-result').classList.add('hidden');
  _aiEditUpdateInstructionLabel();

  // Sections come from the live page's headings (skip the H1 title).
  const sectionSel = document.getElementById('ai-edit-section');
  const headings = _parseHeadings(state.currentPageRaw).filter(h => h.level >= 2);
  sectionSel.innerHTML = headings.length
    ? headings.map(h => `<option value="${escHtml(h.title)}">${escHtml(h.title)}</option>`).join('')
    : '<option value="">(no sections)</option>';

  const reconcileSel = document.getElementById('ai-edit-reconcile');
  reconcileSel.innerHTML = '<option value="">Loading…</option>';
  openModal('modal-ai-edit');
  try {
    const pages = _flattenWikiPages(await api('GET', '/api/wiki')).filter(p => p !== path);
    reconcileSel.innerHTML = pages.length
      ? pages.map(p => `<option value="${escHtml(p)}">${escHtml(p)}</option>`).join('')
      : '<option value="">(no other pages)</option>';
  } catch {
    reconcileSel.innerHTML = '<option value="">(failed to load pages)</option>';
  }
}

function initAiEdit() {
  const scope = document.getElementById('ai-edit-scope');
  const action = document.getElementById('ai-edit-action');
  if (!scope || !action) return;
  scope.addEventListener('change', () => {
    document.getElementById('ai-edit-section-wrap').classList.toggle('hidden', scope.value !== 'section');
  });
  action.addEventListener('change', () => {
    document.getElementById('ai-edit-reconcile-wrap').classList.toggle('hidden', action.value !== 'reconcile');
    _aiEditUpdateInstructionLabel();
  });
  document.getElementById('btn-ai-edit-generate').addEventListener('click', runAiEdit);
  document.getElementById('btn-ai-edit-regen').addEventListener('click', runAiEdit);
  document.getElementById('btn-ai-edit-apply').addEventListener('click', applyAiEdit);
  document.getElementById('btn-ai-edit-discard').addEventListener('click', () => closeModal('modal-ai-edit'));
}

async function runAiEdit() {
  if (!_aiEdit) return;
  const scope = document.getElementById('ai-edit-scope').value;
  const actionVal = document.getElementById('ai-edit-action').value;
  const heading = document.getElementById('ai-edit-section').value;
  const instruction = document.getElementById('ai-edit-instruction').value.trim();
  const reconcileWith = document.getElementById('ai-edit-reconcile').value;

  if (scope === 'section' && !heading) { showToast('Pick a section to edit.', 'error'); return; }
  if (actionVal === 'custom' && !instruction) { showToast('Enter an instruction for a custom edit.', 'error'); return; }
  if (actionVal === 'reconcile' && !reconcileWith) { showToast('Pick a page to reconcile with.', 'error'); return; }

  const result   = document.getElementById('ai-edit-result');
  const statusEl  = document.getElementById('ai-edit-status');
  const previewEl = document.getElementById('ai-edit-preview');
  const diffEl    = document.getElementById('ai-edit-diff');
  const applyBtn  = document.getElementById('btn-ai-edit-apply');
  const regenBtn  = document.getElementById('btn-ai-edit-regen');
  const genBtn    = document.getElementById('btn-ai-edit-generate');

  result.classList.remove('hidden');
  statusEl.textContent = 'Generating…';
  previewEl.innerHTML = '<span class="spinner spinner-dark"></span>';
  diffEl.innerHTML = '';
  applyBtn.disabled = true; regenBtn.disabled = true; genBtn.disabled = true;
  _aiEdit.proposed = null;

  const controller = new AbortController();
  _activeStreamControllers.add(controller);
  _aiEdit.controller = controller;   // so Discard / closing the modal can abort it
  let acc = '', fullContent = null, errored = false;
  try {
    const opts = {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${state.authToken}`,
        ...(state.activeOrgId ? { 'X-Org-Context': state.activeOrgId } : {}),
      },
      body: JSON.stringify({
        path: _aiEdit.path, scope, action: actionVal,
        heading: scope === 'section' ? heading : null,
        instruction,
        reconcile_with: actionVal === 'reconcile' ? reconcileWith : null,
      }),
      signal: controller.signal,
    };
    const res = await streamFetch('/api/ops/wiki/edit/stream', opts);
    if (res.status === 401) { showLoginScreen(); return; }
    if (!res.ok) {
      const d = await res.json().catch(() => ({}));
      throw new Error(d.detail || `Edit failed (${res.status})`);
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop();
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const evt = JSON.parse(line.slice(6));
        if (evt.type === 'chunk') {
          acc += evt.text;
          previewEl.innerHTML = renderMd(_stripPreviewFence(acc));
        } else if (evt.type === 'error') {
          errored = true;
          previewEl.innerHTML = `<div class="diff-line removed">${escHtml(evt.message)}</div>`;
        } else if (evt.type === 'done') {
          fullContent = evt.full_content;
        }
      }
    }
  } catch (e) {
    errored = true;
    previewEl.innerHTML = `<div class="diff-line removed">${escHtml(e.message)}</div>`;
  } finally {
    _activeStreamControllers.delete(controller);
    genBtn.disabled = false; regenBtn.disabled = false;
  }

  if (errored || fullContent == null) {
    statusEl.textContent = errored ? 'Generation failed.' : '';
    return;
  }
  _aiEdit.proposed = fullContent;
  statusEl.textContent = 'Review the changes below, then Apply.';
  // Authoritative render from the clean, fence-stripped content the server
  // returned — this is exactly what Apply will write.
  previewEl.innerHTML = renderMd(fullContent);
  diffEl.innerHTML = renderLineDiffHtml(state.currentPageRaw || '', fullContent);
  applyBtn.disabled = false;
}

async function applyAiEdit() {
  if (!_aiEdit || _aiEdit.proposed == null) return;
  const applyBtn = document.getElementById('btn-ai-edit-apply');
  applyBtn.disabled = true;
  applyBtn.textContent = 'Applying…';
  try {
    await api('PUT', `/api/wiki/${_aiEdit.path}`, { content: _aiEdit.proposed });
    showToast('Page updated.', 'success');
    const path = _aiEdit.path;
    closeModal('modal-ai-edit');
    _aiEdit = null;
    await loadWikiPage(path, false);
  } catch (e) {
    showToast(`Apply failed: ${e.message}`, 'error');
    applyBtn.disabled = false;
  } finally {
    applyBtn.textContent = 'Apply';
  }
}

// ── Wiki import (admins / supervisors, into an empty org) ────────────────────
// Uploads a previously exported bundle (.zip) to /api/ops/import for the active
// org. Uses a raw fetch (multipart) rather than api(), which forces JSON.
async function importWikiBundle(file) {
  const orgLabel = state.userOrgName || 'this organization';
  const ok = await uiConfirm({
    title: 'Import wiki',
    message: `Import only works on an empty wiki. If "${file.name}" contains an `
      + `AGENTS.md schema, it will overwrite ${orgLabel}'s current schema. Continue?`,
    confirmText: 'Import',
  });
  if (!ok) return;
  showToast('Importing wiki…');
  try {
    const fd = new FormData();
    fd.append('file', file);
    const headers = { Authorization: `Bearer ${state.authToken}` };
    if (state.activeOrgId) headers['X-Org-Context'] = state.activeOrgId;
    const res = await fetch('/api/ops/import', { method: 'POST', headers, body: fd });
    const data = await res.json().catch(() => ({}));
    if (res.status === 401) { showLoginScreen(); return; }
    if (!res.ok) throw new Error(data.detail || `Import failed (${res.status})`);
    const bits = [`Imported ${data.pages_imported} page(s)`];
    if (data.schema_imported) bits.push('schema');
    if (data.embeddings_reused) bits.push(`${data.embeddings_reused} embeddings`);
    showToast(bits.join(' · ') + '.', 'success');
    await loadWikiTree();
    document.querySelector('.tab[data-tab="wiki"]')?.click();
  } catch (e) {
    showToast(`Import failed: ${e.message}`, 'error');
  }
}

// ── Sidebar: Sources list ──────────────────────────────────────────────────
async function loadSourcesList() {
  const list = document.getElementById('sources-list');
  try {
    const files = await api('GET', '/api/documents');
    list.innerHTML = '';
    if (!files.length) {
      list.innerHTML = '<div class="loading-row" style="opacity:0.6">No documents uploaded yet.</div>';
      return;
    }
    for (const f of files) {
      list.appendChild(buildSourceItem(f));
    }
  } catch (e) {
    list.innerHTML = `<div class="loading-row" style="color:#FF8080">${escHtml(e.message)}</div>`;
  }
}

function buildSourceItem(f) {
  const item = document.createElement('div');
  item.className = 'tree-item src-selectable' + (state.activeSource === f.name ? ' active' : '');
  item.dataset.name = f.name;
  const isPendingReview = f.ingest_status === 'pending_review';
  const isActive = f.ingest_status === 'processing' || f.ingest_status === 'queued' || f.ingest_status === 'queued_write' || f.ingest_status === 'writing';
  const canDelete = state.userRole === 'admin' || state.userRole === 'supervisor' || !!state.perms.can_delete_files;
  item.innerHTML = `
    ${canDelete ? `<input type="checkbox" class="src-checkbox" aria-label="Select ${escHtml(f.name)}" tabindex="-1">` : ''}
    ${sourceFileIcon(f.name)}
    <span class="label">${escHtml(f.name)}</span>
    ${isPendingReview ? '<span class="src-review-dot" title="Awaiting review"></span>' : ''}
  `;
  const cbEl = item.querySelector('.src-checkbox');
  if (cbEl) {
    cbEl.addEventListener('change', e => {
      e.stopPropagation();
      updateSourcesToolbar();
    });
  }
  item.addEventListener('click', e => {
    if (e.target.classList.contains('src-checkbox')) return;
    // If any checkboxes are checked, toggle this item's checkbox
    const anyChecked = document.querySelectorAll('#sources-list .src-checkbox:checked').length > 0;
    if (anyChecked) {
      const cb = item.querySelector('.src-checkbox');
      if (cb) { cb.checked = !cb.checked; updateSourcesToolbar(); }
      return;
    }
    if (isPendingReview) {
      api('GET', `/api/ops/status/${encodeURIComponent(f.name)}`).catch(() => null)
        .then(job => { if (job?.plan) openReviewModal(f.name, job.plan, job.plan_chat_history || []); });
    } else {
      openSourceView(f.name);
    }
  });
  return item;
}

function ingestBadgeHtml(status) {
  if (!status) return '';
  const map = {
    queued:         ['badge-queued', '&hellip;'],
    processing:     ['badge-processing', '<span class="spinner-tiny"></span>'],
    queued_write:   ['badge-queued', '&hellip;'],
    writing:        ['badge-processing', '<span class="spinner-tiny"></span>'],
    pending_review: ['badge-review', 'Review ›'],
    done:           ['badge-done', '&#10003;'],
    error:          ['badge-error', '!'],
    cancelled:      ['badge-cancelled', '&ndash;'],
  };
  const [cls, icon] = map[status] || ['badge-queued', escHtml(status)];
  return `<span class="badge ${cls} tree-badge">${icon}</span>`;
}

// ── Ingest polling ─────────────────────────────────────────────────────────
function startPolling(filename) {
  if (state.pollingJobs[filename]) return;

  const id = setInterval(async () => {
    try {
      const job = await api('GET', `/api/ops/status/${encodeURIComponent(filename)}`);
      updateSourceBadge(filename, job.status);

      if (job.status === 'pending_review') {
        stopPolling(filename);
        clearUploadRow(filename);
        notifSetReview(filename);
      } else if (job.status === 'done') {
        stopPolling(filename);
        clearUploadRow(filename);
        notifSetDone(filename);
        showToast(`"${filename}" ingested successfully.`, 'success');
        const writerSid = state.writerIngestSessions[filename];
        if (writerSid) {
          delete state.writerIngestSessions[filename];
          try {
            await api('DELETE', `/api/ops/writer/${encodeURIComponent(writerSid)}`);
          } catch { /* draft cleanup best-effort */ }
        }
        await loadWikiTree();
        document.querySelector('.tab[data-tab="wiki"]').click();
      } else if (job.status === 'error') {
        stopPolling(filename);
        clearUploadRow(filename);
        delete state.writerIngestSessions[filename];
        notifSetError(filename, job.message || 'Ingest failed');
        showToast(job.message ? `Ingest failed: ${job.message}` : `Ingest failed for "${filename}".`, 'error');
      } else if (job.status === 'cancelled') {
        stopPolling(filename);
        clearUploadRow(filename);
        delete state.writerIngestSessions[filename];
        notifSetCancelled(filename);
        showToast(`Ingest cancelled for "${filename}".`, 'info');
        await loadSourcesList();
      }
    } catch {
      stopPolling(filename);
    }
  }, 2500);

  state.pollingJobs[filename] = id;
}

function stopPolling(filename) {
  clearInterval(state.pollingJobs[filename]);
  delete state.pollingJobs[filename];
}

function clearUploadRow(filename) {
  const progress = document.getElementById('upload-progress');
  if (!progress) return;
  progress.querySelectorAll('.upload-item').forEach(row => {
    if (row.querySelector('.fname')?.textContent === filename) row.remove();
  });
}

function updateSourceBadge(filename, status) {
  const item = document.querySelector(`#sources-list .tree-item[data-name="${CSS.escape(filename)}"]`);
  if (!item) return;

  // Remove the review dot if present
  item.querySelector('.src-review-dot')?.remove();

  const isActive = status === 'processing' || status === 'queued' || status === 'queued_write' || status === 'writing';
  const isPendingReview = status === 'pending_review';
  const isTerminal = !isActive && !isPendingReview;

  // Add review dot for pending_review
  if (isPendingReview) {
    const dot = document.createElement('span');
    dot.className = 'src-review-dot';
    dot.title = 'Awaiting review';
    item.appendChild(dot);
  }
}

// ── Source file download ───────────────────────────────────────────────────
function canDownloadFiles() {
  return state.userRole === 'admin' || state.userRole === 'supervisor' || !!state.perms.can_download_files;
}

// Download the original uploaded file. Uses fetch (not a plain <a href>) so the
// Bearer token + org context travel with the request — the endpoint is behind
// require_permission("can_download_files"). The blob is saved client-side.
async function downloadDocument(filename, _retried = false) {
  try {
    const headers = {};
    if (state.authToken)   headers['Authorization']  = `Bearer ${state.authToken}`;
    if (state.activeOrgId) headers['X-Org-Context']  = state.activeOrgId;
    const res = await fetch(`/api/documents/${encodeURIComponent(filename)}`, { headers });
    if (res.status === 401 && !_retried && await tryRefreshToken()) {
      return downloadDocument(filename, true);
    }
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Download failed (${res.status})`);
    }
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  } catch (e) {
    showToast(`Could not download "${filename}": ${e.message}`, 'error');
  }
}

// ── Source file view ───────────────────────────────────────────────────────
function openSourceView(filename) {
  state.activeSource = filename;
  state.activePage = null;
  setActiveTreeItem('sources-list', null, filename);
  showPageView();

  const meta = document.getElementById('page-meta');
  if (meta) { meta.classList.add('hidden'); meta.innerHTML = ''; }

  const bc = document.getElementById('page-breadcrumb');
  bc.innerHTML = `raw <span style="opacity:0.4">/</span> <strong>${escHtml(filename)}</strong>`;

  // Download action — available for every uploaded file type (incl. .md/.txt),
  // not just binaries that can't be previewed inline.
  const actions = document.getElementById('page-actions');
  if (canDownloadFiles()) {
    actions.innerHTML = `<button class="btn btn-ghost btn-sm" id="btn-source-download">
      <svg viewBox="0 0 16 16" fill="none" style="width:14px;height:14px"><path d="M8 2v8m0 0L5 7m3 3l3-3M3 13h10" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>
      Download</button>`;
    document.getElementById('btn-source-download')
      .addEventListener('click', () => downloadDocument(filename));
  } else {
    actions.innerHTML = '';
  }

  const content = document.getElementById('page-content');
  const ext = filename.split('.').pop().toLowerCase();

  if (['txt', 'md'].includes(ext)) {
    content.innerHTML = '<div class="content-loading"><span class="spinner spinner-dark"></span> Loading…</div>';
    api('GET', `/api/documents/${filename}`)
      .then(text => {
        content.innerHTML = ext === 'md'
          ? renderMd(text)
          : `<pre style="white-space:pre-wrap;font-family:var(--font-mono);font-size:13px;color:var(--text-secondary);padding:28px 44px">${escHtml(text)}</pre>`;
      })
      .catch(e => { content.innerHTML = `<p style="color:var(--danger);padding:28px">${escHtml(e.message)}</p>`; });
  } else {
    content.innerHTML = `
      <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;padding:80px 24px;color:var(--text-muted)">
        <svg viewBox="0 0 48 48" fill="none" style="width:52px;height:52px;margin-bottom:18px;opacity:0.35"><rect x="4" y="8" width="40" height="32" rx="4" stroke="currentColor" stroke-width="2"/><path d="M16 24h16M16 30h10" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>
        <p style="font-size:14px;font-weight:600;color:var(--text-secondary);margin-bottom:6px">${escHtml(filename)}</p>
        <p style="font-size:12.5px;margin-bottom:20px">Binary file — preview not available</p>
        ${canDownloadFiles() ? '<button class="btn btn-ghost btn-sm" id="btn-binary-download">Download</button>' : ''}
      </div>`;
    const binBtn = document.getElementById('btn-binary-download');
    if (binBtn) binBtn.addEventListener('click', () => downloadDocument(filename));
  }
}

async function reIngest(filename) {
  updateSourceBadge(filename, 'queued');
  if (!notifJobs[filename]) notifAddUpload(filename);
  else notifSetStep(filename, 0);
  try {
    await api('POST', '/api/ops/ingest', { filename });
    notifSetStep(filename, 1);
    startPolling(filename);
  } catch (e) {
    updateSourceBadge(filename, 'error');
    notifSetError(filename, e.message);
    showToast(`Re-ingest failed: ${e.message}`, 'error');
  }
}

async function cancelIngest(filename) {
  try {
    await api('DELETE', `/api/ops/ingest/${encodeURIComponent(filename)}`);
    stopPolling(filename);
    updateSourceBadge(filename, 'cancelled');
    showToast(`Ingest cancelled for "${filename}".`, 'info');
    await loadSourcesList();
  } catch (e) {
    showToast(`Could not cancel ingest: ${e.message}`, 'error');
  }
}

// ── Source selection & deletion ────────────────────────────────────────────
function updateSourcesToolbar() {
  const checked = [...document.querySelectorAll('#sources-list .src-checkbox:checked')];
  const toolbar = document.getElementById('sources-toolbar');
  const count = document.getElementById('sources-sel-count');
  if (checked.length) {
    toolbar.classList.remove('hidden');
    count.textContent = `${checked.length} selected`;
  } else {
    toolbar.classList.add('hidden');
  }
}

function openDeleteSourcesModal() {
  const checked = [...document.querySelectorAll('#sources-list .src-checkbox:checked')];
  if (!checked.length) return;
  const names = checked.map(cb => cb.closest('.tree-item').dataset.name);
  const msg = document.getElementById('delete-sources-msg');
  if (names.length === 1) {
    msg.innerHTML = `You are about to delete <strong>${escHtml(names[0])}</strong>.`;
  } else {
    msg.innerHTML = `You are about to delete <strong>${names.length} files</strong>: ${names.map(n => escHtml(n)).join(', ')}.`;
  }
  openModal('modal-delete-sources');
}

async function confirmDeleteSources() {
  closeModal('modal-delete-sources');
  const checked = [...document.querySelectorAll('#sources-list .src-checkbox:checked')];
  const names = checked.map(cb => cb.closest('.tree-item').dataset.name);
  let failed = 0;
  for (const name of names) {
    try {
      await api('DELETE', `/api/documents/${encodeURIComponent(name)}`);
    } catch {
      failed++;
      showToast(`Failed to delete "${name}"`, 'error');
    }
  }
  if (names.length - failed > 0) {
    showToast(`Deleted ${names.length - failed} file(s). Starting recalibration…`, 'info');
  }
  await loadSourcesList();
  updateSourcesToolbar();
  // Trigger recalibration, passing deleted filenames so the agent can purge derived knowledge
  try {
    await api('POST', '/api/ops/recalibrate', { deleted_files: names });
  } catch (e) {
    if (!e.message.includes('already running')) {
      showToast(`Recalibration could not start: ${e.message}`, 'error');
      return;
    }
  }
  showRecalibrateOverlay();
  startRecalibratePolling();
}

// ── Ingest review ──────────────────────────────────────────────────────────
let _reviewFilename = null;
let _reviewPlan = [];

function openReviewModal(filename, plan, seededHistory = []) {
  _reviewFilename = filename;
  _reviewPlan = plan;

  renderReviewPlan(plan);
  document.getElementById('review-notes').value = '';
  // Wire reject button (always the same action)
  document.getElementById('btn-reject-ingest').onclick = rejectIngest;

  // Clear chat panel then render any pre-seeded messages (e.g. conflict openers)
  const chatPanel = document.getElementById('plan-chat-messages');
  chatPanel.innerHTML = '';
  if (seededHistory.length) {
    seededHistory.forEach(h => appendPlanChatMessage(h.role, h.content));
  } else {
    chatPanel.innerHTML = '<div class="plan-chat-hint">Ask the planner to add, remove, or change pages — or ask it to clarify its reasoning.</div>';
  }
  document.getElementById('plan-chat-input').value = '';

  openModal('modal-review');
}

function renderReviewPlan(plan) {
  _reviewPlan = plan;
  const list = document.getElementById('review-plan-list');

  if (!plan.length) {
    list.innerHTML = `
      <div class="plan-empty">
        <svg viewBox="0 0 48 48" fill="none" width="40" height="40">
          <circle cx="24" cy="24" r="20" stroke="currentColor" stroke-width="1.8" opacity="0.3"/>
          <path d="M16 24l6 6 10-12" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
        <p class="plan-empty-title">No changes needed</p>
        <p class="plan-empty-desc">The planner reviewed the document and determined the wiki is already up to date. You can ask the planner to look more closely, or dismiss.</p>
      </div>`;
    updateReviewSubtitle(0, 0);
    // Swap footer buttons
    document.getElementById('btn-approve-ingest').textContent = 'Dismiss';
    document.getElementById('btn-approve-ingest').onclick = () => {
      closeModal('modal-review');
      notifSetDone(_reviewFilename);
    };
    document.getElementById('btn-reject-ingest').style.display = 'none';
    return;
  }

  // Restore normal footer buttons
  document.getElementById('btn-approve-ingest').textContent = 'Approve & Write';
  document.getElementById('btn-approve-ingest').onclick = approveIngest;
  document.getElementById('btn-reject-ingest').style.display = '';

  list.innerHTML = plan.map((entry, i) => `
    <label class="plan-card" for="plan-check-${i}">
      <input type="checkbox" id="plan-check-${i}" data-index="${i}" class="plan-checkbox" checked>
      <span class="plan-action ${escHtml(entry.action)}">${escHtml(entry.action)}</span>
      <div class="plan-card-body">
        <div class="plan-path">${escHtml(entry.path)}</div>
        <div class="plan-brief">${escHtml((entry.brief ?? '').split('\n')[0])}</div>
      </div>
    </label>
  `).join('');

  updateReviewSubtitle(plan.length, plan.length);
  list.querySelectorAll('.plan-checkbox').forEach(cb => {
    cb.addEventListener('change', () => {
      const total = _reviewPlan.length;
      const checked = list.querySelectorAll('.plan-checkbox:checked').length;
      updateReviewSubtitle(checked, total);
    });
  });
}

function appendPlanChatMessage(role, text) {
  const panel = document.getElementById('plan-chat-messages');
  // Remove hint if still present
  panel.querySelector('.plan-chat-hint')?.remove();
  const div = document.createElement('div');
  div.className = `plan-chat-msg plan-chat-${role}`;
  if (role === 'assistant') {
    div.innerHTML = marked.parse(text || '');
  } else {
    div.textContent = text;
  }
  panel.appendChild(div);
  panel.scrollTop = panel.scrollHeight;
}

async function sendPlanChat() {
  const input = document.getElementById('plan-chat-input');
  const message = input.value.trim();
  if (!message || !_reviewFilename) return;

  input.value = '';
  input.disabled = true;
  document.getElementById('btn-plan-chat-send').disabled = true;

  appendPlanChatMessage('user', message);

  // Typing indicator
  const panel = document.getElementById('plan-chat-messages');
  const typing = document.createElement('div');
  typing.className = 'plan-chat-msg plan-chat-assistant plan-chat-typing';
  typing.innerHTML = '<span class="spinner-tiny"></span> Thinking…';
  panel.appendChild(typing);
  panel.scrollTop = panel.scrollHeight;

  try {
    const result = await api('POST',
      `/api/ops/ingest/${encodeURIComponent(_reviewFilename)}/plan-chat`,
      { message }
    );
    typing.remove();
    appendPlanChatMessage('assistant', result.reply);

    // If the plan was updated, re-render the plan list
    if (result.plan && result.plan.length !== _reviewPlan.length ||
        JSON.stringify(result.plan) !== JSON.stringify(_reviewPlan)) {
      renderReviewPlan(result.plan);
    }
  } catch (e) {
    typing.remove();
    appendPlanChatMessage('assistant', `Error: ${e.message}`);
  } finally {
    input.disabled = false;
    document.getElementById('btn-plan-chat-send').disabled = false;
    input.focus();
  }
}

function updateReviewSubtitle(checked, total) {
  document.getElementById('review-subtitle').textContent =
    `${_reviewFilename} · ${checked} of ${total} page${total !== 1 ? 's' : ''} selected`;
}

async function approveIngest() {
  const filename = _reviewFilename;
  if (!filename) return;
  const notes = document.getElementById('review-notes').value.trim();

  const selectedPlan = Array.from(
    document.querySelectorAll('#review-plan-list .plan-checkbox:checked')
  ).map(cb => _reviewPlan[parseInt(cb.dataset.index)]);

  if (!selectedPlan.length) {
    showToast('Select at least one page to ingest.', 'error');
    return;
  }

  closeModal('modal-review');
  updateSourceBadge(filename, 'processing');
  notifSetStep(filename, 3); // Writing pages
  try {
    await api('POST', `/api/ops/ingest/${encodeURIComponent(filename)}/approve`,
      { notes, plan: selectedPlan });
    startPolling(filename);
  } catch (e) {
    updateSourceBadge(filename, 'error');
    notifSetError(filename, 'Could not start write');
    showToast(`Could not start ingest: ${e.message}`, 'error');
  }
}

async function rejectIngest() {
  const filename = _reviewFilename;
  if (!filename) return;
  closeModal('modal-review');
  notifSetCancelled(filename);
  await cancelIngest(filename);
}

// ── Helpers ────────────────────────────────────────────────────────────────
function setActiveTreeItem(containerId, path, name) {
  // Clear active state in BOTH lists so switching between wiki/sources never leaves a stale highlight
  ['wiki-tree', 'sources-list'].forEach(id => {
    document.querySelectorAll(`#${id} .tree-item.active`).forEach(el => el.classList.remove('active'));
  });
  document.querySelectorAll(`#${containerId} .tree-item`).forEach(el => {
    const match = path ? el.dataset.path === path : el.dataset.name === name;
    if (match) el.classList.add('active');
  });
}

function showPageView() {
  document.getElementById('welcome-screen').classList.add('hidden');
  document.getElementById('page-view').classList.remove('hidden');
}

// ── Notification Panel ─────────────────────────────────────────────────────
const UPLOAD_STEPS  = ['Upload', 'Plan', 'Review', 'Write', 'Done'];
const RECALIB_STEPS = ['Load', 'Analyze', 'Apply', 'Rebuild', 'Done'];

let notifItems     = [];
let notifNextId    = 1;
let notifPanelOpen = false;
let notifFilter    = 'all';
const notifJobs    = {}; // filename → item id

let _bellBtn, _notifPanel, _bellBadge, _notifBody, _notifEmptyEl;

function initNotifPanel() {
  _bellBtn      = document.getElementById('btn-bell');
  _notifPanel   = document.getElementById('notif-panel');
  _bellBadge    = document.getElementById('bell-badge');
  _notifBody    = document.getElementById('notif-body');
  _notifEmptyEl = document.getElementById('notif-empty');

  _bellBtn.addEventListener('click', e => { e.stopPropagation(); notifTogglePanel(); });
  document.addEventListener('click', e => {
    if (notifIsPanelOpen() && e.isTrusted && !_notifPanel.contains(e.target) && e.target !== _bellBtn && !_bellBtn.contains(e.target)) notifClosePanel();
  });
  document.querySelectorAll('.notif-tab[data-filter]').forEach(tab => {
    tab.addEventListener('click', e => {
      e.stopPropagation();
      notifFilter = tab.dataset.filter;
      document.querySelectorAll('.notif-tab[data-filter]').forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      notifRenderBody();
    });
  });
  document.getElementById('btn-clear-done').addEventListener('click', e => {
    e.stopPropagation();
    const settled = notifItems.filter(notifIsSettled);
    if (!settled.length) return;
    settled.forEach(item => {
      const el = _notifBody.querySelector(`.notif-card[data-id="${item.id}"]`);
      if (el) { el.style.maxHeight = el.offsetHeight + 'px'; el.offsetHeight; el.classList.add('dismissing'); }
      if (item.dbId) api('POST', `/api/notifications/${item.dbId}/read`).catch(() => {});
      if (item.type === 'upload') delete notifJobs[item.name];
    });
    setTimeout(() => { notifItems = notifItems.filter(i => !notifIsSettled(i)); notifRender(); }, 370);
  });
  _notifBody.addEventListener('click', e => {
    const btn = e.target.closest('[data-action]');
    if (!btn) return;
    e.stopPropagation();
    const id = Number(btn.dataset.id);
    const action = btn.dataset.action;
    if      (action === 'cancel')     notifCancelUpload(id);
    else if (action === 'review')     notifReviewUpload(id);
    else if (action === 'retry')      notifRetryUpload(id);
    else if (action === 'dismiss')    notifDismissItem(id);
  });
  document.getElementById('btn-theme').addEventListener('click', () => {
    const isDark = document.documentElement.dataset.theme === 'dark';
    document.documentElement.dataset.theme = isDark ? 'light' : 'dark';
  });
  notifRender();
}

function notifIsPanelOpen() { return _notifPanel && !_notifPanel.classList.contains('hidden'); }
function notifTogglePanel() { notifIsPanelOpen() ? notifClosePanel() : notifOpenPanel(); }
function notifOpenPanel() {
  notifPanelOpen = true;
  // Position panel so caret (right:22px, 12px wide → center at right+28) aligns with bell center
  const rect = _bellBtn.getBoundingClientRect();
  const bellCenterFromRight = window.innerWidth - rect.left - rect.width / 2;
  const panelRight = Math.max(8, Math.round(bellCenterFromRight) - 28);
  _notifPanel.style.right = panelRight + 'px';
  _notifPanel.classList.remove('hidden');
  _bellBtn.classList.add('active');
}
function notifClosePanel() {
  notifPanelOpen = false;
  _notifPanel.classList.add('hidden');
  _bellBtn.classList.remove('active');
}

function notifNeedsAttention(item) {
  if (item.type === 'upload')  return ['active','review','error'].includes(item.status);
  if (item.type === 'recalib') return item.status === 'active' || item.status === 'error';
  if (item.type === 'health')  return item.status === 'error';
  if (item.type === 'system')  return item.status === 'warning' || item.status === 'error';
  return false;
}
function notifIsSettled(item) {
  if (item.type === 'upload') return item.status === 'done' || item.status === 'cancelled';
  return item.status === 'done' || item.status === 'info';
}
function notifFilterItems() {
  if (notifFilter === 'upload') return notifItems.filter(i => i.type === 'upload');
  if (notifFilter === 'system') return notifItems.filter(i => i.type !== 'upload');
  return [...notifItems];
}

function notifUpdateBadge() {
  const attention = notifItems.filter(notifNeedsAttention);
  const pill = document.getElementById('notif-count-pill');
  document.getElementById('tab-badge-all').textContent    = notifItems.length;
  document.getElementById('tab-badge-upload').textContent = notifItems.filter(i => i.type === 'upload').length;
  document.getElementById('tab-badge-system').textContent = notifItems.filter(i => i.type !== 'upload').length;
  if (notifItems.length === 0) {
    _bellBadge.className = 'bell-badge hidden';
    pill.textContent = 'all clear'; pill.className = 'notif-count-pill empty'; return;
  }
  const hasError  = attention.some(i => i.status === 'error');
  const hasReview = notifItems.some(i => i.status === 'review') || attention.some(i => i.status === 'warning');
  if (attention.length === 0) {
    _bellBadge.className = 'bell-badge hidden';
    pill.textContent = `${notifItems.length} done`; pill.className = 'notif-count-pill empty';
  } else {
    _bellBadge.textContent = attention.length > 9 ? '9+' : String(attention.length);
    _bellBadge.className = 'bell-badge ' + (hasError ? 'c-error' : hasReview ? 'c-review' : 'c-active');
    if (hasError)       { pill.textContent = `${attention.length} need attention`; pill.className = 'notif-count-pill c-error'; }
    else if (hasReview) { pill.textContent = `${attention.length} need review`;    pill.className = 'notif-count-pill c-review'; }
    else                { pill.textContent = `${attention.length} processing`;     pill.className = 'notif-count-pill'; }
  }
}

function notifRender() { notifRenderBody(); notifUpdateBadge(); }

function notifRenderBody() {
  const visible = notifFilterItems();
  if (visible.length === 0) {
    _notifBody.innerHTML = '';
    _notifBody.appendChild(_notifEmptyEl);
    _notifEmptyEl.style.display = '';
    return;
  }
  _notifEmptyEl.style.display = 'none';
  if (_notifEmptyEl.parentNode === _notifBody) _notifEmptyEl.remove();
  const active  = visible.filter(notifNeedsAttention);
  const settled = visible.filter(i => !notifNeedsAttention(i));
  _notifBody.innerHTML = '';
  active.forEach(i  => _notifBody.appendChild(notifBuildCard(i)));
  if (active.length && settled.length) {
    const hr = document.createElement('div');
    hr.style.cssText = 'height:1px;background:rgba(255,255,255,0.06);margin:6px 8px;';
    _notifBody.appendChild(hr);
  }
  settled.forEach(i => _notifBody.appendChild(notifBuildCard(i)));
}

function notifBuildCard(item) {
  return item.type === 'upload' ? notifBuildUploadCard(item) : notifBuildSysCard(item);
}

function notifBuildUploadCard(job) {
  const card = document.createElement('div');
  card.className = `notif-card type-upload status-${job.status}`;
  card.dataset.id = job.id;
  const _errMsg = job.errorMsg || '';
  const stepLabel =
    job.status === 'done'      ? 'Done · all pages written'   :
    job.status === 'error'     ? `Failed · ${_errMsg.length > 55 ? _errMsg.slice(0, 55) + '…' : _errMsg || 'unknown error'}` :
    job.status === 'review'    ? 'Waiting for your review'    :
    job.status === 'cancelled' ? 'Cancelled'                  :
    job.step === 0             ? 'Uploading file…'            :
    job.step === 1             ? 'Planning pages…'            :
    job.step === 3             ? 'Writing pages…'             : 'Processing…';
  const sublabelClass = 'card-sublabel' +
    (job.status === 'done'   ? ' s-done'   :
     job.status === 'error'  ? ' s-error'  :
     job.status === 'review' ? ' s-review' : '');
  const indicator =
    job.status === 'active' ? '<span class="spinner-tiny"></span>' :
    job.status === 'review' ? '<span class="dot-review"></span>'   :
    job.status === 'done'   ? '<svg width="10" height="10" viewBox="0 0 10 10" fill="none"><path d="M2 5l2.5 2.5L8 3" stroke="#40C997" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>' :
    job.status === 'error'  ? '<svg width="10" height="10" viewBox="0 0 10 10" fill="none"><path d="M5 3v3M5 7.5h.01" stroke="#FF8080" stroke-width="1.5" stroke-linecap="round"/></svg>' : '';
  let actions = '';
  if (job.status === 'active') {
    actions = `<button class="card-action-btn btn-cancel" data-action="cancel" data-id="${job.id}"><svg viewBox="0 0 12 12" fill="none"><path d="M3 9l6-6M9 9L3 3" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>Cancel</button>`;
  } else if (job.status === 'review') {
    actions = `<button class="card-action-btn btn-review" data-action="review" data-id="${job.id}">Review <svg viewBox="0 0 12 12" fill="none"><path d="M4.5 3L8 6l-3.5 3" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg></button>`;
  } else if (job.status === 'error') {
    actions = `<button class="card-action-btn btn-retry" data-action="retry" data-id="${job.id}"><svg viewBox="0 0 12 12" fill="none"><path d="M10 4A4.5 4.5 0 1 0 9 8" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/><path d="M10 2v3H7" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>Retry</button>
      <button class="card-action-btn" data-action="dismiss" data-id="${job.id}"><svg viewBox="0 0 12 12" fill="none"><path d="M3 9l6-6M9 9L3 3" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg></button>`;
  } else {
    actions = `<button class="card-action-btn" data-action="dismiss" data-id="${job.id}"><svg viewBox="0 0 12 12" fill="none"><path d="M3 9l6-6M9 9L3 3" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg></button>`;
  }
  const pct =
    job.status === 'done'      ? 100 :
    job.status === 'error'     ? Math.round((job.step/4)*65) :
    job.status === 'cancelled' ? Math.round((job.step/4)*55) :
    job.status === 'review'    ? 60 :
    Math.round((job.step/4)*80 + 10);
  const fillCls =
    job.status === 'done'                        ? 'card-progress-fill pf-success'       :
    job.status === 'error'                       ? 'card-progress-fill pf-error'         :
    job.status === 'review'                      ? 'card-progress-fill pf-review'        :
    (job.status === 'active' && job.step === 0)  ? 'card-progress-fill pf-indeterminate' :
    'card-progress-fill';
  const fillStyle = fillCls.includes('indeterminate') ? '' : `style="width:${pct}%"`;
  const ext = job.name.split('.').pop().toLowerCase();
  const stepsHTML = UPLOAD_STEPS.map((s, i) => {
    const cls =
      (job.status === 'error'  && i === job.step) ? 'card-step-node sn-error'  :
      (job.status === 'review' && i === 2)        ? 'card-step-node sn-review' :
      (i < job.step || job.status === 'done')     ? 'card-step-node sn-done'   :
      (i === job.step && job.status === 'active') ? 'card-step-node sn-active' :
      'card-step-node';
    return `<div class="${cls}"><div class="card-step-dot"></div></div>${i < UPLOAD_STEPS.length-1 ? '<div class="card-step-sep"></div>' : ''}`;
  }).join('');
  card.innerHTML = `<div class="upload-inner">
    <div class="card-row1">
      <div class="file-badge ext-${ext}">${ext.slice(0,4).toUpperCase()}</div>
      <div class="card-info">
        <div class="card-name" title="${escHtml(job.name)}">${escHtml(job.name)}</div>
        <div class="${sublabelClass}" title="${job.status === 'error' && _errMsg ? escHtml(_errMsg) : ''}">${indicator} ${stepLabel}</div>
      </div>
      <div class="card-actions">${actions}</div>
    </div>
    <div class="card-steps">${stepsHTML}</div>
    <div class="card-progress"><div class="${fillCls}" ${fillStyle}></div></div>
  </div>`;
  return card;
}

function notifBuildSysCard(item) {
  const card = document.createElement('div');
  card.className = `notif-card type-${item.type} status-${item.status}`;
  card.dataset.id = item.id;
  const iconKey = item.status === 'error' ? 'error' : item.type;
  const iconClass = { health:'ic-health', recalib:'ic-recalib', system:'ic-system', query:'ic-query', error:'ic-error' }[iconKey] || 'ic-system';
  const icons = {
    health:  '<svg viewBox="0 0 16 16" fill="none"><path d="M2 7h3l2-4 3 8 2-4h2" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"/></svg>',
    recalib: '<svg viewBox="0 0 16 16" fill="none"><path d="M13.5 8A5.5 5.5 0 112.5 8" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/><path d="M13.5 5v3h-3" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>',
    system:  '<svg viewBox="0 0 16 16" fill="none"><path d="M8 2L1.5 14h13L8 2z" stroke="currentColor" stroke-width="1.4" stroke-linejoin="round"/><path d="M8 7v3M8 11.5h.01" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>',
    query:   '<svg viewBox="0 0 16 16" fill="none"><path d="M13 3H3v10h10V3z" stroke="currentColor" stroke-width="1.3"/><path d="M6 6h4M6 9h2" stroke="currentColor" stroke-width="1.3" stroke-linecap="round"/><path d="M11 11l2 2" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/></svg>',
    error:   '<svg viewBox="0 0 16 16" fill="none"><circle cx="8" cy="8" r="5.5" stroke="currentColor" stroke-width="1.4"/><path d="M8 5v3.5M8 10.5h.01" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>',
  };
  const spinnerSuffix = (item.type === 'recalib' && item.status === 'active')
    ? ' <span style="display:inline-flex;align-items:center;vertical-align:middle;margin-left:5px"><span class="spinner-tiny" style="border-color:rgba(155,109,255,0.18);border-top-color:#9B6DFF"></span></span>'
    : '';
  let actions = '';
  if (item.action) {
    actions += `<button class="card-action-btn btn-view" data-action="sys-action" data-key="${item.action.key}" data-id="${item.id}">${escHtml(item.action.label)} <svg viewBox="0 0 12 12" fill="none"><path d="M4.5 3L8 6l-3.5 3" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg></button>`;
  }
  if (item.status !== 'active') {
    actions += `<button class="card-action-btn" data-action="dismiss" data-id="${item.id}"><svg viewBox="0 0 12 12" fill="none"><path d="M3 9l6-6M9 9L3 3" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg></button>`;
  }
  let progressHTML = '';
  if (item.type === 'recalib' && item.status === 'active' && item.progress != null) {
    const recalibSteps = RECALIB_STEPS.map((s, i) => {
      const done   = i < Math.round(item.progress / 25);
      const active = i === Math.round(item.progress / 25);
      const cls    = done ? 'card-step-node sn-done' : active ? 'card-step-node sn-active' : 'card-step-node';
      return `<div class="${cls}" style="color:rgba(155,109,255,0.8)"><div class="card-step-dot"></div></div>${i < RECALIB_STEPS.length-1 ? '<div class="card-step-sep"></div>' : ''}`;
    }).join('');
    progressHTML = `<div class="sys-progress-row">
      <div class="card-steps" style="margin-bottom:6px">${recalibSteps}</div>
      <div class="sys-progress-bar"><div class="sys-progress-fill" style="width:${item.progress}%"></div></div>
    </div>`;
  }
  card.innerHTML = `<div class="sys-inner"><div class="sys-row">
    <div class="sys-icon-wrap ${iconClass}">${icons[iconKey] || ''}</div>
    <div class="sys-content">
      <div class="sys-title">${escHtml(item.title)}${spinnerSuffix}</div>
      <div class="sys-body">${escHtml(item.body)}</div>
      ${item.timestamp ? `<div class="sys-meta">${escHtml(item.timestamp)}</div>` : ''}
      ${progressHTML}
    </div>
    <div class="sys-actions">${actions}</div>
  </div></div>`;
  return card;
}

function notifCancelUpload(id) {
  const item = notifItems.find(i => i.id === id); if (!item) return;
  item.status = 'cancelled';
  notifAnimateDismiss(id, () => { notifItems = notifItems.filter(i => i.id !== id); delete notifJobs[item.name]; notifRender(); });
  cancelIngest(item.name);
}
function notifReviewUpload(id) {
  const item = notifItems.find(i => i.id === id); if (!item) return;
  if (item.dbId) {
    api('POST', `/api/notifications/${item.dbId}/read`).catch(() => {});
    item.dbId = null;
  }
  api('GET', `/api/ops/status/${encodeURIComponent(item.name)}`).then(job => {
    if (job) openReviewModal(item.name, job.plan || [], job.plan_chat_history || []);
  }).catch(() => {});
}
function notifRetryUpload(id) {
  const item = notifItems.find(i => i.id === id); if (!item) return;
  item.status = 'active'; item.step = 0; notifRender(); notifRingBell();
  reIngest(item.name);
}
function notifDismissItem(id) {
  const item = notifItems.find(i => i.id === id);
  if (item && item.dbId) {
    api('POST', `/api/notifications/${item.dbId}/read`).catch(() => {});
  }
  notifAnimateDismiss(id, () => {
    if (item && item.type === 'upload') delete notifJobs[item.name];
    notifItems = notifItems.filter(i => i.id !== id);
    notifRender();
  });
}
function notifAnimateDismiss(id, cb) {
  const el = _notifBody.querySelector(`.notif-card[data-id="${id}"]`);
  if (!el) { cb(); return; }
  el.style.maxHeight = el.offsetHeight + 'px';
  el.offsetHeight;
  el.classList.add('dismissing');
  setTimeout(cb, 370);
}
function notifRingBell() {
  if (!_bellBtn) return;
  _bellBtn.classList.remove('ringing'); void _bellBtn.offsetWidth; _bellBtn.classList.add('ringing');
  setTimeout(() => _bellBtn.classList.remove('ringing'), 700);
  _bellBadge.classList.remove('pop'); void _bellBadge.offsetWidth; _bellBadge.classList.add('pop');
  setTimeout(() => _bellBadge.classList.remove('pop'), 400);
}

function notifAddUpload(filename) {
  const ext = filename.split('.').pop().toLowerCase();
  const id = notifNextId++;
  notifItems.unshift({ id, type: 'upload', name: filename, ext, step: 0, status: 'active' });
  notifJobs[filename] = id;
  notifRender(); notifRingBell();
}
function notifSetStep(filename, stepIndex) {
  const id = notifJobs[filename]; if (id == null) return;
  const item = notifItems.find(i => i.id === id); if (!item) return;
  item.step = stepIndex; item.status = 'active';
  notifRender();
}
function notifSetReview(filename) {
  const id = notifJobs[filename]; if (id == null) return;
  const item = notifItems.find(i => i.id === id); if (!item) return;
  item.step = 2; item.status = 'review';
  notifRender(); notifRingBell();
}
function notifSetDone(filename) {
  const id = notifJobs[filename]; if (id == null) return;
  const item = notifItems.find(i => i.id === id); if (!item) return;
  item.step = 4; item.status = 'done';
  notifRender(); notifRingBell();
}
function notifSetError(filename, msg) {
  const id = notifJobs[filename]; if (id == null) return;
  const item = notifItems.find(i => i.id === id); if (!item) return;
  item.status = 'error'; item.errorMsg = msg;
  notifRender(); notifRingBell();
}
function notifSetCancelled(filename) {
  const id = notifJobs[filename]; if (id == null) return;
  const item = notifItems.find(i => i.id === id); if (!item) return;
  item.status = 'cancelled';
  notifRender();
  setTimeout(() => {
    notifItems = notifItems.filter(i => i.id !== id);
    delete notifJobs[filename];
    notifRender();
  }, 4000);
}

function notifAddSystem({ type = 'system', title, body, status = 'info', progress, action, timestamp, dbId }) {
  const id = notifNextId++;
  notifItems.unshift({ id, dbId: dbId || null, type, title, body, status, progress, action, timestamp: timestamp || 'just now' });
  notifRender(); notifRingBell();
  return id;
}
function notifUpdateSystem(id, updates) {
  const item = notifItems.find(i => i.id === id); if (!item) return;
  Object.assign(item, updates);
  notifRender();
}


// ── Upload ─────────────────────────────────────────────────────────────────
function initUpload() {
  const zone = document.getElementById('drop-zone');
  const fileInput = document.getElementById('file-input');

  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag-over'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('drag-over');
    const files = Array.from(e.dataTransfer.files);
    if (files.length) uploadFiles(files);
  });
  fileInput.addEventListener('change', () => {
    const files = Array.from(fileInput.files);
    if (files.length) uploadFiles(files);
    fileInput.value = '';
  });
}

async function uploadFiles(files) {
  closeModal('modal-upload');
  document.querySelector('.tab[data-tab="sources"]').click();

  for (const file of files) {
    notifAddUpload(file.name);

    try {
      const fd = new FormData();
      fd.append('file', file);
      const headers = state.authToken ? { 'Authorization': `Bearer ${state.authToken}` } : {};
      if (state.activeOrgId) headers['X-Org-Context'] = state.activeOrgId;
      const res = await fetch('/api/documents/upload', { method: 'POST', body: fd, headers });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        const msg = (err && err.detail) || res.statusText;
        if (res.status === 429) {
          notifAddSystem({ type: 'system', title: 'Limit reached', body: msg, status: 'error' });
          throw Object.assign(new Error(msg), { isQuota: true });
        }
        throw new Error(msg);
      }

      notifSetStep(file.name, 1); // Planning
      await loadSourcesList();
      startPolling(file.name);
    } catch (e) {
      notifSetError(file.name, e.isQuota ? (e.message || 'Limit reached') : e.message);
    }
  }
}

// ── Query Panel ────────────────────────────────────────────────────────────
function autoResizeChatInput() {
  const el = document.getElementById('query-input');
  el.style.height = 'auto';
  const next = Math.min(el.scrollHeight, 150);
  el.style.height = next + 'px';
  el.style.overflowY = el.scrollHeight > 150 ? 'auto' : 'hidden';
}

function initQueryPanel() {
  const input = document.getElementById('query-input');
  input.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); doChat(); }
  });
  input.addEventListener('input', autoResizeChatInput);
  document.getElementById('btn-ask').addEventListener('click', doChat);
  document.getElementById('btn-new-chat').addEventListener('click', newChat);
  // Delegated handler for inline source-ref anchors rendered inside bubbles
  document.getElementById('chat-messages').addEventListener('click', e => {
    const ref = e.target.closest('[data-source-ref]');
    if (ref) {
      e.preventDefault();
      loadWikiPage(ref.dataset.sourceRef);
      document.querySelector('.tab[data-tab="wiki"]').click();
    }
  });
}

async function newChat() {
  if (state.chatSessionId) {
    await api('DELETE', `/api/ops/chat/${state.chatSessionId}`).catch(() => {});
    setChatSessionId(null);
  }
  document.getElementById('chat-messages').innerHTML = '';
}

function setChatSessionId(id) {
  state.chatSessionId = id;
  if (id) localStorage.setItem('chat_session_id', id);
  else    localStorage.removeItem('chat_session_id');
}

// Render an array of stored messages ({role, text, sources}) into the chat panel.
// Shared between the regular-chat boot path and the writer-resume path.
function renderChatHistory(messages) {
  const container = document.getElementById('chat-messages');
  container.innerHTML = '';
  for (const m of messages || []) {
    if (!m || !m.text) continue;
    appendChatBubble(m.role || 'user', m.text, m.sources || []);
  }
}

async function restoreChatHistoryIfAny() {
  if (!state.chatSessionId) return;
  try {
    const data = await api('GET', `/api/ops/chat/${encodeURIComponent(state.chatSessionId)}/history`);
    // Don't rehydrate writer-mode sessions through the regular-chat boot path —
    // those are restored explicitly via the writer picker.
    if (data.mode && data.mode !== 'chat') {
      setChatSessionId(null);
      return;
    }
    renderChatHistory(data.messages);
  } catch (e) {
    // Session expired / deleted server-side — quietly forget it.
    setChatSessionId(null);
  }
}

// Render markdown then replace any <a href="source-path"> elements the LLM wrote
// with proper [N] source-ref anchors, using actual DOM parsing so no regex edge
// cases around href attributes can corrupt the output.
function _matchSource(value, sources) {
  return sources.findIndex(s => {
    const filename = s.split('/').pop();
    const stem = filename.replace(/\.md$/i, '');
    return value === s || value.endsWith('/' + s) || value === filename || value === stem;
  });
}

function _readableLabel(displayText) {
  if (!displayText) return '';
  let text = displayText.trim();
  if (!text) return '';
  if (text.includes('/')) text = text.split('/').pop();
  if (/\.md$/i.test(text)) return text.replace(/\.md$/i, '').replace(/[-_]/g, ' ').trim();
  if (text.includes(' ') && text.length <= 80) return text;
  if ((text.includes('-') || text.includes('_')) && !text.includes(' ')) return text.replace(/[-_]/g, ' ').trim();
  return '';
}

function _makeSourceRefAnchor(sources, idx, displayText = '') {
  const s = sources[idx];
  const a = document.createElement('a');
  a.className = 'source-ref';
  a.dataset.sourceRef = s;
  a.title = s.split('/').pop();
  const label = _readableLabel(displayText);
  if (label) {
    a.appendChild(document.createTextNode(label + ' '));
    const sup = document.createElement('sup');
    sup.textContent = `[${idx + 1}]`;
    a.appendChild(sup);
  } else {
    a.textContent = `[${idx + 1}]`;
  }
  return a;
}

function renderWithSourceAnchors(text, sources) {
  const div = document.createElement('div');
  div.innerHTML = renderMd(text);
  if (sources && sources.length) {
    // Pass 1: replace <a href="source-path"> anchors the LLM wrote as markdown links
    div.querySelectorAll('a[href]').forEach(a => {
      let href = '';
      try { href = decodeURIComponent(a.getAttribute('href') || ''); } catch { href = a.getAttribute('href') || ''; }
      href = href.replace(/\\/g, '/').replace(/^[./]+/, '').replace(/^(?:wiki|data\/wiki)\//i, '').replace(/[?#].*$/, '');
      const idx = _matchSource(href, sources);
      if (idx >= 0) a.replaceWith(_makeSourceRefAnchor(sources, idx, a.textContent));
    });
    // Pass 2: replace [[wikilink]] spans the LLM wrote — renderMd converts these to
    // <span class="wiki-link" data-query="stem"> elements; match them against sources
    // so they become proper [N] source-ref anchors instead of triggering a wiki search.
    div.querySelectorAll('span.wiki-link[data-query]').forEach(span => {
      const idx = _matchSource(span.dataset.query, sources);
      if (idx >= 0) span.replaceWith(_makeSourceRefAnchor(sources, idx, span.textContent));
      // Unmatched spans: strip to plain text so they don't trigger a broken wiki search
      // from inside a chat bubble (LLM used wiki syntax in a chat context).
      else span.replaceWith(document.createTextNode(span.textContent));
    });
  } else {
    // No sources — strip all wiki-link spans to plain text
    div.querySelectorAll('span.wiki-link').forEach(span => {
      span.replaceWith(document.createTextNode(span.textContent));
    });
  }
  return div.innerHTML;
}

function appendChatBubble(role, text, sources = [], savedTo = null, isError = false) {
  const container = document.getElementById('chat-messages');
  const msg = document.createElement('div');
  msg.className = `chat-msg chat-msg-${role}`;

  const bubble = document.createElement('div');
  bubble.className = 'chat-bubble' + (isError ? ' chat-bubble-error' : '');
  if (role === 'assistant' && !isError) {
    bubble.innerHTML = renderWithSourceAnchors(text, sources);
  } else {
    bubble.textContent = text;
  }
  msg.appendChild(bubble);

  if (sources && sources.length) {
    const src = document.createElement('div');
    src.className = 'chat-sources';
    src.innerHTML = 'Sources: ' + sources.map((s, i) => {
      const name = escHtml(s.split('/').pop());
      return `<a class="source-ref" data-source-ref="${escHtml(s)}" title="${name}">[${i + 1}]</a>`;
    }).join(' ');
    msg.appendChild(src);
  }

  if (savedTo) {
    const notice = document.createElement('div');
    notice.className = 'chat-saved-notice';
    notice.innerHTML = `<svg viewBox="0 0 16 16" fill="none"><path d="M3 8l3.5 3.5L13 5" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg> Saved to wiki: ${escHtml(savedTo)}`;
    msg.appendChild(notice);
  }

  container.appendChild(msg);
  container.scrollTop = container.scrollHeight;
  return msg;
}

const _thinkingWords = [
  'Consulting the void…',  'Summoning squirrels…',  'Shaking magic 8-ball…',  'Rolling dice…',
  'Bribing the algorithm…',  'Sacrificing rubber duck…',  'Arguing with myself…',  'Translating dolphin thoughts…',
  'Waiting for pigeons…',  'Polishing crystal ball…',  'Spinning for clarity…',  'Staring into nothing…',
  'Whispering to servers…',  'Consulting vibes…',  'Vibing harder…',  'Adding more chaos…',  'Chasing lost thoughts…',
  'Herding invisible cats…',  'Untangling spaghetti logic…',  'Wrestling conceptual octopus…',  'Inflating brain balloons…',
  'Deflating bad ideas…',  'Mental reboot…',  'Pressing random buttons…',  'Rearranging neurons…',
  'Internal googling…',  'Checking brain cache…',  'Looking under couch…',  'Borrowing brain cell…',
  'Misplacing answers…',  'Blaming the compiler…',  'Opening mental tabs…',  'Closing wrong tab…',
  'Downloading more RAM…',  'Uploading confusion…',  'Rebooting imagination…',  'Installing common sense…',
  'Ignoring updates…',  'Speedrunning overthinking…',  'Achieving nothing efficiently…',
  'Inventing new problems…',  'Solving wrong problem…',  'Thinking about snacks…',
  'Tickling the matrix…',    'Negotiating with ghosts…',    'Petting cosmic hamsters…',    'Greasing thought gears…',
  'Consulting moonbeams…',    'Juggling paradoxes…',    'Warming up neurons…',    'Poking the universe…',
  'Awakening gremlins…',    'Untying brain knots…',    'Borrowing wizard energy…',    'Summoning better thoughts…',
  'Distracting tiny goblins…',    'Recalibrating nonsense…',    'Orbiting bad ideas…',    'Refilling brain juice…',
  'Milkshaking reality…',    'Dusting off synapses…',    'Chasing rogue electrons…',    'Gremlin negotiations…',
  'Rebooting goblin mode…',    'Shuffling braincards…',    'Confusing the compiler…',    'Petting server hamsters…',
  'Pinging the cosmos…',    'Consulting raccoons…',    'Rotating thought cubes…',    'Defragging consciousness…',
  'Tickling probabilities…',    'Rehydrating neurons…',    'Overclocking intuition…',    'Untangling timelines…',
  'Consulting cave spirits…',    'Stirring quantum soup…',    'Chasing shiny thoughts…',    'Reheating old ideas…',
  'Summoning backup neurons…',    'Releasing confusion particles…',    'Shaking thought snowglobes…',    'Mining for epiphanies…',
  'Pondering aggressively…',    'Loitering intellectually…',    'Greasing mental hinges…',    'Calibrating whimsy…',
  'Tuning reality antennas…',    'Compressing overthinking…',    'Searching alternate timelines…',    'Vibecoding internally…',
  'Pacifying brain squirrels…',    'Folding conceptual laundry…',    'Spinning uncertainty…',    'Charging idea crystals…',
  'Consulting basement wizards…',    'Rearranging moon dust…',    'Reversing polarity…',    'Crossing thought streams…',
  'Escaping thought labyrinth…',    'Feeding the gremlins…',    'Reinventing wheels…',    'Optimizing chaos…',
];

function appendLoadingBubble() {
  const container = document.getElementById('chat-messages');
  const msg = document.createElement('div');
  msg.className = 'chat-msg chat-msg-assistant';
  const bubble = document.createElement('div');
  bubble.className = 'chat-bubble chat-bubble-loading';
  const wordEl = document.createElement('span');
  wordEl.className = 'chat-thinking';
  wordEl.textContent = _thinkingWords[Math.floor(Math.random() * _thinkingWords.length)];
  bubble.appendChild(wordEl);
  msg.appendChild(bubble);
  // Cycle through whimsical words every 1.8 s
  msg._thinkingTimer = setInterval(() => {
    wordEl.style.animation = 'none';
    void wordEl.offsetWidth; // reflow to restart animation
    wordEl.style.animation = '';
    wordEl.textContent = _thinkingWords[Math.floor(Math.random() * _thinkingWords.length)];
  }, 1800);
  container.appendChild(msg);
  container.scrollTop = container.scrollHeight;
  return msg;
}

// Strip markdown link syntax [text](url) down to just the display text.
// Used in streaming mode so LLM-written source links appear as plain text.
function stripMdLinks(text) {
  return text.replace(/\[([^\]]*)\]\([^)]*\)/g, '$1');
}

async function doChat() {
  if (state.writerMode) return doWriterChat();
  const input = document.getElementById('query-input');
  const message = input.value.trim();
  if (!message) return;
  input.value = '';
  input.style.height = '';
  input.style.overflowY = 'hidden';

  const saveToWiki = document.getElementById('query-save').checked;
  const btn = document.getElementById('btn-ask');

  appendChatBubble('user', message);
  const loadingBubble = appendLoadingBubble();
  btn.disabled = true;
  input.disabled = true;

  // Track this fetch so signout aborts the connection (and the server-side
  // Bedrock streaming call) instead of letting it consume tokens orphaned.
  const controller = new AbortController();
  _activeStreamControllers.add(controller);

  const opts = {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    signal: controller.signal,
  };
  if (state.authToken) opts.headers['Authorization'] = `Bearer ${state.authToken}`;
  if (state.activeOrgId) opts.headers['X-Org-Context'] = state.activeOrgId;
  opts.body = JSON.stringify({ session_id: state.chatSessionId, message, save_to_wiki: saveToWiki });

  try {
    if (!state.streamEnabled) {
      // ── Non-streaming: full response at once, source refs linked ──────────
      const res = await streamFetch('/api/ops/chat', opts);
      if (res.status === 401) { showLoginScreen(); return; }
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        const msg = err.detail || res.statusText;
        if (res.status === 429) throw Object.assign(new Error(msg), { isQuota: true });
        throw new Error(msg);
      }
      const data = await res.json();
      setChatSessionId(data.session_id);
      clearInterval(loadingBubble._thinkingTimer);
      loadingBubble.remove();
      appendChatBubble('assistant', data.answer, data.sources || [], data.saved_to);
      if (data.saved_to) await loadWikiTree();
    } else {
      // ── Streaming: chunks arrive live, source refs shown as plain text ────
      const res = await streamFetch('/api/ops/chat/stream', opts);
      if (res.status === 401) { showLoginScreen(); return; }
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        const msg = err.detail || res.statusText;
        if (res.status === 429) throw Object.assign(new Error(msg), { isQuota: true });
        throw new Error(msg);
      }

      const container = document.getElementById('chat-messages');
      const msg = loadingBubble;
      const bubble = msg.querySelector('.chat-bubble');

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let fullText = '';
      let sources = [];
      let savedTo = null;
      let firstChunk = true;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop();
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          let data;
          try { data = JSON.parse(line.slice(6)); } catch { continue; }
          if (data.type === 'meta') {
            setChatSessionId(data.session_id);
            sources = data.sources || [];
          } else if (data.type === 'chunk') {
            if (firstChunk) {
              clearInterval(msg._thinkingTimer);
              bubble.classList.remove('chat-bubble-loading');
              bubble.innerHTML = '';
              firstChunk = false;
            }
            fullText += data.text;
            // Streaming: strip markdown links so refs appear as plain [N] text
            bubble.innerHTML = renderMd(stripMdLinks(fullText));
            container.scrollTop = container.scrollHeight;
          } else if (data.type === 'done') {
            savedTo = data.saved_to;
          }
        }
      }

      // Post-processing: now that we have the full text and all sources, do a
      // proper DOM-based source anchor pass (same as non-streaming) so LLM-written
      // links like [Page](concepts/page.md) become correct [N] source-ref anchors
      // instead of plain text or broken hrefs.
      bubble.innerHTML = renderWithSourceAnchors(fullText, sources);

      if (sources.length) {
        const src = document.createElement('div');
        src.className = 'chat-sources';
        src.innerHTML = 'Sources: ' + sources.map((s, i) => {
          const name = escHtml(s.split('/').pop());
          return `<a class="source-ref" data-source-ref="${escHtml(s)}" title="${name}">[${i + 1}]</a>`;
        }).join(' ');
        msg.appendChild(src);
      }

      if (savedTo) {
        const notice = document.createElement('div');
        notice.className = 'chat-saved-notice';
        notice.innerHTML = `<svg viewBox="0 0 16 16" fill="none"><path d="M3 8l3.5 3.5L13 5" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg> Saved to wiki: ${escHtml(savedTo)}`;
        msg.appendChild(notice);
        await loadWikiTree();
      }
    }
  } catch (e) {
    if (loadingBubble) {
      clearInterval(loadingBubble._thinkingTimer);
      loadingBubble.remove();
    }
    if (e.isQuota) {
      appendQuotaBubble(e.message);
      notifAddSystem({ type: 'system', title: 'Token limit reached', body: e.message, status: 'error' });
    } else if (e.name === 'AbortError') {
      // User signed out or navigated away — quiet exit.
    } else {
      appendChatBubble('assistant', `Error: ${e.message}`, [], null, true);
    }
  } finally {
    _activeStreamControllers.delete(controller);
    btn.disabled = false;
    input.disabled = false;
    input.focus();
  }
}

function appendQuotaBubble(message) {
  const container = document.getElementById('chat-messages');
  const el = document.createElement('div');
  el.className = 'chat-msg chat-msg-assistant';
  el.innerHTML = `<div class="chat-bubble chat-bubble-quota">
    <svg viewBox="0 0 20 20" fill="none" style="width:16px;height:16px;flex-shrink:0;margin-top:1px">
      <path d="M10 3a7 7 0 100 14A7 7 0 0010 3zm0 4v4m0 2v.5" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/>
    </svg>
    <span>${escHtml(message)}</span>
  </div>`;
  container.appendChild(el);
  container.scrollTop = container.scrollHeight;
}

// ── Lint ───────────────────────────────────────────────────────────────────
async function runLint() {
  openModal('modal-lint');
  const result = document.getElementById('lint-result');
  result.innerHTML = '<div class="content-loading"><span class="spinner spinner-dark"></span> Analyzing wiki health…</div>';

  try {
    const data = await api('POST', '/api/ops/lint');
    const score = data.health_score ?? '–';
    const color = typeof score === 'number'
      ? (score >= 80 ? 'var(--success)' : score >= 50 ? 'var(--warning)' : 'var(--danger)')
      : 'var(--text-muted)';

    let html = `
      <div class="health-score-block">
        <div class="health-score" style="color:${color}">${score}</div>
        <div class="health-label">Health Score / 100</div>
      </div>`;

    // Grouped by remediation path so it's clear what recalibration will fix
    // vs. what needs you. `link_gaps` splits broken [[wikilinks]] into
    // auto_create (>= min_refs → recalibration makes the page) and manual.
    const bd = data.breakdown || {};
    const gaps = data.link_gaps || { auto_create: [], manual: [] };
    const note = (t) => `<p style="font-size:12px;color:var(--text-muted);margin:6px 0 4px">${t}</p>`;
    const subhead = (t) => `<div style="font-weight:600;font-size:13px;margin:12px 0 4px">${t}</div>`;
    const gapList = (arr, tail) => `<ul class="suggestions-list">` + arr.map(g =>
      `<li><code>[[${escHtml(g.slug)}]]</code> — ${tail(g)}</li>`).join('') + `</ul>`;

    if (typeof score === 'number') {
      html += `<p style="text-align:center;color:var(--text-muted);font-size:12px;margin:-8px 0 12px">${data.flagged_pages}/${data.total_pages} pages have structural issues</p>`;
    }

    // ── Bucket 1: recalibration will fix these ───────────────────────────────
    const RECAL_LABELS = {
      stub: 'Stub pages (very short)',
      stale: 'Stale pages (not reviewed in 90+ days)',
      orphan: 'Orphan pages (nothing links to them)',
      duplicate_candidate: 'Duplicate page titles',
    };
    const recalRows = Object.entries(RECAL_LABELS)
      .filter(([k]) => bd[k])
      .map(([k, label]) => `<div class="health-bd-row"><span>${label}</span><span class="health-bd-count">${bd[k]}</span></div>`)
      .join('');
    html += `<div class="health-breakdown"><div class="lint-section-title">Recalibration will fix these</div>`;
    if (recalRows || gaps.auto_create.length) {
      html += recalRows;
      if (gaps.auto_create.length) {
        html += `<div class="health-bd-row"><span>Missing pages it will create</span><span class="health-bd-count">${gaps.auto_create.length}</span></div>`;
        html += gapList(gaps.auto_create, g => `linked from ${g.ref_count} pages`);
      }
      html += note('Run Master Recalibration to repair these automatically, then re-check the score.');
    } else {
      html += note('Nothing here needs recalibration.');
    }
    html += `</div>`;

    // ── Bucket 2: needs your attention (recalibration won't auto-fix) ─────────
    const missingConcepts = (gaps.manual || []).filter(g => !g.malformed);
    const malformed = (gaps.manual || []).filter(g => g.malformed);
    const issues = data.issues || [];
    html += `<div class="health-breakdown" style="margin-top:16px"><div class="lint-section-title">Needs your attention</div>`
          + note("Recalibration won't auto-fix these — use AI Writer / AI Edit, or edit the page.");
    if (missingConcepts.length) {
      html += subhead(`Missing concept pages — single reference (${missingConcepts.length})`)
            + note('Create with AI Writer if the concept deserves a page, or remove the dangling link.')
            + gapList(missingConcepts, g => `from ${escHtml(g.referenced_by[0] || '')}`);
    }
    if (malformed.length) {
      html += subhead(`Malformed links (${malformed.length})`)
            + note('These look like garbled / oversized link targets — fix via AI Edit on the source page.')
            + gapList(malformed, g => `in ${escHtml(g.referenced_by[0] || '')}`);
    }
    if (issues.length) {
      html += subhead(`AI review — observations & suggestions (${issues.length})`);
      for (const issue of issues) {
        html += `<div class="issue-card">
            <div class="issue-type ${escHtml(issue.type || '')}">${escHtml((issue.type || '').replace(/_/g, ' '))}</div>
            <div class="issue-desc">${escHtml(issue.description || '')}</div>
            ${issue.affected_pages?.length ? `<div class="issue-pages">${escHtml(issue.affected_pages.join(', '))}</div>` : ''}
          </div>`;
      }
    }
    if (!missingConcepts.length && !malformed.length && !issues.length) {
      html += `<p style="color:var(--success);font-size:13px;padding:4px 0">Nothing needs manual attention &#10003;</p>`;
    }
    html += `</div>`;

    if (data.suggestions && data.suggestions.length) {
      html += `<div class="lint-section-title" style="margin-top:16px">Overall suggestions</div><ul class="suggestions-list">`;
      for (const s of data.suggestions) html += `<li>${escHtml(s)}</li>`;
      html += '</ul>';
    }

    result.innerHTML = html;
  } catch (e) {
    result.innerHTML = `<p style="color:var(--danger)">${escHtml(e.message)}</p>`;
  }
}

// ── Search ─────────────────────────────────────────────────────────────────
let searchTimer;
function initSearch() {
  const input = document.getElementById('search-input');
  const dropdown = document.getElementById('search-results');

  input.addEventListener('input', () => {
    clearTimeout(searchTimer);
    const q = input.value.trim();
    if (!q) { dropdown.classList.add('hidden'); return; }
    searchTimer = setTimeout(() => doSearch(q), 300);
  });

  document.addEventListener('click', e => {
    if (!e.target.closest('.header-search')) dropdown.classList.add('hidden');
  });

  document.addEventListener('keydown', e => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
      e.preventDefault();
      input.focus();
      input.select();
    }
  });
}

async function doSearch(q) {
  const dropdown = document.getElementById('search-results');
  dropdown.classList.remove('hidden');
  dropdown.innerHTML = '<div class="search-empty">Searching…</div>';
  try {
    const results = await api('GET', `/api/wiki/search?q=${encodeURIComponent(q)}`);
    if (!results.length) {
      dropdown.innerHTML = '<div class="search-empty">No results found</div>';
      return;
    }
    dropdown.innerHTML = results.slice(0, 8).map(r => `
      <div class="search-result" data-path="${escHtml(r.path)}">
        <div class="sr-path">${escHtml(r.path)}</div>
        <div class="sr-snippet">${escHtml(r.snippet)}</div>
      </div>
    `).join('');
    dropdown.querySelectorAll('.search-result').forEach(el => {
      el.addEventListener('click', () => {
        loadWikiPage(el.dataset.path);
        document.querySelector('.tab[data-tab="wiki"]').click();
        dropdown.classList.add('hidden');
        document.getElementById('search-input').value = '';
      });
    });
  } catch {
    dropdown.classList.add('hidden');
  }
}

// ── Modal helpers ──────────────────────────────────────────────────────────
function openModal(id) { document.getElementById(id).classList.remove('hidden'); }
function closeModal(id) {
  document.getElementById(id).classList.add('hidden');
  // Reset graph overlay error state (but keep it hidden if graph is already loaded)
  if (id === 'modal-graph') {
    const msg = document.getElementById('graph-overlay-msg');
    if (msg) { msg.className = ''; msg.textContent = 'Building graph…'; }
    const ov = document.getElementById('graph-overlay');
    if (ov) ov.classList.remove('transparent');
  }
  // Closing the shared dialog via backdrop/Esc/Cancel resolves it as cancelled.
  if (id === 'modal-dialog') _appDlgSettle(_appDlgMode === 'prompt' ? null : false);
  // Abort an in-flight AI-edit stream so closing the modal stops the server-side
  // generation (and its Bedrock token spend) instead of leaving it running.
  if (id === 'modal-ai-edit' && _aiEdit && _aiEdit.controller) {
    try { _aiEdit.controller.abort(); } catch {}
    _aiEdit.controller = null;
  }
}

function initModals() {
  document.querySelectorAll('[data-close]').forEach(btn => {
    btn.addEventListener('click', () => closeModal(btn.dataset.close));
  });
  document.querySelectorAll('.modal').forEach(modal => {
    modal.querySelector('.modal-backdrop')?.addEventListener('click', () => closeModal(modal.id));
  });
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape') document.querySelectorAll('.modal:not(.hidden)').forEach(m => closeModal(m.id));
  });
  // Wire the reusable dialog's buttons once.
  document.getElementById('app-dlg-confirm')?.addEventListener('click', _appDlgConfirm);
  document.getElementById('app-dlg-cancel')?.addEventListener('click', () => closeModal('modal-dialog'));
  document.getElementById('app-dlg-input')?.addEventListener('keydown', e => {
    if (e.key === 'Enter') { e.preventDefault(); _appDlgConfirm(); }
  });
}

// ── In-app dialogs (replace native confirm/prompt) ──────────────────────────
// `await uiConfirm({...})` → bool; `await uiPrompt({...})` → string|null.
let _appDlgResolve = null;
let _appDlgMode = 'confirm';

function _appDlgSettle(value) {
  const r = _appDlgResolve;
  _appDlgResolve = null;
  if (r) r(value);
}

function uiConfirm({ title = 'Confirm', message = '', confirmText = 'Confirm', danger = false } = {}) {
  return new Promise(resolve => {
    _appDlgSettle(_appDlgMode === 'prompt' ? null : false);
    _appDlgResolve = resolve; _appDlgMode = 'confirm';
    document.getElementById('app-dlg-title').textContent = title;
    document.getElementById('app-dlg-message').textContent = message;
    document.getElementById('app-dlg-input-wrap').classList.add('hidden');
    const c = document.getElementById('app-dlg-confirm');
    c.textContent = confirmText;
    c.className = 'btn btn-sm ' + (danger ? 'btn-danger' : 'btn-primary');
    openModal('modal-dialog'); c.focus();
  });
}

function uiPrompt({ title = '', message = '', label = '', defaultValue = '', confirmText = 'OK' } = {}) {
  return new Promise(resolve => {
    _appDlgSettle(_appDlgMode === 'prompt' ? null : false);
    _appDlgResolve = resolve; _appDlgMode = 'prompt';
    document.getElementById('app-dlg-title').textContent = title;
    document.getElementById('app-dlg-message').textContent = message;
    document.getElementById('app-dlg-input-wrap').classList.remove('hidden');
    document.getElementById('app-dlg-input-label').textContent = label;
    const input = document.getElementById('app-dlg-input');
    input.value = defaultValue;
    const c = document.getElementById('app-dlg-confirm');
    c.textContent = confirmText; c.className = 'btn btn-sm btn-primary';
    openModal('modal-dialog'); input.focus(); input.select();
  });
}

function _appDlgConfirm() {
  const value = _appDlgMode === 'prompt'
    ? document.getElementById('app-dlg-input').value
    : true;
  _appDlgSettle(value);            // resolve first so closeModal's cancel is a no-op
  closeModal('modal-dialog');
}

// ── Tabs ───────────────────────────────────────────────────────────────────
function initTabs() {
  document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      tab.classList.add('active');
      document.getElementById('tab-' + tab.dataset.tab).classList.add('active');
    });
  });
}

// ── Chat panel resize ──────────────────────────────────────────────────────
function initChatResize() {
  const handle = document.getElementById('chat-resize-handle');
  const panel  = document.getElementById('chat-panel');
  const MIN_W  = 260;
  const MAX_W  = 560;

  handle.addEventListener('mousedown', e => {
    e.preventDefault();
    handle.classList.add('dragging');
    const startX     = e.clientX;
    const startWidth = panel.getBoundingClientRect().width;

    function onMove(e) {
      const delta    = startX - e.clientX;           // dragging left = wider
      const newWidth = Math.min(MAX_W, Math.max(MIN_W, startWidth + delta));
      panel.style.width = newWidth + 'px';
    }

    function onUp() {
      handle.classList.remove('dragging');
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup', onUp);
    }

    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
  });
}

// ── Recalibrate ────────────────────────────────────────────────────────────
let recalibratePollingId = null;

function triggerRecalibrate() {
  openModal('modal-recalibrate-confirm');
}

function triggerExport() {
  const embLabel = document.getElementById('export-emb-label');
  const embCheckbox = document.getElementById('export-emb-checkbox');
  if (embLabel) {
    if (state.embeddingEnabled) {
      embLabel.classList.remove('hidden');
      embLabel.style.display = 'flex';
      embCheckbox.checked = true;
    } else {
      embLabel.classList.add('hidden');
      embLabel.style.display = 'none';
    }
  }
  openModal('modal-export-confirm');
}

async function confirmExport() {
  const embCheckbox = document.getElementById('export-emb-checkbox');
  const includeEmbeddings = state.embeddingEnabled && embCheckbox && embCheckbox.checked;

  closeModal('modal-export-confirm');

  const btn = document.getElementById('btn-export');
  const origHTML = btn.innerHTML;
  btn.disabled = true;
  btn.innerHTML = btn.innerHTML.replace('Export', 'Exporting…');

  try {
    const url = `/api/ops/export?include_embeddings=${includeEmbeddings}`;
    const headers = {};
    if (state.authToken) headers['Authorization'] = `Bearer ${state.authToken}`;
    if (state.activeOrgId) headers['X-Org-Context'] = state.activeOrgId;

    const res = await fetch(url, { headers });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || res.statusText);
    }

    const blob = await res.blob();
    const disposition = res.headers.get('Content-Disposition') || '';
    const nameMatch = disposition.match(/filename=([^\s;]+)/);
    const filename = nameMatch ? nameMatch[1] : 'wiki-export.zip';

    const blobUrl = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = blobUrl;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(blobUrl);

    showToast('Export ready — downloading…', 'success');
  } catch (err) {
    showToast(`Export failed: ${err.message}`, 'error');
  } finally {
    btn.disabled = false;
    btn.innerHTML = origHTML;
  }
}

async function confirmRecalibrate() {
  const factInstructions = (document.getElementById('recal-fact-instructions')?.value || '').trim();
  closeModal('modal-recalibrate-confirm');

  try {
    const body = factInstructions ? { fact_instructions: factInstructions } : {};
    await api('POST', '/api/ops/recalibrate', body);
  } catch (e) {
    if (!e.message.includes('already running')) {
      showToast(`Failed to start recalibration: ${e.message}`, 'error');
      return;
    }
  }

  if (factInstructions) {
    const titleEl = document.getElementById('recal-title');
    const subtitleEl = document.getElementById('recal-subtitle');
    if (titleEl) titleEl.textContent = 'Targeted Recalibration';
    if (subtitleEl) subtitleEl.textContent = 'AI is fixing the reported fact(s) in the knowledge base';
  }
  showRecalibrateOverlay();
  startRecalibratePolling();
}

function clearRecalibrateUI() {
  document.getElementById('recal-stage').textContent = 'Starting…';
  document.getElementById('recal-progress-fill').style.width = '0%';
  document.getElementById('recal-pct').textContent = '0%';
  document.getElementById('recal-details').textContent = '';

  const titleEl = document.getElementById('recal-title');
  const subtitleEl = document.getElementById('recal-subtitle');
  if (titleEl) titleEl.textContent = 'Master Recalibration';
  if (subtitleEl) subtitleEl.textContent = 'AI is reorganizing and improving the knowledge base';

  const badge = document.getElementById('recal-targeted-badge');
  if (badge) { badge.textContent = ''; badge.classList.add('hidden'); }

  const textarea = document.getElementById('recal-fact-instructions');
  if (textarea) textarea.value = '';
  const confirmLabel = document.getElementById('recal-confirm-label');
  if (confirmLabel) confirmLabel.textContent = 'Start Recalibration';

  ['recal-pages', 'recal-deleted', 'recal-renamed', 'recal-errors'].forEach(id => {
    const section = document.getElementById(id);
    if (section) section.classList.add('hidden');
  });
  ['recal-pages-list', 'recal-deleted-list', 'recal-renamed-list', 'recal-errors-list'].forEach(id => {
    const list = document.getElementById(id);
    if (list) list.innerHTML = '';
  });
}

function showRecalibrateOverlay() {
  clearRecalibrateUI();
  document.getElementById('overlay-recalibrate').classList.remove('hidden');
  document.getElementById('recal-actions').classList.add('hidden');
  document.getElementById('recal-icon').classList.add('spinning');
}

function hideRecalibrateOverlay() {
  document.getElementById('overlay-recalibrate').classList.add('hidden');
  document.getElementById('recal-icon').classList.remove('spinning');
  stopRecalibratePolling();
}

function startRecalibratePolling() {
  if (recalibratePollingId) return;
  pollRecalibrateStatus(); // immediate first check
  recalibratePollingId = setInterval(pollRecalibrateStatus, 3000);
}

function stopRecalibratePolling() {
  clearInterval(recalibratePollingId);
  recalibratePollingId = null;
}

async function pollRecalibrateStatus() {
  try {
    const job = await api('GET', '/api/ops/recalibrate/status');
    updateRecalibrateUI(job);
    const terminal = job.status === 'done' || job.status === 'done_with_errors' || job.status === 'error';
    if (terminal) {
      stopRecalibratePolling();
      document.getElementById('recal-icon').classList.remove('spinning');
      const btn = document.getElementById('btn-recal-dismiss');
      btn.textContent = job.status === 'error' ? 'Dismiss' : 'Done';
      document.getElementById('recal-actions').classList.remove('hidden');
      if (job.status !== 'error') {
        await loadWikiTree();
        showToast(`Recalibration complete — ${job.details}`, job.status === 'done' ? 'success' : 'info');
      } else {
        showToast(`Recalibration failed: ${job.details}`, 'error');
      }
    }
  } catch {
    // network error while recalibrating — keep polling
  }
}

function updateRecalibrateUI(job) {
  document.getElementById('recal-stage').textContent = job.stage || 'Running…';
  document.getElementById('recal-progress-fill').style.width = `${job.progress}%`;
  document.getElementById('recal-pct').textContent = `${job.progress}%`;
  document.getElementById('recal-details').textContent = job.details || '';

  if (job.fact_instructions) {
    const badge = document.getElementById('recal-targeted-badge');
    if (badge) {
      badge.textContent = `Targeted fix: ${job.fact_instructions}`;
      badge.classList.remove('hidden');
    }
  }

  if (job.pages_improved && job.pages_improved.length) {
    const section = document.getElementById('recal-pages');
    section.classList.remove('hidden');
    const list = document.getElementById('recal-pages-list');
    const existing = new Set(Array.from(list.querySelectorAll('li')).map(li => li.textContent));
    for (const path of job.pages_improved) {
      if (!existing.has(path)) {
        const li = document.createElement('li');
        li.textContent = path;
        list.appendChild(li);
      }
    }
    list.scrollTop = list.scrollHeight;
  }

  if (job.pages_deleted && job.pages_deleted.length) {
    const section = document.getElementById('recal-deleted');
    section.classList.remove('hidden');
    const list = document.getElementById('recal-deleted-list');
    const existing = new Set(Array.from(list.querySelectorAll('li')).map(li => li.textContent));
    for (const path of job.pages_deleted) {
      if (!existing.has(path)) {
        const li = document.createElement('li');
        li.textContent = path;
        list.appendChild(li);
      }
    }
  }

  if (job.pages_renamed && job.pages_renamed.length) {
    const section = document.getElementById('recal-renamed');
    section.classList.remove('hidden');
    const list = document.getElementById('recal-renamed-list');
    const existing = new Set(Array.from(list.querySelectorAll('li')).map(li => li.textContent));
    for (const r of job.pages_renamed) {
      const label = `${r.from} → ${r.to}`;
      if (!existing.has(label)) {
        const li = document.createElement('li');
        li.textContent = label;
        list.appendChild(li);
      }
    }
  }

  if (job.errors && job.errors.length) {
    document.getElementById('recal-errors').classList.remove('hidden');
    document.getElementById('recal-errors-list').innerHTML = job.errors.map(e => `<li>${escHtml(e)}</li>`).join('');
  }
}

async function checkRecalibrateOnStartup() {
  try {
    const job = await api('GET', '/api/ops/recalibrate/status');
    if (job.status === 'running') {
      showRecalibrateOverlay();
      updateRecalibrateUI(job);
      startRecalibratePolling();
    }
  } catch {
    // ignore startup errors
  }
}

// ── Sidebar collapse ───────────────────────────────────────────────────────
function initSidebarCollapse() {
  const sidebar = document.getElementById('sidebar');
  const collapseBtn = document.getElementById('btn-sidebar-collapse');
  const expandTab = document.getElementById('btn-sidebar-expand');

  function collapse() {
    sidebar.classList.add('collapsed');
    expandTab.classList.remove('hidden');
    collapseBtn.title = 'Expand sidebar';
    collapseBtn.setAttribute('aria-label', 'Expand sidebar');
  }

  function expand() {
    sidebar.classList.remove('collapsed');
    expandTab.classList.add('hidden');
    collapseBtn.title = 'Collapse sidebar';
    collapseBtn.setAttribute('aria-label', 'Collapse sidebar');
  }

  collapseBtn.addEventListener('click', collapse);
  expandTab.addEventListener('click', expand);
}

// ── Wiki-link click handler (delegated) ───────────────────────────────────
function initWikiLinks() {
  document.addEventListener('click', async e => {
    // ── [[Wiki link]] spans ───────────────────────────────────────────
    const wikiSpan = e.target.closest('.wiki-link');
    if (wikiSpan) {
      e.preventDefault();
      const query = wikiSpan.dataset.query;
      if (!query) return;
      try {
        const results = await api('GET', `/api/wiki/search?q=${encodeURIComponent(query)}`);
        if (results && results.length) {
          loadWikiPage(results[0].path);
          document.querySelector('.tab[data-tab="wiki"]').click();
        } else {
          showToast(`No wiki page found for "${query}"`, 'info');
        }
      } catch (err) {
        showToast(`Could not resolve wiki link: ${err.message}`, 'error');
      }
      return;
    }

    // ── Markdown <a> links inside rendered page content ───────────────
    const anchor = e.target.closest('#page-content a, #chat-messages a');
    if (!anchor) return;
    const href = anchor.getAttribute('href');
    if (!href) return;

    // Intercept relative .md links — treat them as internal wiki page paths
    if (!href.startsWith('http') && !href.startsWith('#') && !href.startsWith('mailto')) {
      e.preventDefault();
      // Resolve relative path against current page's directory
      let path;
      if (href.startsWith('/')) {
        path = href.replace(/^\//, '').split('?')[0].split('#')[0];
      } else {
        // Use URL resolution to handle ../ correctly
        const base = 'wiki/' + (state.activePage || '');
        const baseDir = base.substring(0, base.lastIndexOf('/') + 1);
        const resolved = new URL(href, 'http://x/' + baseDir).pathname;
        path = resolved.replace(/^\//, '').replace(/^wiki\//, '').split('?')[0].split('#')[0];
      }
      if (path.endsWith('.md')) {
        loadWikiPage(path);
        document.querySelector('.tab[data-tab="wiki"]').click();
      }
    }
  });
}

// ── Knowledge Graph ────────────────────────────────────────────────────────

function _graphOverlay(msg, { error = false, transparent = false } = {}) {
  const el = document.getElementById('graph-overlay');
  const msgEl = document.getElementById('graph-overlay-msg');
  el.classList.remove('hidden', 'transparent');
  if (transparent) el.classList.add('transparent');
  if (error) {
    msgEl.className = 'graph-overlay-error';
    msgEl.textContent = msg;
  } else {
    msgEl.className = '';
    msgEl.textContent = msg;
  }
}

function _graphOverlayHide() {
  document.getElementById('graph-overlay').classList.add('hidden');
}

async function openGraphModal(rebuild = false) {
  openModal('modal-graph');
  const frame    = document.getElementById('graph-frame');
  const subtitle = document.getElementById('graph-subtitle');

  // If we already have a graph loaded, use a translucent overlay while rebuilding
  // so the user can see it's updating rather than getting a blank screen
  const hasGraph = frame.srcdoc && frame.srcdoc.includes('vis.js');
  if (rebuild && hasGraph) {
    _graphOverlay('Rebuilding graph…', { transparent: true });
  } else {
    _graphOverlay('Building graph…');
    subtitle.textContent = '';
  }

  try {
    if (rebuild) await api('POST', '/api/ops/graph/rebuild');

    const [meta, htmlText] = await Promise.all([
      api('GET', '/api/ops/graph'),
      fetch('/api/ops/graph/html', {
        headers: {
          ...(state.authToken ? { 'Authorization': `Bearer ${state.authToken}` } : {}),
          ...(state.activeOrgId ? { 'X-Org-Context': state.activeOrgId } : {}),
        },
      }).then(r => {
        if (!r.ok) throw new Error(`Graph render failed (${r.status})`);
        return r.text();
      }),
    ]);

    if (!meta.nodes?.length) {
      _graphOverlay('No wiki pages yet — ingest a document first.');
      subtitle.textContent = '';
      return;
    }

    subtitle.textContent = `${meta.nodes.length} pages · ${meta.edges.length} links`;
    frame.srcdoc = htmlText;
    _graphOverlayHide();
  } catch (e) {
    _graphOverlay(e.message, { error: true, transparent: hasGraph });
  }
}

// Node clicks inside the iframe bubble up via postMessage
window.addEventListener('message', e => {
  if (e.data && e.data.type === 'graphNavigate' && e.data.path) {
    closeModal('modal-graph');
    loadWikiPage(e.data.path);
    document.querySelector('.tab[data-tab="wiki"]').click();
  }
});

// ── Auth ───────────────────────────────────────────────────────────────────
function showLoginScreen() {
  document.getElementById('login-screen').classList.remove('hidden');
  document.getElementById('app-root').classList.add('hidden');
  state.authToken = null;
  state.refreshToken = null;
  state.userRole = null;
  state.activeOrgId = null;
  state.memberships = [];
  state.userEmail = null;
  state.userOrgName = null;
  state.chatSessionId = null;
  localStorage.removeItem('auth_token');
  localStorage.removeItem('refresh_token');
  localStorage.removeItem('user_role');
  localStorage.removeItem('admin_org_id');
  localStorage.removeItem('active_org_id');
  localStorage.removeItem('memberships');
  localStorage.removeItem('user_email');
  localStorage.removeItem('user_org_name');
  localStorage.removeItem('chat_session_id');
}

function showApp() {
  document.getElementById('login-screen').classList.add('hidden');
  document.getElementById('app-root').classList.remove('hidden');
}

// ── Writer Mode ────────────────────────────────────────────────────────────

async function openWriterPicker() {
  const allowed = state.userRole === 'admin' || state.userRole === 'supervisor' || !!state.perms.can_use_writer;
  if (!allowed) return;
  openModal('modal-writer-picker');
  const listEl = document.getElementById('writer-draft-list');
  listEl.innerHTML = '<div class="content-loading"><span class="spinner spinner-dark"></span> Loading drafts…</div>';
  try {
    const data = await api('GET', '/api/ops/writer/sessions');
    const sessions = data.sessions || [];
    if (!sessions.length) {
      listEl.innerHTML = '<div class="writer-picker-empty">You haven’t started any drafts yet.</div>';
      return;
    }
    listEl.innerHTML = '';
    for (const s of sessions) {
      const row = document.createElement('div');
      row.className = 'writer-draft-row';
      row.innerHTML = `
        <div class="writer-draft-row-main">
          <div class="writer-draft-row-name">${escHtml(s.draft_filename || 'Untitled draft')}</div>
          <div class="writer-draft-row-meta">Last edited: ${s.last_active_at ? new Date(s.last_active_at).toLocaleString() : 'unknown'}</div>
          <div class="writer-draft-row-excerpt">${escHtml(s.excerpt || '(no draft yet)')}</div>
        </div>
        <div class="writer-draft-row-actions">
          <button class="btn btn-secondary btn-sm" data-resume="${escHtml(s.session_id)}">Resume</button>
          <button class="btn btn-ghost btn-sm" data-delete="${escHtml(s.session_id)}">Delete</button>
        </div>`;
      listEl.appendChild(row);
    }
    listEl.querySelectorAll('[data-resume]').forEach(btn => {
      btn.addEventListener('click', () => resumeWriterDraft(btn.dataset.resume));
    });
    listEl.querySelectorAll('[data-delete]').forEach(btn => {
      btn.addEventListener('click', () => deleteWriterDraft(btn.dataset.delete));
    });
  } catch (e) {
    listEl.innerHTML = `<div style="color:var(--danger);padding:8px 0">Failed to load drafts: ${escHtml(e.message)}</div>`;
  }
}

async function deleteWriterDraft(sessionId) {
  if (!(await uiConfirm({
    title: 'Delete draft', message: 'Delete this draft? This cannot be undone.',
    confirmText: 'Delete', danger: true,
  }))) return;
  try {
    await api('DELETE', `/api/ops/writer/${encodeURIComponent(sessionId)}`);
    await openWriterPicker();   // refresh list
  } catch (e) {
    showToast(`Failed to delete: ${e.message}`, 'error');
  }
}

async function resumeWriterDraft(sessionId) {
  state.writerSessionId = sessionId;
  let history = [];
  try {
    const [draft, hist] = await Promise.all([
      api('GET', `/api/ops/writer/${encodeURIComponent(sessionId)}/draft`),
      api('GET', `/api/ops/chat/${encodeURIComponent(sessionId)}/history`).catch(() => ({ messages: [] })),
    ]);
    state.writerDraft = draft.draft_content || '';
    state.writerDraftReady = !!draft.draft_ready;
    state.writerFilename = draft.draft_filename || '';
    history = hist.messages || [];
  } catch (e) {
    showToast(`Failed to load draft: ${e.message}`, 'error');
    return;
  }
  enterWriterView(history);
}

function startNewWriterDraft() {
  state.writerSessionId = null;
  state.writerDraft = '';
  state.writerDraftReady = false;
  state.writerFilename = '';
  enterWriterView([]);
}

function enterWriterView(history = []) {
  closeModal('modal-writer-picker');
  state.writerMode = true;
  document.getElementById('welcome-screen')?.classList.add('hidden');
  document.getElementById('page-view')?.classList.add('hidden');
  document.getElementById('writer-view').classList.remove('hidden');
  document.getElementById('writer-mode-badge').classList.remove('hidden');
  const titleEl = document.getElementById('chat-panel-title-text');
  if (titleEl) titleEl.textContent = 'Writer Conversation';
  const qInput = document.getElementById('query-input');
  if (qInput) qInput.placeholder = 'Describe what you want to document…';
  // Replay any prior conversation so the user sees their context, not a blank panel.
  renderChatHistory(history);

  document.getElementById('writer-filename').value = state.writerFilename;
  renderDraftPreview();
}

async function exitWriterView(force = false) {
  if (!force && state.writerDraft && !(await uiConfirm({
    title: 'Exit writer mode', message: 'Your draft is saved. Exit writer mode?', confirmText: 'Exit',
  }))) return;
  state.writerMode = false;
  state.writerSessionId = null;
  state.writerDraft = '';
  state.writerDraftReady = false;
  state.writerFilename = '';
  document.getElementById('writer-view').classList.add('hidden');
  document.getElementById('writer-mode-badge').classList.add('hidden');
  const titleEl = document.getElementById('chat-panel-title-text');
  if (titleEl) titleEl.textContent = 'Ask the Knowledge Base';
  const qInput = document.getElementById('query-input');
  if (qInput) qInput.placeholder = '';
  document.getElementById('chat-messages').innerHTML = '';
  // Restore prior view
  if (state.activePage) {
    showPageView();
  } else {
    document.getElementById('welcome-screen')?.classList.remove('hidden');
  }
}

function renderDraftPreview() {
  const el = document.getElementById('writer-draft-preview');
  const btn = document.getElementById('btn-writer-ingest');
  const hint = document.getElementById('writer-ingest-hint');

  if (!state.writerDraft) {
    el.innerHTML = '<div class="writer-draft-placeholder">Start chatting on the right — the agent will draft the page here.</div>';
    btn.disabled = true;
    if (hint) hint.textContent = '';
    return;
  }
  el.innerHTML = renderMd(state.writerDraft);
  // Hard gate: ingest is allowed only after the writer agent declares the
  // draft ready via [DRAFT_READY]. Any new draft or section patch in a later
  // turn clears this flag — the agent must re-confirm.
  btn.disabled = !state.writerDraftReady;
  if (hint) {
    hint.textContent = state.writerDraftReady
      ? ''
      : 'The writer agent has not marked this draft as ready. Continue the conversation until it confirms.';
  }
}

async function doWriterChat() {
  const input = document.getElementById('query-input');
  const message = input.value.trim();
  if (!message) return;
  input.value = '';
  input.style.height = '';

  const btn = document.getElementById('btn-ask');
  appendChatBubble('user', message);
  const loadingBubble = appendLoadingBubble();
  btn.disabled = true;
  input.disabled = true;

  // Track this fetch so signout / window unload can abort it server-side.
  // Cancelling the AbortController closes the SSE connection, which causes
  // FastAPI to raise CancelledError in the generator and unwind the Bedrock
  // streaming call.
  const controller = new AbortController();
  _activeStreamControllers.add(controller);

  const opts = {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    signal: controller.signal,
  };
  if (state.authToken) opts.headers['Authorization'] = `Bearer ${state.authToken}`;
  if (state.activeOrgId) opts.headers['X-Org-Context'] = state.activeOrgId;
  opts.body = JSON.stringify({ session_id: state.writerSessionId, message });

  try {
    const res = await streamFetch('/api/ops/writer/chat/stream', opts);
    if (res.status === 401) { showLoginScreen(); return; }
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      const msg = err.detail || res.statusText;
      if (res.status === 429) throw Object.assign(new Error(msg), { isQuota: true });
      throw new Error(msg);
    }

    const msgEl = loadingBubble;
    const bubble = msgEl.querySelector('.chat-bubble');
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let chatText = '';
    let firstChunk = true;
    let lastSectionHeading = null;
    let pendingErrors = [];

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop();
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        let data;
        try { data = JSON.parse(line.slice(6)); } catch { continue; }
        if (data.type === 'meta') {
          state.writerSessionId = data.session_id;
        } else if (data.type === 'chunk') {
          if (firstChunk) {
            clearInterval(msgEl._thinkingTimer);
            bubble.classList.remove('chat-bubble-loading');
            bubble.innerHTML = '';
            firstChunk = false;
          }
          chatText += data.text;
          bubble.innerHTML = renderMd(chatText);
          document.getElementById('chat-messages').scrollTop = document.getElementById('chat-messages').scrollHeight;
        } else if (data.type === 'draft_chunk') {
          // Streaming-in full draft — update live preview
          state.writerDraft = (state.writerDraft && lastSectionHeading === null ? '' : state.writerDraft);
          // We accumulate the streamed full-draft body as it comes in
          if (!msgEl._draftBuffer) msgEl._draftBuffer = '';
          msgEl._draftBuffer += data.text;
          state.writerDraft = msgEl._draftBuffer;
          // New content cancels any prior readiness — agent must re-emit
          // [DRAFT_READY] in this turn (or a later one) to re-enable ingest.
          state.writerDraftReady = false;
          renderDraftPreview();
        } else if (data.type === 'section_chunk') {
          lastSectionHeading = data.heading;
          // A section patch is also "new content" — clear readiness.
          state.writerDraftReady = false;
          // Don't render section patches live (we'll refetch after done)
        } else if (data.type === 'draft_ready') {
          state.writerDraftReady = true;
          renderDraftPreview();
        } else if (data.type === 'error') {
          pendingErrors.push(data.message || 'Section patch failed');
        } else if (data.type === 'done') {
          // Server-side authoritative readiness — overrides anything we inferred.
          if (typeof data.draft_ready === 'boolean') {
            state.writerDraftReady = data.draft_ready;
          }
          if (data.section_heading) {
            // Refetch full draft after server applied section patch
            try {
              const d = await api('GET', `/api/ops/writer/${encodeURIComponent(state.writerSessionId)}/draft`);
              state.writerDraft = d.draft_content || state.writerDraft;
              state.writerFilename = d.draft_filename || state.writerFilename;
              document.getElementById('writer-filename').value = state.writerFilename;
            } catch { /* keep what we have */ }
          }
          renderDraftPreview();
        }
      }
    }

    if (firstChunk) {
      clearInterval(msgEl._thinkingTimer);
      bubble.classList.remove('chat-bubble-loading');
      bubble.innerHTML = chatText ? renderMd(chatText) : '<em>(agent updated the draft)</em>';
    }
    for (const err of pendingErrors) {
      const note = document.createElement('div');
      note.className = 'chat-saved-notice';
      note.style.color = 'var(--warning, #d97706)';
      note.textContent = err;
      msgEl.appendChild(note);
    }
  } catch (e) {
    if (loadingBubble) {
      clearInterval(loadingBubble._thinkingTimer);
      loadingBubble.remove();
    }
    if (e.isQuota) {
      appendQuotaBubble(e.message);
    } else if (e.name === 'AbortError') {
      // User signed out or navigated away — quiet exit, no error bubble.
      if (loadingBubble) loadingBubble.remove();
    } else {
      appendChatBubble('assistant', `Error: ${e.message}`, [], null, true);
    }
  } finally {
    _activeStreamControllers.delete(controller);
    btn.disabled = false;
    input.disabled = false;
    input.focus();
  }
}

let _writerFilenameTimer = null;
function bindWriterFilenameAutosave() {
  const el = document.getElementById('writer-filename');
  el.addEventListener('input', () => {
    state.writerFilename = el.value;
    if (_writerFilenameTimer) clearTimeout(_writerFilenameTimer);
    _writerFilenameTimer = setTimeout(async () => {
      if (!state.writerSessionId) return;
      const name = el.value.trim();
      if (!name || !name.toLowerCase().endsWith('.md')) return;
      try {
        await api('PUT', `/api/ops/writer/${encodeURIComponent(state.writerSessionId)}/draft/filename`, { filename: name });
      } catch { /* ignore transient validation errors as the user types */ }
    }, 500);
  });
}

function doWriterIngest() {
  const name = (document.getElementById('writer-filename').value || '').trim();
  if (!name) { showToast('Enter a filename first (e.g. my-page.md).', 'error'); return; }
  if (!state.writerSessionId) { showToast('No active draft.', 'error'); return; }
  const nameEl = document.getElementById('writer-ingest-confirm-name');
  if (nameEl) nameEl.textContent = name;
  openModal('modal-writer-ingest-confirm');
}

async function confirmWriterIngest() {
  const name = (document.getElementById('writer-filename').value || '').trim();
  if (!name || !state.writerSessionId) {
    closeModal('modal-writer-ingest-confirm');
    return;
  }
  const sessionId = state.writerSessionId;
  closeModal('modal-writer-ingest-confirm');

  // Register the upload notification BEFORE the API call so the user sees the
  // same progress card / step indicators as a regular upload (and any failure
  // is reported through notifSetError, not a blocking alert).
  notifAddUpload(name);
  // Exit writer view immediately (force — the user already chose to ingest, so
  // don't re-prompt). The rest of the flow is observable via the notification
  // card and the Sources tab, just like a regular upload.
  exitWriterView(true);

  try {
    const res = await api('POST', `/api/ops/writer/${encodeURIComponent(sessionId)}/ingest`, { filename: name });
    // Remember which writer draft to delete once this ingest finishes.
    state.writerIngestSessions[res.filename] = sessionId;
    notifSetStep(res.filename, 1); // Planning
    await loadSourcesList?.();
    // Re-use the existing ingest poller so the plan-review notification
    // is surfaced exactly as it is for normal uploads.
    if (typeof startPolling === 'function') startPolling(res.filename);
  } catch (e) {
    // api() has already raised a system "Limit reached" notif for 429s; this
    // sets the inline error on the upload card.
    notifSetError(name, e.isQuota ? 'Limit reached' : e.message);
  }
}

function initWriterMode() {
  document.getElementById('btn-writer')?.addEventListener('click', openWriterPicker);
  document.getElementById('btn-writer-new')?.addEventListener('click', startNewWriterDraft);
  document.getElementById('btn-writer-exit')?.addEventListener('click', () => exitWriterView());
  document.getElementById('btn-writer-ingest')?.addEventListener('click', doWriterIngest);
  document.getElementById('btn-writer-ingest-cancel')?.addEventListener('click', () => closeModal('modal-writer-ingest-confirm'));
  document.getElementById('btn-writer-ingest-confirm')?.addEventListener('click', confirmWriterIngest);
  bindWriterFilenameAutosave();
}

// ── Boot ───────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
  // Inject company name and feature flags from server config
  try {
    const cfg = await fetch('/api/auth/config').then(r => r.json());
    const name = cfg.company_name || '';
    document.querySelectorAll('.brand-mark, .welcome-mark').forEach(el => { el.textContent = name; });
    if (name) document.title = `AI Knowledge Hub \u00b7 ${name}`;
    state.streamEnabled = cfg.chat_stream !== false;
    state.embeddingEnabled = !!cfg.embedding_enabled;
    if (cfg.max_upload_size_mb) {
      state.maxUploadMB = cfg.max_upload_size_mb;
      const hint = document.querySelector('.drop-hint');
      if (hint) hint.textContent = `Supported: PDF, DOCX, TXT, MD · Max ${cfg.max_upload_size_mb} MB per file`;
    }
  } catch { /* non-fatal; branding elements stay empty */ }

  // Login form handler
  document.getElementById('login-form').addEventListener('submit', async e => {
    e.preventDefault();
    const username = document.getElementById('login-username').value.trim();
    const password = document.getElementById('login-password').value;
    const errEl = document.getElementById('login-error');
    errEl.classList.add('hidden');
    try {
      const res = await fetch('/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password }),
      });
      if (!res.ok) {
        errEl.textContent = 'Invalid username or password.';
        errEl.classList.remove('hidden');
        return;
      }
      const data = await res.json();
      state.authToken = data.access_token;
      state.refreshToken = data.refresh_token;
      state.userRole = data.role || null;
      state.userEmail = data.email || null;
      state.userOrgName = data.org_name || null;
      state.memberships = data.memberships || [];
      // active_org_id is set when there's exactly one membership (or admin
      // picks later via the switcher). With multiple memberships the user
      // chooses below before the app boots.
      state.activeOrgId = data.active_org_id || null;
      localStorage.setItem('auth_token', data.access_token);
      localStorage.setItem('refresh_token', data.refresh_token);
      if (data.role) localStorage.setItem('user_role', data.role);
      if (data.email) localStorage.setItem('user_email', data.email);
      if (data.org_name) localStorage.setItem('user_org_name', data.org_name);
      localStorage.setItem('memberships', JSON.stringify(state.memberships));
      if (state.activeOrgId) localStorage.setItem('active_org_id', state.activeOrgId);
      else localStorage.removeItem('active_org_id');

      // Multiple orgs and none auto-selected → enter the first; the org
      // switcher in the header lets them change at any time.
      if (state.userRole !== 'admin' && state.memberships.length > 1 && !state.activeOrgId) {
        state.activeOrgId = state.memberships[0].org_id;
        localStorage.setItem('active_org_id', state.activeOrgId);
      }
      showApp();
      bootApp();
    } catch {
      errEl.textContent = 'Could not connect to server.';
      errEl.classList.remove('hidden');
    }
  });

  // If we have a stored token, validate it then boot; otherwise show login.
  // Admin users have no org_id so /api/wiki fails before the org context is
  // established — use /api/auth/config (always public) as a connectivity check
  // and rely on the 401 path in api() to catch truly expired tokens.
  if (state.authToken) {
    try {
      // Trigger a real authenticated call; admin/wiki 400s are non-fatal here
      await api('GET', state.userRole === 'admin' ? '/api/admin/me' : '/api/wiki');
      showApp();
      bootApp();
    } catch (err) {
      if (err.message && err.message.includes('Session expired')) {
        showLoginScreen();
      } else {
        // Non-401 (e.g. 400 no org context, 500 on startup) — still boot
        showApp();
        bootApp();
      }
    }
  } else {
    showLoginScreen();
  }
});

function applyRoleVisibility() {
  // Elements whose visibility depends only on role (not on permission flags).
  // Must run for ALL users, including admins/supervisors who skip the
  // perms-based hide pass below.
  const privileged = state.userRole === 'admin' || state.userRole === 'supervisor';
  document.getElementById('btn-writer')?.classList.toggle('hidden', !privileged);
}

async function fetchAndApplyPermissions() {
  applyRoleVisibility();
  // Admins/supervisors have full access — no need to hide anything based on perms.
  if (state.userRole === 'admin' || state.userRole === 'supervisor') return;
  try {
    const perms = await api('GET', '/api/ops/permissions');
    state.perms = perms;
    localStorage.setItem('user_perms', JSON.stringify(perms));
    applyPermissions(perms);
  } catch {
    // Non-fatal; fall back to cached perms if available.
    if (Object.keys(state.perms).length) applyPermissions(state.perms);
  }
}

function applyPermissions(perms) {
  function show(id, visible) {
    const el = document.getElementById(id);
    if (el) el.style.display = visible ? '' : 'none';
  }
  function showEl(el, visible) {
    if (el) el.style.display = visible ? '' : 'none';
  }

  const canUpload             = !!perms.can_upload;
  const canUploadWriterDraft  = !!perms.can_upload_writer_draft;
  const canDownload           = !!perms.can_download_files;
  const canDelete             = !!perms.can_delete_files;
  const canChat               = !!perms.can_chat;
  const canUseWriter          = !!perms.can_use_writer;
  const canLint         = !!perms.can_run_lint;
  const canRecalibrate  = !!perms.can_recalibrate;
  const canGraph        = !!perms.can_view_graph;
  const canRebuildGraph = !!perms.can_rebuild_graph;

  show('btn-upload', canUpload);
  show('btn-welcome-upload', canUpload);
  // Writer button: admins/supervisors get it via applyRoleVisibility(); members
  // need can_use_writer. Toggle the .hidden class so it overrides the role pass.
  document.getElementById('btn-writer')?.classList.toggle('hidden', !canUseWriter);
  // Writer-ingest button — hidden when the user lacks can_upload_writer_draft.
  show('btn-writer-ingest', canUploadWriterDraft);

  // Sources tab: visible if user can upload or download
  const sourcesTab = document.querySelector('.tab[data-tab="sources"]');
  showEl(sourcesTab, canUpload || canDownload);

  // Chat panel (aside) — CSS flex layout collapses naturally when hidden
  show('chat-panel', canChat);

  show('btn-lint', canLint);
  show('btn-recalibrate', canRecalibrate);
  show('btn-export', false);
  show('btn-graph', canGraph);
  show('tools-menu-wrap', canLint || canRecalibrate || canGraph);
  show('btn-graph-rebuild', canRebuildGraph);

  // Delete toolbar button in sources list
  show('btn-delete-sources', canDelete);

  // Hide checkboxes in already-rendered source items
  if (!canDelete) {
    document.querySelectorAll('#sources-list .src-checkbox').forEach(cb => {
      cb.style.display = 'none';
    });
  }
}

async function fetchPersistentNotifications() {
  try {
    const items = await api('GET', '/api/notifications');
    _applyServerNotifications(items || []);
  } catch { /* non-fatal */ }
}

// Merge a batch of server notifications into the panel. Idempotent — dedups by
// db id — so it's safe to call from the initial fetch, the live stream's
// backlog (init), and each pushed batch (new).
function _applyServerNotifications(items) {
  if (!items || !items.length) return;
  {
    const _UPLOAD_TYPE_MAP = {
      pending_review: { status: 'review', step: 2 },
      ingest_done:    { status: 'done',   step: 4 },
      ingest_error:   { status: 'error',  step: 2 },
    };
    let added = 0;
    for (const n of items) {
      // Already showing this exact notification — skip so repeated polling
      // doesn't pile up duplicate cards.
      if (n.id && notifItems.some(i => i.dbId === n.id)) continue;
      const ts = n.created_at ? new Date(n.created_at).toLocaleString() : '';
      if (n.type === 'recalib_done') {
        notifAddSystem({ type: 'recalib', title: n.title, body: n.body, status: 'done', timestamp: ts, dbId: n.id });
        added++;
        continue;
      }
      const mapped = _UPLOAD_TYPE_MAP[n.type];
      if (!mapped || !n.link) continue;
      // A local card for this file already exists (e.g. our own in-flight
      // upload). Adopt the server id so dismissing it marks the row read, and
      // upgrade its status to the server's latest (a slow job may have
      // finished server-side while our local poll lagged).
      if (notifJobs[n.link] !== undefined) {
        const existing = notifItems.find(i => i.id === notifJobs[n.link]);
        if (existing) {
          if (!existing.dbId) existing.dbId = n.id;
          if (existing.status === 'active') { existing.status = mapped.status; existing.step = mapped.step; added++; }
        }
        continue;
      }
      const item = {
        id:     notifNextId++,
        dbId:   n.id,
        type:   'upload',
        name:   n.link,
        status: mapped.status,
        step:   mapped.step,
      };
      notifItems.push(item);
      notifJobs[n.link] = item.id;
      added++;
    }
    if (added) { notifRender(); notifRingBell(); }
  }
}

// ── Live notifications via SSE ───────────────────────────────────────────────
// A single long-lived fetch stream pushes notifications in real time (server
// uses Postgres LISTEN/NOTIFY). The connection's `init` event carries the
// backlog and `new` events carry pushes. Reconnects with backoff on drop; each
// (re)connect re-sends the backlog, so anything missed during a gap is caught.
let _notifStreamCtrl = null;
let _notifStreamStop = false;

async function startNotifStream() {
  _notifStreamStop = false;
  let backoff = 1000;
  while (!_notifStreamStop && state.authToken) {
    _notifStreamCtrl = new AbortController();
    try {
      const res = await fetch('/api/notifications/stream', {
        headers: { Authorization: `Bearer ${state.authToken}` },
        signal: _notifStreamCtrl.signal,
      });
      if (res.status === 401) {
        // Token expired mid-session — refresh and reconnect rather than dying
        // permanently (which silently killed all real-time notifications until
        // a page reload). Only give up if the refresh itself fails.
        if (await tryRefreshToken()) continue;
        return;
      }
      if (!res.ok || !res.body) throw new Error(`stream ${res.status}`);
      backoff = 1000;                          // healthy connection — reset backoff

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop();
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          let evt;
          try { evt = JSON.parse(line.slice(6)); } catch { continue; }
          if ((evt.type === 'init' || evt.type === 'new') && Array.isArray(evt.items)) {
            _applyServerNotifications(evt.items);
          }
        }
      }
    } catch {
      if (_notifStreamStop || _notifStreamCtrl?.signal.aborted) return;
      // fall through to backoff + reconnect
    }
    if (_notifStreamStop || !state.authToken) return;
    await new Promise(r => setTimeout(r, backoff));
    backoff = Math.min(backoff * 2, 30000);
  }
}

function stopNotifStream() {
  _notifStreamStop = true;
  if (_notifStreamCtrl) { try { _notifStreamCtrl.abort(); } catch {} }
}

async function bootApp() {
  initTabs();
  initModals();
  initAiEdit();
  initUpload();
  initNotifPanel();
  initQueryPanel();
  initChatResize();
  initSearch();
  initWikiLinks();
  initSidebarCollapse();

  document.getElementById('btn-home').addEventListener('click', () => {
    document.getElementById('welcome-screen').classList.remove('hidden');
    document.getElementById('page-view').classList.add('hidden');
  });
  document.getElementById('btn-home').addEventListener('keydown', e => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); document.getElementById('btn-home').click(); }
  });

  // User menu — populate info and wire toggle + signout
  const userMenuBtn  = document.getElementById('btn-user-menu');
  const userMenuDrop = document.getElementById('user-menu-dropdown');
  if (userMenuBtn && userMenuDrop) {
    document.getElementById('umi-email').textContent   = state.userEmail   || '—';
    document.getElementById('umi-role').textContent    = state.userRole    || '—';
    document.getElementById('umi-org').textContent     = state.userOrgName || (state.userRole === 'admin' ? 'Global admin' : '—');
    // Show the admin dashboard link for admins/supervisors. The welcome
    // "Import Wiki" button is revealed separately by loadWikiTree → it appears
    // only when the org's wiki is empty (see setWelcomeImportVisible).
    if (state.userRole === 'admin' || state.userRole === 'supervisor') {
      const adminLink = document.getElementById('btn-admin-page');
      if (adminLink) adminLink.classList.remove('hidden');
    }
    userMenuBtn.addEventListener('click', e => {
      e.stopPropagation();
      userMenuDrop.classList.toggle('open');
    });
    document.addEventListener('click', () => userMenuDrop.classList.remove('open'));
    userMenuDrop.addEventListener('click', e => e.stopPropagation());
    document.getElementById('btn-signout').addEventListener('click', async () => {
      userMenuDrop.classList.remove('open');
      // Cancel any in-flight LLM streaming requests so the server stops
      // generating tokens against this user's quota the moment they sign out.
      abortAllActiveStreams();
      stopNotifStream();  // close the notification SSE connection
      const refreshToken = state.refreshToken;
      showLoginScreen();
      if (refreshToken) {
        fetch('/api/auth/logout', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ refresh_token: refreshToken }),
        }).catch(() => {});
      }
    });
  }

  // Tools dropdown toggle
  const toolsWrap = document.getElementById('tools-menu-wrap');
  const toolsDrop = document.getElementById('tools-dropdown');
  document.getElementById('btn-tools')?.addEventListener('click', e => {
    e.stopPropagation();
    toolsWrap.classList.toggle('open');
  });
  document.addEventListener('click', () => toolsWrap?.classList.remove('open'));
  toolsDrop?.addEventListener('click', () => toolsWrap.classList.remove('open'));

  document.getElementById('btn-upload').addEventListener('click', () => openModal('modal-upload'));
  document.getElementById('btn-welcome-upload').addEventListener('click', () => openModal('modal-upload'));
  document.getElementById('btn-recent-changes')?.addEventListener('click', openRecentChanges);
  const _importBtn = document.getElementById('btn-welcome-import');
  const _importInput = document.getElementById('wiki-import-file');
  if (_importBtn && _importInput) {
    _importBtn.addEventListener('click', () => { _importInput.value = ''; _importInput.click(); });
    _importInput.addEventListener('change', () => {
      const f = _importInput.files && _importInput.files[0];
      if (f) importWikiBundle(f);
    });
  }
  initWriterMode();
  document.getElementById('btn-graph').addEventListener('click', () => openGraphModal(false));
  document.getElementById('btn-graph-rebuild').addEventListener('click', () => openGraphModal(true));
  document.getElementById('btn-lint').addEventListener('click', runLint);
  document.getElementById('btn-plan-chat-send').addEventListener('click', sendPlanChat);
  document.getElementById('plan-chat-input').addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendPlanChat(); }
  });
  document.getElementById('btn-recalibrate').addEventListener('click', triggerRecalibrate);
  document.getElementById('btn-export').addEventListener('click', triggerExport);
  document.getElementById('btn-export-confirm').addEventListener('click', confirmExport);
  document.getElementById('btn-export-cancel').addEventListener('click', () => closeModal('modal-export-confirm'));
  document.getElementById('btn-recal-confirm').addEventListener('click', confirmRecalibrate);
  document.getElementById('btn-recal-cancel').addEventListener('click', () => closeModal('modal-recalibrate-confirm'));
  document.getElementById('btn-recal-dismiss').addEventListener('click', hideRecalibrateOverlay);
  document.getElementById('recal-fact-instructions')?.addEventListener('input', () => {
    const label = document.getElementById('recal-confirm-label');
    const hasText = document.getElementById('recal-fact-instructions').value.trim().length > 0;
    if (label) label.textContent = hasText ? 'Start Targeted Recalibration' : 'Start Recalibration';
  });
  document.getElementById('btn-delete-sources').addEventListener('click', openDeleteSourcesModal);

  document.getElementById('btn-delete-sources-cancel').addEventListener('click', () => closeModal('modal-delete-sources'));
  document.getElementById('btn-delete-sources-confirm').addEventListener('click', confirmDeleteSources);

  document.getElementById('btn-page-back').addEventListener('click', () => {
    const prev = state.pageHistory.pop();
    if (prev) loadWikiPage(prev, false);
  });

  await initOrgSwitcher();
  await loadOrgScopedData();

  // Keep the notification feed live for the rest of the session via SSE (one
  // stream; notifications are user-scoped, so org switches don't restart it —
  // loadOrgScopedData's one-off fetch covers the switch).
  startNotifStream();
}

// Org-scoped data + render. Safe to re-run on an org switch — unlike bootApp,
// which also binds one-time event listeners. Keep in sync with the boot tail.
async function loadOrgScopedData() {
  await fetchAndApplyPermissions();
  await Promise.all([
    loadWikiTree(),
    loadSourcesList(),
    checkRecalibrateOnStartup(),
    fetchPersistentNotifications(),
    restoreChatHistoryIfAny(),
  ]);
}

async function initOrgSwitcher() {
  try {
    // Admins can switch across every org; members switch between the orgs they
    // belong to. Build the option list as {id, name} either way.
    let orgs;
    if (state.userRole === 'admin') {
      const all = await api('GET', '/api/admin/organizations');
      orgs = (all || []).map(o => ({ id: o.id, name: o.name }));
    } else {
      orgs = (state.memberships || []).map(m => ({
        id: m.org_id,
        name: m.org_name ? `${m.org_name} (${m.role})` : m.role,
      }));
    }
    if (!orgs.length) return;

    // If no/invalid active org is stored, default to the first available.
    if (!state.activeOrgId || !orgs.find(o => o.id === state.activeOrgId)) {
      state.activeOrgId = orgs[0].id;
      localStorage.setItem('active_org_id', state.activeOrgId);
    }

    // Role is per-org for members — sync state.userRole to the active org so
    // UI gating (supervisor tools etc.) matches the org being acted on.
    if (state.userRole !== 'admin') {
      const m = (state.memberships || []).find(x => x.org_id === state.activeOrgId);
      if (m && m.role) {
        state.userRole = m.role;
        localStorage.setItem('user_role', m.role);
      }
    }

    const wrap = document.getElementById('org-switcher-wrap');
    const sel  = document.getElementById('org-switcher-select');
    if (!wrap || !sel) return;

    // Members with a single org don't need a switcher.
    if (state.userRole !== 'admin' && orgs.length < 2) return;

    sel.innerHTML = orgs.map(o =>
      `<option value="${o.id}" ${o.id === state.activeOrgId ? 'selected' : ''}>${escHtml(o.name)}</option>`
    ).join('');
    sel.addEventListener('change', () => { switchActiveOrg(sel.value); });
    wrap.classList.remove('hidden');
  } catch { /* non-fatal */ }
}

// Switch the active org in-app — no page reload, no login flash. Org + role are
// resolved per-request on the server from the X-Org-Context header against
// org_memberships, so the existing token keeps working; we just re-point the
// session, re-gate role-based UI, reset to the wiki home, and reload data.
async function switchActiveOrg(orgId) {
  if (!orgId || orgId === state.activeOrgId) return;
  state.activeOrgId = orgId;
  localStorage.setItem('active_org_id', orgId);

  // Role is per-org for members — re-sync so UI gating matches the new org.
  // Admins keep their global 'admin' role across every org.
  if (state.userRole !== 'admin') {
    const m = (state.memberships || []).find(x => x.org_id === orgId);
    if (m && m.role) {
      state.userRole = m.role;
      localStorage.setItem('user_role', m.role);
    }
    if (m && m.org_name) {
      state.userOrgName = m.org_name;
      localStorage.setItem('user_org_name', m.org_name);
    }
  }

  // Reflect the new role/org in the user menu and re-gate the admin link.
  const roleEl = document.getElementById('umi-role');
  if (roleEl) roleEl.textContent = state.userRole || '—';
  const orgEl = document.getElementById('umi-org');
  if (orgEl) orgEl.textContent = state.userOrgName || (state.userRole === 'admin' ? 'Global admin' : '—');
  const adminLink = document.getElementById('btn-admin-page');
  if (adminLink) adminLink.classList.toggle('hidden',
    !(state.userRole === 'admin' || state.userRole === 'supervisor'));

  // Reset per-org view state so we never show the previous org's content, and
  // land on the wiki home.
  state.activePage = null;
  state.activeSource = null;
  state.pageHistory = [];
  setChatSessionId(null);  // forget (don't delete) the prior org's chat session
  const chatEl = document.getElementById('chat-messages');
  if (chatEl) chatEl.innerHTML = '';
  document.getElementById('welcome-screen')?.classList.remove('hidden');
  document.getElementById('page-view')?.classList.add('hidden');

  await loadOrgScopedData();
}
