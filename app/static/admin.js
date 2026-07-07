/* Admin Dashboard — AI Knowledge Hub */
"use strict";

const API = "/api";
let _token = localStorage.getItem("auth_token") || "";
let _myRole = "";          // "admin" or "supervisor" — set at init
let _myOrgId = "";         // org_id of the currently logged-in user (empty for admin)
let _activeOrgId = "";     // org a multi-org supervisor has selected; sent as X-Org-Context
let _supervisedOrgs = [];  // [{org_id, org_name}] this user supervises (multi-org supervisor only)
let _editUserId = null;
let _editUserOrgId = null;  // track current org of the user being edited
let _templates = [];
let _workspaces = [];
let _orgs = [];
let _userCache = {};  // "id|org_id" → user object, avoids JSON-in-onclick
let _auditPage = 1;

// ── Auth ──────────────────────────────────────────────────────────────────────

async function apiFetch(path, opts = {}) {
  const headers = {
    "Content-Type": "application/json",
    Authorization: `Bearer ${_token}`,
    // A multi-org supervisor scopes every request to their selected org.
    // Explicit per-call headers (e.g. admin viewing another org) still win.
    ...(_activeOrgId ? { "X-Org-Context": _activeOrgId } : {}),
    ...opts.headers,
  };
  const res = await fetch(API + path, { ...opts, headers });
  if (res.status === 401) {
    // Redirect to login, and THROW rather than return null: callers `await`
    // this and dereference the result (e.g. `job.status`), so returning null
    // caused a TypeError instead of a clean failure. Throwing lets their
    // existing try/catch degrade gracefully while the page navigates away.
    window.location.href = "/";
    throw new Error("unauthorized");
  }
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

// ── Toast ─────────────────────────────────────────────────────────────────────

function toast(msg, type = "success") {
  const el = document.getElementById("toast");
  el.textContent = msg;
  el.className = `toast show ${type}`;
  clearTimeout(el._t);
  el._t = setTimeout(() => el.classList.remove("show"), 3500);
}

// ── Tabs ──────────────────────────────────────────────────────────────────────

document.querySelectorAll(".tab").forEach(tab => {
  tab.addEventListener("click", () => {
    // Stop the Jobs auto-refresh whenever we leave (or re-enter) a tab.
    if (_jobsPollTimer) { clearTimeout(_jobsPollTimer); _jobsPollTimer = null; }
    document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
    document.querySelectorAll(".section").forEach(s => s.classList.remove("active"));
    tab.classList.add("active");
    document.getElementById(`sec-${tab.dataset.tab}`).classList.add("active");
    document.querySelector(".main").classList.toggle("schema-active", tab.dataset.tab === "schema");
    document.body.classList.toggle("schema-mode", tab.dataset.tab === "schema");
    const loaders = {
      users: loadUsers,
      orgs: loadOrgs,
      templates: loadTemplates,
      workspaces: loadWorkspaces,
      usage: loadUsage,
      "org-limits": loadOrgLimits,
      audit: loadAudit,
      history: loadHistory,
      jobs: loadJobs,
      schema: loadSchema,
    };
    loaders[tab.dataset.tab]?.();
  });
});

// ── Supervisor org switcher ─────────────────────────────────────────────────
// A supervisor of 2+ orgs sees a header dropdown to pick the active org. The
// selection is sent as X-Org-Context (see apiFetch) so every endpoint scopes to
// that org. Admins keep their existing per-feature org selectors; single-org
// supervisors don't need a switcher (their one org is implicit).

function _setupSupervisorOrgSwitcher(me) {
  if (me.role === "admin") return;
  _supervisedOrgs = (me.memberships || []).filter(m => m.role === "supervisor");
  const sel = document.getElementById("adm-org-switcher");
  if (!sel || _supervisedOrgs.length < 2) return;

  // Restore the last selection if it's still a supervised org, else use the first.
  const saved = localStorage.getItem("admin_active_org");
  const valid = _supervisedOrgs.some(o => o.org_id === saved);
  _activeOrgId = valid ? saved : _supervisedOrgs[0].org_id;
  _myOrgId = _activeOrgId;
  localStorage.setItem("admin_active_org", _activeOrgId);

  sel.innerHTML = _supervisedOrgs
    .map(o => `<option value="${escHtml(o.org_id)}">${escHtml(o.org_name || o.org_id)}</option>`)
    .join("");
  sel.value = _activeOrgId;
  sel.style.display = "";
  sel.addEventListener("change", () => {
    _activeOrgId = sel.value;
    _myOrgId = _activeOrgId;
    localStorage.setItem("admin_active_org", _activeOrgId);
    _reloadActiveSection();
  });
}

function _reloadActiveSection() {
  // Re-clicking the active tab re-runs its loader with the new org context.
  document.querySelector(".tab.active")?.click();
}

// Every .modal-backdrop shares the same CSS z-index, so when one modal opens on
// top of another (e.g. "Edit" from the Manage Members list) DOM order alone
// decides which is in front — and the nested modal is often earlier in the DOM,
// so it opened *behind* its parent. openModal() gives each newly-opened modal a
// z-index above every currently-open one, so the most recent is always on top.
const _MODAL_BASE_Z = 1000;
function openModal(id) {
  const el = document.getElementById(id);
  if (!el) return;
  let maxZ = _MODAL_BASE_Z;
  document.querySelectorAll(".modal-backdrop.open").forEach(m => {
    const z = parseInt(m.style.zIndex, 10);
    if (!Number.isNaN(z)) maxZ = Math.max(maxZ, z);
  });
  el.style.zIndex = String(maxZ + 10);
  el.classList.add("open");
}

function closeModal(id) {
  const el = document.getElementById(id);
  if (el) { el.classList.remove("open"); el.style.zIndex = ""; }
  if (id === "modal-dialog") _dlgSettle(_dlgMode === "prompt" ? null : false);
}
document.querySelectorAll(".modal-backdrop").forEach(m => {
  m.addEventListener("click", e => { if (e.target === m) closeModal(m.id); });
});

// ── In-app dialogs (replace native confirm/prompt) ──────────────────────────
// Promise-based: `await uiConfirm({...})` → bool; `await uiPrompt({...})` →
// string|null. Backed by the #modal-dialog element; closing via backdrop/Esc/
// Cancel resolves to the cancelled value so callers never hang.

let _dlgResolve = null;
let _dlgMode = "confirm";

function _dlgSettle(value) {
  const r = _dlgResolve;
  _dlgResolve = null;
  if (r) r(value);
}

function uiConfirm({ title = "Confirm", message = "", confirmText = "Confirm", danger = false } = {}) {
  return new Promise(resolve => {
    _dlgSettle(_dlgMode === "prompt" ? null : false);  // settle any prior dialog
    _dlgResolve = resolve;
    _dlgMode = "confirm";
    document.getElementById("dlg-title").textContent = title;
    document.getElementById("dlg-message").textContent = message;
    document.getElementById("dlg-input-wrap").style.display = "none";
    const c = document.getElementById("dlg-confirm");
    c.textContent = confirmText;
    c.className = "btn " + (danger ? "btn-danger" : "btn-primary");
    openModal("modal-dialog");
    c.focus();
  });
}

function uiPrompt({ title = "", message = "", label = "", defaultValue = "", confirmText = "OK" } = {}) {
  return new Promise(resolve => {
    _dlgSettle(_dlgMode === "prompt" ? null : false);
    _dlgResolve = resolve;
    _dlgMode = "prompt";
    document.getElementById("dlg-title").textContent = title;
    document.getElementById("dlg-message").textContent = message;
    document.getElementById("dlg-input-wrap").style.display = "";
    document.getElementById("dlg-input-label").textContent = label;
    const input = document.getElementById("dlg-input");
    input.value = defaultValue;
    const c = document.getElementById("dlg-confirm");
    c.textContent = confirmText;
    c.className = "btn btn-primary";
    openModal("modal-dialog");
    input.focus(); input.select();
  });
}

function _dlgConfirmClick() {
  const value = _dlgMode === "prompt"
    ? document.getElementById("dlg-input").value
    : true;
  const dlg = document.getElementById("modal-dialog");
  dlg.classList.remove("open");
  dlg.style.zIndex = "";
  _dlgSettle(value);
}

document.getElementById("dlg-confirm").addEventListener("click", _dlgConfirmClick);
document.getElementById("dlg-cancel").addEventListener("click", () => closeModal("modal-dialog"));
document.getElementById("dlg-input").addEventListener("keydown", e => {
  if (e.key === "Enter") { e.preventDefault(); _dlgConfirmClick(); }
});
document.addEventListener("keydown", e => {
  if (e.key === "Escape" && document.getElementById("modal-dialog").classList.contains("open")) {
    closeModal("modal-dialog");
  }
});

// ── USERS ─────────────────────────────────────────────────────────────────────

async function loadUsers() {
  document.getElementById("users-table").innerHTML = '<div class="loading">Loading…</div>';
  try {
    const [users, templates, workspaces, orgs] = await Promise.all([
      apiFetch("/admin/users"),
      apiFetch("/admin/templates"),
      apiFetch("/admin/workspaces"),
      apiFetch("/admin/organizations"),
    ]);
    _templates  = templates  || [];
    _workspaces = workspaces || [];
    _orgs       = orgs       || [];

    if (!users || !users.length) {
      document.getElementById("users-table").innerHTML = '<div class="empty">No users found.</div>';
      return;
    }

    // Cache by (id, org_id): the per-membership Edit modal looks rows up this way.
    _userCache = {};
    users.forEach(u => { _userCache[`${u.id}|${u.org_id || ""}`] = u; });

    // Group the per-membership rows into one entry per identity. Admins and
    // floating identities have a single row with org_id = null.
    const byId = new Map();
    for (const u of users) {
      let g = byId.get(u.id);
      if (!g) {
        g = { id: u.id, email: u.email, last_login_at: u.last_login_at,
              isAdmin: u.role === "admin", memberships: [] };
        byId.set(u.id, g);
      }
      if (u.role === "admin") g.isAdmin = true;
      if (u.org_id) g.memberships.push(u);
    }

    const rows = [...byId.values()].map(renderUserRow).join("");
    document.getElementById("users-table").innerHTML = `
      <table class="users-table"><thead><tr>
        <th style="width:24px"></th><th>Email</th><th>Organizations &amp; roles</th><th>Last login</th>
      </tr></thead><tbody>${rows}</tbody></table>`;
  } catch (e) {
    document.getElementById("users-table").innerHTML = `<div class="empty">${escHtml(e.message)}</div>`;
  }
}

function _roleBadge(role) {
  const cls = role === "admin" ? "badge-admin" : role === "supervisor" ? "badge-supervisor" : "badge-member";
  return `<span class="badge ${cls}">${escHtml(role || "member")}</span>`;
}

// One <tr> per identity (the summary), plus a hidden detail <tr> toggled on click.
function renderUserRow(g) {
  const last = g.last_login_at ? new Date(g.last_login_at).toLocaleDateString() : "never";

  let summary;
  if (g.isAdmin) {
    summary = `<span class="badge badge-admin">★ Global admin</span>`;
  } else if (!g.memberships.length) {
    summary = `<span style="color:var(--text-muted);font-size:0.82rem">— no org access yet —</span>`;
  } else {
    summary = g.memberships.map(m => {
      const susp = m.is_suspended ? ` <span class="badge badge-suspended">suspended</span>` : "";
      return `<span class="org-chip">${escHtml(m.org_name || "?")} · ${escHtml(m.role)}${susp}</span>`;
    }).join(" ");
  }

  return `<tr class="user-row" onclick='toggleUserDetail("${g.id}")' style="cursor:pointer">
      <td class="caret" id="caret-${g.id}">▸</td>
      <td>${escHtml(g.email)}</td>
      <td>${summary}</td>
      <td style="color:var(--text-muted)">${last}</td>
    </tr>
    <tr class="user-detail" id="udetail-${g.id}" style="display:none">
      <td></td>
      <td colspan="3">${renderUserDetail(g)}</td>
    </tr>`;
}

function renderUserDetail(g) {
  const reset = `<button class="btn btn-secondary btn-sm" onclick='event.stopPropagation();openResetPasswordFor("${g.id}")'>Reset password</button>`;

  if (g.isAdmin) {
    return `<div class="user-detail-box">
      <p style="color:var(--text-muted);font-size:0.85rem;margin-bottom:10px">Global admin — full platform access across all organizations. Not tied to an org.</p>
      ${reset}
    </div>`;
  }
  if (!g.memberships.length) {
    return `<div class="user-detail-box">
      <p style="color:var(--text-muted);font-size:0.85rem;margin-bottom:10px">No organization access yet. Add this user from the <b>Organizations</b> tab → <b>Manage members</b>.</p>
      ${reset}
    </div>`;
  }

  const tplMap = Object.fromEntries(_templates.map(t => [t.id, t.name]));
  const memberRows = g.memberships.map(m => {
    const tpl = m.permission_template_id && tplMap[m.permission_template_id]
      ? escHtml(tplMap[m.permission_template_id])
      : (m.role === "supervisor" ? "full access" : "—");
    const status = m.is_suspended
      ? `<span class="badge badge-suspended">suspended</span>`
      : `<span style="color:var(--text-muted)">active</span>`;
    return `<tr>
      <td>${escHtml(m.org_name || "?")}</td>
      <td>${_roleBadge(m.role)}</td>
      <td style="color:var(--text-muted)">${tpl}</td>
      <td>${status}</td>
      <td style="color:var(--text-muted)">${m.uploads_today ?? 0} / ${fmtNum(m.tokens_today ?? 0)}</td>
      <td style="text-align:right;white-space:nowrap">
        <button class="btn btn-secondary btn-sm" onclick='event.stopPropagation();openEditUser("${m.id}","${m.org_id}")'>Edit</button>
        <button class="btn btn-danger btn-sm" onclick='event.stopPropagation();removeMembershipFromUsers("${m.id}","${m.org_id}","${escHtml(m.email)}","${escHtml(m.org_name || "")}")'>Remove</button>
      </td>
    </tr>`;
  }).join("");

  return `<div class="user-detail-box">
    <table class="sub-table" style="width:100%;margin-bottom:10px"><thead><tr>
      <th>Organization</th><th>Role</th><th>Template</th><th>Status</th><th>Uploads / tokens today</th><th></th>
    </tr></thead><tbody>${memberRows}</tbody></table>
    ${reset}
  </div>`;
}

function toggleUserDetail(id) {
  const row = document.getElementById(`udetail-${id}`);
  const caret = document.getElementById(`caret-${id}`);
  if (!row) return;
  const open = row.style.display !== "none";
  row.style.display = open ? "none" : "";
  if (caret) caret.textContent = open ? "▸" : "▾";
}

function openResetPasswordFor(userId, orgId = "") {
  _editUserId = userId;
  _editUserOrgId = orgId;
  resetPassword();  // reuses the existing reset-password modal flow
}

async function removeMembershipFromUsers(userId, orgId, email, orgName) {
  if (!(await uiConfirm({
    title: "Remove member",
    message: `Remove ${email} from "${orgName}"? Their account and other org memberships are kept.`,
    confirmText: "Remove", danger: true,
  }))) return;
  try {
    await apiFetch(`/admin/users/${userId}/memberships/${orgId}`, { method: "DELETE" });
    toast("Membership removed");
    loadUsers();
  } catch (e) { toast(e.message, "error"); }
}

function openEditUser(userId, orgId) {
  const u = _userCache[`${userId}|${orgId || ""}`];
  if (!u) { toast("User data not loaded — please refresh", "error"); return; }
  _editUserId    = u.id;
  _editUserOrgId = u.org_id;

  document.getElementById("edit-user-title").textContent = `Edit — ${u.email}`;
  document.getElementById("eu-email").value = u.email;
  document.getElementById("eu-role").value  = u.role;
  document.getElementById("eu-suspended").checked = !!u.is_suspended;

  // Org selector — allows transferring user to a different org
  const orgSel = document.getElementById("eu-org");
  orgSel.innerHTML = '<option value="">— Keep current org —</option>' +
    (_orgs || []).map(o =>
      `<option value="${o.id}" ${u.org_id === o.id ? "selected" : ""}>${escHtml(o.name)}</option>`
    ).join("");

  const tplSel = document.getElementById("eu-template");
  tplSel.innerHTML = '<option value="">— None (read-only) —</option>' +
    (_templates || []).map(t =>
      `<option value="${t.id}" ${u.permission_template_id === t.id ? "selected" : ""}>${escHtml(t.name)}${t.is_builtin ? " ★" : ""}</option>`
    ).join("");

  const wsSel = document.getElementById("eu-workspace");
  wsSel.innerHTML = '<option value="">— Default —</option>' +
    (_workspaces || []).map(w =>
      `<option value="${w.id}" ${u.workspace_id === w.id ? "selected" : ""}>${escHtml(w.name)}</option>`
    ).join("");

  document.getElementById("eu-max_tokens").value = tokensToMillionsStr(u.max_tokens_per_day);
  document.getElementById("eu-max_msgs").value   = u.max_chat_messages_per_day ?? "";

  onEditUserRoleChange(u.role);
  openModal("modal-edit-user");
}

function onEditUserRoleChange(role) {
  // Supervisors (and admins) have full access — a permission template is moot.
  const fullAccess = role === "supervisor" || role === "admin";
  const tpl = document.getElementById("eu-template");
  tpl.disabled = fullAccess;
  if (fullAccess) tpl.value = "";
}

async function saveEditUser() {
  try {
    // No membership to edit: either a global admin, or a floating identity that
    // hasn't been added to any org yet.
    if (!_editUserOrgId) {
      const cu = _userCache[`${_editUserId}|`];
      toast(cu && cu.role === "admin"
        ? "Admin users are global — no per-org settings to edit"
        : "This user has no org membership — add them from the Organizations tab → Manage members.",
        "error");
      return;
    }
    const newOrgId = document.getElementById("eu-org").value || null;
    // Move this membership to another org if the org selector changed. The
    // source org is this row's membership, passed via X-Org-Context.
    if (newOrgId && newOrgId !== _editUserOrgId) {
      await apiFetch(`/admin/users/${_editUserId}/transfer-org`, {
        method: "PUT",
        headers: { "X-Org-Context": _editUserOrgId },
        body: JSON.stringify({ new_org_id: newOrgId }),
      });
      // The membership now lives in newOrgId — target it for the profile update.
      _editUserOrgId = newOrgId;
    }
    const toInt = id => { const v = document.getElementById(id)?.value; return v ? parseInt(v) : null; };
    await apiFetch(`/admin/users/${_editUserId}`, {
      method: "PUT",
      body: JSON.stringify({
        org_id: _editUserOrgId,
        role: document.getElementById("eu-role").value,
        permission_template_id: document.getElementById("eu-template").value || null,
        workspace_id: document.getElementById("eu-workspace").value || null,
        is_suspended: document.getElementById("eu-suspended").checked,
        max_tokens_per_day: millionsStrToTokens(document.getElementById("eu-max_tokens").value),
        max_chat_messages_per_day: toInt("eu-max_msgs"),
      }),
    });
    closeModal("modal-edit-user");
    toast("User updated");
    loadUsers();
  } catch (e) { toast(e.message, "error"); }
}

function resetPassword() {
  closeModal("modal-edit-user");
  document.getElementById("rp-password").value = "";
  document.getElementById("rp-confirm").value = "";
  document.getElementById("rp-strength-fill").style.width = "0";
  document.getElementById("rp-strength-fill").style.background = "";
  document.getElementById("rp-strength-label").textContent = "";
  document.getElementById("rp-strength-label").style.color = "";
  document.getElementById("rp-match-label").textContent = "";

  const u = _userCache[`${_editUserId}|${_editUserOrgId || ""}`];
  const label = u ? u.email : "this user";
  document.getElementById("rp-user-label").textContent = `for ${label}`;

  openModal("modal-reset-pw");
  setTimeout(() => document.getElementById("rp-password").focus(), 60);
}

function togglePwVisibility(inputId, btn) {
  const input = document.getElementById(inputId);
  const isHidden = input.type === "password";
  input.type = isHidden ? "text" : "password";
  btn.querySelector(".eye-icon").style.opacity = isHidden ? "0.4" : "1";
}

function updatePwStrength() {
  const pw = document.getElementById("rp-password").value;
  const fill = document.getElementById("rp-strength-fill");
  const lbl  = document.getElementById("rp-strength-label");

  if (!pw) { fill.style.width = "0"; lbl.textContent = ""; return; }

  let score = 0;
  if (pw.length >= 8)  score++;
  if (pw.length >= 12) score++;
  if (/[A-Z]/.test(pw) && /[a-z]/.test(pw)) score++;
  if (/\d/.test(pw)) score++;
  if (/[^A-Za-z0-9]/.test(pw)) score++;

  const levels = [
    { pct: "20%", color: "#e05555", text: "Very weak",  textColor: "#e05555" },
    { pct: "40%", color: "#e08c3a", text: "Weak",       textColor: "#e08c3a" },
    { pct: "60%", color: "#d4b93a", text: "Fair",       textColor: "#d4b93a" },
    { pct: "80%", color: "#5aab7b", text: "Strong",     textColor: "#5aab7b" },
    { pct: "100%",color: "#2fa86e", text: "Very strong",textColor: "#2fa86e" },
  ];
  const lvl = levels[Math.max(0, Math.min(score - 1, 4))];
  fill.style.width = lvl.pct;
  fill.style.background = lvl.color;
  lbl.textContent = lvl.text;
  lbl.style.color = lvl.textColor;

  checkPwMatch();
}

function checkPwMatch() {
  const pw  = document.getElementById("rp-password").value;
  const cnf = document.getElementById("rp-confirm").value;
  const lbl = document.getElementById("rp-match-label");
  if (!cnf) { lbl.textContent = ""; return; }
  if (pw === cnf) {
    lbl.textContent = "✓ Passwords match";
    lbl.style.color = "var(--success, #2fa86e)";
  } else {
    lbl.textContent = "✗ Passwords do not match";
    lbl.style.color = "var(--danger, #e05555)";
  }
}

async function doResetPassword() {
  const pw  = document.getElementById("rp-password").value;
  const cnf = document.getElementById("rp-confirm").value;
  if (pw.length < 8) { toast("Password must be at least 8 characters", "error"); return; }
  if (pw !== cnf) { toast("Passwords do not match", "error"); return; }
  const btn = document.getElementById("rp-submit-btn");
  btn.disabled = true;
  try {
    await apiFetch(`/admin/users/${_editUserId}/reset-password`, {
      method: "POST", body: JSON.stringify({ new_password: pw }),
    });
    closeModal("modal-reset-pw");
    toast("Password reset successfully");
  } catch (e) {
    toast(e.message, "error");
  } finally {
    btn.disabled = false;
  }
}

function openCreateUser() {
  // Add User now creates the identity only — email + password + Admin toggle.
  // Org access (role + template) is granted from the Organizations tab.
  document.getElementById("cu-email").value = "";
  document.getElementById("cu-password").value = "";
  document.getElementById("cu-is-admin").checked = false;
  onCreateUserAdminChange(false);
  openModal("modal-create-user");
}

function onCreateUserAdminChange(isAdmin) {
  document.getElementById("cu-admin-note").style.display = isAdmin ? "" : "none";
  document.getElementById("cu-member-note").style.display = isAdmin ? "none" : "";
}

async function createUser() {
  const email    = document.getElementById("cu-email").value.trim();
  const password = document.getElementById("cu-password").value;
  const isAdmin  = document.getElementById("cu-is-admin").checked;
  if (!email || !password) { toast("Email and password required", "error"); return; }
  try {
    await apiFetch("/admin/users", {
      method: "POST",
      // No org_id → backend creates an admin (if flagged) or a floating identity.
      body: JSON.stringify({ email, password, role: isAdmin ? "admin" : "member" }),
    });
    closeModal("modal-create-user");
    toast(isAdmin ? "Admin created" : "User created — add them to an organization to grant access");
    loadUsers();
  } catch (e) { toast(e.message, "error"); }
}

// ── PERMISSION TEMPLATES ──────────────────────────────────────────────────────

const _BOOL_FLAGS = [
  "can_upload", "can_upload_writer_draft", "can_delete_files", "can_download_files",
  "can_view_wiki", "can_edit_wiki", "can_delete_wiki_pages",
  "can_query", "can_chat", "can_use_writer",
  "can_recalibrate", "can_run_lint", "can_manage_schema",
  "can_view_audit_log", "can_view_graph", "can_rebuild_graph",
  "can_manage_workspace", "can_approve_ingest", "can_cancel_ingest",
];
// Plain integer count fields (loaded/saved as-is).
const _NUM_FIELDS = [
  "max_upload_size_mb", "max_uploads_per_day", "max_uploads_per_week",
  "max_queries_per_day", "max_chat_messages_per_day",
];
// Token-budget fields — input is in millions of tokens; stored as raw tokens.
const _TOKEN_FIELDS = ["max_tokens_per_day", "max_tokens_per_week"];

// ── ORGANIZATIONS ─────────────────────────────────────────────────────────────

async function loadOrgs() {
  document.getElementById("orgs-table").innerHTML = '<div class="loading">Loading…</div>';
  try {
    _orgs = await apiFetch("/admin/organizations") || [];
    if (!_orgs.length) {
      document.getElementById("orgs-table").innerHTML = '<div class="empty">No organizations found.</div>';
      return;
    }
    const rows = _orgs.map(o => {
      const sups = (o.supervisor_emails && o.supervisor_emails.length)
        ? escHtml(o.supervisor_emails.join(", "))
        : '<span style="color:var(--text-muted);font-size:0.8rem">—</span>';
      return `<tr>
      <td><b>${escHtml(o.name)}</b></td>
      <td style="color:var(--text-muted)">${escHtml(o.slug || "")}</td>
      <td style="color:var(--text-muted)">${sups}</td>
      <td style="color:var(--text-muted)">${o.user_count ?? 0} users</td>
      <td style="color:var(--text-muted)">${o.max_members != null ? o.max_members : "unlimited"} max</td>
      <td style="color:var(--text-muted)">${o.created_at ? new Date(o.created_at).toLocaleDateString() : "—"}</td>
      <td>
        <button class="btn btn-secondary btn-sm" onclick='openOrgMembers("${o.id}", "${escHtml(o.name)}")'>Manage members</button>
        <button class="btn btn-secondary btn-sm" onclick='cloneOrg("${o.id}", "${escHtml(o.name)}")'>Clone</button>
        ${o.id !== "00000000-0000-0000-0000-000000000001"
          ? `<button class="btn btn-danger btn-sm" onclick='deleteOrg("${o.id}", "${escHtml(o.name)}")'>Delete</button>`
          : `<span style="color:var(--text-muted);font-size:0.8rem">default</span>`}
      </td>
    </tr>`;
    }).join("");
    document.getElementById("orgs-table").innerHTML = `
      <table><thead><tr>
        <th>Name</th><th>Slug</th><th>Supervisor</th><th>Members</th><th>Limit</th><th>Created</th><th></th>
      </tr></thead><tbody>${rows}</tbody></table>`;
  } catch (e) {
    document.getElementById("orgs-table").innerHTML = `<div class="empty">${escHtml(e.message)}</div>`;
  }
}

function openCreateOrg() {
  document.getElementById("co-name").value = "";
  document.getElementById("co-supervisor-email").value = "";
  const f = document.getElementById("co-import-file");
  if (f) f.value = "";
  openModal("modal-create-org");
}

async function createOrg() {
  const name            = document.getElementById("co-name").value.trim();
  const supervisorEmail = document.getElementById("co-supervisor-email").value.trim();
  if (!name) { toast("Organization name is required", "error"); return; }
  const importFile = document.getElementById("co-import-file")?.files?.[0] || null;

  let created;
  try {
    created = await apiFetch("/admin/organizations", {
      method: "POST",
      body: JSON.stringify({ name, supervisor_email: supervisorEmail || null }),
    });
  } catch (e) { toast(e.message, "error"); return; }
  toast(`Organization "${name}" created`);

  // Optionally populate the brand-new (empty) org from an uploaded export.
  if (importFile && created?.id) {
    try {
      const fd = new FormData();
      fd.append("file", importFile);
      const res = await fetch(API + "/ops/import", {
        method: "POST",
        headers: { Authorization: `Bearer ${_token}`, "X-Org-Context": created.id },
        body: fd,
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
      toast(`Imported ${data.pages_imported} page(s) into "${name}"`);
    } catch (e) {
      toast(`Org created, but import failed: ${e.message}`, "error");
    }
  }

  closeModal("modal-create-org");
  loadOrgs();
}

async function cloneOrg(id, name) {
  const newName = await uiPrompt({
    title: `Clone "${name}"`,
    message: "Copies the wiki pages, embeddings and schema (AGENTS.md). Does NOT copy members, permission templates, workspaces, uploaded source files, or history.",
    label: "Name for the new organization",
    defaultValue: `${name} (copy)`,
    confirmText: "Clone",
  });
  if (newName === null) return;  // cancelled
  const trimmed = newName.trim();
  if (!trimmed) { toast("New organization name is required", "error"); return; }
  toast(`Cloning "${name}"…`);
  try {
    await apiFetch(`/admin/organizations/${id}/clone`, {
      method: "POST",
      body: JSON.stringify({ new_name: trimmed }),
    });
    toast(`Cloned "${name}" → "${trimmed}"`);
    loadOrgs();
  } catch (e) { toast(e.message, "error"); }
}

async function deleteOrg(id, name) {
  if (!(await uiConfirm({
    title: "Delete organization",
    message: `Delete organization "${name}" and all its data? This cannot be undone.`,
    confirmText: "Delete", danger: true,
  }))) return;
  try {
    await apiFetch(`/admin/organizations/${id}`, { method: "DELETE" });
    toast(`Organization "${name}" deleted`);
    loadOrgs();
  } catch (e) { toast(e.message, "error"); }
}


// ── ORGANIZATION MEMBERS ────────────────────────────────────────────────────────

let _omOrgId = null;
let _omOrgName = "";

async function openOrgMembers(orgId, orgName) {
  _omOrgId = orgId;
  _omOrgName = orgName;
  document.getElementById("om-org-name").textContent = orgName;
  document.getElementById("om-list").innerHTML = '<div class="loading">Loading…</div>';
  openModal("modal-org-members");
  try {
    const [allUsers, identities, templates, orgs] = await Promise.all([
      apiFetch("/admin/users"),
      apiFetch("/admin/identities"),
      _templates.length ? Promise.resolve(_templates) : apiFetch("/admin/templates"),
      _orgs.length ? Promise.resolve(_orgs) : apiFetch("/admin/organizations"),
    ]);
    _templates = templates || _templates;
    _orgs = orgs || _orgs;

    // Members of this org (one /admin/users row per membership).
    const members = (allUsers || []).filter(u => u.org_id === orgId);
    // Cache so the shared Edit User modal (openEditUser) can find them.
    members.forEach(u => { _userCache[`${u.id}|${u.org_id || ""}`] = u; });
    renderOrgMembers(members);

    // Member-picker: existing non-admin identities not already in this org.
    const memberIds = new Set(members.map(m => m.id));
    const candidates = (identities || []).filter(i => !i.is_admin && !memberIds.has(i.id));
    const userSel = document.getElementById("om-user-select");
    userSel.innerHTML = candidates.length
      ? candidates.map(c => `<option value="${c.id}">${escHtml(c.email)}</option>`).join("")
      : '<option value="">(no available users — create one from the Users tab)</option>';

    const tplSel = document.getElementById("om-template");
    tplSel.innerHTML = '<option value="">— None (read-only) —</option>' +
      _templates.map(t => `<option value="${t.id}">${escHtml(t.name)}${t.is_builtin ? " ★" : ""}</option>`).join("");
    document.getElementById("om-role").value = "member";
    onOrgMemberRoleChange("member");
  } catch (e) {
    document.getElementById("om-list").innerHTML = `<div class="empty">${escHtml(e.message)}</div>`;
  }
}

function onOrgMemberRoleChange(role) {
  // Supervisors have full org access, so a permission template is meaningless.
  const isSup = role === "supervisor";
  const tpl = document.getElementById("om-template");
  tpl.disabled = isSup;
  if (isSup) tpl.value = "";
  const note = document.getElementById("om-supervisor-note");
  if (note) note.style.display = isSup ? "" : "none";
}

function renderOrgMembers(members) {
  const el = document.getElementById("om-list");
  if (!members.length) { el.innerHTML = '<div class="empty">No members yet.</div>'; return; }
  const tplMap = Object.fromEntries(_templates.map(t => [t.id, t.name]));
  el.innerHTML = `<table><thead><tr>
      <th>Email</th><th>Role</th><th>Template</th><th>Status</th><th></th>
    </tr></thead><tbody>${members.map(u => `<tr>
      <td>${escHtml(u.email)}</td>
      <td>${escHtml(u.role || "member")}</td>
      <td style="color:var(--text-muted)">${u.permission_template_id && tplMap[u.permission_template_id] ? escHtml(tplMap[u.permission_template_id]) : "—"}</td>
      <td style="color:var(--text-muted)">${u.is_suspended ? "suspended" : "active"}</td>
      <td>
        <button class="btn btn-secondary btn-sm" onclick='openEditUser("${u.id}","${u.org_id || ""}")'>Edit</button>
        <button class="btn btn-danger btn-sm" onclick='removeOrgMember("${u.id}","${escHtml(u.email)}")'>Remove</button>
      </td>
    </tr>`).join("")}</tbody></table>`;
}

async function addOrgMember() {
  const userId = document.getElementById("om-user-select").value;
  const role   = document.getElementById("om-role").value;
  const tplId  = document.getElementById("om-template").value || null;
  if (!userId) { toast("Select a user to add", "error"); return; }
  try {
    await apiFetch(`/admin/users/${userId}/memberships`, {
      method: "POST",
      body: JSON.stringify({ org_id: _omOrgId, role, permission_template_id: tplId }),
    });
    toast("Member added");
    await openOrgMembers(_omOrgId, _omOrgName);  // refresh list + picker
    loadOrgs();                                   // refresh member count
  } catch (e) { toast(e.message, "error"); }
}

async function removeOrgMember(userId, email) {
  if (!(await uiConfirm({
    title: "Remove member",
    message: `Remove ${email} from "${_omOrgName}"? Their account and other org memberships are kept.`,
    confirmText: "Remove", danger: true,
  }))) return;
  try {
    await apiFetch(`/admin/users/${userId}/memberships/${_omOrgId}`, { method: "DELETE" });
    toast("Member removed");
    await openOrgMembers(_omOrgId, _omOrgName);
    loadOrgs();
  } catch (e) { toast(e.message, "error"); }
}


// ── PERMISSION TEMPLATES ───────────────────────────────────────────────────────

async function loadTemplates() {
  document.getElementById("templates-table").innerHTML = '<div class="loading">Loading…</div>';
  try {
    _templates = await apiFetch("/admin/templates") || [];
    if (!_templates.length) {
      document.getElementById("templates-table").innerHTML = '<div class="empty">No templates found.</div>';
      return;
    }
    const showOrg = _myRole === "admin";
    const rows = _templates.map(t => {
      const builtinBadge = t.is_builtin ? '<span class="badge badge-builtin">built-in</span>' : "";
      const perms = _BOOL_FLAGS.filter(f => t[f]).map(f =>
        `<span style="font-size:0.72rem;color:var(--text-muted)">${f.replace("can_","")}</span>`
      ).join(" · ");
      const orgCell = showOrg
        ? `<td style="color:var(--text-muted);font-size:0.82rem">${escHtml(t.org_name || "—")}</td>`
        : "";
      return `<tr>
        <td><b>${escHtml(t.name)}</b> ${builtinBadge}</td>
        <td style="color:var(--text-muted);font-size:0.82rem">${escHtml(t.description || "")}</td>
        ${orgCell}
        <td style="font-size:0.78rem;max-width:400px">${perms || "—"}</td>
        <td>
          ${!t.is_builtin ? `<button class="btn btn-secondary btn-sm" onclick="openEditTemplate('${t.id}')">Edit</button>` : ""}
          ${!t.is_builtin ? `<button class="btn btn-danger btn-sm" onclick="deleteTemplate('${t.id}','${escHtml(t.name)}')">Delete</button>` : ""}
          <button class="btn btn-secondary btn-sm" onclick="cloneTemplate('${t.id}')">Clone</button>
        </td>
      </tr>`;
    }).join("");
    const orgHeader = showOrg ? "<th>Organization</th>" : "";
    document.getElementById("templates-table").innerHTML = `
      <table><thead><tr>
        <th>Name</th><th>Description</th>${orgHeader}<th>Allowed permissions</th><th></th>
      </tr></thead><tbody>${rows}</tbody></table>`;
  } catch (e) {
    document.getElementById("templates-table").innerHTML = `<div class="empty">${escHtml(e.message)}</div>`;
  }
}

function openCreateTemplate() {
  document.getElementById("tpl-modal-title").textContent = "New Permission Template";
  document.getElementById("tpl-id").value = "";
  document.getElementById("tpl-name").value = "";
  document.getElementById("tpl-desc").value = "";
  _BOOL_FLAGS.forEach(f => {
    const el = document.getElementById(`tpl-${f}`);
    if (el) el.checked = f === "can_view_wiki" || f === "can_view_graph" || f === "can_download_files";
  });
  _NUM_FIELDS.forEach(f => { const el = document.getElementById(`tpl-${f}`); if (el) el.value = ""; });
  _TOKEN_FIELDS.forEach(f => { const el = document.getElementById(`tpl-${f}`); if (el) el.value = ""; });
  openModal("modal-template");
}

function openEditTemplate(id) {
  const t = _templates.find(x => x.id === id);
  if (!t) return;
  document.getElementById("tpl-modal-title").textContent = `Edit Template — ${t.name}`;
  document.getElementById("tpl-id").value = t.id;
  document.getElementById("tpl-name").value = t.name;
  document.getElementById("tpl-desc").value = t.description;
  _BOOL_FLAGS.forEach(f => { const el = document.getElementById(`tpl-${f}`); if (el) el.checked = !!t[f]; });
  _NUM_FIELDS.forEach(f => { const el = document.getElementById(`tpl-${f}`); if (el) el.value = t[f] ?? ""; });
  _TOKEN_FIELDS.forEach(f => { const el = document.getElementById(`tpl-${f}`); if (el) el.value = tokensToMillionsStr(t[f]); });
  openModal("modal-template");
}

function _collectTemplateBody() {
  const body = {
    name: document.getElementById("tpl-name").value.trim(),
    description: document.getElementById("tpl-desc").value.trim(),
  };
  _BOOL_FLAGS.forEach(f => { const el = document.getElementById(`tpl-${f}`); if (el) body[f] = el.checked; });
  _NUM_FIELDS.forEach(f => {
    const el = document.getElementById(`tpl-${f}`);
    if (el) body[f] = el.value !== "" ? parseInt(el.value) : null;
  });
  _TOKEN_FIELDS.forEach(f => {
    const el = document.getElementById(`tpl-${f}`);
    if (el) body[f] = millionsStrToTokens(el.value);
  });
  return body;
}

async function saveTemplate() {
  const id = document.getElementById("tpl-id").value;
  const body = _collectTemplateBody();
  if (!body.name) { toast("Template name is required", "error"); return; }
  try {
    if (id) {
      await apiFetch(`/admin/templates/${id}`, { method: "PUT", body: JSON.stringify(body) });
    } else {
      await apiFetch("/admin/templates", { method: "POST", body: JSON.stringify(body) });
    }
    closeModal("modal-template");
    toast(id ? "Template updated" : "Template created");
    loadTemplates();
  } catch (e) { toast(e.message, "error"); }
}

async function deleteTemplate(id, name) {
  let msg = `Delete template "${name}"?`;
  try {
    const { count } = await apiFetch(`/admin/templates/${id}/user-count`);
    if (count > 0) {
      const noun = count === 1 ? "user is" : "users are";
      msg += `\n\n${count} ${noun} assigned to this template. They will be switched to the "read_only" built-in template.`;
    }
  } catch (_) { /* fall through with basic message if count fetch fails */ }
  if (!(await uiConfirm({ title: "Delete template", message: msg, confirmText: "Delete", danger: true }))) return;
  try {
    await apiFetch(`/admin/templates/${id}`, { method: "DELETE" });
    toast("Template deleted");
    loadTemplates();
  } catch (e) { toast(e.message, "error"); }
}

async function cloneTemplate(id) {
  const name = await uiPrompt({ title: "Clone template", label: "Name for cloned template", confirmText: "Clone" });
  if (!name) return;
  try {
    await apiFetch(`/admin/templates/${id}/clone`, { method: "POST", body: JSON.stringify({ name }) });
    toast("Template cloned");
    loadTemplates();
  } catch (e) { toast(e.message, "error"); }
}

// ── WORKSPACES ────────────────────────────────────────────────────────────────

async function loadWorkspaces() {
  document.getElementById("workspaces-table").innerHTML = '<div class="loading">Loading…</div>';
  try {
    _workspaces = await apiFetch("/admin/workspaces") || [];
    if (!_workspaces.length) {
      document.getElementById("workspaces-table").innerHTML = '<div class="empty">No workspaces. Create one to isolate user data.</div>';
      return;
    }
    const showOrg = _myRole === "admin";
    const rows = _workspaces.map(w => {
      const orgCell = showOrg
        ? `<td style="color:var(--text-muted);font-size:0.82rem">${escHtml(w.org_name || "—")}</td>`
        : "";
      return `<tr>
        <td><b>${escHtml(w.name)}</b></td>
        ${orgCell}
        <td style="color:var(--text-muted);font-size:0.82rem">${escHtml(w.owner_email || "—")}</td>
        <td style="color:var(--text-muted);font-size:0.82rem;font-family:monospace">${escHtml(w.s3_prefix)}</td>
        <td style="color:var(--text-muted)">${w.created_at ? new Date(w.created_at).toLocaleDateString() : "—"}</td>
        <td>
          <button class="btn btn-secondary btn-sm" onclick="copyWorkspace('${w.id}','${escHtml(w.name)}')">Copy</button>
          <button class="btn btn-danger btn-sm" onclick="deleteWorkspace('${w.id}','${escHtml(w.name)}')">Delete</button>
        </td>
      </tr>`;
    }).join("");
    const orgHeader = showOrg ? "<th>Organization</th>" : "";
    document.getElementById("workspaces-table").innerHTML = `
      <table><thead><tr>
        <th>Name</th>${orgHeader}<th>Owner</th><th>S3 Prefix</th><th>Created</th><th></th>
      </tr></thead><tbody>${rows}</tbody></table>`;
  } catch (e) {
    document.getElementById("workspaces-table").innerHTML = `<div class="empty">${escHtml(e.message)}</div>`;
  }
}

function openCreateWorkspace() {
  document.getElementById("ws-modal-title").textContent = "New Workspace";
  document.getElementById("ws-name").value = "";
  openModal("modal-workspace");
}

async function saveWorkspace() {
  const name = document.getElementById("ws-name").value.trim();
  if (!name) { toast("Workspace name required", "error"); return; }
  try {
    await apiFetch("/admin/workspaces", { method: "POST", body: JSON.stringify({ name }) });
    closeModal("modal-workspace");
    toast("Workspace created");
    loadWorkspaces();
  } catch (e) { toast(e.message, "error"); }
}

async function copyWorkspace(id, srcName) {
  const name = await uiPrompt({
    title: `Copy workspace "${srcName}"`,
    label: "New workspace name", confirmText: "Copy",
  });
  if (!name) return;
  try {
    await apiFetch(`/admin/workspaces/${id}/copy`, { method: "POST", body: JSON.stringify({ new_name: name }) });
    toast("Workspace copied");
    loadWorkspaces();
  } catch (e) { toast(e.message, "error"); }
}

async function deleteWorkspace(id, name) {
  if (!(await uiConfirm({
    title: "Delete workspace",
    message: `Delete workspace "${name}" and all its S3 files? This cannot be undone.`,
    confirmText: "Delete", danger: true,
  }))) return;
  try {
    await apiFetch(`/admin/workspaces/${id}`, { method: "DELETE" });
    toast("Workspace deleted");
    loadWorkspaces();
  } catch (e) { toast(e.message, "error"); }
}

// ── USAGE ─────────────────────────────────────────────────────────────────────

let _usageData = null;

async function loadUsage() {
  const btn = document.getElementById("usage-refresh-btn");
  if (btn) btn.disabled = true;
  document.getElementById("usage-stats").innerHTML = '<div class="loading">Loading…</div>';
  const days = document.getElementById("usage-days")?.value || 30;
  try {
    const data = await apiFetch(`/admin/usage?days=${days}`);
    if (!data) return;
    _usageData = data;
    _populateUsageFilterOptions(data);
    renderUsage();
  } catch (e) {
    document.getElementById("usage-stats").innerHTML = `<div class="empty">${escHtml(e.message)}</div>`;
  } finally {
    if (btn) btn.disabled = false;
  }
}

function _populateUsageFilterOptions(data) {
  const ops    = [...new Set(data.token_usage.map(r => r.operation).filter(Boolean))].sort();
  const models = [...new Set(data.token_usage.map(r => r.model_id).filter(Boolean))].sort();

  const opSel = document.getElementById("usage-filter-op");
  const prevOp = opSel.value;
  opSel.innerHTML = '<option value="">All operations</option>' +
    ops.map(o => `<option value="${escHtml(o)}"${o === prevOp ? " selected" : ""}>${escHtml(o)}</option>`).join("");

  const modelSel = document.getElementById("usage-filter-model");
  const prevModel = modelSel.value;
  modelSel.innerHTML = '<option value="">All models</option>' +
    models.map(m => `<option value="${escHtml(m)}"${m === prevModel ? " selected" : ""}>${escHtml(m)}</option>`).join("");
}

function _applyUsageFilters(rows) {
  const userFilter  = document.getElementById("usage-filter-user")?.value.toLowerCase() || "";
  const opFilter    = document.getElementById("usage-filter-op")?.value || "";
  const modelFilter = document.getElementById("usage-filter-model")?.value || "";
  return rows.filter(r =>
    (!userFilter  || (r.email  || "").toLowerCase().includes(userFilter)) &&
    (!opFilter    || r.operation === opFilter) &&
    (!modelFilter || r.model_id  === modelFilter)
  );
}

function _groupUsageRows(rows, groupBy) {
  if (groupBy === "detail") return rows;
  const keyFn = {
    "user":       r => r.user_id || r.email,
    "user+op":    r => `${r.user_id}||${r.operation}`,
    "user+model": r => `${r.user_id}||${r.model_id}`,
  }[groupBy];
  const map = new Map();
  for (const r of rows) {
    const k = keyFn(r);
    if (!map.has(k)) {
      map.set(k, {
        ...r,
        operation: groupBy === "user"       ? "(all)" : r.operation,
        model_id:  groupBy === "user+op"    ? "(all)" : groupBy === "user" ? "(all)" : r.model_id,
      });
    } else {
      const agg = map.get(k);
      agg.tokens_in    += r.tokens_in;
      agg.tokens_out   += r.tokens_out;
      agg.tokens_total += r.tokens_total;
      agg.call_count   += r.call_count;
    }
  }
  return [...map.values()];
}

function renderUsage() {
  if (!_usageData) return;
  const groupBy  = document.getElementById("usage-groupby")?.value || "detail";
  const filtered = _applyUsageFilters(_usageData.token_usage);
  const rows     = _groupUsageRows(filtered, groupBy).sort((a, b) => b.tokens_total - a.tokens_total);

  const totIn      = rows.reduce((s, r) => s + r.tokens_in, 0);
  const totOut     = rows.reduce((s, r) => s + r.tokens_out, 0);
  const totTotal   = rows.reduce((s, r) => s + r.tokens_total, 0);
  const totCalls   = rows.reduce((s, r) => s + r.call_count, 0);
  const totUploads = _usageData.upload_counts.reduce((s, r) => s + r.upload_count, 0);
  const uniqUsers  = new Set(filtered.map(r => r.user_id)).size;

  const statsHtml = `<div class="stat-grid">
    <div class="stat-card" title="${fmtNum(totTotal)}"><div class="val">${fmtNumCompact(totTotal)}</div><div class="lbl">Total tokens</div></div>
    <div class="stat-card" title="${fmtNum(totIn)}"><div class="val">${fmtNumCompact(totIn)}</div><div class="lbl">Tokens in</div></div>
    <div class="stat-card" title="${fmtNum(totOut)}"><div class="val">${fmtNumCompact(totOut)}</div><div class="lbl">Tokens out</div></div>
    <div class="stat-card" title="${fmtNum(totCalls)}"><div class="val">${fmtNumCompact(totCalls)}</div><div class="lbl">LLM calls</div></div>
    <div class="stat-card"><div class="val">${uniqUsers}</div><div class="lbl">Active users</div></div>
    <div class="stat-card"><div class="val">${totUploads}</div><div class="lbl">Uploads</div></div>
  </div>`;

  const tokenRows = rows.map(r => `<tr>
    <td>${escHtml(r.email || "—")}</td>
    <td><code style="font-size:0.78rem">${escHtml(r.operation || "—")}</code></td>
    <td style="color:var(--text-muted);font-size:0.78rem">${escHtml(r.model_id || "—")}</td>
    <td style="text-align:right">${fmtNum(r.tokens_in)}</td>
    <td style="text-align:right">${fmtNum(r.tokens_out)}</td>
    <td style="text-align:right"><b>${fmtNum(r.tokens_total)}</b></td>
    <td style="text-align:right;color:var(--text-muted)">${fmtNum(r.call_count)}</td>
  </tr>`).join("") || '<tr><td colspan="7" style="text-align:center;padding:20px;color:var(--text-muted)">No matching data</td></tr>';

  const uploadRows = _usageData.upload_counts.map(r => `<tr>
    <td>${escHtml(r.email || "—")}</td>
    <td style="text-align:right">${r.upload_count}</td>
    <td style="color:var(--text-muted);font-size:0.8rem">${r.last_upload_at ? new Date(r.last_upload_at).toLocaleString() : "—"}</td>
  </tr>`).join("") || '<tr><td colspan="3" style="text-align:center;padding:20px;color:var(--text-muted)">No data</td></tr>';

  document.getElementById("usage-stats").innerHTML = statsHtml +
    `<h3 style="margin:4px 0 10px;font-size:0.92rem;display:flex;align-items:center;gap:8px">
       Token Usage
       <span style="font-size:0.78rem;color:var(--text-muted);font-weight:400">${rows.length} row${rows.length !== 1 ? "s" : ""}</span>
     </h3>
     <div style="overflow-x:auto">
       <table><thead><tr>
         <th>User</th><th>Operation</th><th>Model</th>
         <th style="text-align:right">Tokens In</th>
         <th style="text-align:right">Tokens Out</th>
         <th style="text-align:right">Total</th>
         <th style="text-align:right">Calls</th>
       </tr></thead><tbody>${tokenRows}</tbody></table>
     </div>
     <h3 style="margin:20px 0 10px;font-size:0.92rem">Upload Counts</h3>
     <div style="overflow-x:auto">
       <table><thead><tr>
         <th>User</th><th style="text-align:right">Uploads</th><th>Last Upload</th>
       </tr></thead><tbody>${uploadRows}</tbody></table>
     </div>`;
}

// ── ORG LIMITS ────────────────────────────────────────────────────────────────

async function loadOrgLimits() {
  document.getElementById("org-limits-form").innerHTML = '<div class="loading">Loading…</div>';
  try {
    const data = await apiFetch("/admin/org-limits");
    if (!data) return;
    if (_myRole === "admin" && Array.isArray(data)) {
      // Admin view: table of all orgs with inline edit
      const rows = data.map(o => `<tr>
        <td><b>${escHtml(o.name)}</b></td>
        <td><input type="number" min="1" placeholder="unlimited" value="${o.max_uploads_per_day_org ?? ""}"
          style="width:100px" onchange="saveOrgLimitsInline('${o.id}',this.closest('tr'))"></td>
        <td><input type="number" min="0" step="0.1" placeholder="unlimited" value="${tokensToMillionsStr(o.max_tokens_per_day_org)}"
          style="width:120px" onchange="saveOrgLimitsInline('${o.id}',this.closest('tr'))"
          title="In millions of tokens (e.g. 1.5 = 1,500,000)"></td>
        <td><input type="number" min="1" placeholder="unlimited" value="${o.max_members ?? ""}"
          style="width:90px" onchange="saveOrgLimitsInline('${o.id}',this.closest('tr'))"></td>
        <td><input type="number" min="1" value="${o.revision_retention_days ?? 21}"
          style="width:80px" onchange="saveOrgLimitsInline('${o.id}',this.closest('tr'))"></td>
        <td><button class="btn btn-primary btn-sm" onclick="saveOrgLimitsInline('${o.id}',this.closest('tr'))">Save</button></td>
      </tr>`).join("");
      document.getElementById("org-limits-form").innerHTML = `
        <div style="overflow-x:auto">
          <table><thead><tr>
            <th>Organization</th><th>Max uploads/day</th><th title="In millions of tokens">Max tokens/day (M)</th><th>Max members</th><th title="Days of revert history to keep">Retention (days)</th><th></th>
          </tr></thead><tbody>${rows}</tbody></table>
        </div>
        <p style="margin-top:12px;font-size:0.82rem;color:var(--text-muted)">
          Org-level limits apply on top of per-user template limits (whichever is stricter wins). Retention controls how long change-history is kept for the History tab; older actions are auto-pruned and become non-revertable.
        </p>`;
    } else {
      const org = data;
      document.getElementById("org-limits-form").innerHTML = `
        <div class="form-grid" style="max-width:600px">
          <div class="field">
            <label>Max uploads / day (org-wide, blank = unlimited)</label>
            <input type="number" id="ol-max_uploads" min="1" value="${org.max_uploads_per_day_org ?? ""}" placeholder="unlimited">
          </div>
          <div class="field">
            <label>Max tokens / day (millions, org-wide, blank = unlimited)</label>
            <input type="number" id="ol-max_tokens" min="0" step="0.1" value="${tokensToMillionsStr(org.max_tokens_per_day_org)}" placeholder="unlimited">
            <p style="font-size:0.78rem;color:var(--text-muted);margin-top:3px">Enter in millions. 1 = 1,000,000 tokens; 1.5 = 1,500,000.</p>
          </div>
          <div class="field">
            <label>Max members (blank = unlimited)</label>
            <input type="number" id="ol-max_members" min="1" value="${org.max_members ?? ""}" placeholder="unlimited">
          </div>
          <div class="field">
            <label>Revert history retention (days)</label>
            <input type="number" id="ol-retention_days" min="1" value="${org.revision_retention_days ?? 21}">
            <p style="font-size:0.78rem;color:var(--text-muted);margin-top:3px">How long change-history is kept on the History tab. Older actions are auto-pruned and become non-revertable. Default 21.</p>
          </div>
          <div class="field" style="display:flex;align-items:flex-end">
            <button class="btn btn-primary" onclick="saveOrgLimits()">Save Limits</button>
          </div>
        </div>
        <p style="margin-top:12px;font-size:0.82rem;color:var(--text-muted)">
          Org-level limits apply on top of per-user template limits (whichever is stricter wins).
        </p>`;
    }
  } catch (e) {
    document.getElementById("org-limits-form").innerHTML = `<div class="empty">${escHtml(e.message)}</div>`;
  }
}

async function saveOrgLimitsInline(orgId, row) {
  const inputs = row.querySelectorAll("input[type=number]");
  const toInt = el => el.value ? parseInt(el.value) : null;
  try {
    await apiFetch("/admin/org-limits", {
      method: "PUT",
      body: JSON.stringify({
        org_id: orgId,
        max_uploads_per_day_org: toInt(inputs[0]),
        max_tokens_per_day_org:  millionsStrToTokens(inputs[1].value),
        max_members:             toInt(inputs[2]),
        revision_retention_days: toInt(inputs[3]),
      }),
    });
    toast("Limits saved");
  } catch (e) { toast(e.message, "error"); }
}

async function saveOrgLimits() {
  const toInt = id => { const v = document.getElementById(id).value; return v ? parseInt(v) : null; };
  try {
    await apiFetch("/admin/org-limits", {
      method: "PUT",
      body: JSON.stringify({
        max_uploads_per_day_org: toInt("ol-max_uploads"),
        max_tokens_per_day_org:  millionsStrToTokens(document.getElementById("ol-max_tokens").value),
        max_members:             toInt("ol-max_members"),
        revision_retention_days: toInt("ol-retention_days"),
      }),
    });
    toast("Organization limits saved");
  } catch (e) { toast(e.message, "error"); }
}

// ── AUDIT LOG ─────────────────────────────────────────────────────────────────

let _auditDebounce;
let _auditSort = "desc";  // "desc" = newest first, "asc" = oldest first
let _auditEntriesById = new Map();  // cached for the details modal

document.getElementById?.("audit-search")?.addEventListener("input", () => {
  clearTimeout(_auditDebounce);
  _auditDebounce = setTimeout(() => { _auditPage = 1; loadAudit(); }, 300);
});

async function _initAuditOrgFilter() {
  const sel = document.getElementById("audit-org-filter");
  if (!sel || _myRole !== "admin") return;
  sel.style.display = "";
  if (sel.options.length > 1) return; // already populated
  try {
    const orgs = await apiFetch("/admin/organizations");
    if (!orgs) return;
    orgs.forEach(o => {
      const opt = document.createElement("option");
      opt.value = o.id;
      opt.textContent = o.name;
      sel.appendChild(opt);
    });
  } catch (_) {}
}

function _onAuditOrgChange() {
  _auditPage = 1;
  // Operation list is org-scoped; refresh it on org change.
  document.getElementById("audit-operation-filter").innerHTML = '<option value="">All operations</option>';
  _auditOperationsLoaded = false;
  loadAudit();
}

let _auditOperationsLoaded = false;
async function _loadAuditOperations() {
  if (_auditOperationsLoaded) return;
  const sel = document.getElementById("audit-operation-filter");
  if (!sel) return;
  const orgFilter = document.getElementById("audit-org-filter")?.value || "";
  const params = orgFilter ? `?org_filter=${encodeURIComponent(orgFilter)}` : "";
  try {
    const ops = await apiFetch(`/admin/audit-log/operations${params}`);
    if (Array.isArray(ops)) {
      ops.forEach(op => {
        const opt = document.createElement("option");
        opt.value = op;
        opt.textContent = op;
        sel.appendChild(opt);
      });
    }
  } catch (_) { /* leave empty */ }
  _auditOperationsLoaded = true;
}

function _toggleAuditSort() {
  _auditSort = _auditSort === "desc" ? "asc" : "desc";
  document.getElementById("audit-sort-icon").textContent = _auditSort === "desc" ? "↓" : "↑";
  document.getElementById("audit-sort-btn").lastChild.textContent =
    _auditSort === "desc" ? " Newest first" : " Oldest first";
  _auditPage = 1;
  loadAudit();
}

async function loadAudit() {
  document.getElementById("audit-table").innerHTML = '<div class="loading">Loading…</div>';
  const search = document.getElementById("audit-search")?.value || "";
  const operation = document.getElementById("audit-operation-filter")?.value || "";
  const orgFilter = document.getElementById("audit-org-filter")?.value || "";
  await _initAuditOrgFilter();
  await _loadAuditOperations();
  try {
    const params = new URLSearchParams({
      page: _auditPage, limit: 50, search, sort: _auditSort,
    });
    if (operation) params.set("operation", operation);
    if (orgFilter) params.set("org_filter", orgFilter);
    const data = await apiFetch(`/admin/audit-log?${params}`);
    if (!data) return;

    if (!data.entries.length) {
      document.getElementById("audit-table").innerHTML = '<div class="empty">No entries found.</div>';
      document.getElementById("audit-pager").innerHTML = "";
      return;
    }

    _auditEntriesById = new Map(data.entries.map(e => [String(e.id), e]));

    const isAdmin = _myRole === "admin";
    const headers = isAdmin
      ? "<th>Time</th><th>Org</th><th>User</th><th>Operation</th><th>Summary</th>"
      : "<th>Time</th><th>User</th><th>Operation</th><th>Summary</th>";

    const rows = data.entries.map(e => {
      const time = `<td style="color:var(--text-muted);font-size:0.8rem;white-space:nowrap">${e.created_at ? new Date(e.created_at).toLocaleString() : "—"}</td>`;
      const orgCol = isAdmin ? `<td style="font-size:0.8rem;color:var(--text-muted)">${escHtml(e.org_name || "—")}</td>` : "";
      const userCol = `<td style="font-size:0.8rem;color:var(--text-muted)">${escHtml(e.user_email || "—")}</td>`;
      const op = `<td><code style="font-size:0.8rem">${escHtml(e.operation)}</code></td>`;
      // Show only the first line; full text + JSON details live in the modal.
      const firstLine = (e.raw_text || "").split("\n")[0].slice(0, 120);
      const ellipsis = (e.raw_text || "").length > firstLine.length ? "…" : "";
      const summary = `<td style="color:var(--text-muted);font-size:0.82rem;max-width:540px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${escHtml(firstLine)}${ellipsis}</td>`;
      return `<tr style="cursor:pointer" onclick="openAuditDetails('${e.id}')">${time}${orgCol}${userCol}${op}${summary}</tr>`;
    }).join("");

    document.getElementById("audit-table").innerHTML = `
      <div style="color:var(--text-muted);font-size:0.82rem;margin-bottom:10px">${data.total} total entries</div>
      <div style="overflow-x:auto;max-height:480px;overflow-y:auto;border:1px solid var(--border);border-radius:6px">
        <table><thead style="position:sticky;top:0;z-index:1"><tr style="background:var(--surface)">${headers}</tr></thead>
        <tbody>${rows}</tbody></table>
      </div>`;

    // Pager
    const totalPages = Math.max(1, Math.ceil(data.total / data.limit));
    let pagerHtml = "";
    if (_auditPage > 1)
      pagerHtml += `<button class="btn btn-secondary btn-sm" onclick="_auditPage--;loadAudit()">← Prev</button>`;
    pagerHtml += `<span style="color:var(--text-muted);font-size:0.85rem;padding:0 8px">Page ${_auditPage} / ${totalPages}</span>`;
    if (_auditPage < totalPages)
      pagerHtml += `<button class="btn btn-secondary btn-sm" onclick="_auditPage++;loadAudit()">Next →</button>`;
    document.getElementById("audit-pager").innerHTML = pagerHtml;
  } catch (e) {
    document.getElementById("audit-table").innerHTML = `<div class="empty">${escHtml(e.message)}</div>`;
  }
}

function openAuditDetails(id) {
  const entry = _auditEntriesById.get(String(id));
  if (!entry) return;
  document.getElementById("audit-details-op").textContent = entry.operation;
  const when = entry.created_at ? new Date(entry.created_at).toLocaleString() : "—";
  const parts = [when];
  if (entry.user_email) parts.push(entry.user_email);
  if (entry.org_name)   parts.push(entry.org_name);
  document.getElementById("audit-details-meta").textContent = parts.join(" · ");
  document.getElementById("audit-details-text").textContent = entry.raw_text || "(no message)";
  const json = entry.details && Object.keys(entry.details).length > 0
    ? JSON.stringify(entry.details, null, 2)
    : "(no structured details)";
  document.getElementById("audit-details-json").textContent = json;
  openModal("modal-audit-details");
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function escHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, c =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function fmtNum(n) {
  if (n == null) return "—";
  return Number(n).toLocaleString();
}

// ── Token ↔ millions conversion (UI only; DB stores raw tokens) ──────────────
// 1 unit in the input = 1,000,000 tokens. Empty input → null (unlimited).
function tokensToMillionsStr(tokens) {
  if (tokens == null) return "";
  return (Number(tokens) / 1_000_000).toString();
}

function millionsStrToTokens(s) {
  if (s == null || s === "") return null;
  const m = parseFloat(s);
  if (!isFinite(m) || m < 0) return null;
  return Math.round(m * 1_000_000);
}

const _compactFmt = new Intl.NumberFormat(undefined, { notation: "compact", maximumFractionDigits: 1 });
function fmtNumCompact(n) {
  if (n == null) return "—";
  const num = Number(n);
  return Math.abs(num) >= 10000 ? _compactFmt.format(num) : num.toLocaleString();
}

// ── Init ──────────────────────────────────────────────────────────────────────

(async () => {
  if (!_token) { window.location.href = "/"; return; }

  // Check role — only admin and supervisor may use this dashboard
  try {
    const me = await apiFetch("/admin/me");
    if (!me || !["admin", "supervisor"].includes(me.role)) {
      window.location.href = "/"; return;
    }
    _myRole  = me.role;
    _myOrgId = me.org_id;

    // A supervisor of multiple orgs gets an admin-like experience scoped to the
    // orgs they supervise: a header switcher selects the active org, which is
    // sent as X-Org-Context on every request.
    _setupSupervisorOrgSwitcher(me);

    // Populate user menu dropdown
    const emailEl = document.getElementById("adm-umi-email");
    const roleEl  = document.getElementById("adm-umi-role");
    const labelEl = document.getElementById("adm-user-label");
    if (emailEl) emailEl.textContent = me.email || "—";
    if (roleEl)  roleEl.textContent  = me.role  || "—";
    if (labelEl) labelEl.textContent = me.email ? me.email.split("@")[0] : me.role;

    // Wire dropdown toggle
    const btn  = document.getElementById("adm-user-btn");
    const drop = document.getElementById("adm-user-dropdown");
    if (btn && drop) {
      btn.addEventListener("click", e => { e.stopPropagation(); drop.classList.toggle("open"); });
      document.addEventListener("click", () => drop.classList.remove("open"));
      drop.addEventListener("click", e => e.stopPropagation());
    }

    // Wire sign-out
    const signoutBtn = document.getElementById("adm-btn-signout");
    if (signoutBtn) {
      signoutBtn.addEventListener("click", () => {
        localStorage.removeItem("auth_token");
        localStorage.removeItem("refresh_token");
        localStorage.removeItem("user_role");
        localStorage.removeItem("user_email");
        localStorage.removeItem("user_org_name");
        localStorage.removeItem("user_perms");
        window.location.href = "/";
      });
    }

    // Update page title with company name if available
    try {
      const cfg = await apiFetch("/auth/config");
      if (cfg && cfg.company_name) {
        document.getElementById("hdr-title").textContent = `${cfg.company_name} — Admin`;
        document.title = `Admin Dashboard — ${cfg.company_name}`;
      }
    } catch (_) {}

    // Supervisors cannot create/delete orgs — hide the Organizations tab
    if (_myRole !== "admin") {
      document.querySelectorAll(".tab[data-tab='orgs']").forEach(t => t.style.display = "none");
    }
  } catch (_) {
    window.location.href = "/"; return;
  }
  await loadUsers();
  await loadTemplates();

  // Schema editor button wiring
  document.getElementById("btn-schema-preview").addEventListener("click", previewSchemaValidation);
  document.getElementById("btn-schema-save").addEventListener("click", saveSchema);
  document.getElementById("btn-schema-revert").addEventListener("click", () => {
    openModal("modal-schema-revert");
  });
  document.getElementById("btn-schema-rules").addEventListener("click", () => {
    loadSchemaRules();   // fetch + render the rule list (once)
    openModal("modal-schema-rules");
  });
  document.getElementById("btn-schema-revert-confirm").addEventListener("click", revertSchemaToDefault);
  document.getElementById("schema-org-select").addEventListener("change", _onSchemaOrgChange);
  // Live preview refresh on typing (debounced)
  let _previewTimer = null;
  document.getElementById("schema-editor").addEventListener("input", () => {
    const activeTab = document.querySelector(".schema-rtab[data-ltab].active");
    if (activeTab && activeTab.dataset.ltab === "preview") {
      clearTimeout(_previewTimer);
      _previewTimer = setTimeout(_refreshPreview, 300);
    }
  });
})();

// ── Schema Editor ─────────────────────────────────────────────────────────────

let _schemaOriginal = "";
let _schemaOrgSchemas = [];   // [{org_id, org_name, content}] — admin only
let _schemaViewingOrgId = ""; // "" = current user's org (supervisor context)

// ── Left-pane tab switching (Editor / Preview) ────────────────────────────────

document.querySelectorAll(".schema-rtab[data-ltab]").forEach(tab => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".schema-rtab[data-ltab]").forEach(t => t.classList.remove("active"));
    document.querySelectorAll(".schema-lpanel").forEach(p => p.classList.remove("active"));
    tab.classList.add("active");
    document.getElementById(`lpanel-${tab.dataset.ltab}`).classList.add("active");
    if (tab.dataset.ltab === "preview") _refreshPreview();
  });
});

function _refreshPreview() {
  const content = document.getElementById("schema-editor").value;
  const el = document.getElementById("schema-preview-body");
  if (typeof marked !== "undefined") {
    el.innerHTML = marked.parse(content);
  } else {
    el.textContent = content;
  }
}

// ── Divider drag ──────────────────────────────────────────────────────────────

(function initSchemaDivider() {
  const layout   = document.getElementById("schema-layout");
  const divider  = document.getElementById("schema-divider");
  const leftPane = document.getElementById("schema-editor-pane");
  let dragging = false, startX = 0, startW = 0;

  divider.addEventListener("mousedown", e => {
    dragging = true;
    startX = e.clientX;
    startW = leftPane.getBoundingClientRect().width;
    divider.classList.add("dragging");
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
    e.preventDefault();
  });

  document.addEventListener("mousemove", e => {
    if (!dragging) return;
    const totalW = layout.getBoundingClientRect().width;
    const newW = Math.max(200, Math.min(totalW - 220, startW + (e.clientX - startX)));
    leftPane.style.flex = `0 0 ${newW}px`;
  });

  document.addEventListener("mouseup", () => {
    if (!dragging) return;
    dragging = false;
    divider.classList.remove("dragging");
    document.body.style.cursor = "";
    document.body.style.userSelect = "";
  });
})();

// ── Load / org dropdown ───────────────────────────────────────────────────────

async function loadSchema() {
  try {
    if (_myRole === "admin") {
      await _loadAdminOrgSchemas();
    } else {
      const data = await apiFetch("/ops/schema");
      _schemaOriginal = (data && data.content != null) ? data.content : "";
      document.getElementById("schema-editor").value = _schemaOriginal;
      _resetSchemaDiffPane();
    }
  } catch (e) {
    toast("Failed to load schema: " + e.message, "error");
  }
}

// Acceptance rules are rendered from GET /ops/schema/rules (the same SCHEMA_RULES
// the validator uses) so the guidance can't drift from what's enforced. Fetched
// once per session; backtick spans in each description render as inline <code>.
let _schemaRulesLoaded = false;
async function loadSchemaRules() {
  if (_schemaRulesLoaded) return;
  const host = document.getElementById("schema-rules-list");
  if (!host) return;
  try {
    const data = await apiFetch("/ops/schema/rules");
    const rules = (data && data.rules) || [];
    const fmt = (desc) => escHtml(desc).replace(/`([^`]+)`/g, "<code>$1</code>");
    const block = (cls, label, sev) => {
      const items = rules.filter(r => r.severity === sev);
      if (!items.length) return "";
      return `<p class="schema-rules-h ${cls}">${label}</p><ul>` +
        items.map(r => `<li>${fmt(r.description)}</li>`).join("") + "</ul>";
    };
    host.innerHTML =
      block("req", "Required — save is rejected if any of these fail", "error") +
      block("warn", "Warnings — the save still goes through, but you'll be flagged", "warning");
    _schemaRulesLoaded = true;
  } catch (e) {
    host.innerHTML = `<p class="schema-rules-intro" style="margin:8px 0 0">Couldn't load validation rules: ${escHtml(e.message)}</p>`;
  }
}

async function _loadAdminOrgSchemas() {
  const sel = document.getElementById("schema-org-select");
  sel.style.display = "";
  try {
    _schemaOrgSchemas = await apiFetch("/ops/schema/orgs") || [];
  } catch {
    _schemaOrgSchemas = [];
  }
  // Populate dropdown
  sel.innerHTML = _schemaOrgSchemas.map(o =>
    `<option value="${escHtml(o.org_id)}">${escHtml(o.org_name)}</option>`
  ).join("");
  if (_schemaOrgSchemas.length === 0) {
    sel.innerHTML = "<option>No organizations</option>";
    return;
  }
  _onSchemaOrgChange();
}

function _onSchemaOrgChange() {
  const sel = document.getElementById("schema-org-select");
  const orgId = sel.value;
  const entry = _schemaOrgSchemas.find(o => o.org_id === orgId);
  _schemaViewingOrgId = orgId;
  _schemaOriginal = entry ? entry.content : "";
  document.getElementById("schema-editor").value = _schemaOriginal;
  const lbl = document.getElementById("schema-org-label");
  lbl.textContent = entry ? entry.org_name : "";
  lbl.style.display = entry ? "" : "none";
  _resetSchemaDiffPane();
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function _resetSchemaDiffPane() {
  document.getElementById("schema-diff-body").innerHTML = `
    <div class="schema-diff-hint">
      <svg viewBox="0 0 16 16" fill="none" width="18" height="18"><path d="M2 4h5M2 8h5M2 12h5M9 4h5M9 8h5M9 12h5" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/></svg>
      Click <strong>Preview &amp; Validate</strong> to see a diff of your edits and run validation checks before saving.
    </div>`;
  document.getElementById("schema-val-summary").innerHTML = "";
  document.getElementById("btn-schema-save").disabled = true;
  // Reset preview pane placeholder
  document.getElementById("schema-preview-body").innerHTML =
    '<p style="color:var(--text-muted);font-size:0.85rem">Switch to this tab while editing to see a live markdown render of the schema.</p>';
}

function _computeDiff(oldText, newText) {
  const oldLines = oldText.split("\n");
  const newLines = newText.split("\n");
  const m = oldLines.length, n = newLines.length;
  const result = [];
  let oi = 0, ni = 0;
  const lcs = Array.from({ length: m + 1 }, () => new Array(n + 1).fill(0));
  for (let i = m - 1; i >= 0; i--) {
    for (let j = n - 1; j >= 0; j--) {
      if (oldLines[i] === newLines[j]) lcs[i][j] = 1 + lcs[i + 1][j + 1];
      else lcs[i][j] = Math.max(lcs[i + 1][j], lcs[i][j + 1]);
    }
  }
  while (oi < m || ni < n) {
    if (oi < m && ni < n && oldLines[oi] === newLines[ni]) {
      result.push({ type: "context", text: oldLines[oi] }); oi++; ni++;
    } else if (ni < n && (oi >= m || lcs[oi][ni + 1] >= lcs[oi + 1][ni])) {
      result.push({ type: "added", text: newLines[ni] }); ni++;
    } else {
      result.push({ type: "removed", text: oldLines[oi] }); oi++;
    }
  }
  return result;
}

function _renderDiff(diffLines) {
  const CONTEXT_PAD = 3;
  const changed = new Set();
  diffLines.forEach((l, i) => { if (l.type !== "context") changed.add(i); });
  const visible = new Set();
  changed.forEach(i => {
    for (let d = -CONTEXT_PAD; d <= CONTEXT_PAD; d++) {
      if (i + d >= 0 && i + d < diffLines.length) visible.add(i + d);
    }
  });
  const frag = document.createDocumentFragment();
  let lastVisible = -1;
  diffLines.forEach((line, i) => {
    if (!visible.has(i)) return;
    if (lastVisible !== -1 && i > lastVisible + 1) {
      const hunk = document.createElement("div");
      hunk.className = "diff-line hunk";
      hunk.textContent = `@@ … ${i - lastVisible - 1} lines hidden …`;
      frag.appendChild(hunk);
    }
    const div = document.createElement("div");
    div.className = `diff-line ${line.type}`;
    div.textContent = (line.type === "added" ? "+ " : line.type === "removed" ? "- " : "  ") + line.text;
    frag.appendChild(div);
    lastVisible = i;
  });
  if (!changed.size) {
    const div = document.createElement("div");
    div.className = "diff-line context";
    div.textContent = "  (no changes)";
    frag.appendChild(div);
  }
  const body = document.getElementById("schema-diff-body");
  body.innerHTML = "";
  body.appendChild(frag);
}

function _renderValidation(result) {
  const errCount  = (result.errors   || []).length;
  const warnCount = (result.warnings || []).length;
  let html = "";
  if (errCount)  html += `<span class="schema-val-badge error">${errCount} error${errCount > 1 ? "s" : ""}</span>`;
  if (warnCount) html += `<span class="schema-val-badge warning">${warnCount} warning${warnCount > 1 ? "s" : ""}</span>`;
  if (!errCount && !warnCount) html = `<span class="schema-val-badge ok">Validation passed</span>`;
  document.getElementById("schema-val-summary").innerHTML = html;
  const body = document.getElementById("schema-diff-body");
  if (errCount || warnCount) {
    const block = document.createElement("div");
    block.className = "schema-val-block";
    if (errCount) {
      const h = document.createElement("p"); h.className = "schema-val-heading error"; h.textContent = "Errors"; block.appendChild(h);
      result.errors.forEach(msg => { const p = document.createElement("p"); p.className = "schema-val-item"; p.textContent = "✗ " + msg; block.appendChild(p); });
    }
    if (warnCount) {
      const h = document.createElement("p"); h.className = "schema-val-heading warning"; h.textContent = "Warnings"; block.appendChild(h);
      result.warnings.forEach(msg => { const p = document.createElement("p"); p.className = "schema-val-item"; p.textContent = "⚠ " + msg; block.appendChild(p); });
    }
    body.insertBefore(block, body.firstChild);
    body.scrollTop = 0;
  }
  document.getElementById("btn-schema-save").disabled = errCount > 0;
}

// ── Actions ───────────────────────────────────────────────────────────────────

async function previewSchemaValidation() {
  const content = document.getElementById("schema-editor").value;
  const btn = document.getElementById("btn-schema-preview");
  btn.disabled = true; btn.textContent = "Validating…";
  const extraHeaders = (_myRole === "admin" && _schemaViewingOrgId)
    ? { "X-Org-Context": _schemaViewingOrgId } : {};
  try {
    const result = await apiFetch("/ops/schema/validate", { method: "POST", body: JSON.stringify({ content }), headers: extraHeaders });
    _renderDiff(_computeDiff(_schemaOriginal, content));
    _renderValidation(result);
  } catch (e) {
    toast("Validation failed: " + e.message, "error");
  } finally {
    btn.disabled = false; btn.textContent = "Preview & Validate";
  }
}

async function saveSchema() {
  const content = document.getElementById("schema-editor").value;
  const btn = document.getElementById("btn-schema-save");
  btn.disabled = true; btn.textContent = "Saving…";
  // If admin is viewing a specific org, set X-Org-Context for this request
  const extraHeaders = (_myRole === "admin" && _schemaViewingOrgId)
    ? { "X-Org-Context": _schemaViewingOrgId } : {};
  try {
    await apiFetch("/ops/schema", { method: "PUT", body: JSON.stringify({ content }), headers: extraHeaders });
    _schemaOriginal = content;
    // Update cached org schema if admin
    const entry = _schemaOrgSchemas.find(o => o.org_id === _schemaViewingOrgId);
    if (entry) entry.content = content;
    toast("Schema saved successfully.", "success");
    btn.disabled = false; btn.textContent = "Save Schema";
  } catch (e) {
    toast("Save failed: " + e.message, "error");
    btn.disabled = false; btn.textContent = "Save Schema";
  }
}

async function revertSchemaToDefault() {
  closeModal("modal-schema-revert");
  try {
    const data = await apiFetch("/ops/schema/default");
    const text = (data && data.content != null) ? data.content : "";
    document.getElementById("schema-editor").value = text;
    _resetSchemaDiffPane();
    toast("Default schema loaded. Review and save when ready.", "success");
  } catch (e) {
    toast("Failed to load default schema: " + e.message, "error");
  }
}


// ── HISTORY (CHANGE TRACKING + REVERT) ──────────────────────────────────────

let _historyPage = 1;
let _pendingRevertId = null;
let _pendingRevertToken = "";
let _historyViewingOrgId = ""; // "" = current user's own org (supervisor); admins must pick

const HISTORY_TYPE_LABELS = {
  manual_edit: "Manual edit",
  manual_delete: "Manual delete",
  document_upload: "Document upload",
  document_delete: "Document delete",
  upload_ingest: "Upload + ingest",
  saved_query: "Saved Q&A",
  recalibrate: "Recalibration",
  targeted_recalibrate: "Targeted recalibration",
  schema_update: "Schema update",
  revert: "Revert",
};

// Per-type accent colour (same language as the member-facing activity feed).
const HISTORY_TYPE_TONE = {
  manual_edit: "#4d9fff", manual_delete: "#e05252",
  document_upload: "#4caf50", document_delete: "#e05252",
  upload_ingest: "#4caf50", saved_query: "#00c9b1",
  recalibrate: "#9b6dff", targeted_recalibrate: "#9b6dff",
  schema_update: "#f59e0b", revert: "#f59e0b",
};

// Coloured pill for an action type, using admin's translucent-badge idiom.
function _typeBadge(actionType) {
  const label = HISTORY_TYPE_LABELS[actionType] || actionType;
  const tone = HISTORY_TYPE_TONE[actionType] || "#8892a4";
  return `<span class="badge" style="background:${tone}22;color:${tone}">${_esc(label)}</span>`;
}

// Small coloured pill for a revision op (create / update / delete).
function _opBadge(op) {
  const tone = { create: "#4caf50", update: "#4d9fff", delete: "#e05252" }[op] || "#8892a4";
  return `<span class="badge" style="background:${tone}22;color:${tone}">${_esc(op)}</span>`;
}

// Human-readable byte size, e.g. 1536 → "1.5 KB".
function _fmtBytes(n) {
  if (n == null) return "—";
  if (n < 1024) return n + " B";
  if (n < 1048576) return (n / 1024).toFixed(n < 10240 ? 1 : 0) + " KB";
  return (n / 1048576).toFixed(1) + " MB";
}

function _historyOrgHeaders() {
  return (_myRole === "admin" && _historyViewingOrgId)
    ? { "X-Org-Context": _historyViewingOrgId }
    : {};
}

async function _initHistoryOrgFilter() {
  const sel = document.getElementById("history-org-filter");
  if (!sel || _myRole !== "admin") return true;
  sel.style.display = "";
  if (sel.options.length <= 1) {
    try {
      const orgs = await apiFetch("/admin/organizations");
      if (Array.isArray(orgs)) {
        orgs.forEach(o => {
          const opt = document.createElement("option");
          opt.value = o.id;
          opt.textContent = o.name;
          sel.appendChild(opt);
        });
      }
    } catch (_) { /* leave empty */ }
  }
  if (!_historyViewingOrgId) {
    // Admin must choose an org first — render a hint and bail.
    document.getElementById("history-table").innerHTML =
      `<p style="color:var(--text-muted);padding:20px;text-align:center">Select an organization above to view its change history.</p>`;
    document.getElementById("history-pager").innerHTML = "";
    return false;
  }
  return true;
}

function _onHistoryOrgChange() {
  _historyViewingOrgId = document.getElementById("history-org-filter").value;
  _historyPage = 1;
  loadHistory();
}

async function loadHistory() {
  const ready = await _initHistoryOrgFilter();
  if (!ready) return;
  const action_type = document.getElementById("history-type-filter").value;
  const params = new URLSearchParams({ page: _historyPage, limit: 50 });
  if (action_type) params.append("action_type", action_type);
  const body = document.getElementById("history-table");
  body.innerHTML = `<div class="loading">Loading…</div>`;
  try {
    const data = await apiFetch(`/admin/history?${params}`, { headers: _historyOrgHeaders() });
    _renderHistoryTable(data);
  } catch (e) {
    body.innerHTML = `<p style="color:var(--danger)">Failed to load history: ${e.message}</p>`;
  }
}

function _renderHistoryTable(data) {
  const body = document.getElementById("history-table");
  if (!data.entries || !data.entries.length) {
    body.innerHTML = `<p style="color:var(--text-muted);padding:20px;text-align:center">No history yet.</p>`;
    document.getElementById("history-pager").innerHTML = "";
    return;
  }
  const rows = data.entries.map(e => {
    const when = e.started_at ? new Date(e.started_at).toLocaleString() : "—";
    const label = HISTORY_TYPE_LABELS[e.action_type] || e.action_type;
    const statusChip = _statusChip(e.status);
    const revertBtn = (e.status === "done" || e.status === "error")
      ? `<button class="btn btn-danger btn-sm" onclick="openRevertModal('${e.id}', '${_escAttr(e.summary || label)}')">Revert</button>`
      : `<button class="btn btn-secondary btn-sm" disabled title="Cannot revert (${e.status})">Revert</button>`;
    return `<tr>
      <td>${when}</td>
      <td>${_esc(e.user_email || "—")}</td>
      <td>${_typeBadge(e.action_type)}</td>
      <td>${_esc(e.summary || "—")}</td>
      <td>${e.revision_count}</td>
      <td>${statusChip}</td>
      <td style="white-space:nowrap">
        <button class="btn btn-secondary btn-sm" onclick="viewHistoryDetails('${e.id}')">View</button>
        ${revertBtn}
      </td>
    </tr>`;
  }).join("");
  body.innerHTML = `
    <div style="color:var(--text-muted);font-size:0.82rem;margin-bottom:10px">${data.total} total actions</div>
    <div style="overflow-x:auto;max-height:520px;overflow-y:auto;border:1px solid var(--border);border-radius:6px">
      <table><thead style="position:sticky;top:0;z-index:1"><tr style="background:var(--surface)">
        <th>When</th><th>Who</th><th>Type</th><th>Summary</th><th>Changes</th><th>Status</th><th></th>
      </tr></thead><tbody>${rows}</tbody></table>
    </div>`;

  const totalPages = Math.max(1, Math.ceil(data.total / data.limit));
  let pagerHtml = "";
  if (_historyPage > 1)
    pagerHtml += `<button class="btn btn-secondary btn-sm" onclick="_historyPage--;loadHistory()">← Prev</button>`;
  pagerHtml += `<span style="color:var(--text-muted);font-size:0.85rem;padding:0 8px">Page ${_historyPage} / ${totalPages}</span>`;
  if (_historyPage < totalPages)
    pagerHtml += `<button class="btn btn-secondary btn-sm" onclick="_historyPage++;loadHistory()">Next →</button>`;
  document.getElementById("history-pager").innerHTML = pagerHtml;
}

function _statusChip(status) {
  // Hex (not var()) so the `${color}22` alpha suffix yields a valid 8-digit colour.
  const colors = {
    done: "#4caf50", error: "#e05252",
    reverted: "#f59e0b", running: "#6c63ff",
  };
  const color = colors[status] || "#8892a4";
  return `<span style="display:inline-block;padding:2px 8px;border-radius:10px;font-size:0.75rem;font-weight:600;background:${color}22;color:${color}">${status}</span>`;
}

function _esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, c =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

function _escAttr(s) {
  return _esc(s).replace(/'/g, "&#39;");
}

async function viewHistoryDetails(actionId) {
  try {
    const data = await apiFetch(`/admin/history/${actionId}`, { headers: _historyOrgHeaders() });
    const label = HISTORY_TYPE_LABELS[data.action_type] || data.action_type;
    document.getElementById("history-details-title").textContent = `${label}: ${data.summary || "(no summary)"}`;
    const started = data.started_at ? new Date(data.started_at).toLocaleString() : "—";
    document.getElementById("history-details-meta").textContent =
      `${started} · ${data.user_email || "system"} · ${data.status}`;

    const revs = data.revisions || [];
    const body = document.getElementById("history-details-revisions");
    if (!revs.length) {
      body.innerHTML = `<p style="color:var(--text-muted);padding:8px">No revisions recorded.</p>`;
    } else {
      body.innerHTML = `<table style="width:100%;font-size:0.82rem"><thead><tr>
        <th style="text-align:left">Op</th><th style="text-align:left">Kind</th>
        <th style="text-align:left">Target</th><th style="text-align:right">Before</th>
        <th style="text-align:right">After</th>
      </tr></thead><tbody>${revs.map(r => `<tr>
        <td>${_opBadge(r.op)}</td>
        <td style="color:var(--text-muted)">${_esc(r.target_kind)}</td>
        <td style="font-family:monospace">${_esc(r.target_key)}</td>
        <td style="text-align:right;color:var(--text-muted)">${_fmtBytes(r.size_before)}</td>
        <td style="text-align:right;color:var(--text-muted)">${_fmtBytes(r.size_after)}</td>
      </tr>`).join("")}</tbody></table>`;
    }
    openModal("modal-history-details");
  } catch (e) {
    toast("Failed to load details: " + e.message, "error");
  }
}

async function openRevertModal(actionId, summary) {
  _pendingRevertId = actionId;
  _pendingRevertToken = (summary || "REVERT").slice(0, 40);
  document.getElementById("history-revert-summary").textContent = summary || "(no summary)";
  document.getElementById("history-revert-confirm-token").textContent = _pendingRevertToken;
  document.getElementById("history-revert-confirm-input").value = "";
  document.getElementById("btn-history-revert-confirm").disabled = true;
  document.getElementById("history-revert-discard-list").innerHTML = `<p style="color:var(--text-muted)">Loading impact…</p>`;
  openModal("modal-history-revert");

  try {
    const data = await apiFetch(`/admin/history/${actionId}/preview-revert`, { headers: _historyOrgHeaders() });
    const others = (data.actions_to_discard || []).filter(a => a.id !== actionId);
    let html;
    if (others.length === 0) {
      html = `<p>Only this action will be reverted — nothing happened after it.</p>`;
    } else {
      html = `<p>${others.length} later action${others.length === 1 ? "" : "s"} will also be discarded:</p><ul style="margin:6px 0 0 18px">${others.map(a => {
        const label = HISTORY_TYPE_LABELS[a.action_type] || a.action_type;
        const when = a.started_at ? new Date(a.started_at).toLocaleString() : "—";
        return `<li><strong>${when}</strong> — ${_esc(label)}: ${_esc(a.summary || "")}</li>`;
      }).join("")}</ul>`;
    }
    document.getElementById("history-revert-discard-list").innerHTML = html;
  } catch (e) {
    document.getElementById("history-revert-discard-list").innerHTML =
      `<p style="color:var(--danger)">Could not preview impact: ${_esc(e.message)}</p>`;
  }
}

function _updateRevertConfirmBtn() {
  const typed = document.getElementById("history-revert-confirm-input").value;
  document.getElementById("btn-history-revert-confirm").disabled = (typed !== _pendingRevertToken);
}

document.getElementById("btn-history-revert-confirm").addEventListener("click", async () => {
  if (!_pendingRevertId) return;
  const btn = document.getElementById("btn-history-revert-confirm");
  btn.disabled = true;
  btn.textContent = "Starting…";
  try {
    // POST returns 202 with {revert_action_id, status, progress_done, progress_total}.
    // The actual work runs in a server-side background task; we poll for progress.
    const job = await apiFetch(
      `/admin/history/${_pendingRevertId}/revert`,
      { method: "POST", headers: _historyOrgHeaders() },
    );
    closeModal("modal-history-revert");
    _trackRevertJob(job.revert_action_id, job.progress_total || 0);
  } catch (e) {
    toast("Revert failed: " + e.message, "error");
  } finally {
    btn.textContent = "Revert";
    _updateRevertConfirmBtn();
  }
});


// ── Revert job tracker ─────────────────────────────────────────────────────
// Shows a sticky toast that updates with progress every 1.5s. When the job
// finishes (status='done' or 'error') the toast clears and we either reload
// the history or surface the error message.

const _revertPollers = new Map(); // revert_action_id → setTimeout handle

function _trackRevertJob(revertActionId, total) {
  const containerId = `revert-progress-${revertActionId}`;
  // Pin the org at revert-start. The poll must keep scoping to THIS revert's
  // org; re-reading _historyOrgHeaders() each tick meant switching the History
  // org filter mid-revert pointed the poll at the wrong org → 404 → a false
  // "lost contact with revert job" while the revert was actually fine.
  const orgHeaders = _historyOrgHeaders();
  const initial = total
    ? `Reverting… 0 / ${total} action(s)`
    : "Reverting… starting up";
  _showRevertProgress(containerId, initial);

  async function poll() {
    try {
      const job = await apiFetch(
        `/admin/history/revert-job/${revertActionId}`,
        { headers: orgHeaders },
      );
      if (job.status === "running") {
        const label = job.progress_total
          ? `Reverting… ${job.progress_done} / ${job.progress_total} change(s)`
          : "Reverting…";
        _showRevertProgress(containerId, label);
        _revertPollers.set(revertActionId, setTimeout(poll, 1500));
        return;
      }
      _clearRevertProgress(containerId);
      _revertPollers.delete(revertActionId);
      if (job.status === "done") {
        toast(
          `Revert finished — undid ${job.progress_done} action(s).`,
          "success",
        );
        _historyPage = 1;
        loadHistory();
      } else {
        toast(
          `Revert failed: ${job.error_message || "unknown error"}`,
          "error",
        );
        loadHistory();
      }
    } catch (e) {
      _clearRevertProgress(containerId);
      _revertPollers.delete(revertActionId);
      toast("Lost contact with revert job: " + e.message, "error");
    }
  }

  // Kick off the first poll immediately so the toast text updates fast.
  poll();
}

function _showRevertProgress(id, label) {
  let el = document.getElementById(id);
  if (!el) {
    el = document.createElement("div");
    el.id = id;
    el.style.cssText =
      "position:fixed;bottom:16px;right:16px;background:var(--surface,#222);" +
      "color:var(--text,#eee);border:1px solid var(--border,#444);" +
      "border-radius:6px;padding:10px 14px;font-size:0.85rem;" +
      "box-shadow:0 4px 12px rgba(0,0,0,0.4);z-index:9999";
    document.body.appendChild(el);
  }
  el.textContent = label;
}

function _clearRevertProgress(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}


// ── Jobs tab ────────────────────────────────────────────────────────────────
// Unified live view of all active background work. Polls every ~3s while the
// tab is open (like the other live views). Admins must pick an org; supervisors
// are scoped to their own org via the standard X-Org-Context plumbing.

let _jobsPollTimer = null;
let _jobsViewingOrgId = ""; // "" = own org (supervisor); admins must pick

const JOBS_TYPE_LABELS = {
  ingest: "Ingest",
  recalibrate: "Recalibration",
  revert: "Revert",
};

const JOBS_STATUS_LABELS = {
  queued: "Queued (planning)",
  processing: "Planning",
  pending_review: "Awaiting review",
  queued_write: "Queued for write",
  writing: "Writing pages",
  running: "Running",
};

function _jobsOrgHeaders() {
  return (_myRole === "admin" && _jobsViewingOrgId)
    ? { "X-Org-Context": _jobsViewingOrgId }
    : {};
}

async function _initJobsOrgFilter() {
  const sel = document.getElementById("jobs-org-filter");
  if (!sel || _myRole !== "admin") return true;
  sel.style.display = "";
  if (sel.options.length <= 1) {
    try {
      const orgs = await apiFetch("/admin/organizations");
      if (Array.isArray(orgs)) {
        orgs.forEach(o => {
          const opt = document.createElement("option");
          opt.value = o.id;
          opt.textContent = o.name;
          sel.appendChild(opt);
        });
      }
    } catch (_) { /* leave empty */ }
  }
  if (!_jobsViewingOrgId) {
    document.getElementById("jobs-table").innerHTML =
      `<p style="color:var(--text-muted);padding:20px;text-align:center">Select an organization above to view its active jobs.</p>`;
    return false;
  }
  return true;
}

function _onJobsOrgChange() {
  _jobsViewingOrgId = document.getElementById("jobs-org-filter").value;
  loadJobs();
}

async function loadJobs() {
  if (_jobsPollTimer) { clearTimeout(_jobsPollTimer); _jobsPollTimer = null; }
  const ready = await _initJobsOrgFilter();
  if (!ready) return;
  try {
    const data = await apiFetch(`/admin/jobs`, { headers: _jobsOrgHeaders() });
    _renderJobsTable(data);
  } catch (e) {
    document.getElementById("jobs-table").innerHTML =
      `<p style="color:var(--danger)">Failed to load jobs: ${e.message}</p>`;
  }
  // Reschedule only while the Jobs tab is still the active one.
  if (document.querySelector(".tab.active")?.dataset.tab === "jobs") {
    _jobsPollTimer = setTimeout(loadJobs, 3000);
  }
}

function _renderJobsTable(data) {
  const body = document.getElementById("jobs-table");
  const jobs = data.jobs || [];
  if (!jobs.length) {
    body.innerHTML = `<p style="color:var(--text-muted);padding:20px;text-align:center">No active jobs right now.</p>`;
    return;
  }
  const rows = jobs.map(j => {
    const typeLabel = JOBS_TYPE_LABELS[j.type] || j.type;
    let statusText = JOBS_STATUS_LABELS[j.status] || j.status;
    if (j.progress) statusText += ` (${j.progress})`;
    if (j.cancel_requested) statusText += " · cancelling…";
    const queue = j.queue_position != null ? `#${j.queue_position}` : "—";
    const when = j.started_at ? new Date(j.started_at).toLocaleString() : "—";

    let action;
    if (!j.cancellable) {
      action = `<button class="btn btn-secondary btn-sm" disabled title="${_escAttr(j.cancel_reason || "This job cannot be cancelled.")}">Cancel</button>`;
    } else if (j.cancel_requested) {
      action = `<button class="btn btn-secondary btn-sm" disabled>Cancelling…</button>`;
    } else {
      action = `<button class="btn btn-danger btn-sm" onclick="_cancelJob('${j.type}', '${_escAttr(j.id)}', '${_escAttr(j.target)}')">Cancel</button>`;
    }

    return `<tr>
      <td>${_esc(typeLabel)}</td>
      <td style="font-family:monospace">${_esc(j.target || "—")}</td>
      <td>${_esc(j.user_email || "—")}</td>
      <td>${_esc(statusText)}</td>
      <td>${when}</td>
      <td style="text-align:center">${queue}</td>
      <td style="white-space:nowrap">${action}</td>
    </tr>`;
  }).join("");

  body.innerHTML = `
    <div style="color:var(--text-muted);font-size:0.82rem;margin-bottom:10px">${jobs.length} active job(s)</div>
    <div style="overflow-x:auto;border:1px solid var(--border);border-radius:6px">
      <table><thead><tr style="background:var(--surface)">
        <th>Type</th><th>File / Target</th><th>User</th><th>Status</th><th>Started</th><th>Queue #</th><th></th>
      </tr></thead><tbody>${rows}</tbody></table>
    </div>`;
}

async function _cancelJob(type, id, target) {
  if (!(await uiConfirm({
    title: "Cancel job",
    message: `Cancel this ${JOBS_TYPE_LABELS[type] || type} job?\n\n${target || id}`,
    confirmText: "Cancel job", danger: true,
  }))) return;
  try {
    if (type === "ingest") {
      await apiFetch(`/admin/jobs/ingest/${encodeURIComponent(id)}/cancel`,
        { method: "POST", headers: _jobsOrgHeaders() });
    } else if (type === "recalibrate") {
      await apiFetch(`/admin/jobs/recalibrate/cancel`,
        { method: "POST", headers: _jobsOrgHeaders() });
    } else {
      return;
    }
    toast("Cancellation requested.", "success");
    loadJobs();
  } catch (e) {
    toast("Cancel failed: " + e.message, "error");
  }
}
