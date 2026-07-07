"""Frontend tests for inline AI editing (app.js).

The "AI Edit" button appears on a wiki page for users who can edit; it opens a
modal that streams an AI rewrite, shows a diff, and applies via PUT /api/wiki.
Network is stubbed (route interception) so nothing touches the shared DB.
"""
import json

import pytest

_PAGE_MD = "# Demo\n\n## Background\n\nOriginal body.\n"
_PAGE_URL = "**/api/wiki/concepts/demo.md"


@pytest.fixture
def app_page(page, live_server_url):
    page.goto("/")
    page.fill('input[placeholder*="sername"], input[name="username"]', "admin")
    page.fill('input[placeholder*="assword"], input[name="password"]', "admin")
    page.click('button[type="submit"], button:has-text("Login"), button:has-text("Sign in")')
    page.wait_for_selector("#app-root", state="visible")
    return page


def _route_page(page, put_seen):
    """Serve the demo page markdown for GET; record + 200 for PUT (apply)."""
    def handler(route):
        if route.request.method == "PUT":
            put_seen.append(route.request.post_data)
            route.fulfill(status=200, content_type="application/json",
                          body='{"path": "concepts/demo.md", "updated": true}')
        else:
            route.fulfill(status=200, content_type="text/plain; charset=utf-8", body=_PAGE_MD)
    page.route(_PAGE_URL, handler)


@pytest.mark.frontend
def test_ai_edit_button_shown_for_admin(app_page):
    page = app_page
    _route_page(page, [])
    page.evaluate("async () => { await loadWikiPage('concepts/demo.md'); }")
    page.wait_for_selector("#btn-ai-edit", state="visible", timeout=5000)
    assert page.locator("#btn-ai-edit").is_visible()


@pytest.mark.frontend
def test_ai_edit_button_hidden_for_readonly_member(app_page):
    page = app_page
    # A read-only member (no can_edit_wiki) must not get the edit affordance.
    has_btn = page.evaluate("""() => {
        state.userRole = 'member';
        state.perms = {};
        state.currentPageRaw = '# X';
        renderPageActions('concepts/x.md');
        return !!document.getElementById('btn-ai-edit');
    }""")
    assert has_btn is False


@pytest.mark.frontend
def test_ai_edit_section_dropdown_populates_from_headings(app_page):
    page = app_page
    _route_page(page, [])
    page.evaluate("async () => { await loadWikiPage('concepts/demo.md'); }")
    page.click("#btn-ai-edit")
    page.wait_for_selector("#modal-ai-edit:not(.hidden)", timeout=3000)
    # Switch scope to "section" → the section <select> reveals, built from headings.
    page.select_option("#ai-edit-scope", "section")
    page.wait_for_selector("#ai-edit-section-wrap:not(.hidden)", timeout=2000)
    opts = page.locator("#ai-edit-section option").all_inner_texts()
    assert "Background" in opts


@pytest.mark.frontend
def test_ai_edit_generate_shows_diff_then_applies(app_page):
    page = app_page
    put_seen = []
    _route_page(page, put_seen)
    # Tree fetch (for the reconcile picker) → empty.
    page.route("**/api/wiki", lambda route: route.fulfill(
        status=200, content_type="application/json", body="[]"))
    # Stub the streaming edit endpoint with a canned SSE body.
    proposed = "# Demo\n\n## Background\n\nImproved, expanded body.\n"
    sse = (
        f'data: {json.dumps({"type": "meta", "sources": [], "scope": "page", "heading": None})}\n\n'
        f'data: {json.dumps({"type": "chunk", "text": proposed})}\n\n'
        f'data: {json.dumps({"type": "done", "full_content": proposed, "scope": "page", "heading": None})}\n\n'
    )
    page.route("**/api/ops/wiki/edit/stream", lambda route: route.fulfill(
        status=200, content_type="text/event-stream", body=sse))

    page.evaluate("async () => { await loadWikiPage('concepts/demo.md'); }")
    page.click("#btn-ai-edit")
    page.wait_for_selector("#modal-ai-edit:not(.hidden)", timeout=3000)
    page.click("#btn-ai-edit-generate")

    # Diff renders and Apply enables once the stream's done event arrives.
    page.wait_for_selector("#btn-ai-edit-apply:not([disabled])", timeout=5000)
    diff_html = page.locator("#ai-edit-diff").inner_html()
    assert "Improved, expanded body." in diff_html

    page.click("#btn-ai-edit-apply")
    # Apply PUTs the proposed content to /api/wiki/{path}.
    page.wait_for_function("() => document.querySelector('#modal-ai-edit').classList.contains('hidden')",
                           timeout=5000)
    assert put_seen, "Apply did not PUT the rewrite"
    assert "Improved, expanded body." in put_seen[0]


@pytest.mark.frontend
def test_ai_edit_stream_refreshes_token_on_401(app_page):
    """B2: a stream that 401s mid-session refreshes the access token and retries
    instead of bouncing the user to login."""
    page = app_page
    _route_page(page, [])
    page.route("**/api/wiki", lambda r: r.fulfill(
        status=200, content_type="application/json", body="[]"))
    page.route("**/api/auth/refresh", lambda r: r.fulfill(
        status=200, content_type="application/json",
        body=json.dumps({"access_token": "refreshed-token"})))

    proposed = "# Demo\n\n## Background\n\nRefreshed body.\n"
    sse = (
        f'data: {json.dumps({"type": "meta", "sources": [], "scope": "page", "heading": None})}\n\n'
        f'data: {json.dumps({"type": "chunk", "text": proposed})}\n\n'
        f'data: {json.dumps({"type": "done", "full_content": proposed, "scope": "page", "heading": None})}\n\n'
    )
    calls = {"n": 0}

    def edit_handler(route):
        calls["n"] += 1
        if calls["n"] == 1:
            route.fulfill(status=401, content_type="application/json", body='{"detail":"expired"}')
        else:
            route.fulfill(status=200, content_type="text/event-stream", body=sse)

    page.route("**/api/ops/wiki/edit/stream", edit_handler)

    page.evaluate("async () => { await loadWikiPage('concepts/demo.md'); }")
    page.click("#btn-ai-edit")
    page.wait_for_selector("#modal-ai-edit:not(.hidden)", timeout=3000)
    page.click("#btn-ai-edit-generate")
    page.wait_for_selector("#btn-ai-edit-apply:not([disabled])", timeout=5000)
    assert calls["n"] == 2   # 401 → refreshed → retried → succeeded


@pytest.mark.frontend
def test_eschtml_escapes_quotes(app_page):
    """R2: escHtml must escape quotes (not just & < >) since it's used inside
    double-quoted HTML attributes (e.g. renderMd's data-query)."""
    out = app_page.evaluate("(s) => escHtml(s)", "a\"b'c<d>&e")
    assert out == "a&quot;b&#39;c&lt;d&gt;&amp;e"


@pytest.mark.frontend
def test_render_md_sanitizes_untrusted_html(app_page):
    """renderMd must neutralize HTML/script in wiki + AI-generated content (XSS),
    while keeping our wiki-link spans."""
    page = app_page
    if not page.evaluate("typeof DOMPurify !== 'undefined'"):
        pytest.skip("DOMPurify not loaded (offline test env)")
    out = page.evaluate(r"""() => ({
        script: renderMd('# Hi\n\n<img src=x onerror="alert(1)">\n\n<script>alert(2)</script>'),
        link: renderMd('[click](javascript:alert(1))'),
        wikilink: renderMd('See [[concepts/foo.md|Foo]].'),
    })""")
    assert "<script" not in out["script"].lower()
    assert "onerror" not in out["script"].lower()
    assert "javascript:" not in out["link"].lower()
    assert "wiki-link" in out["wikilink"]   # our span survives sanitization


@pytest.mark.frontend
def test_ai_edit_discard_aborts_inflight_stream(app_page):
    """Closing the AI-edit modal aborts the in-flight stream so server-side
    generation (and its token spend) stops."""
    page = app_page
    aborted = page.evaluate("""() => {
        const ctrl = new AbortController();
        _aiEdit = { path: 'x', controller: ctrl };
        document.getElementById('modal-ai-edit').classList.remove('hidden');
        closeModal('modal-ai-edit');
        return ctrl.signal.aborted;
    }""")
    assert aborted is True


@pytest.mark.frontend
def test_ai_edit_preview_renders_markdown_when_model_wraps_in_code_fence(app_page):
    """Regression: the model often wraps its whole rewrite in a ```markdown
    fence. The preview must still render real markdown (headings/lists), not a
    raw code block."""
    page = app_page
    _route_page(page, [])
    page.route("**/api/wiki", lambda route: route.fulfill(
        status=200, content_type="application/json", body="[]"))
    clean = "# Demo\n\n## Background\n\nBody.\n\n- one\n- two\n"
    fenced = "```markdown\n" + clean + "```"   # what the model streams
    sse = (
        f'data: {json.dumps({"type": "meta", "sources": [], "scope": "page", "heading": None})}\n\n'
        f'data: {json.dumps({"type": "chunk", "text": fenced})}\n\n'
        f'data: {json.dumps({"type": "done", "full_content": clean, "scope": "page", "heading": None})}\n\n'
    )
    page.route("**/api/ops/wiki/edit/stream", lambda route: route.fulfill(
        status=200, content_type="text/event-stream", body=sse))

    page.evaluate("async () => { await loadWikiPage('concepts/demo.md'); }")
    page.click("#btn-ai-edit")
    page.wait_for_selector("#modal-ai-edit:not(.hidden)", timeout=3000)
    page.click("#btn-ai-edit-generate")
    page.wait_for_selector("#btn-ai-edit-apply:not([disabled])", timeout=5000)

    # Rendered as markdown: headings + list items present, no top-level code block.
    assert page.locator("#ai-edit-preview h1, #ai-edit-preview h2").count() >= 2
    assert page.locator("#ai-edit-preview li").count() == 2
    assert page.locator("#ai-edit-preview > pre").count() == 0
