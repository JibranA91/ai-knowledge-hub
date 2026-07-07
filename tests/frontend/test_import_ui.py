"""Frontend tests for the main-app wiki import affordance (app.js).

The 'Import Wiki' button is shown to admins/supervisors on the welcome screen;
selecting a file opens the in-app confirm dialog (not a native one) before any
upload. The actual import is covered by integration tests.
"""
import pytest


@pytest.fixture
def app_page(page, live_server_url):
    page.goto("/")
    page.fill('input[placeholder*="sername"], input[name="username"]', "admin")
    page.fill('input[placeholder*="assword"], input[name="password"]', "admin")
    page.click('button[type="submit"], button:has-text("Login"), button:has-text("Sign in")')
    page.wait_for_selector("#app-root", state="visible")
    return page


@pytest.mark.frontend
def test_import_button_visible_for_admin_when_wiki_empty(app_page):
    # Fresh org → empty wiki → admin sees the import affordance. Revealed
    # asynchronously by loadWikiTree once it confirms there are no pages.
    app_page.wait_for_selector("#btn-welcome-import", state="visible", timeout=5000)
    assert app_page.locator("#btn-welcome-import").is_visible()


@pytest.mark.frontend
def test_import_button_hidden_when_wiki_not_empty(app_page):
    page = app_page
    # Wait for the empty-wiki reveal first (confirms boot finished).
    page.wait_for_selector("#btn-welcome-import", state="visible", timeout=5000)
    # Make /api/wiki report a populated wiki, then re-run the tree load. Stubbing
    # the response (rather than seeding the DB) keeps this test from leaking a
    # page into the session-scoped DB shared by the other frontend tests.
    page.route("**/api/wiki", lambda route: route.fulfill(
        status=200, content_type="application/json",
        body='[{"type":"file","name":"seed.md","path":"concepts/seed.md"}]',
    ))
    try:
        page.evaluate("async () => { await loadWikiTree(); }")
        page.wait_for_selector("#btn-welcome-import", state="hidden", timeout=5000)
        assert not page.locator("#btn-welcome-import").is_visible()
    finally:
        page.unroute("**/api/wiki")


@pytest.mark.frontend
def test_selecting_file_opens_in_app_confirm(app_page):
    page = app_page
    # Minimal empty-zip bytes — content is irrelevant; we cancel before upload.
    page.set_input_files("#wiki-import-file", files=[{
        "name": "wiki.zip", "mimeType": "application/zip",
        "buffer": b"PK\x05\x06" + b"\x00" * 18,
    }])
    page.wait_for_selector("#modal-dialog:not(.hidden)", timeout=2000)
    assert "AGENTS.md" in page.locator("#app-dlg-message").inner_text()
    page.click("#app-dlg-cancel")  # don't actually upload
