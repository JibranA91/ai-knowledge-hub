"""Frontend tests for the in-app dialog component (uiConfirm / uiPrompt).

These replace native confirm()/prompt(). The component is implemented twice —
once in the admin SPA (admin.js, #modal-dialog with #dlg-* ids) and once in the
main app (app.js, #modal-dialog with #app-dlg-* ids) — so we exercise both.

We drive the promises deterministically: start the dialog (store the promise on
window WITHOUT awaiting it), interact with the modal, then read the promise back
(Playwright auto-awaits a returned promise) and assert the resolved value.
"""
import pytest


@pytest.fixture
def admin_page(page, live_server_url):
    page.goto("/")
    page.fill('input[placeholder*="sername"], input[name="username"]', "admin")
    page.fill('input[placeholder*="assword"], input[name="password"]', "admin")
    page.click('button[type="submit"], button:has-text("Login"), button:has-text("Sign in")')
    page.wait_for_selector("#app-root", state="visible")
    page.goto("/admin")
    page.wait_for_selector(".tabs", state="visible")
    return page


@pytest.fixture
def app_page(page, live_server_url):
    page.goto("/")
    page.fill('input[placeholder*="sername"], input[name="username"]', "admin")
    page.fill('input[placeholder*="assword"], input[name="password"]', "admin")
    page.click('button[type="submit"], button:has-text("Login"), button:has-text("Sign in")')
    page.wait_for_selector("#app-root", state="visible")
    return page


# ── No native dialogs remain in the shipped JS ────────────────────────────

@pytest.mark.frontend
def test_no_native_dialogs_in_bundles(admin_page):
    page = admin_page
    for name in ("/static/admin.js", "/static/app.js"):
        src = page.evaluate(f"async () => (await fetch('{name}')).text()")
        for native in ("confirm(", "prompt(", "alert("):
            assert native not in src, f"{native} still present in {name}"


# ── Admin SPA dialog (#dlg-*) ─────────────────────────────────────────────

@pytest.mark.frontend
def test_admin_confirm_resolves_true(admin_page):
    page = admin_page
    page.evaluate("() => { window.__p = uiConfirm({ title: 't', message: 'm' }); }")
    page.wait_for_selector("#modal-dialog.open", timeout=2000)
    page.click("#dlg-confirm")
    assert page.evaluate("() => window.__p") is True


@pytest.mark.frontend
def test_admin_confirm_resolves_false_on_cancel(admin_page):
    page = admin_page
    page.evaluate("() => { window.__p = uiConfirm({ title: 't', message: 'm' }); }")
    page.wait_for_selector("#modal-dialog.open", timeout=2000)
    page.click("#dlg-cancel")
    assert page.evaluate("() => window.__p") is False


@pytest.mark.frontend
def test_admin_prompt_returns_value(admin_page):
    page = admin_page
    page.evaluate("() => { window.__p = uiPrompt({ title: 't', label: 'Name' }); }")
    page.wait_for_selector("#modal-dialog.open", timeout=2000)
    page.fill("#dlg-input", "Hello World")
    page.click("#dlg-confirm")
    assert page.evaluate("() => window.__p") == "Hello World"


@pytest.mark.frontend
def test_admin_prompt_returns_null_on_cancel(admin_page):
    page = admin_page
    page.evaluate("() => { window.__p = uiPrompt({ title: 't', label: 'Name' }); }")
    page.wait_for_selector("#modal-dialog.open", timeout=2000)
    page.click("#dlg-cancel")
    assert page.evaluate("() => window.__p") is None


@pytest.mark.frontend
def test_clone_button_opens_in_app_dialog(admin_page):
    """The Org Clone action uses the in-app prompt (input shown), not a native one."""
    page = admin_page
    page.click(".tab[data-tab='orgs']")
    page.wait_for_timeout(600)
    page.click("#orgs-table button:has-text('Clone')")
    page.wait_for_selector("#modal-dialog.open", timeout=2000)
    assert page.locator("#dlg-input-wrap").is_visible()   # prompt mode
    assert page.locator("#dlg-input").input_value() != ""  # prefilled "<name> (copy)"
    page.click("#dlg-cancel")  # don't actually create a clone


# ── Main app dialog (#app-dlg-*) ──────────────────────────────────────────

@pytest.mark.frontend
def test_app_confirm_resolves_true(app_page):
    page = app_page
    page.evaluate("() => { window.__p = uiConfirm({ title: 't', message: 'm' }); }")
    page.wait_for_selector("#modal-dialog:not(.hidden)", timeout=2000)
    page.click("#app-dlg-confirm")
    assert page.evaluate("() => window.__p") is True


@pytest.mark.frontend
def test_app_confirm_resolves_false_on_cancel(app_page):
    page = app_page
    page.evaluate("() => { window.__p = uiConfirm({ title: 't', message: 'm' }); }")
    page.wait_for_selector("#modal-dialog:not(.hidden)", timeout=2000)
    page.click("#app-dlg-cancel")
    assert page.evaluate("() => window.__p") is False


@pytest.mark.frontend
def test_app_prompt_returns_value(app_page):
    page = app_page
    page.evaluate("() => { window.__p = uiPrompt({ title: 't', label: 'Name' }); }")
    page.wait_for_selector("#modal-dialog:not(.hidden)", timeout=2000)
    page.fill("#app-dlg-input", "Draft name")
    page.click("#app-dlg-confirm")
    assert page.evaluate("() => window.__p") == "Draft name"
