"""Frontend auth tests using Playwright."""
import pytest


@pytest.mark.frontend
def test_login_valid(page, live_server_url):
    page.goto("/")
    page.fill('input[placeholder*="sername"], input[name="username"]', "admin")
    page.fill('input[placeholder*="assword"], input[name="password"]', "admin")
    page.click('button[type="submit"], button:has-text("Login"), button:has-text("Sign in")')
    page.wait_for_selector("#app-root", state="visible")

    # After login the main app UI should be visible (not the login form)
    assert page.locator('input[name="username"]').count() == 0 or \
           page.locator(".app, #app, main, .wiki-container").count() > 0


@pytest.mark.frontend
def test_login_invalid_shows_error(page):
    page.goto("/")
    page.fill('input[placeholder*="sername"], input[name="username"]', "admin")
    page.fill('input[placeholder*="assword"], input[name="password"]', "wrong-password")
    page.click('button[type="submit"], button:has-text("Login"), button:has-text("Sign in")')
    page.wait_for_timeout(1500)

    # Should show an error (login form still present or error element visible)
    has_error = (
        page.locator(".error, .toast-error, [class*='error']").count() > 0
        or page.locator('input[name="password"]').count() > 0
    )
    assert has_error, "Expected an error message or login form to remain after bad password"


@pytest.mark.frontend
def test_logout_returns_to_login(logged_in_page):
    page = logged_in_page
    # The sign-out button is inside a user-menu dropdown — open it first
    user_menu = page.locator('#btn-user-menu, button.btn-user-menu')
    if user_menu.count() > 0:
        user_menu.first.click()
        page.wait_for_timeout(300)
    page.click('#btn-signout, button:has-text("Sign out"), button:has-text("Logout"), [data-action="logout"]')
    page.wait_for_timeout(1000)

    # Login form should be visible again
    assert (
        page.locator('input[name="password"], input[placeholder*="assword"]').count() > 0
    )


@pytest.mark.frontend
def test_token_persisted_in_localstorage(logged_in_page):
    """After login the auth token should be in localStorage."""
    page = logged_in_page
    token = page.evaluate("localStorage.getItem('auth_token')")
    assert token and len(token) > 0


# ── 401 / refresh / retry ─────────────────────────────────────────────────

@pytest.mark.frontend
def test_api_retries_after_401_with_valid_refresh_token(logged_in_page):
    """401 on first request → tryRefreshToken() called → original request retried."""
    page = logged_in_page
    wiki_calls = {"n": 0}

    def handle_wiki(route):
        wiki_calls["n"] += 1
        if wiki_calls["n"] == 1:
            route.fulfill(status=401, content_type="application/json",
                          body='{"detail":"Unauthorized"}')
        else:
            route.fulfill(status=200, content_type="application/json",
                          body='{"pages":[]}')

    def handle_refresh(route):
        route.fulfill(status=200, content_type="application/json",
                      body='{"access_token":"new-access-token"}')

    page.route("**/api/wiki", handle_wiki)
    page.route("**/api/auth/refresh", handle_refresh)
    page.evaluate("state.refreshToken = 'valid-refresh-token'")

    # The app loads the wiki tree once right after login; that request can race
    # with this test's route (on CI it did, inflating the count to 3). Let it
    # settle, then reset so we count ONLY the explicit api() call below.
    page.wait_for_timeout(500)
    wiki_calls["n"] = 0

    result = page.evaluate("""(async () => {
        try {
            return { ok: true, data: await api('GET', '/api/wiki') };
        } catch (e) {
            return { ok: false, error: e.message };
        }
    })()""")

    assert result["ok"], f"api() threw unexpectedly: {result.get('error')}"
    assert result["data"] == {"pages": []}
    assert wiki_calls["n"] == 2, "Expected original request + one retry"
    assert page.evaluate("state.authToken") == "new-access-token"

    page.unroute("**/api/wiki", handle_wiki)
    page.unroute("**/api/auth/refresh", handle_refresh)


@pytest.mark.frontend
def test_api_shows_login_screen_when_refresh_fails(logged_in_page):
    """401 + failed refresh → showLoginScreen() called, error thrown."""
    page = logged_in_page

    page.route("**/api/wiki",
               lambda r: r.fulfill(status=401, content_type="application/json",
                                   body='{"detail":"Unauthorized"}'))
    page.route("**/api/auth/refresh",
               lambda r: r.fulfill(status=401, content_type="application/json",
                                   body='{"detail":"Invalid refresh token"}'))

    page.evaluate("state.refreshToken = 'expired-refresh-token'")

    result = page.evaluate("""(async () => {
        try {
            await api('GET', '/api/wiki');
            return { threw: false };
        } catch (e) {
            return { threw: true, message: e.message };
        }
    })()""")

    assert result["threw"]
    assert "session" in result["message"].lower() or "expired" in result["message"].lower()
    assert page.locator('input[name="password"], input[placeholder*="assword"]').count() > 0


@pytest.mark.frontend
def test_api_does_not_refresh_when_no_refresh_token(logged_in_page):
    """Without a stored refresh token, no refresh attempt is made on 401."""
    page = logged_in_page
    refresh_calls = {"n": 0}

    def handle_refresh(route):
        refresh_calls["n"] += 1
        route.fulfill(status=200, content_type="application/json",
                      body='{"access_token":"should-not-get-this"}')

    page.route("**/api/wiki",
               lambda r: r.fulfill(status=401, content_type="application/json",
                                   body='{"detail":"Unauthorized"}'))
    page.route("**/api/auth/refresh", handle_refresh)
    page.evaluate("state.refreshToken = null")

    page.evaluate("""(async () => {
        try { await api('GET', '/api/wiki'); } catch {}
    })()""")
    page.wait_for_timeout(300)

    assert refresh_calls["n"] == 0, "Refresh endpoint must not be called without a refresh token"

    page.unroute("**/api/wiki")
    page.unroute("**/api/auth/refresh", handle_refresh)


@pytest.mark.frontend
def test_new_access_token_persisted_to_localstorage(logged_in_page):
    """After a successful token refresh, the new access token is saved to localStorage."""
    page = logged_in_page
    wiki_calls = {"n": 0}

    def handle_wiki(route):
        wiki_calls["n"] += 1
        if wiki_calls["n"] == 1:
            route.fulfill(status=401, content_type="application/json",
                          body='{"detail":"Unauthorized"}')
        else:
            route.fulfill(status=200, content_type="application/json", body='{}')

    page.route("**/api/wiki", handle_wiki)
    page.route("**/api/auth/refresh",
               lambda r: r.fulfill(status=200, content_type="application/json",
                                   body='{"access_token":"freshly-issued-token"}'))

    page.evaluate("state.refreshToken = 'some-refresh-token'")
    page.evaluate("api('GET', '/api/wiki')")
    page.wait_for_timeout(500)

    assert page.evaluate("localStorage.getItem('auth_token')") == "freshly-issued-token"

    page.unroute("**/api/wiki", handle_wiki)
    page.unroute("**/api/auth/refresh")
