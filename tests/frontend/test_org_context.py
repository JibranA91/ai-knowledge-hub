"""Frontend tests for the generalized X-Org-Context header (app.js).

Multi-org means the active org is sent on every request for ALL users (not just
admins), driven by state.activeOrgId. These tests intercept /api/wiki and
inspect the outgoing header.
"""
import pytest


@pytest.mark.frontend
def test_api_sends_x_org_context_for_active_org(logged_in_page):
    page = logged_in_page
    captured = {}

    def handle(route):
        captured["org"] = route.request.headers.get("x-org-context")
        route.fulfill(status=200, content_type="application/json", body="{}")

    page.route("**/api/wiki", handle)
    page.evaluate("state.activeOrgId = 'org-xyz-123'")
    page.evaluate("api('GET','/api/wiki')")
    page.wait_for_timeout(300)
    page.unroute("**/api/wiki", handle)

    assert captured.get("org") == "org-xyz-123"


@pytest.mark.frontend
def test_api_omits_x_org_context_when_no_active_org(logged_in_page):
    page = logged_in_page
    captured = {"org": "sentinel"}

    def handle(route):
        captured["org"] = route.request.headers.get("x-org-context")
        route.fulfill(status=200, content_type="application/json", body="{}")

    page.route("**/api/wiki", handle)
    page.evaluate("state.activeOrgId = null")
    page.evaluate("api('GET','/api/wiki')")
    page.wait_for_timeout(300)
    page.unroute("**/api/wiki", handle)

    # Header must not be sent when there is no active org.
    assert captured["org"] in (None, "")
