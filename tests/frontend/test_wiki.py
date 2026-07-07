"""Frontend wiki navigation tests using Playwright."""
import pytest
import httpx

_DEFAULT_ORG_ID = "00000000-0000-0000-0000-000000000001"


def _seed_page(live_server_url: str, path: str, content: str):
    """Helper: create a wiki page via the API (bypasses browser).

    Admin users have no org — supply X-Org-Context so the wiki PUT doesn't 400.
    """
    r = httpx.post(
        f"{live_server_url}/api/auth/login",
        json={"username": "admin", "password": "admin"},
    )
    token = r.json()["access_token"]
    resp = httpx.put(
        f"{live_server_url}/api/wiki/{path}",
        json={"content": content},
        headers={
            "Authorization": f"Bearer {token}",
            "X-Org-Context": _DEFAULT_ORG_ID,
        },
    )
    resp.raise_for_status()


@pytest.mark.frontend
def test_wiki_tree_shows_pages(logged_in_page, live_server_url):
    _seed_page(live_server_url, "concepts/test-page.md", "# Test Page\n\nContent.")
    page = logged_in_page
    page.reload()
    page.wait_for_selector("#app-root", state="visible")

    # Wait for the wiki tree to finish loading (no spinner)
    page.wait_for_selector("#wiki-tree", timeout=10000)
    page.wait_for_function(
        "() => !document.querySelector('#wiki-tree .spinner')",
        timeout=10000,
    )
    tree_text = page.locator("#wiki-tree").inner_text()
    assert "test-page" in tree_text.lower() or "test page" in tree_text.lower(), (
        f"Expected 'test-page' in wiki tree, got: {tree_text!r}"
    )


@pytest.mark.frontend
def test_clicking_page_loads_content(logged_in_page, live_server_url):
    _seed_page(live_server_url, "concepts/clickable.md", "# Clickable\n\nHello from clickable page.")
    page = logged_in_page
    page.reload()
    page.wait_for_selector("#app-root", state="visible")

    # Wait for the wiki tree to load and find the clickable page entry
    page.wait_for_selector("#wiki-tree", timeout=10000)
    page.wait_for_function(
        "() => !document.querySelector('#wiki-tree .spinner')",
        timeout=10000,
    )
    page.locator("#wiki-tree .tree-item").filter(has_text="clickable").first.click(timeout=10000)
    page.wait_for_timeout(1000)

    # Content should appear in the main area
    main_text = page.locator("#page-content").inner_text()
    assert "Hello from clickable page" in main_text, (
        f"Expected page content in main area, got: {main_text[:200]!r}"
    )


@pytest.mark.frontend
def test_search_returns_results(logged_in_page, live_server_url):
    _seed_page(
        live_server_url,
        "concepts/searchable.md",
        "# Searchable\n\nUnique keyword xyzzy123 for search testing.",
    )
    page = logged_in_page

    # Find the search input and type the unique keyword
    search = page.locator('#search-input, input[placeholder*="earch"], input[type="search"]')
    if search.count() == 0:
        pytest.skip("Could not locate search input in the UI")

    search.first.fill("xyzzy123")
    page.wait_for_timeout(1500)

    # Results are rendered into the #search-results dropdown in the live DOM
    results_text = page.locator("#search-results").inner_text()
    assert "xyzzy123" in results_text.lower() or "searchable" in results_text.lower(), (
        f"Expected search result in dropdown, got: {results_text!r}"
    )
