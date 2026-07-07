"""Frontend tests for Step 3 — populate a new org from an export in the
create-org modal (admin SPA).
"""
import io
import json
import uuid
import zipfile

import pytest


def _bundle() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("manifest.json", json.dumps({"format_version": 1}))
        zf.writestr("schema/AGENTS.md", "# Imported Schema")
        zf.writestr("wiki/concepts/imp.md", "# Imported Page\n\nbody")
    return buf.getvalue()


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


@pytest.mark.frontend
def test_create_org_modal_has_import_field(admin_page):
    page = admin_page
    page.click(".tab[data-tab='orgs']")
    page.wait_for_timeout(400)
    page.evaluate("openCreateOrg()")
    page.wait_for_selector("#modal-create-org.open", timeout=2000)
    assert page.locator("#co-import-file").count() == 1


@pytest.mark.frontend
def test_create_org_with_bundle_populates_it(admin_page):
    page = admin_page
    org_name = f"Imp Org {uuid.uuid4().hex[:8]}"

    page.click(".tab[data-tab='orgs']")
    page.wait_for_timeout(400)
    page.evaluate("openCreateOrg()")
    page.wait_for_selector("#modal-create-org.open", timeout=2000)
    page.fill("#co-name", org_name)
    page.set_input_files("#co-import-file", files=[{
        "name": "wiki.zip", "mimeType": "application/zip", "buffer": _bundle(),
    }])
    page.click("#modal-create-org button:has-text('Create Organization')")
    # createOrg awaits the import (incl. graph rebuild) before closing the modal.
    # The backdrop is display:none when not .open, so wait for it to go hidden.
    page.wait_for_selector("#modal-create-org", state="hidden", timeout=10000)

    # The new org's wiki should contain the imported page.
    tree_json = page.evaluate(
        "async (name) => {"
        "  const orgs = await apiFetch('/admin/organizations');"
        "  const o = (orgs || []).find(x => x.name === name);"
        "  if (!o) return 'NO_ORG';"
        "  const tree = await apiFetch('/wiki', { headers: { 'X-Org-Context': o.id } });"
        "  return JSON.stringify(tree);"
        "}",
        org_name,
    )
    assert tree_json not in (None, "NO_ORG")
    assert "imp" in tree_json
