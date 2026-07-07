"""Frontend tests for the admin dashboard's multi-org user management (admin.js).

Covers:
  - Add User modal is identity-only (no org / template fields, has Admin toggle).
  - Organizations tab exposes "Manage members" and no longer "Change supervisor".
  - Users tab groups memberships into one row per identity, expandable to a
    per-membership detail table.

The admin SPA reads its bearer token from localStorage, so we log in at "/"
first, then navigate to "/admin". admin.js exposes its top-level functions
(apiFetch, openCreateUser, loadUsers) as globals we can drive via evaluate().
"""
import pytest

_DEFAULT_ORG_ID = "00000000-0000-0000-0000-000000000001"


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
def test_add_user_modal_is_identity_only(admin_page):
    page = admin_page
    page.evaluate("openCreateUser()")
    page.wait_for_timeout(200)
    # Org / template / per-user-limit fields are gone…
    assert page.locator("#cu-org").count() == 0
    assert page.locator("#cu-template").count() == 0
    # …replaced by an Admin toggle alongside email + password.
    assert page.locator("#cu-is-admin").count() == 1
    assert page.locator("#cu-email").count() == 1
    assert page.locator("#cu-password").count() == 1


@pytest.mark.frontend
def test_orgs_tab_has_manage_members_and_no_change_supervisor(admin_page):
    page = admin_page
    page.click(".tab[data-tab='orgs']")
    page.wait_for_timeout(600)
    html = page.locator("#orgs-table").inner_html()
    assert "Manage members" in html
    assert "Change supervisor" not in html


@pytest.mark.frontend
def test_edit_user_modal_opens_above_members_modal(admin_page):
    """Regression: clicking Edit inside the Manage Members modal must bring the
    Edit User modal to the foreground. All .modal-backdrop share one CSS z-index,
    so without openModal's stacking the nested edit modal (earlier in the DOM)
    opened *behind* the members modal and couldn't be used."""
    page = admin_page

    # Seed a member in the default org so the members list has a row to edit.
    user_id = page.evaluate(f"""async () => {{
        const email = 'stk_' + Date.now() + '@test.com';
        const u = await apiFetch('/admin/users', {{
            method: 'POST',
            body: JSON.stringify({{ email, password: 'password123', role: 'member' }})
        }});
        await apiFetch('/admin/users/' + u.id + '/memberships', {{
            method: 'POST',
            body: JSON.stringify({{ org_id: '{_DEFAULT_ORG_ID}', role: 'member' }})
        }});
        return u.id;
    }}""")

    # Open Manage Members (loads + caches the member), then open Edit for them.
    page.evaluate(
        """async ([uid, orgId]) => {
            await openOrgMembers(orgId, 'Default Organization');
            openEditUser(uid, orgId);
        }""",
        [user_id, _DEFAULT_ORG_ID],
    )
    page.wait_for_selector("#modal-edit-user.open", timeout=5000)

    # Both modals are open; the edit modal must stack ABOVE the members modal.
    z_edit = page.evaluate(
        "parseInt(getComputedStyle(document.getElementById('modal-edit-user')).zIndex, 10)")
    z_members = page.evaluate(
        "parseInt(getComputedStyle(document.getElementById('modal-org-members')).zIndex, 10)")
    assert z_edit > z_members

    # And the edit modal is genuinely on top: a click at its centre lands inside it.
    on_top = page.evaluate("""() => {
        const m = document.getElementById('modal-edit-user');
        const r = m.getBoundingClientRect();
        const el = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
        return m.contains(el);
    }""")
    assert on_top


@pytest.mark.frontend
def test_users_tab_groups_memberships_and_expands(admin_page):
    page = admin_page

    # Seed: one identity that is a member in the default org and a supervisor in
    # a brand-new org — via the admin API exposed in the page (apiFetch global).
    email = page.evaluate(f"""async () => {{
        const email = 'grp_' + Date.now() + '@test.com';
        const u  = await apiFetch('/admin/users', {{
            method: 'POST',
            body: JSON.stringify({{ email, password: 'password123', role: 'member' }})
        }});
        const ob = await apiFetch('/admin/organizations', {{
            method: 'POST',
            body: JSON.stringify({{ name: 'UI Org ' + Date.now() }})
        }});
        await apiFetch('/admin/users/' + u.id + '/memberships', {{
            method: 'POST',
            body: JSON.stringify({{ org_id: '{_DEFAULT_ORG_ID}', role: 'member' }})
        }});
        await apiFetch('/admin/users/' + u.id + '/memberships', {{
            method: 'POST',
            body: JSON.stringify({{ org_id: ob.id, role: 'supervisor' }})
        }});
        return email;
    }}""")

    page.evaluate("loadUsers()")
    page.wait_for_timeout(800)

    # Grouping: exactly ONE row for this identity (not one per membership).
    row = page.locator(f"tr.user-row:has-text('{email}')")
    assert row.count() == 1
    # Its summary cell shows a chip per membership.
    assert row.locator(".org-chip").count() == 2

    # Expanding the row reveals a per-membership detail table.
    row.click()
    page.wait_for_timeout(300)
    visible_detail = page.locator("tr.user-detail:visible")
    assert visible_detail.count() == 1
    detail_html = visible_detail.inner_html()
    assert "supervisor" in detail_html
