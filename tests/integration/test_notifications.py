"""Integration tests for notification recipient resolution.

Focus: supervisors are resolved from org_memberships (the source of truth since
migration 023), not the legacy users table. Requires PostgreSQL.
"""
import pytest

from app.services import notif_svc
from app.services import orgs as orgs_svc


async def _new_org(name: str) -> str:
    org_id = await orgs_svc.create_org(name)
    assert org_id
    return org_id


# ── get_org_supervisors ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_supervisor_via_membership_is_found(default_user):
    """A user granted supervisor in another org via add_membership must be
    returned for THAT org, even though users.org_id points elsewhere — the
    case the old users-table query silently missed."""
    org_a = default_user["org_id"]
    org_b = await _new_org("Notif Org B")

    # Created in org A as supervisor (legacy users.org_id = org A).
    uid = await orgs_svc.create_user(org_a, "multi-sup@test.com", "password123",
                                     role="supervisor")
    # Granted supervisor in org B purely via membership.
    await orgs_svc.add_membership(uid, org_b, role="supervisor")

    sups_b = await notif_svc.get_org_supervisors(org_b)
    assert uid in [s["id"] for s in sups_b]


@pytest.mark.asyncio
async def test_promoted_member_is_found(default_user):
    """A member promoted to supervisor via update_membership is found."""
    org = default_user["org_id"]
    uid = await orgs_svc.create_user(org, "promote@test.com", "password123",
                                     role="member")
    assert uid not in [s["id"] for s in await notif_svc.get_org_supervisors(org)]

    await orgs_svc.update_membership(uid, org, {"role": "supervisor"})
    assert uid in [s["id"] for s in await notif_svc.get_org_supervisors(org)]


@pytest.mark.asyncio
async def test_member_is_not_a_supervisor(default_user):
    org = default_user["org_id"]
    uid = await orgs_svc.create_user(org, "plain-member@test.com", "password123",
                                     role="member")
    assert uid not in [s["id"] for s in await notif_svc.get_org_supervisors(org)]


@pytest.mark.asyncio
async def test_suspended_supervisor_excluded(default_user):
    org = default_user["org_id"]
    uid = await orgs_svc.create_user(org, "susp-sup@test.com", "password123",
                                     role="supervisor")
    assert uid in [s["id"] for s in await notif_svc.get_org_supervisors(org)]

    await orgs_svc.update_membership(uid, org, {"role": "supervisor", "is_suspended": True})
    assert uid not in [s["id"] for s in await notif_svc.get_org_supervisors(org)]


@pytest.mark.asyncio
async def test_supervisors_are_org_scoped(default_user):
    """A supervisor in org A is not returned for org B."""
    org_a = default_user["org_id"]
    org_b = await _new_org("Notif Org Scoped")
    uid = await orgs_svc.create_user(org_a, "scoped-sup@test.com", "password123",
                                     role="supervisor")
    assert uid not in [s["id"] for s in await notif_svc.get_org_supervisors(org_b)]


# ── create_for_supervisors fan-out ────────────────────────────────────────

@pytest.mark.asyncio
async def test_create_for_supervisors_fans_out_and_excludes(default_user):
    org = default_user["org_id"]
    sup1 = await orgs_svc.create_user(org, "sup1@test.com", "password123", role="supervisor")
    sup2 = await orgs_svc.create_user(org, "sup2@test.com", "password123", role="supervisor")

    await notif_svc.create_for_supervisors(
        org_id=org, type="recalib_done", title="Done", body="ok",
        exclude_user_id=sup1,
    )

    assert len(await notif_svc.get_unread(sup2)) == 1
    assert len(await notif_svc.get_unread(sup1)) == 0  # excluded
