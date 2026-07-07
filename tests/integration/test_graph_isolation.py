"""Integration test: the wiki graph cache is isolated per org.

Regression guard for a cross-tenant leak — a single process-global graph would
serve whichever org last loaded it to every other org. Requires PostgreSQL.
"""
import pytest

from app.context import UserContext, current_user
from app.services import graph as graph_mod
from app.services import orgs as orgs_svc
from app.services.graph import get_graph, reset_graph_cache
from app.services.wiki_db import upsert_wiki_page


def _ctx(org_id: str, user_id: str = "00000000-0000-0000-0000-000000000000") -> None:
    current_user.set(UserContext(user_id=user_id, org_id=org_id, email="sys", role="admin"))


@pytest.mark.asyncio
async def test_graph_is_isolated_per_org(default_user):
    reset_graph_cache()
    org_a = default_user["org_id"]
    org_b = await orgs_svc.create_org("Graph Isolation Org B")

    # Org A: build its graph from its pages.
    _ctx(org_a, default_user["id"])
    await upsert_wiki_page("concepts/alpha.md", "# Alpha\n\nbody")
    await get_graph().rebuild()
    assert "concepts/alpha.md" in get_graph()._meta

    # Org B's read path (ensure_loaded) must NOT return org A's loaded graph.
    # This is the cross-tenant leak the global singleton caused.
    _ctx(org_b, default_user["id"])
    await upsert_wiki_page("concepts/gamma.md", "# Gamma\n\nbody")
    g_b = await get_graph().ensure_loaded()
    assert "concepts/alpha.md" not in g_b._meta

    await get_graph().rebuild()
    assert "concepts/gamma.md" in get_graph()._meta
    assert "concepts/alpha.md" not in get_graph()._meta

    # A's graph is untouched by B's rebuild (separate instances).
    _ctx(org_a, default_user["id"])
    assert "concepts/alpha.md" in get_graph()._meta
    assert "concepts/gamma.md" not in get_graph()._meta


@pytest.mark.asyncio
async def test_get_graph_returns_distinct_instances_per_org(default_user):
    reset_graph_cache()
    org_b = await orgs_svc.create_org("Graph Distinct Org B")

    _ctx(default_user["org_id"], default_user["id"])
    g_a = get_graph()
    _ctx(org_b, default_user["id"])
    g_b = get_graph()
    assert g_a is not g_b


@pytest.mark.asyncio
async def test_update_pages_on_cold_cache_preserves_existing_links(default_user):
    """Regression (B1): an incremental update_pages on a COLD per-org cache must
    reload the full graph first. Otherwise _save() persists the (nearly empty)
    in-memory graph over the real one, wiping every node/edge except the touched
    page — the cross-replica-failover and post-revert failure mode."""
    reset_graph_cache()
    _ctx(default_user["org_id"], default_user["id"])
    await upsert_wiki_page("concepts/a.md", "# A\n\nSee [B](b.md).\n")
    await upsert_wiki_page("concepts/b.md", "# B\n\nbody\n")
    await get_graph().rebuild()
    assert "concepts/b.md" in get_graph()._adj.get("concepts/a.md", set())  # A→B edge

    # Cold cache (fresh replica / post-restart): drop the in-memory instance.
    reset_graph_cache()
    await upsert_wiki_page("concepts/c.md", "# C\n\nbody\n")
    await get_graph().update_pages(["concepts/c.md"])

    g = get_graph()
    assert "concepts/a.md" in g._meta        # pre-existing nodes survive
    assert "concepts/b.md" in g._meta
    assert "concepts/c.md" in g._meta
    assert "concepts/b.md" in g._adj.get("concepts/a.md", set())  # A→B edge survives


@pytest.fixture(autouse=True)
def _reset_graph():
    yield
    graph_mod.reset_graph_cache()
