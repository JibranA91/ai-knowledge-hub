"""Integration test: the lint (health) report surfaces content-gap candidates.

Requires PostgreSQL. The deterministic article-candidate detection runs over the
real wiki pages, independent of the (mocked) qualitative LLM audit.
"""
import pytest


@pytest.mark.asyncio
async def test_lint_reports_article_candidates(client, auth_headers, user_ctx):
    from app.services.wiki_db import upsert_wiki_page
    body = " ".join(["word"] * 200)
    # Two pages both link to [[shared-gap]], which has no page → a content gap.
    await upsert_wiki_page("concepts/a.md", f"# A\n\n{body}\n\nSee [[shared-gap]].")
    await upsert_wiki_page("concepts/b.md", f"# B\n\n{body}\n\nAlso [[shared-gap]].")

    resp = await client.post("/api/ops/lint", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert "article_candidates" in data
    match = next((c for c in data["article_candidates"] if c["slug"] == "shared-gap"), None)
    assert match is not None
    assert match["ref_count"] == 2
