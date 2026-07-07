"""Integration tests for hybrid (BM25 + cosine) page search.

Exercises the real `_HYBRID_SEARCH` SQL in `wiki_db.find_relevant_pages`
against Postgres — the unit tests mock the DB session and the route tests mock
`_find_relevant_pages`, so this is the only coverage of the actual query.

Requires the pgvector extension (the `wiki_pages.embedding` column). The
testcontainers image `postgres:16-alpine` does NOT ship pgvector, so these
tests skip there and only run against a pgvector-enabled Postgres (the local
dev container, or a CI service image with the extension).
"""
import math

import pytest
import pytest_asyncio
import sqlalchemy as sa
from unittest.mock import AsyncMock, patch

from app.services import embeddings, wiki_db

# pgvector embedding dimension (matches migration 005 + EMBEDDING_DIMENSIONS).
_DIM = 1536


def _unit_vec(*leading: float) -> list[float]:
    """Build a _DIM-length vector from the given leading components (rest zero)."""
    return list(leading) + [0.0] * (_DIM - len(leading))


async def _embedding_column_present(test_engine) -> bool:
    async with test_engine.begin() as conn:
        result = await conn.execute(sa.text("""
            SELECT COUNT(*) FROM information_schema.columns
            WHERE table_name = 'wiki_pages' AND column_name = 'embedding'
        """))
        return (result.scalar() or 0) > 0


@pytest_asyncio.fixture
async def require_pgvector(test_engine):
    """Skip unless the embedding column exists, and reset wiki_db's cached flag.

    `_embedding_col_ready` is a module-global set on first use; a prior test (or
    a prior run against a non-pgvector DB) may have cached False, so we force a
    re-check against this DB.
    """
    if not await _embedding_column_present(test_engine):
        pytest.skip("pgvector / wiki_pages.embedding not available in this Postgres")
    wiki_db._embedding_col_ready = None
    yield
    wiki_db._embedding_col_ready = None


async def _seed_page(test_engine, org_id: str, path: str, title: str,
                     content: str, vec: list[float]) -> None:
    async with test_engine.begin() as conn:
        await conn.execute(
            sa.text("""
                INSERT INTO wiki_pages (org_id, path, title, content, embedding)
                VALUES (CAST(:org AS UUID), :path, :title, :content,
                        CAST(:emb AS vector))
            """),
            {"org": org_id, "path": path, "title": title, "content": content,
             "emb": embeddings.vec_to_pg(vec)},
        )


@pytest.mark.asyncio
async def test_hybrid_search_normalized_bm25_contributes(
    test_engine, default_user, user_ctx, require_pgvector
):
    """A strong keyword match with only modest cosine similarity must surface.

    This is the regression guard for ts_rank min-max normalization. With the
    query vector == beta's vector:

      beta   keyword=none  cosine=1.0  → 0.4*0    + 0.6*1.0  = 0.60  (kept)
      alpha  keyword=best  cosine=0.2  → 0.4*1.0  + 0.6*0.2  = 0.52  (kept)
      gamma  keyword=none  cosine=0.0  → 0.4*0    + 0.6*0.0  = 0.00  (dropped, < 0.45)

    Under the OLD raw-ts_rank scoring, alpha's tiny ts_rank (~0.06) gave
    0.4*0.06 + 0.6*0.2 ≈ 0.144 < min_score(0.45) and it was dropped. Asserting
    alpha is present — and ranked below the pure-semantic beta — proves the
    normalized BM25 term carries its full 0.4 weight.
    """
    org_id = default_user["org_id"]

    query_vec = _unit_vec(1.0)                              # along dim 0
    await _seed_page(test_engine, org_id, "beta.md", "Beta",
                     "beta beta concepts and notes", _unit_vec(1.0))        # cosine 1.0, no "alpha"
    await _seed_page(test_engine, org_id, "alpha.md", "Alpha",
                     "alpha alpha alpha topic", _unit_vec(0.2, math.sqrt(1 - 0.04)))  # cosine 0.2, strong "alpha"
    await _seed_page(test_engine, org_id, "gamma.md", "Gamma",
                     "gamma unrelated material", _unit_vec(0.0, 1.0))       # cosine 0.0, no "alpha"

    with patch.object(embeddings, "is_enabled", return_value=True), \
         patch.object(embeddings, "embed_text", new_callable=AsyncMock,
                      return_value=query_vec):
        paths = await wiki_db.find_relevant_pages("alpha", limit=8)

    assert "gamma.md" not in paths                # below min_score, excluded
    assert "alpha.md" in paths                    # normalized BM25 lifts it past threshold
    assert "beta.md" in paths
    assert paths.index("beta.md") < paths.index("alpha.md")  # 0.60 ranks above 0.52


@pytest.mark.asyncio
async def test_hybrid_search_null_embedding_uses_bm25_only(
    test_engine, default_user, user_ctx, require_pgvector
):
    """A page ingested before pgvector (NULL embedding) still scores via BM25.

    With cosine = 0, a page's score is capped at 0.4*bm25_norm ≤ 0.4 — always
    below min_score(0.45) — so a NULL-embedding page on a pure keyword match
    stays *out* regardless of its BM25 rank, while an embedded page with strong
    cosine clears the line. Verifies the COALESCE path and that NULL embeddings
    don't error.
    """
    org_id = default_user["org_id"]

    query_vec = _unit_vec(1.0)
    # NULL-embedding page: keyword match but no vector → cosine term 0 → score
    # ≤ 0.4 < min_score, dropped (independent of its ts_rank).
    async with test_engine.begin() as conn:
        await conn.execute(
            sa.text("""
                INSERT INTO wiki_pages (org_id, path, title, content)
                VALUES (CAST(:org AS UUID), 'legacy.md', 'Legacy',
                        'widget widget widget legacy')
            """),
            {"org": org_id},
        )
    # Embedded page: same keyword + cosine 1.0 → score ≥ 0.6, always kept.
    await _seed_page(test_engine, org_id, "current.md", "Current",
                     "widget reference", _unit_vec(1.0))

    with patch.object(embeddings, "is_enabled", return_value=True), \
         patch.object(embeddings, "embed_text", new_callable=AsyncMock,
                      return_value=query_vec):
        paths = await wiki_db.find_relevant_pages("widget", limit=8)

    assert "current.md" in paths
    assert "legacy.md" not in paths
