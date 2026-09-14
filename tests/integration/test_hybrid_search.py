"""Integration tests for hybrid (BM25 + cosine) page search.

Exercises the real `_HYBRID_SEARCH` SQL in `wiki_db.find_relevant_pages`
against Postgres — the unit tests mock the DB session and the route tests mock
`_find_relevant_pages`, so this is the only coverage of the actual query.

Requires the pgvector extension (the `wiki_pages.embedding` column). Both the
integration testcontainer and CI service use `pgvector/pgvector:pg16`.
A missing embedding column is a setup failure, not a reason to skip coverage.
"""
import math

import pytest
import pytest_asyncio
import sqlalchemy as sa
from unittest.mock import AsyncMock, patch

from app import model
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
async def require_pgvector(test_engine, monkeypatch):
    """Require the embedding column, and reset wiki_db's cached flag.

    `_embedding_col_ready` is a module-global set on first use; a prior test (or
    a prior run against a non-pgvector DB) may have cached False, so we force a
    re-check against this DB.
    """
    monkeypatch.setattr(model.settings, "MODEL_EMBEDDING", "titanembedv1")
    monkeypatch.setattr(model.settings, "LLM_PROVIDER", "bedrock")
    assert await _embedding_column_present(test_engine), (
        "Integration tests require pgvector and wiki_pages.embedding. "
        "Use pgvector/pgvector:pg16 and run migrations on a fresh test database."
    )
    wiki_db._embedding_col_ready = None
    yield
    wiki_db._embedding_col_ready = None


async def _seed_page(test_engine, org_id: str, path: str, title: str,
                     content: str, vec: list[float]) -> None:
    async with test_engine.begin() as conn:
        await conn.execute(
            sa.text("""
                INSERT INTO wiki_pages (org_id, path, title, content, embedding, embedding_space)
                VALUES (CAST(:org AS UUID), :path, :title, :content,
                        CAST(:emb AS vector), :space)
            """),
            {"org": org_id, "path": path, "title": title, "content": content,
             "emb": embeddings.vec_to_pg(vec), "space": model.embedding_identity()},
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
    """A page without a vector retains full keyword-search weight."""
    org_id = default_user["org_id"]
    query_vec = _unit_vec(1.0)
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
    assert "legacy.md" in paths


@pytest.mark.asyncio
@pytest.mark.parametrize("space", [None, "old-provider:same-size-model"])
async def test_search_and_export_exclude_incompatible_vectors(
    test_engine, default_user, user_ctx, require_pgvector, space
):
    org_id = default_user["org_id"]
    vec = _unit_vec(1.0)
    await _seed_page(test_engine, org_id, "old.md", "Old", "unrelated old page", vec)
    await _seed_page(test_engine, org_id, "keyword.md", "Widget", "widget widget", vec)
    await _seed_page(test_engine, org_id, "current.md", "Current", "current text", vec)
    async with test_engine.begin() as conn:
        await conn.execute(sa.text("""
            UPDATE wiki_pages SET embedding_space = :space
            WHERE org_id = CAST(:org AS UUID) AND path IN ('old.md', 'keyword.md')
        """), {"space": space, "org": org_id})
    with patch.object(embeddings, "embed_text", AsyncMock(return_value=vec)):
        assert "old.md" not in await wiki_db.find_relevant_pages("widget")
        assert "keyword.md" in await wiki_db.find_relevant_pages("widget")
        assert [p["path"] for p in await wiki_db.semantic_search_wiki(vec)] == ["current.md"]
    exported = {p["path"]: p["embedding"] for p in await wiki_db.list_wiki_pages_for_export(True)}
    assert exported["old.md"] is None
    assert exported["keyword.md"] is None
    assert exported["current.md"] == vec


@pytest.mark.asyncio
async def test_rewrite_replaces_embedding_provenance_and_failed_embed_invalidates_it(
    test_engine, default_user, user_ctx, require_pgvector
):
    vec = _unit_vec(1.0)
    with patch.object(embeddings, "embed_text", AsyncMock(return_value=vec)):
        await wiki_db.upsert_wiki_page("page.md", "# Widget\nwidget information")
    assert len(await wiki_db.semantic_search_wiki(vec)) == 1
    with patch.object(embeddings, "embed_text", AsyncMock(return_value=None)):
        await wiki_db.upsert_wiki_page("page.md", "# Changed\nnew content")
    assert await wiki_db.semantic_search_wiki(vec) == []
    assert (await wiki_db.list_wiki_pages_for_export(True))[0]["embedding"] is None
    with patch.object(embeddings, "embed_text", AsyncMock(return_value=vec)):
        await wiki_db.upsert_wiki_page("page.md", "# Rebuilt\ncurrent content")
    assert len(await wiki_db.semantic_search_wiki(vec)) == 1


@pytest.mark.asyncio
async def test_switching_to_same_dimension_model_excludes_old_vectors(
    test_engine, default_user, user_ctx, require_pgvector, monkeypatch
):
    vec = _unit_vec(1.0)
    with patch.object(embeddings, "embed_text", AsyncMock(return_value=vec)):
        await wiki_db.upsert_wiki_page("old.md", "# Old\nold facts")
        monkeypatch.setattr(model.settings, "MODEL_EMBEDDING", "cohereembedv4")
        await wiki_db.upsert_wiki_page("new.md", "# New\nnew facts")
        assert [p["path"] for p in await wiki_db.semantic_search_wiki(vec)] == ["new.md"]
        assert "old.md" in await wiki_db.find_relevant_pages("old")
    assert (await wiki_db.list_wiki_pages_for_export(True))[1]["embedding"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("transfer", ["clone", "import"])
async def test_clone_and_import_preserve_embedding_identity(
    test_engine, default_user, user_ctx, require_pgvector, transfer
):
    from dataclasses import replace
    from app.context import current_user
    from app.services import orgs, wiki_import
    from app.services.wiki_engine import WikiEngine

    vec = _unit_vec(1.0)
    with patch.object(embeddings, "embed_text", AsyncMock(return_value=vec)):
        await wiki_db.upsert_wiki_page("source.md", "# Source\nsource content")
    if transfer == "clone":
        target = await orgs.clone_org(default_user["org_id"], "Embedding clone")
        bundle = None
    else:
        engine = object.__new__(WikiEngine)
        bundle, _, _ = await engine.build_wiki_export(True)
        target = await orgs.create_org("Embedding import")
    token = current_user.set(replace(current_user.get(), org_id=target))
    try:
        if bundle is not None:
            with patch.object(embeddings, "embed_text", AsyncMock(side_effect=AssertionError("should reuse vector"))):
                result = await wiki_import.import_bundle(bundle)
                assert result["embeddings_reused"] == 1
        assert [p["path"] for p in await wiki_db.semantic_search_wiki(vec)] == ["source.md"]
    finally:
        current_user.reset(token)
