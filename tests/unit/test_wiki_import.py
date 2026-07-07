"""Unit tests for wiki_import pure helpers (no DB)."""
from app.config import settings
from app.services.wiki_import import _is_unsafe_rel, _embeddings_compatible


# ── zip-slip guard ─────────────────────────────────────────────────────────

def test_is_unsafe_rel_flags_traversal_and_absolute():
    assert _is_unsafe_rel("../evil.md")
    assert _is_unsafe_rel("a/../../b.md")
    assert _is_unsafe_rel("/abs.md")
    assert _is_unsafe_rel("c:\\windows")     # backslash + drive colon
    assert _is_unsafe_rel("")                 # empty


def test_is_unsafe_rel_allows_normal_paths():
    assert not _is_unsafe_rel("concepts/x.md")
    assert not _is_unsafe_rel("a/b/c.md")
    assert not _is_unsafe_rel("page.md")


# ── embedding-compatibility gate ───────────────────────────────────────────

def test_embeddings_compatible_when_model_and_dims_match():
    manifest = {
        "embedding_dimensions": settings.EMBEDDING_DIMENSIONS,
        "embedding_model": settings.BEDROCK_EMBEDDING_MODEL_ID or "",
    }
    assert _embeddings_compatible(manifest, {"p": [0.1, 0.2]}) is True


def test_embeddings_incompatible_when_no_vectors():
    manifest = {
        "embedding_dimensions": settings.EMBEDDING_DIMENSIONS,
        "embedding_model": settings.BEDROCK_EMBEDDING_MODEL_ID or "",
    }
    assert _embeddings_compatible(manifest, {}) is False


def test_embeddings_incompatible_on_dimension_mismatch():
    manifest = {"embedding_dimensions": 99999, "embedding_model": settings.BEDROCK_EMBEDDING_MODEL_ID or ""}
    assert _embeddings_compatible(manifest, {"p": [0.1]}) is False


def test_embeddings_incompatible_on_model_mismatch():
    manifest = {"embedding_dimensions": settings.EMBEDDING_DIMENSIONS, "embedding_model": "some-other-model"}
    assert _embeddings_compatible(manifest, {"p": [0.1]}) is False
