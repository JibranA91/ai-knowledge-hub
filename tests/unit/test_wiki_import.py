"""Unit tests for wiki_import pure helpers (no DB)."""
from app import model
import pytest
from app.config import settings
from app.services.wiki_import import _is_unsafe_rel, _embeddings_compatible


@pytest.fixture(autouse=True)
def embedding_config(monkeypatch):
    monkeypatch.setattr(settings, "MODEL_EMBEDDING", "titanembedv1")
    monkeypatch.setattr(settings, "LLM_PROVIDER", "bedrock")


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
        "embedding_model": model.model_id_for(model.Role.EMBEDDING),
        "embedding_provider": model.provider_name(),
        "embedding_space": model.embedding_identity(),
    }
    assert _embeddings_compatible(manifest, {"p": [0.1] * 1536}) is True


def test_embeddings_with_unknown_provenance_are_not_reused():
    manifest = {"embedding_dimensions": 1536,
                "embedding_model": model.model_id_for(model.Role.EMBEDDING)}
    assert not _embeddings_compatible(manifest, {"p": [0.1] * 1536})


def test_import_does_not_trust_manifest_dimensions_over_actual_vector():
    manifest = {"embedding_dimensions": 1536,
                "embedding_model": model.model_id_for(model.Role.EMBEDDING),
                "embedding_provider": model.provider_name(),
                "embedding_space": model.embedding_identity()}
    assert not _embeddings_compatible(manifest, {"p": [0.1] * 1024})


def test_embeddings_incompatible_when_no_vectors():
    manifest = {
        "embedding_dimensions": settings.EMBEDDING_DIMENSIONS,
        "embedding_model": model.model_id_for(model.Role.EMBEDDING),
    }
    assert _embeddings_compatible(manifest, {}) is False


def test_embeddings_incompatible_on_dimension_mismatch():
    manifest = {"embedding_dimensions": 99999, "embedding_model": model.model_id_for(model.Role.EMBEDDING)}
    assert _embeddings_compatible(manifest, {"p": [0.1]}) is False


def test_embeddings_incompatible_on_model_mismatch():
    manifest = {"embedding_dimensions": settings.EMBEDDING_DIMENSIONS, "embedding_model": "some-other-model"}
    assert _embeddings_compatible(manifest, {"p": [0.1]}) is False
