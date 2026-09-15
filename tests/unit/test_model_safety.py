"""Regression coverage for model configuration and embedding safety."""
from unittest.mock import AsyncMock, patch

import pytest

from app import model
from app.config import Settings
from app.providers.base import UnknownModelError
from app.providers.bedrock import BedrockConverseClient


@pytest.fixture
def clean_settings(monkeypatch):
    for name in Settings.model_fields:
        monkeypatch.delenv(name, raising=False)
    settings = Settings(_env_file=None, MODEL_EMBEDDING="titanembedv1")
    monkeypatch.setattr(model, "settings", settings)
    monkeypatch.setattr("app.providers.bedrock.settings", settings)
    return settings


def test_embeddings_default_to_disabled(clean_settings):
    assert Settings(_env_file=None).MODEL_EMBEDDING == ""


@pytest.mark.parametrize("role", [r for r in model.Role if r != model.Role.EMBEDDING])
def test_required_role_cannot_be_blank(clean_settings, role):
    setattr(clean_settings, model._ROLE_SETTING[role], "  ")
    with pytest.raises(UnknownModelError, match=role.value):
        model.validate_configuration()


@pytest.mark.parametrize("role,name", [
    (model.Role.QUERY, "titanembedv1"),
    (model.Role.INGEST_PLAN, "titanembedv1"),
    (model.Role.EMBEDDING, "haiku45"),
])
def test_rejects_model_with_wrong_capability(clean_settings, role, name):
    setattr(clean_settings, model._ROLE_SETTING[role], name)
    with pytest.raises(UnknownModelError, match=role.value):
        model.validate_configuration()


@pytest.mark.parametrize("name", ["titanembedv2", "cohereembeden", "cohereembedml"])
def test_rejects_embedding_model_incompatible_with_storage(clean_settings, name):
    clean_settings.MODEL_EMBEDDING = name
    with pytest.raises(UnknownModelError, match="1536"):
        model.validate_configuration()


def test_setting_dimensions_does_not_resize_database(clean_settings):
    clean_settings.EMBEDDING_DIMENSIONS = 1024
    with pytest.raises(UnknownModelError, match="1536"):
        model.validate_configuration()


@pytest.mark.asyncio
@pytest.mark.parametrize("vector", [[0.1] * 1024, [float("nan")] * 1536,
                                   [float("inf")] * 1536, [0.0] * 1536])
async def test_bad_provider_vector_falls_back_without_reaching_db(clean_settings, vector):
    with patch.object(model._provider(), "embed_sync", return_value=vector):
        assert await model.embed("hello") is None


@pytest.mark.asyncio
async def test_valid_provider_vector_is_preserved(clean_settings):
    vector = [0.1] * 1536
    with patch.object(model._provider(), "embed_sync", return_value=vector):
        assert await model.embed("hello") == vector


@pytest.mark.asyncio
@pytest.mark.parametrize("partial", [False, True])
async def test_stream_surfaces_worker_failure_and_releases_slot(clean_settings, partial):
    import asyncio
    from unittest.mock import MagicMock

    def events():
        if partial:
            yield {"contentBlockDelta": {"delta": {"text": "partial"}}}
        raise RuntimeError("upstream stream failed")

    stream = MagicMock()
    stream.__iter__.side_effect = events
    client = MagicMock()
    client.converse_stream.return_value = {"stream": stream}
    with patch.object(BedrockConverseClient, "_make_client", return_value=client):
        service = BedrockConverseClient("test-safety")
    semaphore = asyncio.Semaphore(1)
    chunks = []
    with patch.object(service, "_sem", return_value=semaphore), \
         patch("app.providers.usage.record", new_callable=AsyncMock):
        with pytest.raises(RuntimeError, match="upstream stream failed"):
            async for chunk in service.converse_stream("system", []):
                chunks.append(chunk)
    assert chunks == (["partial"] if partial else [])
    assert not semaphore.locked()
    stream.close.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["request", "event"])
async def test_stream_reports_request_and_in_band_errors(clean_settings, failure):
    from unittest.mock import MagicMock
    client = MagicMock()
    if failure == "request":
        client.converse_stream.side_effect = RuntimeError("connection refused")
    else:
        client.converse_stream.return_value = {
            "stream": iter([{"modelStreamErrorException": {"message": "interrupted"}}])}
    with patch.object(BedrockConverseClient, "_make_client", return_value=client):
        service = BedrockConverseClient("test-errors")
    with pytest.raises(RuntimeError):
        async for _ in service.converse_stream("system", []):
            pass


def test_embedding_identity_tracks_model_and_provider_not_alias(clean_settings):
    import json
    original = model.embedding_identity()
    clean_settings.MODEL_EMBEDDING = "Titan-Embed-v1"
    assert model.embedding_identity() == original
    clean_settings.LLM_PROVIDER = "openai"
    clean_settings.MODEL_EMBEDDING = "embed3small"
    assert json.loads(model.embedding_identity()) == ["openai", "text-embedding-3-small", 1536]
    assert model.embedding_identity() != original
    clean_settings.MODEL_EMBEDDING = "embed3large"
    assert json.loads(model.embedding_identity())[1] == "text-embedding-3-large"


def test_default_configuration_is_valid(clean_settings):
    assert set(model.validate_configuration()) == set(model.Role)


def test_disabled_embedding_is_optional(clean_settings):
    clean_settings.MODEL_EMBEDDING = ""
    assert model.Role.EMBEDDING not in model.validate_configuration()
