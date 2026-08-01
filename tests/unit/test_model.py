"""Unit tests for app/model.py — the central LLM access layer.

No network: the provider is replaced with a fake throughout.
"""
import pytest
from unittest.mock import MagicMock, patch


class _FakeProvider:
    name = "fake"

    def __init__(self):
        self.chat_calls = []
        self.converse_calls = []
        self.embed_calls = []

    def chat_model(self, model_id, max_tokens=4096):
        self.chat_calls.append((model_id, max_tokens))
        return MagicMock()

    def converse_client(self, model_id):
        self.converse_calls.append(model_id)
        client = MagicMock()
        client.model_id = model_id
        return client

    def embed_sync(self, model_id, text):
        self.embed_calls.append((model_id, text))
        return [0.5, 0.5]


@pytest.fixture
def fake_provider():
    provider = _FakeProvider()
    with patch("app.model._provider", return_value=provider):
        yield provider


# ── role resolution ────────────────────────────────────────────────────────

def test_model_id_for_reads_the_configured_setting():
    from app import model
    with patch("app.model.settings") as s:
        s.BEDROCK_QUERY_MODEL_ID = "some.query.model"
        assert model.model_id_for(model.Role.QUERY) == "some.query.model"


def test_model_id_for_accepts_a_plain_string_role():
    from app import model
    with patch("app.model.settings") as s:
        s.BEDROCK_EDIT_MODEL_ID = "some.edit.model"
        assert model.model_id_for("edit") == "some.edit.model"


def test_model_id_for_rejects_unknown_role():
    from app import model
    with pytest.raises(ValueError, match="Unknown model role"):
        model.model_id_for("not_a_role")


def test_every_role_has_a_distinct_setting():
    from app import model
    settings_used = [model._ROLE_SETTING[r] for r in model.Role]
    assert len(settings_used) == len(set(settings_used))


# ── chat ───────────────────────────────────────────────────────────────────

def test_get_chat_uses_the_role_model_and_tracks_usage(fake_provider):
    from app import model
    from app.providers.usage import TrackedChat
    with patch("app.model.settings") as s:
        s.BEDROCK_INGEST_MODEL_ID = "planner.model"
        llm = model.get_chat(model.Role.INGEST_PLAN, max_tokens=1234)
    assert isinstance(llm, TrackedChat)
    assert fake_provider.chat_calls == [("planner.model", 1234)]
    assert llm._model_id == "planner.model"


def test_get_chat_operation_defaults_to_role_name(fake_provider):
    from app import model
    with patch("app.model.settings") as s:
        s.BEDROCK_QUERY_MODEL_ID = "m"
        assert model.get_chat(model.Role.QUERY)._operation == "query"


def test_get_chat_operation_can_be_overridden(fake_provider):
    """One role can serve several operations — recalibrate analyze vs. write."""
    from app import model
    with patch("app.model.settings") as s:
        s.BEDROCK_RECALIBRATE_MODEL_ID = "m"
        llm = model.get_chat(model.Role.RECALIBRATE, operation="recalibrate_analyze")
    assert llm._operation == "recalibrate_analyze"


def test_get_converse_uses_the_role_model(fake_provider):
    from app import model
    with patch("app.model.settings") as s:
        s.BEDROCK_DRAFT_AGENT_MODEL_ID = "drafter.model"
        client = model.get_converse(model.Role.DRAFT_AGENT)
    assert fake_provider.converse_calls == ["drafter.model"]
    assert client.model_id == "drafter.model"


# ── embeddings ─────────────────────────────────────────────────────────────

def test_embedding_enabled_follows_config():
    from app import model
    with patch("app.model.settings") as s:
        s.BEDROCK_EMBEDDING_MODEL_ID = ""
        assert model.embedding_enabled() is False
        s.BEDROCK_EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"
        assert model.embedding_enabled() is True


@pytest.mark.asyncio
async def test_embed_returns_none_when_disabled(fake_provider):
    from app import model
    with patch("app.model.settings") as s:
        s.BEDROCK_EMBEDDING_MODEL_ID = ""
        assert await model.embed("hi") is None
    assert fake_provider.embed_calls == []


@pytest.mark.asyncio
async def test_embed_delegates_to_provider(fake_provider):
    from app import model
    with patch("app.model.settings") as s:
        s.BEDROCK_EMBEDDING_MODEL_ID = "embed.model"
        vec = await model.embed("hi")
    assert vec == [0.5, 0.5]
    assert fake_provider.embed_calls == [("embed.model", "hi")]


@pytest.mark.asyncio
async def test_embed_swallows_provider_errors():
    """Semantic search degrades to BM25 rather than failing the request."""
    from app import model
    boom = MagicMock()
    boom.embed_sync.side_effect = RuntimeError("provider down")
    with patch("app.model._provider", return_value=boom), \
         patch("app.model.settings") as s:
        s.BEDROCK_EMBEDDING_MODEL_ID = "embed.model"
        assert await model.embed("hi") is None


# ── provider registry ──────────────────────────────────────────────────────

def test_default_provider_is_bedrock():
    from app import model
    with patch("app.model.settings") as s:
        s.LLM_PROVIDER = ""
        assert model.provider_name() == "bedrock"


def test_provider_name_is_normalised():
    from app import model
    with patch("app.model.settings") as s:
        s.LLM_PROVIDER = "  Bedrock  "
        assert model.provider_name() == "bedrock"


def test_unknown_provider_raises_with_a_useful_message():
    from app import providers
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        providers.get("gpt5-turbo-max")


def test_bedrock_provider_satisfies_the_protocol():
    from app import providers
    from app.providers.base import Provider
    assert isinstance(providers.get("bedrock"), Provider)
    assert "bedrock" in providers.available()
