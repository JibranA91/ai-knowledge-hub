"""Unit tests for app/model.py — the central LLM access layer.

No network: the provider is replaced with a fake throughout.
"""
import pytest
from unittest.mock import MagicMock, patch


class _FakeProvider:
    name = "fake"

    # A tiny catalogue so name→ID mapping can be asserted without depending on
    # Bedrock's real one.
    CATALOG = {"fastmodel": "fake.fast-v1", "bigmodel": "fake.big-v1"}

    def __init__(self):
        self.chat_calls = []
        self.converse_calls = []
        self.embed_calls = []

    def resolve_model(self, name):
        from app.providers.base import UnknownModelError
        key = "".join(c for c in name.lower() if c.isalnum())
        if key in self.CATALOG:
            return self.CATALOG[key]
        if "." in name:
            return name  # raw ID passthrough
        raise UnknownModelError(f"Unknown model name {name!r} for the fake provider.")

    def known_models(self):
        return sorted(self.CATALOG)

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

def test_model_name_for_reads_the_configured_setting():
    from app import model
    with patch("app.model.settings") as s:
        s.MODEL_QUERY = "fastmodel"
        assert model.model_name_for(model.Role.QUERY) == "fastmodel"


def test_model_id_for_maps_the_name_through_the_provider(fake_provider):
    """Config carries a name; the provider turns it into a vendor model ID."""
    from app import model
    with patch("app.model.settings") as s:
        s.MODEL_QUERY = "fastmodel"
        assert model.model_id_for(model.Role.QUERY) == "fake.fast-v1"


def test_model_id_for_accepts_a_plain_string_role(fake_provider):
    from app import model
    with patch("app.model.settings") as s:
        s.MODEL_EDIT = "bigmodel"
        assert model.model_id_for("edit") == "fake.big-v1"


def test_model_id_for_rejects_unknown_role():
    from app import model
    with pytest.raises(ValueError, match="Unknown model role"):
        model.model_id_for("not_a_role")


def test_model_id_for_names_the_role_when_the_model_name_is_bad(fake_provider):
    """The error has to say which role is misconfigured, not just which name."""
    from app import model
    from app.providers.base import UnknownModelError
    with patch("app.model.settings") as s:
        s.MODEL_QUERY = "nosuchmodel"
        with pytest.raises(UnknownModelError, match="role 'query'"):
            model.model_id_for(model.Role.QUERY)


def test_model_id_for_is_empty_when_the_role_is_unset(fake_provider):
    from app import model
    with patch("app.model.settings") as s:
        s.MODEL_EMBEDDING = ""
        assert model.model_id_for(model.Role.EMBEDDING) == ""


def test_every_role_has_a_distinct_setting():
    from app import model
    settings_used = [model._ROLE_SETTING[r] for r in model.Role]
    assert len(settings_used) == len(set(settings_used))


# ── startup validation ─────────────────────────────────────────────────────

def test_validate_configuration_resolves_every_configured_role(fake_provider):
    from app import model
    with patch("app.model.settings") as s:
        for setting in model._ROLE_SETTING.values():
            setattr(s, setting, "fastmodel")
        resolved = model.validate_configuration()
    assert set(resolved) == set(model.Role)
    assert set(resolved.values()) == {"fake.fast-v1"}


def test_validate_configuration_skips_unset_optional_roles(fake_provider):
    from app import model
    with patch("app.model.settings") as s:
        for setting in model._ROLE_SETTING.values():
            setattr(s, setting, "fastmodel")
        s.MODEL_EMBEDDING = ""
        resolved = model.validate_configuration()
    assert model.Role.EMBEDDING not in resolved


def test_validate_configuration_reports_every_bad_role_at_once(fake_provider):
    """One restart per typo would be a miserable way to fix a config."""
    from app import model
    from app.providers.base import UnknownModelError
    with patch("app.model.settings") as s:
        for setting in model._ROLE_SETTING.values():
            setattr(s, setting, "fastmodel")
        s.MODEL_QUERY = "bogus1"
        s.MODEL_EDIT = "bogus2"
        with pytest.raises(UnknownModelError) as exc:
            model.validate_configuration()
    assert "query" in str(exc.value)
    assert "edit" in str(exc.value)


# ── chat ───────────────────────────────────────────────────────────────────

def test_get_chat_uses_the_role_model_and_tracks_usage(fake_provider):
    from app import model
    from app.providers.usage import TrackedChat
    with patch("app.model.settings") as s:
        s.MODEL_INGEST_PLAN = "planner.model"
        llm = model.get_chat(model.Role.INGEST_PLAN, max_tokens=1234)
    assert isinstance(llm, TrackedChat)
    assert fake_provider.chat_calls == [("planner.model", 1234)]
    assert llm._model_id == "planner.model"


def test_get_chat_operation_defaults_to_role_name(fake_provider):
    from app import model
    with patch("app.model.settings") as s:
        s.MODEL_QUERY = "fastmodel"
        assert model.get_chat(model.Role.QUERY)._operation == "query"


def test_get_chat_operation_can_be_overridden(fake_provider):
    """One role can serve several operations — recalibrate analyze vs. write."""
    from app import model
    with patch("app.model.settings") as s:
        s.MODEL_RECALIBRATE = "fastmodel"
        llm = model.get_chat(model.Role.RECALIBRATE, operation="recalibrate_analyze")
    assert llm._operation == "recalibrate_analyze"


def test_get_converse_uses_the_role_model(fake_provider):
    from app import model
    with patch("app.model.settings") as s:
        s.MODEL_DRAFT_AGENT = "drafter.model"
        client = model.get_converse(model.Role.DRAFT_AGENT)
    assert fake_provider.converse_calls == ["drafter.model"]
    assert client.model_id == "drafter.model"


# ── embeddings ─────────────────────────────────────────────────────────────

def test_embedding_enabled_follows_config():
    from app import model
    with patch("app.model.settings") as s:
        s.MODEL_EMBEDDING = ""
        assert model.embedding_enabled() is False
        s.MODEL_EMBEDDING = "amazon.titan-embed-text-v2:0"
        assert model.embedding_enabled() is True


@pytest.mark.asyncio
async def test_embed_returns_none_when_disabled(fake_provider):
    from app import model
    with patch("app.model.settings") as s:
        s.MODEL_EMBEDDING = ""
        assert await model.embed("hi") is None
    assert fake_provider.embed_calls == []


@pytest.mark.asyncio
async def test_embed_delegates_to_provider(fake_provider):
    from app import model
    with patch("app.model.settings") as s:
        s.MODEL_EMBEDDING = "embed.model"
        vec = await model.embed("hi")
    assert vec == [0.5, 0.5]
    assert fake_provider.embed_calls == [("embed.model", "hi")]


@pytest.mark.asyncio
async def test_embed_swallows_provider_errors():
    """Semantic search degrades to BM25 rather than failing the request."""
    from app import model
    boom = MagicMock()
    boom.resolve_model.return_value = "embed.model"
    boom.embed_sync.side_effect = RuntimeError("provider down")
    with patch("app.model._provider", return_value=boom), \
         patch("app.model.settings") as s:
        s.MODEL_EMBEDDING = "embed.model"
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
