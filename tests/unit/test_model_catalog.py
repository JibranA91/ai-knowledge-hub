"""Catalogue contracts: one mapping, no raw-ID bypass or provider substitution."""
import copy
import json

import pytest

from app import model_catalog as catalog, providers
from app.providers.base import UnknownModelError


@pytest.mark.parametrize("name", ["haiku45", "haiku-4.5", "Haiku 4.5", "haiku_4_5"])
def test_same_name_resolves_by_provider(name):
    assert catalog.resolve(name, "bedrock", geo="us") == "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    assert catalog.resolve(name, "anthropic") == "claude-haiku-4-5-20251001"


@pytest.mark.parametrize("provider,name", [
    ("bedrock", "haiku4.5x"), ("bedrock", "anthropic.claude-haiku-4-5-20251001-v1:0"),
    ("bedrock", "us.anthropic.claude-haiku-4-5-20251001-v1:0"), ("bedrock", "arn:aws:bedrock:test"),
    ("bedrock", "vendor.unknown:0"), ("anthropic", "claude-haiku-4-5-20251001"),
    ("anthropic", "claude-new-snapshot"), ("openai", "gpt-4.1-mini-2025-04-14"),
    ("openai", "gpt-5-new-snapshot"), ("openai", "o3"), ("openai", "text-embedding-3-small"),
])
def test_raw_ids_and_typos_are_rejected(provider, name):
    with pytest.raises(UnknownModelError, match="app/model_catalog.yaml"):
        catalog.resolve(name, provider)


@pytest.mark.parametrize("name,provider", [
    ("haiku45", "openai"), ("gpt41mini", "bedrock"), ("llama4maverick", "anthropic"),
    ("embed3small", "anthropic"), ("titanembedv1", "openai"),
])
def test_unsupported_pair_never_substitutes_a_model(name, provider):
    with pytest.raises(UnknownModelError, match="No mapping"):
        catalog.resolve(name, provider)


@pytest.mark.parametrize("geo,prefix", [("us", "us."), ("global", "global."), ("", ""), (" US. ", "us.")])
def test_bedrock_profile_geography(geo, prefix):
    assert catalog.resolve("sonnet45", "bedrock", geo=geo) == prefix + "anthropic.claude-sonnet-4-5-20250929-v1:0"
    assert catalog.resolve("titanembedv1", "bedrock", geo=geo) == "amazon.titan-embed-text-v1"
    assert catalog.resolve("haiku45", "anthropic", geo=geo) == "claude-haiku-4-5-20251001"


@pytest.mark.parametrize("name,geo", [("llama4maverick", "global"), ("haiku45", "eu"), ("opus41", "global")])
def test_invalid_geography_fails_locally(name, geo):
    with pytest.raises(UnknownModelError, match="BEDROCK_INFERENCE_GEO"):
        catalog.resolve(name, "bedrock", geo=geo)


def test_registered_provider_catalogue_is_complete_and_valid():
    entries = catalog.load_catalog()
    assert {p for entry in entries.values() for p in entry.providers} == set(providers.available())
    for provider in providers.available():
        assert catalog.known_models(provider)
        for name in catalog.known_models(provider):
            binding = entries[name].providers[provider]
            if binding.unavailable_reason:
                with pytest.raises(UnknownModelError, match="does not yet parse"):
                    catalog.resolve(name, provider)
            else:
                assert catalog.resolve(name, provider) == binding.model_id


@pytest.mark.parametrize("name,provider,caps", [
    ("haiku45", "bedrock", {"embedding"}), ("titanembedv1", "bedrock", {"chat"}),
    ("embed3small", "openai", {"tools"}), ("gpt41mini", "openai", {"embedding"}),
])
def test_role_capability_mismatch(name, provider, caps):
    with pytest.raises(UnknownModelError, match="does not support"):
        catalog.resolve(name, provider, capabilities=caps)


@pytest.mark.parametrize("role,caps", [
    ("ingest_plan", {"chat"}), ("query", {"chat", "converse"}),
    ("recalibrate", {"converse"}), ("draft_agent", {"chat"}),
])
def test_facade_enforces_required_capabilities_before_client_construction(role, caps, monkeypatch):
    from unittest.mock import Mock
    from app import model
    from app.config import Settings
    entry = catalog.ModelEntry(providers={"bedrock": catalog.Binding(model_id="limited", capabilities=caps)})
    monkeypatch.setattr(catalog, "load_catalog", lambda: {"limited": entry})
    settings = Settings(_env_file=None, MODEL_DEFAULT="limited")
    monkeypatch.setattr(model, "settings", settings)
    provider = Mock()
    monkeypatch.setattr(model, "_provider", lambda: provider)
    with pytest.raises(UnknownModelError, match=role):
        model.get_chat(role) if role in {"ingest_plan", "recalibrate"} else model.get_converse(role)
    provider.chat_model.assert_not_called()
    provider.converse_client.assert_not_called()


@pytest.mark.parametrize("name", ["titanembedv2", "cohereembeden", "cohereembedml"])
def test_embedding_storage_constraint(name):
    with pytest.raises(UnknownModelError, match="1536"):
        catalog.resolve(name, "bedrock", capabilities={"embedding"})


def test_catalogue_only_addition_reaches_resolution_and_payload_metadata(monkeypatch):
    data = {name: entry.model_dump(mode="json") for name, entry in catalog.load_catalog().items()}
    data["customchat"] = {"providers": {"openai": {
        "model_id": "custom-snapshot", "capabilities": ["chat", "tools", "converse", "stream"],
        "temperature": False,
    }}}
    parsed = catalog.parse_catalog(json.dumps(data))
    monkeypatch.setattr(catalog, "load_catalog", lambda: parsed)
    assert catalog.resolve("Custom Chat", "openai", capabilities={"tools"}) == "custom-snapshot"
    from app.providers.openai import _sampling
    assert _sampling("custom-snapshot", 0.3) == {}


@pytest.mark.parametrize("text", ["", "[]", "name: {}\nname: {}", "name: {providers: {bedrock: {}, bedrock: {}}}",
                                "Bad-Name: {}", "name: {providers: {}}", "name: {unexpected: true}"])
def test_malformed_catalogue_fails(text):
    with pytest.raises(UnknownModelError, match="Invalid app/model_catalog.yaml"):
        catalog.parse_catalog(text)


@pytest.mark.parametrize("change", [
    {"capabilities": ["magic"]}, {"dimensions": [1536]}, {"model_id": " "},
    {"capabilities": ["embedding"], "dimensions": []},
    {"capabilities": ["embedding", "chat"], "dimensions": [1536]},
    {"capabilities": ["tools"]}, {"capabilities": ["stream"]},
    {"inference_geos": ["invalid.geo"]}, {"unknown": True},
])
def test_invalid_binding_metadata_fails(change):
    binding = {"model_id": "example", "capabilities": ["chat"], **change}
    with pytest.raises(UnknownModelError):
        catalog.parse_catalog(json.dumps({"example": {"providers": {"bedrock": binding}}}))


@pytest.mark.parametrize("case", ["provider", "duplicate_id", "geo"])
def test_ambiguous_or_unsupported_metadata_fails(case):
    data = {"example": {"providers": {"openai": {"model_id": "example", "capabilities": ["chat"]}}}}
    if case == "provider":
        data["example"]["providers"]["typo"] = data["example"]["providers"].pop("openai")
    elif case == "duplicate_id":
        data["duplicate"] = copy.deepcopy(data["example"])
    else:
        data["example"]["providers"]["openai"]["inference_geos"] = ["us"]
    with pytest.raises(UnknownModelError):
        catalog.parse_catalog(json.dumps(data))
