"""Config carries model *names* now; the old raw-ID vars must keep working.

A deployment whose .env predates the name catalogue should upgrade without
touching its config, so these tests pin the promotion rules in
Settings.model_post_init.
"""
import pytest

from app.config import LEGACY_MODEL_VARS, Settings


def _settings(monkeypatch, **env) -> Settings:
    """Build a Settings from a clean env — no .env file, only what's passed."""
    for var in list(Settings.model_fields):
        monkeypatch.delenv(var, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return Settings(_env_file=None)


def test_defaults_are_model_names_not_ids(monkeypatch):
    s = _settings(monkeypatch)
    assert s.MODEL_QUERY == "haiku45"
    assert s.MODEL_RECALIBRATE == "sonnet45"
    assert "." not in s.MODEL_QUERY  # a name, not a vendor ID


def test_legacy_var_is_promoted_onto_its_replacement(monkeypatch):
    s = _settings(monkeypatch,
                  BEDROCK_QUERY_MODEL_ID="us.anthropic.claude-opus-4-7")
    assert s.MODEL_QUERY == "us.anthropic.claude-opus-4-7"


def test_every_legacy_var_has_a_promotion_target(monkeypatch):
    for legacy, current in LEGACY_MODEL_VARS.items():
        s = _settings(monkeypatch, **{legacy: "us.anthropic.some-model-v1:0"})
        assert getattr(s, current) == "us.anthropic.some-model-v1:0", legacy


def test_an_explicit_new_var_wins_over_the_legacy_one(monkeypatch):
    """Setting both means the operator is mid-migration — honour the new one."""
    s = _settings(monkeypatch,
                  MODEL_QUERY="sonnet5",
                  BEDROCK_QUERY_MODEL_ID="us.anthropic.claude-opus-4-7")
    assert s.MODEL_QUERY == "sonnet5"


def test_new_var_wins_even_when_set_to_its_own_default(monkeypatch):
    """The check is 'was it set', not 'is it non-default' — those differ."""
    s = _settings(monkeypatch,
                  MODEL_QUERY="haiku45",
                  BEDROCK_QUERY_MODEL_ID="us.anthropic.claude-opus-4-7")
    assert s.MODEL_QUERY == "haiku45"


def test_legacy_var_left_unset_does_not_clobber_the_default(monkeypatch):
    s = _settings(monkeypatch)
    assert s.MODEL_EDIT == "sonnet45"


def test_oldest_writer_var_still_reaches_the_ingest_write_role(monkeypatch):
    """BEDROCK_WRITER_MODEL_ID → BEDROCK_INGEST_WRITER_MODEL_ID → MODEL_INGEST_WRITE."""
    s = _settings(monkeypatch, BEDROCK_WRITER_MODEL_ID="us.meta.llama4-scout-17b-instruct-v1:0")
    assert s.MODEL_INGEST_WRITE == "us.meta.llama4-scout-17b-instruct-v1:0"


def test_newer_writer_var_beats_the_oldest_one(monkeypatch):
    s = _settings(monkeypatch,
                  BEDROCK_WRITER_MODEL_ID="us.meta.llama4-scout-17b-instruct-v1:0",
                  BEDROCK_INGEST_WRITER_MODEL_ID="us.meta.llama4-maverick-17b-instruct-v1:0")
    assert s.MODEL_INGEST_WRITE == "us.meta.llama4-maverick-17b-instruct-v1:0"


@pytest.mark.parametrize("legacy,current", list(LEGACY_MODEL_VARS.items()))
def test_promotion_map_points_at_real_fields(legacy, current):
    assert legacy in Settings.model_fields
    assert current in Settings.model_fields


def test_promotion_map_covers_every_role():
    """A new role must not silently lose its legacy var."""
    from app.model import Role, _ROLE_SETTING
    targets = set(LEGACY_MODEL_VARS.values())
    assert {_ROLE_SETTING[r] for r in Role} == targets


def test_shared_default_applies_to_all_text_roles_only(monkeypatch):
    s = _settings(monkeypatch, MODEL_DEFAULT=" sonnet45 ")
    for current in LEGACY_MODEL_VARS.values():
        assert getattr(s, current) == ("titanembedv1" if current == "MODEL_EMBEDDING" else "sonnet45")


@pytest.mark.parametrize("default", ["", "  "])
def test_empty_shared_default_preserves_previous_defaults(monkeypatch, default):
    s = _settings(monkeypatch, MODEL_DEFAULT=default)
    assert s.MODEL_QUERY == "haiku45"
    assert s.MODEL_INGEST_WRITE == "llama4maverick"
    assert s.MODEL_RECALIBRATE == "sonnet45"


@pytest.mark.parametrize("override", ["haiku45", ""])
def test_explicit_role_overrides_shared_default_even_when_empty(monkeypatch, override):
    s = _settings(monkeypatch, MODEL_DEFAULT="sonnet45", MODEL_QUERY=override,
                  BEDROCK_QUERY_MODEL_ID="legacy.model")
    assert s.MODEL_QUERY == override


@pytest.mark.parametrize("legacy,current", list(LEGACY_MODEL_VARS.items()))
def test_legacy_model_overrides_shared_default(monkeypatch, legacy, current):
    s = _settings(monkeypatch, MODEL_DEFAULT="sonnet45", **{legacy: "legacy.model"})
    assert getattr(s, current) == "legacy.model"


def test_oldest_writer_and_disabled_embeddings_survive_shared_default(monkeypatch):
    s = _settings(monkeypatch, MODEL_DEFAULT="sonnet45", BEDROCK_WRITER_MODEL_ID="legacy.writer",
                  BEDROCK_EMBEDDING_MODEL_ID="")
    assert s.MODEL_INGEST_WRITE == "legacy.writer"
    assert s.MODEL_EMBEDDING == ""


def test_shared_connection_settings_mask_api_key(monkeypatch):
    s = _settings(monkeypatch, LLM_API_KEY="private-test-key", LLM_BASE_URL="http://localhost:11434/v1")
    assert s.LLM_API_KEY.get_secret_value() == "private-test-key"
    assert "private-test-key" not in repr(s)
    assert "private-test-key" not in s.model_dump_json()
    assert s.LLM_BASE_URL == "http://localhost:11434/v1"


@pytest.mark.parametrize("url", ["file:///tmp/model", "localhost:1234", "https://host/?key=secret",
                                 "https://user:secret@host/v1", "https://host/#secret"])
def test_shared_endpoint_rejects_unsafe_or_malformed_urls(monkeypatch, url):
    with pytest.raises(ValueError, match="LLM_BASE_URL"):
        _settings(monkeypatch, LLM_BASE_URL=url)
