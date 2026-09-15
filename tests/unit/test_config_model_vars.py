"""Names-only configuration, explicit precedence and obsolete-key rejection."""
import os

import pytest

from app.config import Settings, TEXT_MODEL_SETTINGS


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for key in list(os.environ):
        if key.upper() in Settings.model_fields or (key.upper().startswith("BEDROCK_") and key.upper().endswith("_MODEL_ID")):
            monkeypatch.delenv(key, raising=False)


def test_defaults_share_one_text_model_and_opt_in_embeddings():
    settings = Settings(_env_file=None)
    assert settings.MODEL_DEFAULT == "haiku45"
    assert all(getattr(settings, key) == "haiku45" for key in TEXT_MODEL_SETTINGS)
    assert settings.MODEL_EMBEDDING == ""


@pytest.mark.parametrize("override", ["haiku45", ""])
def test_explicit_role_overrides_shared_default_even_when_empty(override):
    settings = Settings(_env_file=None, MODEL_DEFAULT=" sonnet45 ", MODEL_QUERY=override)
    assert settings.MODEL_QUERY == override
    assert settings.MODEL_EDIT == "sonnet45"
    assert settings.MODEL_EMBEDDING == ""


@pytest.mark.parametrize("default", ["", "  "])
def test_blank_default_does_not_silently_restore_hidden_defaults(default):
    settings = Settings(_env_file=None, MODEL_DEFAULT=default)
    assert all(getattr(settings, key) == "" for key in TEXT_MODEL_SETTINGS)


def test_text_settings_cover_all_non_embedding_roles():
    from app.model import Role, _ROLE_SETTING
    assert set(TEXT_MODEL_SETTINGS) == {_ROLE_SETTING[r] for r in Role if r != Role.EMBEDDING}


OBSOLETE_KEYS = [
    "BEDROCK_INGEST_MODEL_ID", "BEDROCK_INGEST_WRITER_MODEL_ID", "BEDROCK_WRITER_MODEL_ID",
    "BEDROCK_QUERY_MODEL_ID", "BEDROCK_RECALIBRATE_MODEL_ID", "BEDROCK_DRAFT_AGENT_MODEL_ID",
    "BEDROCK_EDIT_MODEL_ID", "BEDROCK_EMBEDDING_MODEL_ID",
]


@pytest.mark.parametrize("key", OBSOLETE_KEYS)
@pytest.mark.parametrize("source", ["init", "environment", "dotenv"])
@pytest.mark.parametrize("value", ["", "private-obsolete-value"])
def test_obsolete_settings_rejected_even_when_shadowed(key, source, value, monkeypatch, tmp_path):
    kwargs = {"MODEL_DEFAULT": "haiku45", "MODEL_QUERY": "sonnet45", "LLM_API_KEY": "private-current-key"}
    path = None
    if source == "init":
        kwargs[key] = value
    elif source == "environment":
        monkeypatch.setenv(key.lower(), value)
    else:
        path = tmp_path / ".env"
        path.write_text(f"{key.lower()}={value}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Obsolete model settings") as exc:
        Settings(_env_file=path, **kwargs)
    assert key in str(exc.value)
    assert "MODEL_DEFAULT" in str(exc.value)
    assert "private-obsolete-value" not in str(exc.value)
    assert "private-current-key" not in str(exc.value)
    assert key not in Settings.model_fields


def test_environment_overrides_dotenv_but_init_wins(monkeypatch, tmp_path):
    path = tmp_path / ".env"
    path.write_text("MODEL_DEFAULT=haiku45\nMODEL_EDIT=haiku45\nUNRELATED_COMPOSE_VAR=ok\n", encoding="utf-8")
    monkeypatch.setenv("MODEL_DEFAULT", "sonnet45")
    monkeypatch.setenv("MODEL_EDIT", "sonnet45")
    settings = Settings(_env_file=path, MODEL_EDIT="opus45")
    assert settings.MODEL_QUERY == "sonnet45"
    assert settings.MODEL_EDIT == "opus45"


def test_stale_dotenv_rejected_despite_valid_environment(monkeypatch, tmp_path):
    path = tmp_path / ".env"
    path.write_text("BEDROCK_WRITER_MODEL_ID=\n", encoding="utf-8")
    monkeypatch.setenv("MODEL_DEFAULT", "haiku45")
    with pytest.raises(ValueError, match="BEDROCK_WRITER_MODEL_ID"):
        Settings(_env_file=path)


def test_shared_connection_settings_mask_api_key():
    settings = Settings(_env_file=None, LLM_API_KEY="private-test-key", LLM_BASE_URL="http://localhost:11434/v1")
    assert settings.LLM_API_KEY.get_secret_value() == "private-test-key"
    assert "private-test-key" not in repr(settings)
    assert "private-test-key" not in settings.model_dump_json()
    assert settings.LLM_BASE_URL == "http://localhost:11434/v1"


@pytest.mark.parametrize("url", ["file:///tmp/model", "localhost:1234", "https://host/?key=secret",
                                 "https://user:secret@host/v1", "https://host/#secret"])
def test_shared_endpoint_rejects_unsafe_or_malformed_urls(url):
    with pytest.raises(ValueError, match="LLM_BASE_URL") as exc:
        Settings(_env_file=None, LLM_BASE_URL=url, LLM_API_KEY="private-key")
    assert "private-key" not in str(exc.value)
