"""Connection checks use synthetic prompts and never need a database or real SDK call."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from app import check_models, model
from app.config import Settings


@pytest.fixture
def configured(monkeypatch):
    for name in Settings.model_fields:
        monkeypatch.delenv(name, raising=False)
    settings = Settings(_env_file=None, MODEL_DEFAULT="haiku45", MODEL_EMBEDDING="")
    monkeypatch.setattr(model, "settings", settings)
    monkeypatch.setattr("app.providers.bedrock.settings", settings)
    return settings


def test_offline_command_never_constructs_clients(configured, monkeypatch, capsys):
    provider = model._provider()
    for method in ("chat_model", "converse_client", "embed_sync"):
        monkeypatch.setattr(provider, method, Mock(side_effect=AssertionError("must stay offline")))
    assert check_models.main([]) == 0
    output = capsys.readouterr().out
    assert "NOT TESTED" in output
    assert "SKIP embedding: disabled" in output
    assert "credentials and capabilities were not tested live" in output


@pytest.mark.parametrize("args", [["--live"], ["--yes"], ["--role", "invalid"]])
def test_command_requires_explicit_live_consent(configured, args):
    with pytest.raises(SystemExit) as exc:
        check_models.main(args)
    assert exc.value.code == 2


@pytest.mark.parametrize("setting,value", [("MODEL_QUERY", ""), ("MODEL_QUERY", "typo"),
                                          ("LLM_PROVIDER", "not-registered")])
def test_invalid_configuration_exits_nonzero(configured, monkeypatch, setting, value, capsys):
    monkeypatch.setattr(configured, setting, value)
    assert check_models.main([]) == 1
    assert "Configuration failed" in capsys.readouterr().out


def test_shared_auth_not_silently_ignored_by_bedrock(configured, monkeypatch, capsys):
    monkeypatch.setattr(configured, "LLM_BASE_URL", "http://localhost:1234")
    assert check_models.main([]) == 1
    assert "use AWS credentials" in capsys.readouterr().out


def test_plan_deduplicates_model_and_capability(configured):
    plan = check_models.connection_plan(model, model.validate_configuration(), list(model.Role))
    assert len(plan) == 4  # tools, LangChain chat, converse, streaming
    stream = next(roles for (_, check), roles in plan.items() if check == "stream")
    assert stream == [model.Role.QUERY, model.Role.DRAFT_AGENT, model.Role.EDIT]


def test_live_command_limits_roles_and_calls_synthetic_probe(configured, monkeypatch, capsys):
    fake = AsyncMock()
    monkeypatch.setattr(check_models, "probe", fake)
    assert check_models.main(["--live", "--yes", "--role", "query"]) == 0
    assert [c.args[2] for c in fake.call_args_list] == ["converse", "stream"]
    assert "may incur charges" in capsys.readouterr().out


def test_live_failure_is_nonzero_redacted_and_other_checks_continue(configured, monkeypatch, capsys):
    fake = AsyncMock(side_effect=[RuntimeError("Bearer secret-key https://secret-host"), None])
    monkeypatch.setattr(check_models, "probe", fake)
    assert check_models.main(["--live", "--yes", "--role", "query"]) == 1
    output = capsys.readouterr().out
    assert "FAIL" in output and "PASS" in output
    assert "secret-key" not in output and "secret-host" not in output
    assert fake.await_count == 2


def test_probe_failure_reports_actionable_reason(configured, monkeypatch, capsys):
    monkeypatch.setattr(check_models, "probe", AsyncMock(side_effect=check_models.ProbeFailure(
        "Model did not return the requested structured tool call")))
    assert check_models.main(["--live", "--yes", "--role", "ingest_plan"]) == 1
    assert "requested structured tool call" in capsys.readouterr().out


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", ["OK", "", "   "])
async def test_converse_probe_validates_nonempty_response(configured, monkeypatch, reply):
    client = SimpleNamespace(converse=AsyncMock(return_value=reply))
    monkeypatch.setattr(model, "get_converse", Mock(return_value=client))
    if reply.strip():
        await check_models.probe(model, model.Role.QUERY, "converse")
    else:
        with pytest.raises(ValueError):
            await check_models.probe(model, model.Role.QUERY, "converse")
    assert client.converse.call_args.kwargs == {"max_tokens": 64, "operation": ""}


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, "empty", "partial"])
async def test_stream_probe_drains_and_closes_stream(configured, monkeypatch, failure):
    closed = []

    async def stream(*args, **kwargs):
        assert kwargs["operation"] == ""
        try:
            if failure != "empty":
                yield "OK"
            if failure == "partial":
                raise RuntimeError("failed after partial text")
        finally:
            closed.append(True)

    monkeypatch.setattr(model, "get_converse", Mock(return_value=SimpleNamespace(converse_stream=stream)))
    if failure:
        with pytest.raises((ValueError, RuntimeError)):
            await check_models.probe(model, model.Role.QUERY, "stream")
    else:
        await check_models.probe(model, model.Role.QUERY, "stream")
    assert closed == [True]


@pytest.mark.asyncio
@pytest.mark.parametrize("calls", [[], [{"name": "wrong", "args": {}}],
                                  [{"name": "connection_probe", "args": {"value": "ok"}}]])
async def test_tools_probe_requires_actual_structured_call(configured, monkeypatch, calls):
    client = Mock()
    client.bind_tools.return_value = client
    client.ainvoke = AsyncMock(return_value=SimpleNamespace(tool_calls=calls))
    getter = Mock(return_value=client)
    monkeypatch.setattr(model, "get_chat", getter)
    if calls and calls[0]["name"] == "connection_probe":
        await check_models.probe(model, model.Role.INGEST_PLAN, "tools")
    else:
        with pytest.raises(ValueError):
            await check_models.probe(model, model.Role.INGEST_PLAN, "tools")
    assert getter.call_args.kwargs["operation"] == ""
    assert client.bind_tools.call_args.args[0][0]["name"] == "connection_probe"


@pytest.mark.asyncio
@pytest.mark.parametrize("vector", [None, [0.1] * 1024, [0.1] * 1536])
async def test_embedding_probe_checks_actual_vector(configured, monkeypatch, vector):
    monkeypatch.setattr(model, "embed", AsyncMock(return_value=vector))
    if vector and len(vector) == 1536:
        await check_models.probe(model, model.Role.EMBEDDING, "embedding")
    else:
        with pytest.raises(ValueError):
            await check_models.probe(model, model.Role.EMBEDDING, "embedding")


@pytest.mark.asyncio
async def test_untracked_chat_probe_does_not_write_usage_to_db(configured, monkeypatch):
    client = SimpleNamespace(ainvoke=AsyncMock(return_value=SimpleNamespace(
        content=[{"type": "text", "text": "OK"}], usage_metadata={"input_tokens": 4, "output_tokens": 1})))
    monkeypatch.setattr(model._provider(), "chat_model", Mock(return_value=client))
    db_write = AsyncMock(side_effect=AssertionError("no database writes"))
    monkeypatch.setattr("app.services.usage_log.record", db_write)
    await check_models.probe(model, model.Role.RECALIBRATE, "chat")
    db_write.assert_not_called()


def test_command_restores_logging_and_tracing(configured, monkeypatch):
    import logging
    import os
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    previous = logging.root.manager.disable
    assert check_models.main([]) == 0
    assert os.environ["LANGSMITH_TRACING"] == "true"
    assert logging.root.manager.disable == previous


@pytest.mark.parametrize("source", ["environment", "dotenv"])
def test_cli_reports_obsolete_setting_at_import_without_disclosing_values(configured, tmp_path, source):
    import os
    from pathlib import Path
    import subprocess
    import sys

    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    env["LLM_API_KEY"] = "private-current-key"
    if source == "environment":
        env["BEDROCK_QUERY_MODEL_ID"] = "private-obsolete-value"
    else:
        (tmp_path / ".env").write_text("BEDROCK_QUERY_MODEL_ID=private-obsolete-value\n", encoding="utf-8")
    result = subprocess.run([sys.executable, "-m", "app.check_models"], cwd=tmp_path, env=env,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 1
    assert "BEDROCK_QUERY_MODEL_ID" in result.stdout
    assert "MODEL_DEFAULT" in result.stdout
    assert "private-" not in result.stdout + result.stderr
