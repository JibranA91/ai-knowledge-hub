"""Unit tests for app/providers/bedrock.py.

boto3 calls are fully mocked — no AWS credentials required.
"""
import asyncio
import pytest
from unittest.mock import MagicMock, patch


@pytest.mark.parametrize("kind", ["chat", "converse", "embedding"])
@pytest.mark.parametrize("explicit_credentials", [False, True])
def test_clients_use_only_aws_connection_settings(monkeypatch, kind, explicit_credentials):
    from app import model
    from app.config import Settings
    from app.providers import bedrock

    for name in Settings.model_fields:
        monkeypatch.delenv(name, raising=False)
    settings = Settings(_env_file=None, LLM_PROVIDER="bedrock", LLM_API_KEY="unused-test-key",
                        LLM_BASE_URL="https://unused.example", AWS_REGION="us-west-2",
                        MODEL_EMBEDDING="titanembedv1")
    monkeypatch.setattr(model, "settings", settings)
    monkeypatch.setattr(bedrock, "settings", settings)
    credentials = ({"aws_access_key_id": "test-aws-id", "aws_secret_access_key": "test-aws-secret",
                    "aws_session_token": "test-session-token"} if explicit_credentials else {})
    monkeypatch.setattr(bedrock, "get_credentials", lambda: credentials.copy())
    constructor = MagicMock()
    monkeypatch.setattr(bedrock, "ChatBedrockConverse", constructor)
    monkeypatch.setattr(bedrock.boto3, "client", constructor)
    constructor.return_value.invoke_model.return_value = {
        "body": MagicMock(read=lambda: b'{"embedding": [0.1]}')}

    resolved = model.validate_configuration()
    assert model.provider_name() == "bedrock"
    if kind == "chat":
        model.get_chat(model.Role.INGEST_WRITE)
    elif kind == "converse":
        model.get_converse(model.Role.QUERY)
    else:
        model._provider().embed_sync(resolved[model.Role.EMBEDDING], "test")

    constructor.assert_called_once()
    kwargs = constructor.call_args.kwargs
    assert kwargs["region_name"] == "us-west-2"
    for name in ("aws_access_key_id", "aws_secret_access_key", "aws_session_token"):
        if explicit_credentials:
            assert kwargs[name] == credentials[name]
        else:
            assert name not in kwargs
    assert not {"api_key", "base_url", "endpoint_url"} & kwargs.keys()
    assert "unused-test-key" not in repr(constructor.call_args)
    assert "unused.example" not in repr(constructor.call_args)


def _make_service(mock_client=None):
    """Build a BedrockConverseClient without real boto3 initialisation."""
    from app.providers.bedrock import BedrockConverseClient, _semaphores
    service = BedrockConverseClient.__new__(BedrockConverseClient)
    service.model_id = "test-model"
    service._botocore_config = MagicMock()
    service.client = mock_client or MagicMock()
    _semaphores.setdefault("test-model", asyncio.Semaphore(20))
    return service


def _bedrock_response(text: str) -> dict:
    return {"output": {"message": {"content": [{"text": text}]}}}


# ── converse ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_converse_returns_text():
    mock_client = MagicMock()
    mock_client.converse.return_value = _bedrock_response("Hello, world!")
    service = _make_service(mock_client)

    with patch("app.providers.bedrock.settings") as mock_settings:
        mock_settings.ASSUMED_ROLE_ARN = ""
        result = await service.converse("system", [{"role": "user", "content": [{"text": "hi"}]}])

    assert result == "Hello, world!"


@pytest.mark.asyncio
async def test_converse_calls_bedrock_with_model_id():
    mock_client = MagicMock()
    mock_client.converse.return_value = _bedrock_response("ok")
    service = _make_service(mock_client)

    with patch("app.providers.bedrock.settings") as mock_settings:
        mock_settings.ASSUMED_ROLE_ARN = ""
        await service.converse("sys", [], max_tokens=512)

    call_kwargs = mock_client.converse.call_args[1]
    assert call_kwargs["modelId"] == "test-model"
    assert call_kwargs["inferenceConfig"]["maxTokens"] == 512


@pytest.mark.asyncio
async def test_converse_uses_system_prompt():
    mock_client = MagicMock()
    mock_client.converse.return_value = _bedrock_response("ok")
    service = _make_service(mock_client)

    with patch("app.providers.bedrock.settings") as mock_settings:
        mock_settings.ASSUMED_ROLE_ARN = ""
        await service.converse("my system prompt", [])

    call_kwargs = mock_client.converse.call_args[1]
    assert call_kwargs["system"][0]["text"] == "my system prompt"


@pytest.mark.asyncio
async def test_converse_recreates_client_when_role_assumed():
    """When ASSUMED_ROLE_ARN is set, client should be recreated on each call."""
    mock_client = MagicMock()
    mock_client.converse.return_value = _bedrock_response("ok")
    service = _make_service(mock_client)

    new_client = MagicMock()
    new_client.converse.return_value = _bedrock_response("from-new-client")

    with patch("app.providers.bedrock.settings") as mock_settings, \
         patch.object(service, "_make_client", return_value=new_client) as mock_make:
        mock_settings.ASSUMED_ROLE_ARN = "arn:aws:iam::123:role/Test"
        result = await service.converse("sys", [])

    mock_make.assert_called_once()
    assert result == "from-new-client"


# ── converse_stream ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_converse_stream_yields_chunks():
    events = [
        {"contentBlockDelta": {"delta": {"text": "Hello"}}},
        {"contentBlockDelta": {"delta": {"text": " world"}}},
    ]
    mock_client = MagicMock()
    mock_client.converse_stream.return_value = {"stream": iter(events)}
    service = _make_service(mock_client)

    with patch("app.providers.bedrock.settings") as mock_settings:
        mock_settings.ASSUMED_ROLE_ARN = ""
        chunks = []
        async for chunk in service.converse_stream("sys", []):
            chunks.append(chunk)

    assert chunks == ["Hello", " world"]


@pytest.mark.asyncio
async def test_converse_stream_empty_response():
    mock_client = MagicMock()
    mock_client.converse_stream.return_value = {"stream": iter([])}
    service = _make_service(mock_client)

    with patch("app.providers.bedrock.settings") as mock_settings:
        mock_settings.ASSUMED_ROLE_ARN = ""
        chunks = []
        async for chunk in service.converse_stream("sys", []):
            chunks.append(chunk)

    assert chunks == []


@pytest.mark.asyncio
async def test_converse_stream_skips_non_text_events():
    events = [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockDelta": {"delta": {"text": "text"}}},
        {"contentBlockStop": {}},
        {"messageStop": {"stopReason": "end_turn"}},
    ]
    mock_client = MagicMock()
    mock_client.converse_stream.return_value = {"stream": iter(events)}
    service = _make_service(mock_client)

    with patch("app.providers.bedrock.settings") as mock_settings:
        mock_settings.ASSUMED_ROLE_ARN = ""
        chunks = []
        async for chunk in service.converse_stream("sys", []):
            chunks.append(chunk)

    assert chunks == ["text"]


@pytest.mark.asyncio
async def test_converse_stream_multiple_chunks_assembled():
    events = [
        {"contentBlockDelta": {"delta": {"text": "A"}}},
        {"contentBlockDelta": {"delta": {"text": "B"}}},
        {"contentBlockDelta": {"delta": {"text": "C"}}},
    ]
    mock_client = MagicMock()
    mock_client.converse_stream.return_value = {"stream": iter(events)}
    service = _make_service(mock_client)

    with patch("app.providers.bedrock.settings") as mock_settings:
        mock_settings.ASSUMED_ROLE_ARN = ""
        chunks = []
        async for chunk in service.converse_stream("sys", []):
            chunks.append(chunk)

    assert "".join(chunks) == "ABC"


@pytest.mark.asyncio
async def test_converse_stream_stops_worker_when_consumer_closes_early():
    """B3: when the consumer closes the stream early (SSE client disconnect), the
    worker thread must stop pulling from Bedrock — not keep running until the read
    timeout, leaking a thread-pool slot. aclose() signals stop and awaits the
    worker, so its consumed-count is frozen by the time aclose() returns."""
    import time
    consumed = {"n": 0}

    def gen():
        for i in range(50):
            consumed["n"] += 1
            time.sleep(0.02)  # slow the worker so 'stop' is observable
            yield {"contentBlockDelta": {"delta": {"text": f"c{i}"}}}

    mock_client = MagicMock()
    mock_client.converse_stream.return_value = {"stream": gen()}
    service = _make_service(mock_client)

    with patch("app.providers.bedrock.settings") as mock_settings:
        mock_settings.ASSUMED_ROLE_ARN = ""
        agen = service.converse_stream("sys", [])
        first = await agen.__anext__()   # consume one chunk
        await agen.aclose()              # simulate consumer disconnect

    assert first == "c0"
    n1 = consumed["n"]
    await asyncio.sleep(0.2)             # ~10 more would be consumed if not stopped
    assert consumed["n"] == n1           # worker stopped + was reclaimed by aclose
    assert n1 < 50                       # did not run the stream to completion


# -- embedding payload shaping --------------------------------------------

def test_titan_embed_request_body():
    import json
    from app.providers.bedrock import _build_request_body
    body = json.loads(_build_request_body("amazon.titan-embed-text-v2:0", "hello"))
    assert body["inputText"] == "hello"


def test_cohere_embed_request_body():
    import json
    from app.providers.bedrock import _build_request_body
    body = json.loads(_build_request_body("cohere.embed-english-v3", "hello"))
    assert body["texts"] == ["hello"]


def test_titan_parse_response():
    import json
    from app.providers.bedrock import _parse_response_body
    vec = [0.1, 0.2, 0.3]
    body = json.dumps({"embedding": vec}).encode()
    assert _parse_response_body("amazon.titan-embed-text-v2:0", body) == vec


def test_cohere_parse_response():
    import json
    from app.providers.bedrock import _parse_response_body
    vec = [0.4, 0.5, 0.6]
    body = json.dumps({"embeddings": [vec]}).encode()
    assert _parse_response_body("cohere.embed-english-v3", body) == vec
