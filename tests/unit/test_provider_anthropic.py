"""Exercise the real Anthropic and LangChain SDKs against fake HTTP responses."""
import asyncio
import json
from unittest.mock import AsyncMock, Mock

import anthropic as sdk
import httpx2
import pytest
from langchain_anthropic import ChatAnthropic
from pydantic import SecretStr

from app import check_models, model, providers
from app.config import Settings
from app.providers import anthropic as adapter
from app.providers.base import Provider, UnknownModelError


@pytest.fixture
def settings(monkeypatch):
    for field in Settings.model_fields:
        monkeypatch.delenv(field, raising=False)
    configured = Settings(_env_file=None, LLM_PROVIDER="anthropic", LLM_API_KEY="test-key",
                          MODEL_DEFAULT="haiku45", MODEL_EMBEDDING="")
    monkeypatch.setattr(adapter, "settings", configured)
    monkeypatch.setattr(model, "settings", configured)
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    return configured


@pytest.fixture
def accounting(monkeypatch):
    record = AsyncMock()
    monkeypatch.setattr("app.providers.usage.record", record)
    return record


@pytest.fixture
async def transport(monkeypatch):
    clients = []
    requests = []

    def install(handler):
        def handle(request):
            requests.append(request)
            return handler(request)

        def client(**options):
            instance = sdk.AsyncAnthropic(**options, http_client=httpx2.AsyncClient(
                transport=httpx2.MockTransport(handle)))
            clients.append(instance)
            return instance

        monkeypatch.setattr(adapter, "AsyncAnthropic", client)
        monkeypatch.setattr(ChatAnthropic, "_async_client", property(lambda self: client(**self._client_params)))
        return requests, clients

    yield install
    for client in clients:
        await client.close()


def message(content=None):
    return {"id": "msg_test", "type": "message", "role": "assistant", "model": "claude-haiku-4-5-20251001",
            "content": content or [{"type": "text", "text": "OK"}], "stop_reason": "end_turn",
            "stop_sequence": None, "usage": {"input_tokens": 5, "output_tokens": 2,
                "cache_read_input_tokens": 3, "cache_creation_input_tokens": 4}}


def frames(text="OK", complete=True):
    start = message()
    start.update(content=[], stop_reason=None)
    events = [
        {"type": "message_start", "message": start},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}},
    ]
    if complete:
        events.extend([
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
             "usage": {"output_tokens": 7}},
            {"type": "message_stop"},
        ])
    return [f"event: {e['type']}\ndata: {json.dumps(e)}\n\n".encode() for e in events]


class EventStream(httpx2.AsyncByteStream):
    def __init__(self, events, failure=None, wait=False):
        self.events = events
        self.failure = failure
        self.wait = wait
        self.closed = False
        self.waiting = asyncio.Event()

    async def __aiter__(self):
        for event in self.events:
            yield event
        if self.wait:
            self.waiting.set()
            await asyncio.Event().wait()
        if self.failure:
            raise self.failure

    async def aclose(self):
        self.closed = True


def test_registered_configuration_needs_no_aws(settings, monkeypatch):
    monkeypatch.setattr(adapter, "AsyncAnthropic", Mock(side_effect=AssertionError("offline")))
    assert isinstance(providers.get("anthropic"), Provider)
    assert "anthropic" in providers.available()
    assert set(model.validate_configuration()) == set(model.Role) - {model.Role.EMBEDDING}


@pytest.mark.parametrize("name,expected", [("Haiku 4.5", "claude-haiku-4-5-20251001"),
                                         ("sonnet45", "claude-sonnet-4-5-20250929"),
                                         ("opus45", "claude-opus-4-5-20251101"),
                                         ("claude-new-snapshot", "claude-new-snapshot")])
def test_model_resolution(settings, name, expected):
    assert providers.get("anthropic").resolve_model(name) == expected


@pytest.mark.parametrize("name", ["haikuu45", "us.anthropic.claude-haiku-4-5-20251001-v1:0",
                                  "llama4maverick", "titanembedv1", "https://host/claude-test"])
def test_rejects_incompatible_model_ids(settings, name):
    with pytest.raises(UnknownModelError):
        providers.get("anthropic").resolve_model(name)


def test_missing_key_is_actionable_without_ambient_credentials(settings, monkeypatch):
    settings.LLM_API_KEY = SecretStr("")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ambient-secret")
    with pytest.raises(UnknownModelError, match="requires LLM_API_KEY"):
        model.validate_configuration()


def test_embeddings_must_be_explicitly_disabled(settings):
    settings.MODEL_EMBEDDING = "haiku45"
    with pytest.raises(UnknownModelError, match="MODEL_EMBEDDING="):
        model.validate_configuration()
    with pytest.raises(NotImplementedError):
        providers.get("anthropic").embed_sync("anything", "text")


@pytest.mark.asyncio
async def test_converse_wire_format_credentials_and_usage(settings, transport, accounting, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://untrusted.invalid")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "ambient-token-must-not-be-sent")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ambient-key-must-not-be-sent")
    requests, clients = transport(lambda r: httpx2.Response(200, json=message([
        {"type": "text", "text": "first"}, {"type": "text", "text": "second"}])))
    client = model.get_converse(model.Role.QUERY)
    text = await client.converse("system", [{"role": "user", "content": [{"text": "one"}, {"text": "two"}]}],
                                 max_tokens=50, operation="query", temperature=0.2)
    assert text == "firstsecond"
    request = requests[0]
    assert str(request.url) == "https://api.anthropic.com/v1/messages"
    assert request.headers["x-api-key"] == "test-key"
    assert "authorization" not in request.headers
    body = json.loads(request.content)
    assert body["system"] == "system" and body["temperature"] == 0.2 and body["max_tokens"] == 50
    assert body["messages"] == [{"role": "user", "content": [
        {"type": "text", "text": "one"}, {"type": "text", "text": "two"}]}]
    accounting.assert_awaited_once_with(client.model_id, 12, 2, "query")
    assert clients[0].is_closed()


@pytest.mark.asyncio
async def test_custom_endpoint_and_tool_call_through_langchain(settings, transport, accounting):
    settings.LLM_BASE_URL = "https://gateway.example/anthropic"
    reply = message([{"type": "tool_use", "id": "tool_test", "name": "connection_probe", "input": {"value": "ok"}}])
    reply["stop_reason"] = "tool_use"
    requests, _ = transport(lambda r: httpx2.Response(200, json=reply))
    await check_models.probe(model, model.Role.INGEST_PLAN, "tools")
    assert str(requests[0].url) == "https://gateway.example/anthropic/v1/messages"
    body = json.loads(requests[0].content)
    assert body["tools"][0]["input_schema"]["required"] == ["value"]
    assert requests[0].headers["x-api-key"] == "test-key"
    assert accounting.call_args.args[3] == ""  # probe never persists usage


@pytest.mark.asyncio
async def test_normal_langchain_usage_is_tracked(settings, transport, accounting):
    transport(lambda r: httpx2.Response(200, json=message()))
    reply = await model.get_chat(model.Role.RECALIBRATE).ainvoke([{"role": "user", "content": "hello"}])
    assert reply.content == "OK"
    accounting.assert_awaited_once_with(model.model_id_for(model.Role.RECALIBRATE), 12, 2, "recalibrate")


@pytest.mark.asyncio
async def test_streaming_wire_and_usage(settings, transport, accounting):
    upstream = EventStream(frames())
    requests, clients = transport(lambda r: httpx2.Response(200, headers={"content-type": "text/event-stream"}, stream=upstream))
    client = model.get_converse(model.Role.QUERY)
    chunks = [c async for c in client.converse_stream("system", [{"role": "user", "content": [{"text": "hello"}]}],
                                                     operation="chat")]
    assert chunks == ["OK"]
    assert json.loads(requests[0].content)["stream"] is True
    accounting.assert_awaited_once_with(client.model_id, 12, 7, "chat")
    assert upstream.closed and clients[0].is_closed()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["eof", "network", "event"])
async def test_partial_stream_never_becomes_success(settings, transport, accounting, failure, monkeypatch):
    gate = asyncio.Semaphore(1)
    monkeypatch.setattr(adapter, "_semaphore", lambda model_id: gate)
    events = frames(complete=False)
    if failure == "event":
        events.append(b'event: error\ndata: {"type":"error","error":{"type":"overloaded_error","message":"retry"}}\n\n')
    upstream = EventStream(events, httpx2.ReadError("lost connection") if failure == "network" else None)
    requests, clients = transport(lambda r: httpx2.Response(200, headers={"content-type": "text/event-stream"}, stream=upstream))
    chunks = []
    client = model.get_converse(model.Role.QUERY)
    with pytest.raises((RuntimeError, httpx2.ReadError, sdk.APIError)):
        async for chunk in client.converse_stream("system", [{"role": "user", "content": [{"text": "hello"}]}]):
            chunks.append(chunk)
    assert chunks == ["OK"] and len(requests) == 1  # no replay after partial output
    assert upstream.closed and clients[0].is_closed()
    accounting.assert_not_called()
    assert not adapter._semaphore(client.model_id).locked()


@pytest.mark.asyncio
async def test_stream_cancellation_closes_connection(settings, transport, accounting, monkeypatch):
    gate = asyncio.Semaphore(1)
    monkeypatch.setattr(adapter, "_semaphore", lambda model_id: gate)
    upstream = EventStream(frames(complete=False), wait=True)
    _, clients = transport(lambda r: httpx2.Response(200, headers={"content-type": "text/event-stream"}, stream=upstream))
    client = model.get_converse(model.Role.QUERY)

    async def consume():
        return [c async for c in client.converse_stream("system", [{"role": "user", "content": [{"text": "hello"}]}])]

    task = asyncio.create_task(consume())
    await asyncio.wait_for(upstream.waiting.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert upstream.closed and clients[0].is_closed()
    assert not adapter._semaphore(client.model_id).locked()
    accounting.assert_not_called()


@pytest.mark.asyncio
async def test_consumer_closing_stream_releases_resources(settings, transport, accounting, monkeypatch):
    gate = asyncio.Semaphore(1)
    monkeypatch.setattr(adapter, "_semaphore", lambda model_id: gate)
    upstream = EventStream(frames(complete=False), wait=True)
    _, clients = transport(lambda r: httpx2.Response(200, headers={"content-type": "text/event-stream"}, stream=upstream))
    client = model.get_converse(model.Role.QUERY)
    stream = client.converse_stream("system", [{"role": "user", "content": [{"text": "hello"}]}])
    assert await anext(stream) == "OK"
    assert gate.locked()
    await stream.aclose()
    assert not gate.locked() and upstream.closed and clients[0].is_closed()
    accounting.assert_not_called()


@pytest.mark.asyncio
async def test_raw_models_do_not_receive_legacy_sampling(settings, transport, accounting):
    settings.MODEL_QUERY = "claude-new-snapshot"
    requests, _ = transport(lambda r: httpx2.Response(200, json=message()))
    await model.get_converse(model.Role.QUERY).converse("sys", [{"role": "user", "content": [{"text": "hello"}]}])
    assert "temperature" not in json.loads(requests[0].content)
    settings.MODEL_RECALIBRATE = "claude-new-snapshot"
    assert model.get_chat(model.Role.RECALIBRATE)._runnable.temperature is None


@pytest.mark.asyncio
@pytest.mark.parametrize("status,count", [(401, 1), (429, 3), (500, 3)])
async def test_retry_policy_and_cleanup(settings, transport, accounting, status, count):
    requests, clients = transport(lambda r: httpx2.Response(status, headers={"retry-after-ms": "1"}, json={
        "type": "error", "error": {"type": "api_error", "message": "synthetic error"}}))
    with pytest.raises(sdk.APIStatusError):
        await model.get_converse(model.Role.QUERY).converse("sys", [{"role": "user", "content": [{"text": "hi"}]}])
    assert len(requests) == count and clients[0].is_closed()
    accounting.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("messages", [[{"role": "system", "content": [{"text": "hi"}]}],
                                     [{"role": "user", "content": [{"image": {}}]}]])
async def test_unsupported_messages_fail_before_request(settings, transport, accounting, messages):
    requests, _ = transport(lambda r: httpx2.Response(200, json=message()))
    with pytest.raises(ValueError):
        await model.get_converse(model.Role.QUERY).converse("sys", messages)
    assert requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("storage", ["local", "s3"])
async def test_aws_refresh_depends_on_storage_for_direct_provider(settings, monkeypatch, storage):
    from app.services import aws_auth
    settings.ASSUMED_ROLE_ARN = "arn:aws:iam::123456789012:role/unused"
    settings.STORAGE_BACKEND = storage
    monkeypatch.setattr(aws_auth, "settings", settings)
    monkeypatch.setattr(aws_auth, "_refresh_task", None)
    assume = Mock()
    monkeypatch.setattr(aws_auth, "_assume_role", assume)
    try:
        await aws_auth.start_refresh_task()
        assert assume.call_count == int(storage == "s3")
    finally:
        await aws_auth.stop_refresh_task()
