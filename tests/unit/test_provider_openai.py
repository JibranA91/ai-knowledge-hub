"""Run the official SDKs against fake HTTP, never a paid model endpoint."""
import asyncio
import json
from unittest.mock import AsyncMock, Mock

import httpx
import openai as sdk
import pytest
from langchain_core.messages import HumanMessage, ToolMessage
from pydantic import SecretStr

from app import check_models, model, providers
from app.config import Settings
from app.providers import openai as adapter
from app.providers.base import Provider, UnknownModelError


@pytest.fixture
def settings(monkeypatch):
    for field in Settings.model_fields:
        monkeypatch.delenv(field, raising=False)
    configured = Settings(_env_file=None, LLM_PROVIDER="openai", LLM_API_KEY="test-key",
                          MODEL_DEFAULT="gpt41mini", MODEL_EMBEDDING="embed3small")
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
def transport(monkeypatch):
    def install(handler):
        requests, clients = [], []

        def handle(request):
            requests.append(request)
            return handler(request)

        def async_client(**options):
            client = sdk.AsyncOpenAI(**options, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)))
            clients.append(client)
            return client

        def sync_client(**options):
            client = sdk.OpenAI(**options, http_client=httpx.Client(transport=httpx.MockTransport(handle)))
            clients.append(client)
            return client

        monkeypatch.setattr(adapter, "AsyncOpenAI", async_client)
        monkeypatch.setattr(adapter, "OpenAI", sync_client)
        return requests, clients
    return install


def response(status="completed", output=None):
    return {"id": "resp_test", "object": "response", "created_at": 1, "model": "gpt-4.1-mini-2025-04-14",
            "status": status, "error": None, "incomplete_details": None,
            "output": output if output is not None else [{"type": "message", "id": "msg_test",
                "status": "completed", "role": "assistant", "content": [
                    {"type": "output_text", "text": "OK", "annotations": []}]}],
            "usage": {"input_tokens": 5, "output_tokens": 7, "total_tokens": 12,
                      "input_tokens_details": {"cached_tokens": 3},
                      "output_tokens_details": {"reasoning_tokens": 2}},
            "parallel_tool_calls": True, "tools": [], "tool_choice": "auto"}


def frames(terminal="completed"):
    events = [{"type": "response.output_text.delta", "sequence_number": 0,
               "item_id": "msg_test", "output_index": 0, "content_index": 0, "delta": "OK"}]
    if terminal:
        events.append({"type": "response." + terminal, "sequence_number": 1, "response": response(terminal)})
    return [f"event: {e['type']}\ndata: {json.dumps(e)}\n\n".encode() for e in events]


class EventStream(httpx.AsyncByteStream):
    def __init__(self, events, failure=None, wait=False):
        self.events, self.failure, self.wait = events, failure, wait
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


MESSAGES = [{"role": "user", "content": [{"text": "hello"}]}]


def test_offline_validation_and_registration(settings, monkeypatch):
    monkeypatch.setattr(adapter, "AsyncOpenAI", Mock(side_effect=AssertionError("offline")))
    monkeypatch.setattr(adapter, "OpenAI", Mock(side_effect=AssertionError("offline")))
    assert isinstance(providers.get("openai"), Provider)
    assert set(model.validate_configuration()) == set(model.Role)
    settings.MODEL_EMBEDDING = ""
    assert model.Role.EMBEDDING not in model.validate_configuration()


@pytest.mark.parametrize("name,expected", [("GPT 4.1 Mini", "gpt-4.1-mini-2025-04-14"),
    ("gpt41", "gpt-4.1-2025-04-14"), ("embed3large", "text-embedding-3-large"),
    ("gpt-5-new-snapshot", "gpt-5-new-snapshot"), ("o3", "o3")])
def test_resolves_models(settings, name, expected):
    assert providers.get("openai").resolve_model(name) == expected


@pytest.mark.parametrize("name", ["haiku45", "us.anthropic.claude-model", "titanembedv1", "https://host/gpt-test"])
def test_rejects_other_provider_ids(settings, name):
    with pytest.raises(UnknownModelError):
        providers.get("openai").resolve_model(name)


@pytest.mark.parametrize("field,value", [("MODEL_EMBEDDING", "gpt41mini"),
    ("MODEL_QUERY", "embed3small"), ("EMBEDDING_DIMENSIONS", 3072)])
def test_capability_mismatches(settings, field, value):
    setattr(settings, field, value)
    with pytest.raises(UnknownModelError):
        model.validate_configuration()


def test_missing_key_does_not_use_ambient_key(settings, monkeypatch):
    settings.LLM_API_KEY = SecretStr("")
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-key")
    with pytest.raises(UnknownModelError, match="requires LLM_API_KEY"):
        model.validate_configuration()


@pytest.mark.asyncio
async def test_converse_payload_credentials_and_usage(settings, transport, accounting, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://untrusted.invalid")
    monkeypatch.setenv("OPENAI_ORG_ID", "ambient-org")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "ambient-project")
    monkeypatch.setenv("OPENAI_CUSTOM_HEADERS", "Authorization: Bearer ambient\nOpenAI-Project: ambient")
    requests, clients = transport(lambda r: httpx.Response(200, json=response()))
    client = model.get_converse(model.Role.QUERY)
    text = await client.converse("system", [*MESSAGES, {"role": "assistant", "content": [{"text": "one"}, {"text": "two"}]}],
                                 max_tokens=50, operation="query", temperature=0.2)
    assert text == "OK"
    request = requests[0]
    assert str(request.url) == "https://api.openai.com/v1/responses"
    assert request.headers["authorization"] == "Bearer test-key"
    assert request.headers.get("OpenAI-Organization", "") == ""
    assert request.headers.get("OpenAI-Project", "") == ""
    body = json.loads(request.content)
    assert body["input"][1] == {"role": "assistant", "content": "onetwo"}
    assert body["instructions"] == "system" and body["max_output_tokens"] == 50
    assert body["temperature"] == 0.2 and body["store"] is False
    accounting.assert_awaited_once_with(client.model_id, 5, 7, "query")
    assert all(c.is_closed() for c in clients)


@pytest.mark.asyncio
@pytest.mark.parametrize("reasoning", [False, True])
async def test_langchain_tools_round_trip_and_usage(settings, transport, accounting, reasoning):
    settings.LLM_BASE_URL = "https://gateway.example/v1"
    reply = response(output=[{"type": "function_call", "id": "fc_test", "call_id": "call_test",
        "name": "connection_probe", "arguments": '{"value":"ok"}', "status": "completed"}])
    if reasoning:
        reply["output"].insert(0, {"type": "reasoning", "id": "rs_test", "summary": [],
                                   "encrypted_content": "opaque-test-value"})
    requests, clients = transport(lambda r: httpx.Response(200, json=reply if len(requests) == 1 else response()))
    llm = model.get_chat(model.Role.INGEST_PLAN)
    tool = {"name": "connection_probe", "description": "Test", "parameters": {
        "type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]}}
    bound = llm.bind_tools([tool])
    first = await bound.ainvoke([HumanMessage(content="probe")])
    assert first.tool_calls[0]["args"] == {"value": "ok"}
    second = await bound.ainvoke([HumanMessage(content="probe"), first,
                                  ToolMessage(content="found", tool_call_id="call_test")])
    assert second.content == "OK"
    body = json.loads(requests[1].content)
    assert body["store"] is False and "previous_response_id" not in body
    assert body["include"] == ["reasoning.encrypted_content"]
    assert any(i.get("type") == "function_call_output" and i["call_id"] == "call_test" for i in body["input"])
    if reasoning:
        assert any(i.get("encrypted_content") == "opaque-test-value" for i in body["input"])
    assert body["tools"][0]["name"] == "connection_probe"
    assert str(requests[0].url) == "https://gateway.example/v1/responses"
    assert all(c.is_closed() for c in clients)
    assert accounting.await_count == 2
    accounting.assert_awaited_with(model.model_id_for(model.Role.INGEST_PLAN), 5, 7, "ingest_plan")


@pytest.mark.asyncio
async def test_diagnostic_probe_does_not_persist_usage(settings, transport, accounting):
    transport(lambda r: httpx.Response(200, json=response()))
    await check_models.probe(model, model.Role.INGEST_WRITE, "chat")
    assert accounting.call_args.args[3] == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["incomplete", "failed"])
async def test_noncompleted_responses_rejected_by_both_clients(settings, transport, accounting, status):
    _, clients = transport(lambda r: httpx.Response(200, json=response(status)))
    with pytest.raises(RuntimeError, match="not completed"):
        await model.get_converse(model.Role.QUERY).converse("sys", MESSAGES)
    with pytest.raises(RuntimeError, match="not completed"):
        await model.get_chat(model.Role.INGEST_WRITE).ainvoke([HumanMessage(content="hello")])
    assert all(c.is_closed() for c in clients)


@pytest.mark.asyncio
async def test_stream_success_closes_and_records_tokens(settings, transport, accounting):
    upstream = EventStream(frames())
    requests, clients = transport(lambda r: httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=upstream))
    client = model.get_converse(model.Role.QUERY)
    assert [c async for c in client.converse_stream("sys", MESSAGES, operation="chat")] == ["OK"]
    assert json.loads(requests[0].content)["stream"] is True
    accounting.assert_awaited_once_with(client.model_id, 5, 7, "chat")
    assert upstream.closed and all(c.is_closed() for c in clients)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["eof", "network", "incomplete", "failed", "error"])
async def test_stream_failure_never_replays_partial_output(settings, transport, accounting, monkeypatch, failure):
    gate = asyncio.Semaphore(1)
    monkeypatch.setattr(adapter, "_semaphore", lambda model_id: gate)
    events = frames(failure if failure in {"incomplete", "failed"} else None)
    if failure == "error":
        events.append(b'event: error\ndata: {"type":"error","code":"server_error","message":"test"}\n\n')
    upstream = EventStream(events, httpx.ReadError("test") if failure == "network" else None)
    requests, clients = transport(lambda r: httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=upstream))
    chunks = []
    with pytest.raises((RuntimeError, httpx.ReadError, sdk.APIError)):
        async for chunk in model.get_converse(model.Role.QUERY).converse_stream("sys", MESSAGES):
            chunks.append(chunk)
    assert chunks == ["OK"] and len(requests) == 1
    assert not gate.locked() and upstream.closed and all(c.is_closed() for c in clients)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [True, False])
async def test_cancel_or_close_releases_stream_and_capacity(settings, transport, accounting, monkeypatch, cancel):
    gate = asyncio.Semaphore(1)
    monkeypatch.setattr(adapter, "_semaphore", lambda model_id: gate)
    upstream = EventStream(frames(None), wait=True)
    _, clients = transport(lambda r: httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=upstream))
    stream = model.get_converse(model.Role.QUERY).converse_stream("sys", MESSAGES)
    assert await anext(stream) == "OK"
    if cancel:
        task = asyncio.create_task(anext(stream))
        await asyncio.wait_for(upstream.waiting.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        await stream.aclose()
    assert not gate.locked() and upstream.closed and all(c.is_closed() for c in clients)
    accounting.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("status,count", [(401, 1), (429, 3), (500, 3)])
async def test_retry_policy(settings, transport, accounting, status, count):
    requests, clients = transport(lambda r: httpx.Response(status, headers={"retry-after-ms": "1"},
        json={"error": {"message": "synthetic", "type": "api_error", "code": "test"}}))
    with pytest.raises(sdk.APIStatusError):
        await model.get_converse(model.Role.QUERY).converse("sys", MESSAGES)
    assert len(requests) == count and all(c.is_closed() for c in clients)


@pytest.mark.asyncio
async def test_raw_reasoning_model_omits_sampling(settings, transport, accounting):
    settings.MODEL_QUERY = "o3"
    settings.MODEL_INGEST_WRITE = "o3"
    requests, _ = transport(lambda r: httpx.Response(200, json=response()))
    await model.get_converse(model.Role.QUERY).converse("sys", MESSAGES)
    await model.get_chat(model.Role.INGEST_WRITE).ainvoke([HumanMessage(content="hi")])
    assert all("temperature" not in json.loads(r.content) for r in requests)


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["embed3small", "embed3large"])
async def test_embeddings_dimensions_and_identity(settings, transport, name):
    settings.MODEL_EMBEDDING = name
    requests, clients = transport(lambda r: httpx.Response(200, json={"object": "list", "model": "test",
        "data": [{"object": "embedding", "index": 0, "embedding": [0.1] * 1536}],
        "usage": {"prompt_tokens": 2, "total_tokens": 2}}))
    assert await model.embed("hello") == [0.1] * 1536
    body = json.loads(requests[0].content)
    assert body["dimensions"] == 1536 and body["encoding_format"] == "float"
    assert str(requests[0].url).endswith("/embeddings")
    assert json.loads(model.embedding_identity()) == ["openai", body["model"], 1536]
    assert all(c.is_closed() for c in clients)


@pytest.mark.asyncio
@pytest.mark.parametrize("vector", [[0.1] * 3072, [0.0] * 1536, []])
async def test_bad_embedding_falls_back(settings, transport, vector):
    transport(lambda r: httpx.Response(200, json={"object": "list", "model": "test",
        "data": [{"object": "embedding", "index": 0, "embedding": vector}],
        "usage": {"prompt_tokens": 2, "total_tokens": 2}}))
    assert await model.embed("hello") is None


@pytest.mark.asyncio
async def test_invalid_content_rejected_before_network(settings, transport):
    requests, _ = transport(lambda r: httpx.Response(200, json=response()))
    with pytest.raises(ValueError, match="text block"):
        await model.get_converse(model.Role.QUERY).converse("sys", [{"role": "user", "content": [{"image": {}}]}])
    assert requests == []


@pytest.mark.asyncio
async def test_empty_output_does_not_become_success(settings, transport, accounting):
    _, clients = transport(lambda r: httpx.Response(200, json=response(output=[])))
    with pytest.raises(ValueError, match="no text"):
        await model.get_converse(model.Role.QUERY).converse("sys", MESSAGES)
    with pytest.raises(ValueError, match="no text"):
        await model.get_chat(model.Role.INGEST_WRITE).ainvoke([HumanMessage(content="hi")])
    assert all(c.is_closed() for c in clients)


@pytest.mark.asyncio
async def test_embedding_failure_falls_back_and_disabled_never_connects(settings, transport):
    requests, clients = transport(lambda r: httpx.Response(401, json={
        "error": {"message": "synthetic", "type": "api_error", "code": "test"}}))
    assert await model.embed("hello") is None
    assert len(requests) == 1 and all(c.is_closed() for c in clients)
    settings.MODEL_EMBEDDING = ""
    assert await model.embed("hello") is None
    assert len(requests) == 1
