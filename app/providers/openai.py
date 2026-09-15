"""Direct Responses and embeddings APIs, with stateless agent tool calling."""
import asyncio
import re
from dataclasses import dataclass, replace
from weakref import WeakKeyDictionary

from langchain_core.messages import AIMessage, convert_to_messages
from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI, OpenAI, omit

from app.config import settings
from app.logger import get_logger
from app.providers import usage
from app.providers.base import UnknownModelError

log = get_logger(__name__)
MODEL_CATALOG = {
    "gpt41mini": "gpt-4.1-mini-2025-04-14",
    "gpt41": "gpt-4.1-2025-04-14",
    "embed3small": "text-embedding-3-small",
    "embed3large": "text-embedding-3-large",
}
EMBEDDING_MODELS = {"text-embedding-3-small", "text-embedding-3-large"}
_semaphores = WeakKeyDictionary()


def _semaphore(model_id):
    per_loop = _semaphores.setdefault(asyncio.get_running_loop(), {})
    if model_id not in per_loop:
        per_loop[model_id] = asyncio.Semaphore(20)
    return per_loop[model_id]


def _client_options():
    key = settings.LLM_API_KEY.get_secret_value().strip()
    if not key:
        raise UnknownModelError("openai requires LLM_API_KEY (an API key, not a ChatGPT subscription)")
    return {"api_key": key, "base_url": settings.LLM_BASE_URL or "https://api.openai.com/v1",
            "organization": "", "project": "", "admin_api_key": "",
            "default_headers": {"Authorization": f"Bearer {key}",
                                "OpenAI-Organization": omit, "OpenAI-Project": omit},
            "timeout": 120.0, "max_retries": 2}


def _messages(messages):
    translated = []
    for message in messages:
        if message.get("role") not in {"user", "assistant"}:
            raise ValueError("Conversation messages must be user or assistant; pass system separately")
        blocks = message.get("content")
        if not isinstance(blocks, list) or not blocks or any(
            not isinstance(block, dict) or set(block) != {"text"}
            or not isinstance(block["text"], str) for block in blocks
        ):
            raise ValueError("Direct conversations currently support non-empty text block lists only")
        translated.append({"role": message["role"], "content": "".join(b["text"] for b in blocks)})
    return translated


def _sampling(model_id, temperature):
    # Reasoning/new raw models may reject temperature; use their API defaults.
    return {"temperature": temperature} if model_id in {"gpt-4.1-mini-2025-04-14", "gpt-4.1-2025-04-14"} else {}


def _require_completed(status):
    if status != "completed":
        raise RuntimeError("Model response was not completed; partial output was not accepted")


async def _record(model_id, response, operation):
    if response.usage:
        # Input/output totals already include cached/reasoning tokens.
        await usage.record(model_id, response.usage.input_tokens, response.usage.output_tokens, operation)


class OpenAIConverseClient:
    def __init__(self, model_id):
        self.model_id = model_id

    async def converse(self, system_prompt, messages, max_tokens=4096, operation="", temperature=0.3):
        payload = _messages(messages)
        async with _semaphore(self.model_id):
            async with AsyncOpenAI(**_client_options()) as client:
                response = await client.responses.create(model=self.model_id, instructions=system_prompt,
                    input=payload, max_output_tokens=max_tokens, store=False,
                    **_sampling(self.model_id, temperature))
        await _record(self.model_id, response, operation)
        _require_completed(response.status)
        if not response.output_text.strip():
            raise ValueError("Model returned no text")
        return response.output_text

    async def converse_stream(self, system_prompt, messages, max_tokens=4096, operation=""):
        payload = _messages(messages)
        async with _semaphore(self.model_id):
            async with AsyncOpenAI(**_client_options()) as client:
                stream = await client.responses.create(model=self.model_id, instructions=system_prompt,
                    input=payload, max_output_tokens=max_tokens, store=False, stream=True,
                    **_sampling(self.model_id, 0.3))
                completed = False
                has_text = False
                async with stream:
                    async for event in stream:
                        if event.type == "response.output_text.delta":
                            has_text = has_text or bool(event.delta.strip())
                            yield event.delta
                        elif event.type in {"response.completed", "response.incomplete", "response.failed"}:
                            await _record(self.model_id, event.response, operation)
                            _require_completed(event.response.status)
                            completed = True
                        elif event.type == "error":
                            raise RuntimeError("Model stream returned an error")
                if not completed or not has_text:
                    raise RuntimeError("Model stream ended without completed text output")


@dataclass(frozen=True)
class OpenAIChat:
    """Keep LangChain tool conversion while owning each call's SDK lifetime."""
    model_id: str
    max_tokens: int
    tools: list | None = None
    tool_options: dict | None = None

    def bind_tools(self, tools, **kwargs):
        return replace(self, tools=list(tools), tool_options=kwargs)

    async def ainvoke(self, messages, **kwargs):
        options = _client_options()
        replay = []
        for message in convert_to_messages(messages):
            if isinstance(message, AIMessage) and "_response_blocks" in message.additional_kwargs:
                metadata = dict(message.additional_kwargs)
                message = message.model_copy(update={"content": metadata.pop("_response_blocks"),
                                                     "additional_kwargs": metadata})
            replay.append(message)
        async with _semaphore(self.model_id):
            with OpenAI(**options) as sync_client:
                async with AsyncOpenAI(**options) as async_client:
                    llm = ChatOpenAI(model=self.model_id, api_key=options["api_key"],
                        base_url=options["base_url"], organization="", openai_proxy="",
                        client=sync_client.chat.completions, root_client=sync_client,
                        async_client=async_client.chat.completions, root_async_client=async_client,
                        use_responses_api=True, use_previous_response_id=False, store=False,
                        include=["reasoning.encrypted_content"],
                        output_version="responses/v1", max_tokens=self.max_tokens,
                        **_sampling(self.model_id, 0.3))
                    if self.tools is not None:
                        llm = llm.bind_tools(self.tools, **(self.tool_options or {}))
                    response = await llm.ainvoke(replay, **kwargs)
        _require_completed(response.response_metadata.get("status"))
        # Existing ingest/recalibration agents consume strings. Preserve tool
        # calls and replay metadata, but normalize Responses text blocks here.
        text = response.content if isinstance(response.content, str) else "".join(
            block["text"] for block in response.content
            if isinstance(block, dict) and block.get("type") == "text")
        if not text.strip() and not response.tool_calls:
            raise ValueError("Model returned no text or tool calls")
        return response.model_copy(update={"content": text, "additional_kwargs": {
            **response.additional_kwargs, "_response_blocks": response.content}})


class OpenAIProvider:
    name = "openai"

    def resolve_model(self, name):
        raw = name.strip()
        key = "".join(c for c in raw.lower() if c.isalnum())
        if key in MODEL_CATALOG:
            return MODEL_CATALOG[key]
        if raw in EMBEDDING_MODELS or re.fullmatch(r"(?:gpt-[a-z0-9][a-z0-9.-]*|o[134](?:-[a-z0-9.-]+)?)", raw):
            return raw
        raise UnknownModelError(f"Unknown OpenAI model {raw!r}. Use {', '.join(self.known_models())} "
                                "or a direct gpt-*/o-series text ID; Bedrock IDs are not supported.")

    def known_models(self):
        return sorted(MODEL_CATALOG)

    def validate_model(self, model_id, *, embedding, dimensions):
        _client_options()
        if embedding:
            if model_id not in EMBEDDING_MODELS or dimensions != 1536:
                raise UnknownModelError("Embeddings require text-embedding-3-small/large with EMBEDDING_DIMENSIONS=1536")
        elif model_id in EMBEDDING_MODELS:
            raise UnknownModelError("An embedding model cannot serve a text role")
        elif model_id not in MODEL_CATALOG.values():
            log.warning("OpenAI raw model capabilities are unverified locally: %s", model_id)

    def chat_model(self, model_id, max_tokens=4096):
        return OpenAIChat(model_id, max_tokens)

    def converse_client(self, model_id):
        return OpenAIConverseClient(model_id)

    def embed_sync(self, model_id, text):
        self.validate_model(model_id, embedding=True, dimensions=settings.EMBEDDING_DIMENSIONS)
        with OpenAI(**_client_options()) as client:
            response = client.embeddings.create(model=model_id, input=text, dimensions=1536, encoding_format="float")
        if len(response.data) != 1 or response.data[0].index != 0:
            raise ValueError("Expected one embedding for one input")
        return response.data[0].embedding
