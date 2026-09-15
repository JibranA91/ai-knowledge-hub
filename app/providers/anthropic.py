"""Direct Anthropic Messages API; no AWS credentials or embedding fallback."""
import asyncio
import re
from weakref import WeakKeyDictionary

from anthropic import AsyncAnthropic
from langchain_anthropic import ChatAnthropic

from app.config import settings
from app.logger import get_logger
from app.providers import usage
from app.providers.base import UnknownModelError

log = get_logger(__name__)

# Explicit API snapshots; Bedrock IDs and geography prefixes are not portable.
MODEL_CATALOG = {
    "haiku45": "claude-haiku-4-5-20251001",
    "sonnet45": "claude-sonnet-4-5-20250929",
    "opus45": "claude-opus-4-5-20251101",
}
_semaphores = WeakKeyDictionary()


def _semaphore(model_id):
    per_loop = _semaphores.setdefault(asyncio.get_running_loop(), {})
    if model_id not in per_loop:
        per_loop[model_id] = asyncio.Semaphore(20)
    return per_loop[model_id]


def _client_options():
    key = settings.LLM_API_KEY.get_secret_value().strip()
    if not key:
        raise UnknownModelError("anthropic requires LLM_API_KEY (a direct Anthropic API key, not AWS credentials)")
    # Explicit base URL prevents ambient ANTHROPIC_BASE_URL from redirecting keys.
    return {"api_key": key, "base_url": settings.LLM_BASE_URL or "https://api.anthropic.com",
            "timeout": 120.0, "max_retries": 2}


def _messages(messages):
    translated = []
    for message in messages:
        if message.get("role") not in {"user", "assistant"}:
            raise ValueError("Anthropic conversation messages must be user or assistant; pass system separately")
        blocks = message.get("content")
        if not isinstance(blocks, list) or not blocks:
            raise ValueError("Expected non-empty text content blocks")
        if any(not isinstance(block, dict) or set(block) != {"text"}
               or not isinstance(block["text"], str) for block in blocks):
            raise ValueError("Direct Anthropic conversations currently support text blocks only")
        translated.append({"role": message["role"], "content": [
            {"type": "text", "text": block["text"]} for block in blocks]})
    return translated


def _text(response):
    text = "".join(block.text for block in response.content if block.type == "text")
    if not text.strip():
        raise ValueError("Anthropic returned no text")
    return text


def _sampling(model_id, temperature):
    # SDK v1 moved legacy sampling fields to extra_body. Unknown/new snapshots
    # use provider defaults instead of receiving potentially unsupported fields.
    return {"temperature": temperature} if model_id in MODEL_CATALOG.values() else {}


async def _record(model_id, tokens, operation):
    # Anthropic reports cache reads/writes separately from uncached input.
    total_in = (tokens.input_tokens + (getattr(tokens, "cache_read_input_tokens", 0) or 0)
                + (getattr(tokens, "cache_creation_input_tokens", 0) or 0))
    await usage.record(model_id, total_in, tokens.output_tokens, operation)


class AnthropicConverseClient:
    def __init__(self, model_id):
        self.model_id = model_id

    async def converse(self, system_prompt, messages, max_tokens=4096, operation="", temperature=0.3):
        payload = _messages(messages)
        async with _semaphore(self.model_id):
            async with AsyncAnthropic(**_client_options()) as client:
                response = await client.messages.create(model=self.model_id, system=system_prompt,
                    messages=payload, max_tokens=max_tokens, extra_body=_sampling(self.model_id, temperature))
        await _record(self.model_id, response.usage, operation)
        return _text(response)

    async def converse_stream(self, system_prompt, messages, max_tokens=4096, operation=""):
        payload = _messages(messages)
        async with _semaphore(self.model_id):
            async with AsyncAnthropic(**_client_options()) as client:
                async with client.messages.stream(model=self.model_id, system=system_prompt,
                        messages=payload, max_tokens=max_tokens, extra_body=_sampling(self.model_id, 0.3)) as stream:
                    completed = False
                    async for event in stream:
                        if event.type == "content_block_delta" and event.delta.type == "text_delta":
                            yield event.delta.text
                        elif event.type == "message_stop":
                            completed = True
                    if not completed:
                        raise RuntimeError("Anthropic stream ended before message_stop")
                    response = await stream.get_final_message()
        await _record(self.model_id, response.usage, operation)


class AnthropicProvider:
    name = "anthropic"

    def resolve_model(self, name):
        raw = name.strip()
        key = "".join(c for c in raw.lower() if c.isalnum())
        if key in MODEL_CATALOG:
            return MODEL_CATALOG[key]
        if re.fullmatch(r"claude-[a-z0-9]+(?:-[a-z0-9]+)*", raw):
            return raw
        raise UnknownModelError(
            f"Unknown Anthropic model {raw!r}. Use {', '.join(self.known_models())} or a direct claude-* ID. "
            "Bedrock IDs are not supported; set MODEL_EMBEDDING= to disable embeddings.")

    def known_models(self):
        return sorted(MODEL_CATALOG)

    def validate_model(self, model_id, *, embedding, dimensions):
        if embedding:
            raise UnknownModelError("Anthropic has no embedding API; set MODEL_EMBEDDING= (keyword search)")
        _client_options()  # Local validation only; never constructs an SDK client.
        if model_id not in MODEL_CATALOG.values():
            log.warning("Anthropic raw model capabilities are unverified locally: %s", model_id)

    def chat_model(self, model_id, max_tokens=4096):
        return ChatAnthropic(model=model_id, max_tokens=max_tokens,
                             **_sampling(model_id, 0.3), **_client_options())

    def converse_client(self, model_id):
        return AnthropicConverseClient(model_id)

    def embed_sync(self, model_id, text):
        raise NotImplementedError("Anthropic does not provide embeddings; set MODEL_EMBEDDING=")
