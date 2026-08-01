"""Amazon Bedrock provider.

This module — together with the rest of `app/providers/` — is the ONLY place in
the codebase allowed to construct a `bedrock-runtime` client or a
`ChatBedrockConverse` model. Everything else imports `app.model`.
`tests/unit/test_no_direct_llm_clients.py` enforces that.
"""
import asyncio
import json
import logging
import threading

import boto3
from botocore.config import Config as BotocoreConfig
from botocore.exceptions import ClientError
from langchain_aws import ChatBedrockConverse
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential, before_sleep_log

from app.config import settings
from app.logger import get_logger
from app.providers import usage
from app.providers.base import UnknownModelError
from app.services.aws_auth import get_credentials

log = get_logger(__name__)

# One semaphore per model_id — caps concurrent Bedrock calls across the whole process.
# Initialised lazily so the event loop exists when first accessed.
_semaphores: dict[str, asyncio.Semaphore] = {}


# ── Model name catalogue ───────────────────────────────────────────────────
# Friendly name → the Bedrock model ID it resolves to. Config carries names
# ("haiku45"), never IDs, so retargeting a role is a one-word change and the
# vendor-specific ID lives here.
#
# Two shapes of ID, because Bedrock has two:
#   • Text/chat models are reached through a *cross-region inference profile*,
#     whose ID is the model ID with a geo prefix — "us.", "eu.", "global.".
#     BEDROCK_INFERENCE_GEO picks the prefix; `geos` records which ones AWS
#     actually publishes for that model, so a bad combination fails with a
#     clear message instead of a Bedrock 400.
#   • Embedding models are plain foundation models with no profile and no
#     prefix (`geos=None`).
#
# Verified against ListFoundationModels + ListInferenceProfiles (us-east-1).
# Availability is per-account: a model your account hasn't been granted still
# resolves here and fails at call time with Bedrock's own AccessDenied.

_US = frozenset({"us"})
_US_GLOBAL = frozenset({"us", "global"})


class _Entry:
    __slots__ = ("model_id", "geos")

    def __init__(self, model_id: str, geos: frozenset[str] | None = _US_GLOBAL):
        self.model_id = model_id
        self.geos = geos          # None → plain foundation model, never prefixed


MODEL_CATALOG: dict[str, _Entry] = {
    # ── Anthropic Claude ───────────────────────────────────────────────────
    "fable5":    _Entry("anthropic.claude-fable-5"),
    "opus5":     _Entry("anthropic.claude-opus-5"),
    "opus48":    _Entry("anthropic.claude-opus-4-8"),
    "opus47":    _Entry("anthropic.claude-opus-4-7"),
    "opus46":    _Entry("anthropic.claude-opus-4-6-v1"),
    "opus45":    _Entry("anthropic.claude-opus-4-5-20251101-v1:0"),
    "opus41":    _Entry("anthropic.claude-opus-4-1-20250805-v1:0", _US),
    "sonnet5":   _Entry("anthropic.claude-sonnet-5"),
    "sonnet46":  _Entry("anthropic.claude-sonnet-4-6"),
    "sonnet45":  _Entry("anthropic.claude-sonnet-4-5-20250929-v1:0"),
    "sonnet4":   _Entry("anthropic.claude-sonnet-4-20250514-v1:0"),
    "haiku45":   _Entry("anthropic.claude-haiku-4-5-20251001-v1:0"),
    "haiku3":    _Entry("anthropic.claude-3-haiku-20240307-v1:0", _US),
    # ── Meta Llama ─────────────────────────────────────────────────────────
    "llama4maverick": _Entry("meta.llama4-maverick-17b-instruct-v1:0", _US),
    "llama4scout":    _Entry("meta.llama4-scout-17b-instruct-v1:0", _US),
    # ── Embeddings (foundation models — no inference profile) ──────────────
    "titanembedv2":   _Entry("amazon.titan-embed-text-v2:0", None),   # 1536 dims
    "titanembedv1":   _Entry("amazon.titan-embed-text-v1", None),     # 1536 dims
    "cohereembedv4":  _Entry("cohere.embed-v4:0", None),
    "cohereembeden":  _Entry("cohere.embed-english-v3", None),        # 1024 dims
    "cohereembedml":  _Entry("cohere.embed-multilingual-v3", None),   # 1024 dims
}


def _normalize(name: str) -> str:
    """Fold a name to its catalogue key: lowercase, punctuation stripped.

    So "haiku45", "haiku-4.5", "Haiku 4.5", and "haiku_4_5" are all the same
    entry — nobody should have to remember which separator we picked.
    """
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _geo() -> str:
    return (settings.BEDROCK_INFERENCE_GEO or "").strip().lower().rstrip(".")


# Leading segment of a real Bedrock model ID: an inference-profile geo, or a
# vendor namespace. Used to tell a deliberate raw ID from a mistyped name —
# "anthropic.claude-opus-5" is an ID, "haiku4.5x" is a typo, and both contain
# a dot, so a bare dot-check would wave the typo through to a Bedrock 400.
_ID_NAMESPACES = frozenset({
    "us", "eu", "apac", "global",                                  # inference-profile geos
    "anthropic", "meta", "amazon", "cohere", "mistral", "ai21",    # vendors
    "stability", "twelvelabs", "deepseek", "writer", "luma", "qwen", "openai",
})


def _looks_like_raw_id(raw: str) -> bool:
    """True when *raw* is a Bedrock model ID or ARN rather than a model name."""
    lowered = raw.lower()
    if lowered.startswith("arn:") or ":" in lowered:
        return True
    return "." in lowered and lowered.split(".", 1)[0] in _ID_NAMESPACES


def _is_throttling(exc: BaseException) -> bool:
    return isinstance(exc, ClientError) and exc.response["Error"]["Code"] in (
        "ThrottlingException",
        "TooManyRequestsException",
    )


def _runtime_client(read_timeout: int = 300, verify: bool | None = None):
    """Build a bedrock-runtime boto3 client using the current credentials."""
    creds = get_credentials()
    kwargs: dict = {
        "region_name": creds.pop("region_name", settings.AWS_REGION),
        "config": BotocoreConfig(read_timeout=read_timeout, connect_timeout=10),
        "verify": settings.BEDROCK_SSL_VERIFY if verify is None else verify,
    }
    for k in ("aws_access_key_id", "aws_secret_access_key", "aws_session_token"):
        if creds.get(k):
            kwargs[k] = creds[k]
    return boto3.client("bedrock-runtime", **kwargs)


# ── Converse client ────────────────────────────────────────────────────────

class BedrockConverseClient:
    """Direct Bedrock Converse access with retry, concurrency cap and usage logging."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        self._botocore_config = BotocoreConfig(read_timeout=300, connect_timeout=10)
        self.client = self._make_client()

    def _sem(self) -> asyncio.Semaphore:
        if self.model_id not in _semaphores:
            _semaphores[self.model_id] = asyncio.Semaphore(settings.BEDROCK_CONCURRENCY)
        return _semaphores[self.model_id]

    def _make_client(self):
        creds = get_credentials()
        kwargs: dict = {
            "region_name": creds.pop("region_name", settings.AWS_REGION),
            "config": self._botocore_config,
            "verify": settings.BEDROCK_SSL_VERIFY,
        }
        for k in ("aws_access_key_id", "aws_secret_access_key", "aws_session_token"):
            if creds.get(k):
                kwargs[k] = creds[k]
        return boto3.client("bedrock-runtime", **kwargs)

    async def converse(self, system_prompt: str, messages: list[dict], max_tokens: int = 4096, operation: str = "", temperature: float = 0.3) -> str:
        log.debug("converse | model=%s | operation=%s | max_tokens=%d | temp=%.1f", self.model_id, operation, max_tokens, temperature)
        async with self._sem():
            text, tokens_in, tokens_out = await asyncio.to_thread(
                self._converse_sync, system_prompt, messages, max_tokens, temperature
            )
        await usage.record(self.model_id, tokens_in, tokens_out, operation)
        return text

    @retry(
        retry=retry_if_exception(_is_throttling),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        before_sleep=before_sleep_log(log, logging.WARNING),
        reraise=True,
    )
    def _converse_sync(self, system_prompt: str, messages: list[dict], max_tokens: int, temperature: float = 0.3) -> tuple[str, int, int]:
        if settings.ASSUMED_ROLE_ARN:
            self.client = self._make_client()
        response = self.client.converse(
            modelId=self.model_id,
            system=[{"text": system_prompt}],
            messages=messages,
            inferenceConfig={"maxTokens": max_tokens, "temperature": temperature},
        )
        text = response["output"]["message"]["content"][0]["text"]
        usage_meta = response.get("usage", {})
        return text, usage_meta.get("inputTokens", 0), usage_meta.get("outputTokens", 0)

    async def converse_stream(self, system_prompt: str, messages: list[dict], max_tokens: int = 4096, operation: str = ""):
        """Async generator that yields text chunks as they arrive from Bedrock."""
        log.debug("converse_stream | model=%s | operation=%s | max_tokens=%d", self.model_id, operation, max_tokens)
        usage_holder: dict = {}
        # Set when the consumer goes away (SSE client disconnect → GeneratorExit
        # at the yield). The worker thread checks it between events and stops
        # pulling from Bedrock, so a cancelled stream doesn't keep a thread-pool
        # slot + Bedrock connection busy until the read timeout.
        stop = threading.Event()
        async with self._sem():
            queue: asyncio.Queue[str | None] = asyncio.Queue()
            loop = asyncio.get_event_loop()

            def _stream():
                try:
                    if settings.ASSUMED_ROLE_ARN:
                        self.client = self._make_client()
                    response = self.client.converse_stream(
                        modelId=self.model_id,
                        system=[{"text": system_prompt}],
                        messages=messages,
                        inferenceConfig={"maxTokens": max_tokens, "temperature": 0.3},
                    )
                    for event in response.get("stream", []):
                        if stop.is_set():
                            break  # consumer disconnected — stop consuming Bedrock
                        if "contentBlockDelta" in event:
                            delta = event["contentBlockDelta"].get("delta", {})
                            if "text" in delta:
                                loop.call_soon_threadsafe(queue.put_nowait, delta["text"])
                        elif "metadata" in event:
                            usage_holder.update(event["metadata"].get("usage", {}))
                finally:
                    loop.call_soon_threadsafe(queue.put_nowait, None)

            task = loop.run_in_executor(None, _stream)
            try:
                while True:
                    chunk = await queue.get()
                    if chunk is None:
                        break
                    yield chunk
            finally:
                # On normal completion this is a no-op; on GeneratorExit/cancel it
                # signals the worker to stop and reclaims the thread instead of
                # leaking it (and the semaphore slot) until Bedrock finishes.
                stop.set()
                try:
                    await task
                except Exception:
                    pass
        if usage_holder:
            await usage.record(
                self.model_id,
                usage_holder.get("inputTokens", 0),
                usage_holder.get("outputTokens", 0),
                operation,
                suffix=" (stream)",
            )


# ── Embeddings ─────────────────────────────────────────────────────────────

def _build_request_body(model_id: str, text: str) -> str:
    """Build the invoke_model request body for the given embedding model."""
    if "titan-embed" in model_id:
        return json.dumps({"inputText": text})
    if "cohere.embed" in model_id:
        return json.dumps({"texts": [text], "input_type": "search_document"})
    # Generic fallback (Titan-compatible)
    return json.dumps({"inputText": text})


def _parse_response_body(model_id: str, body_bytes: bytes) -> list[float]:
    """Extract the embedding vector from the model response."""
    result = json.loads(body_bytes)
    if "titan-embed" in model_id:
        return result["embedding"]
    if "cohere.embed" in model_id:
        return result["embeddings"][0]
    # Generic fallback
    return result.get("embedding") or result.get("embeddings", [[]])[0]


# ── Provider ───────────────────────────────────────────────────────────────

class BedrockProvider:
    name = "bedrock"

    def resolve_model(self, name: str) -> str:
        """Map a friendly model name to a Bedrock model ID.

        Unrecognised values pass through unchanged, so a raw Bedrock ID (or an
        inference-profile ARN) can still be set directly when the catalogue
        doesn't cover what you need.
        """
        raw = (name or "").strip()
        if not raw:
            return ""
        entry = MODEL_CATALOG.get(_normalize(raw))
        if entry is None:
            if _looks_like_raw_id(raw):
                return raw  # a deliberate Bedrock model ID / ARN — pass through
            # Not a known name and not shaped like an ID: almost certainly a
            # typo. Fail loudly; it would otherwise reach Bedrock as a 400.
            raise UnknownModelError(
                f"Unknown model name {raw!r} for the bedrock provider. "
                f"Known names: {', '.join(self.known_models())}. "
                "To use a model that isn't in the catalogue, set its full "
                "Bedrock model ID or inference-profile ARN instead."
            )

        if entry.geos is None:
            return entry.model_id  # foundation model; never prefixed

        geo = _geo()
        if not geo:
            return entry.model_id  # explicitly opted out of inference profiles
        if geo not in entry.geos:
            raise UnknownModelError(
                f"Model {raw!r} has no {geo!r} inference profile on Bedrock "
                f"(available: {', '.join(sorted(entry.geos))}). Set "
                f"BEDROCK_INFERENCE_GEO to one of those, or leave it empty to "
                f"call the foundation model {entry.model_id!r} directly."
            )
        return f"{geo}.{entry.model_id}"

    def known_models(self) -> list[str]:
        return sorted(MODEL_CATALOG)

    def chat_model(self, model_id: str, max_tokens: int = 4096):
        """Create a ChatBedrockConverse runnable (untracked — `app.model` wraps it)."""
        creds = get_credentials()
        kwargs: dict = {
            "model": model_id,
            "max_tokens": max_tokens,
            "config": BotocoreConfig(read_timeout=300, connect_timeout=10),
        }
        for k in ("aws_access_key_id", "aws_secret_access_key", "aws_session_token", "region_name"):
            if creds.get(k):
                kwargs[k] = creds[k]
        if "region_name" not in kwargs:
            kwargs["region_name"] = settings.AWS_REGION
        if not settings.BEDROCK_SSL_VERIFY:
            # ChatBedrockConverse has no `verify` option, so build the runtime
            # client ourselves with TLS verification off and inject it — the model
            # uses a passed-in client as-is instead of creating its own.
            client_kwargs: dict = {
                "region_name": kwargs["region_name"],
                "config": kwargs["config"],
                "verify": False,
            }
            for k in ("aws_access_key_id", "aws_secret_access_key", "aws_session_token"):
                if creds.get(k):
                    client_kwargs[k] = creds[k]
            kwargs["client"] = boto3.client("bedrock-runtime", **client_kwargs)
        return ChatBedrockConverse(**kwargs)

    def converse_client(self, model_id: str) -> BedrockConverseClient:
        return BedrockConverseClient(model_id)

    @retry(
        retry=retry_if_exception(_is_throttling),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        before_sleep=before_sleep_log(log, logging.WARNING),
        reraise=True,
    )
    def embed_sync(self, model_id: str, text: str) -> list[float]:
        log.debug("embed_sync | model=%s | text_len=%d", model_id, len(text))
        client = _runtime_client(read_timeout=120)
        response = client.invoke_model(
            modelId=model_id,
            body=_build_request_body(model_id, text[:8000]),
            contentType="application/json",
            accept="application/json",
        )
        vec = _parse_response_body(model_id, response["body"].read())
        log.debug("embed_sync | dims=%d", len(vec))
        return vec
