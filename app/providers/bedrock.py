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
                event_stream = None
                try:
                    if settings.ASSUMED_ROLE_ARN:
                        self.client = self._make_client()
                    response = self.client.converse_stream(
                        modelId=self.model_id,
                        system=[{"text": system_prompt}],
                        messages=messages,
                        inferenceConfig={"maxTokens": max_tokens, "temperature": 0.3},
                    )
                    event_stream = response.get("stream", [])
                    for event in event_stream:
                        if stop.is_set():
                            break  # consumer disconnected — stop consuming Bedrock
                        if "contentBlockDelta" in event:
                            delta = event["contentBlockDelta"].get("delta", {})
                            if "text" in delta:
                                loop.call_soon_threadsafe(queue.put_nowait, delta["text"])
                        elif "metadata" in event:
                            usage_holder.update(event["metadata"].get("usage", {}))
                        elif any(key.endswith("Exception") for key in event):
                            raise RuntimeError(f"Bedrock streaming error: {event}")
                finally:
                    try:
                        close = getattr(event_stream, "close", None)
                        if close:
                            close()
                    finally:
                        loop.call_soon_threadsafe(queue.put_nowait, None)

            task = loop.run_in_executor(None, _stream)
            try:
                while True:
                    chunk = await queue.get()
                    if chunk is None:
                        break
                    yield chunk
                # Propagate producer failures on normal consumption, including
                # failures after partial output, before callers emit 'done'.
                await asyncio.shield(task)
            finally:
                # On normal completion this is a no-op; on GeneratorExit/cancel it
                # signals the worker to stop and reclaims the thread instead of
                # leaking it (and the semaphore slot) until Bedrock finishes.
                stop.set()
                try:
                    await asyncio.shield(task)
                except Exception:
                    # The normal path above already raised. Cleanup must not
                    # replace GeneratorExit / cancellation with a worker error.
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

    def validate_configuration(self) -> None:
        if settings.LLM_API_KEY.get_secret_value() or settings.LLM_BASE_URL:
            raise UnknownModelError("bedrock: use AWS credentials and AWS_REGION; "
                                    "leave LLM_API_KEY and LLM_BASE_URL empty")

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
