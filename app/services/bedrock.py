import asyncio
import threading

import boto3
from botocore.config import Config as BotocoreConfig
from botocore.exceptions import ClientError
from langchain_aws import ChatBedrockConverse
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential, before_sleep_log
import logging

from app.config import settings
from app.logger import get_logger
from app.services.aws_auth import get_credentials

log = get_logger(__name__)

_FALLBACK_ORG_ID = "00000000-0000-0000-0000-000000000001"
_FALLBACK_USER_ID = ""

# One semaphore per model_id — caps concurrent Bedrock calls across the whole process.
# Initialised lazily so the event loop exists when first accessed.
_semaphores: dict[str, asyncio.Semaphore] = {}


def _caller_context() -> tuple[str, str]:
    """Return (org_id, user_id) from the active request context.

    Falls back to safe defaults for background/startup tasks that have no user context.
    asyncio.create_task inherits the ContextVar snapshot of the creator, so this
    correctly captures the request that triggered the LLM call.
    """
    from app.context import current_user
    ctx = current_user.get(None)
    if ctx is None:
        return (_FALLBACK_ORG_ID, _FALLBACK_USER_ID)
    return (ctx.org_id or _FALLBACK_ORG_ID, ctx.user_id or _FALLBACK_USER_ID)


def _is_throttling(exc: BaseException) -> bool:
    return isinstance(exc, ClientError) and exc.response["Error"]["Code"] in (
        "ThrottlingException",
        "TooManyRequestsException",
    )


# ── LangChain chat LLM factory ─────────────────────────────────────────────

class TrackedChatBedrock:
    """Wraps a LangChain runnable to fire usage_log.record() after each ainvoke."""

    def __init__(self, runnable, model_id: str, operation: str) -> None:
        self._runnable = runnable
        self._model_id = model_id
        self._operation = operation

    async def ainvoke(self, messages, **kwargs):
        log.debug("TrackedChatBedrock.ainvoke | model=%s | operation=%s", self._model_id, self._operation)
        response = await self._runnable.ainvoke(messages, **kwargs)
        usage = getattr(response, "usage_metadata", None) or {}
        tokens_in  = usage.get("input_tokens", 0)
        tokens_out = usage.get("output_tokens", 0)
        log.info("bedrock_usage | model=%s | operation=%s | tokens_in=%d | tokens_out=%d",
                 self._model_id, self._operation, tokens_in, tokens_out)
        if tokens_in or tokens_out:
            from app.services import usage_log
            org_id, user_id = _caller_context()
            await usage_log.record(
                org_id=org_id,
                user_id=user_id,
                model_id=self._model_id,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                operation=self._operation,
            )
        return response

    def bind_tools(self, tools, **kwargs):
        return TrackedChatBedrock(
            self._runnable.bind_tools(tools, **kwargs),
            self._model_id,
            self._operation,
        )

    def __getattr__(self, name: str):
        return getattr(self._runnable, name)


def make_chat_llm(model_id: str, max_tokens: int = 4096, operation: str = "unknown") -> TrackedChatBedrock:
    """Create a ChatBedrockConverse wrapped with automatic usage tracking."""
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
    return TrackedChatBedrock(ChatBedrockConverse(**kwargs), model_id, operation)


class BedrockService:
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
        log.debug("BedrockService.converse | model=%s | operation=%s | max_tokens=%d | temp=%.1f", self.model_id, operation, max_tokens, temperature)
        async with self._sem():
            text, tokens_in, tokens_out = await asyncio.to_thread(
                self._converse_sync, system_prompt, messages, max_tokens, temperature
            )
        log.info("bedrock_usage | model=%s | operation=%s | tokens_in=%d | tokens_out=%d",
                 self.model_id, operation, tokens_in, tokens_out)
        if operation:
            from app.services import usage_log
            org_id, user_id = _caller_context()
            await usage_log.record(
                org_id=org_id,
                user_id=user_id,
                model_id=self.model_id,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                operation=operation,
            )
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
        usage = response.get("usage", {})
        return text, usage.get("inputTokens", 0), usage.get("outputTokens", 0)

    async def converse_stream(self, system_prompt: str, messages: list[dict], max_tokens: int = 4096, operation: str = ""):
        """Async generator that yields text chunks as they arrive from Bedrock."""
        log.debug("BedrockService.converse_stream | model=%s | operation=%s | max_tokens=%d", self.model_id, operation, max_tokens)
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
        if operation and usage_holder:
            tokens_in  = usage_holder.get("inputTokens", 0)
            tokens_out = usage_holder.get("outputTokens", 0)
            log.info("bedrock_usage | model=%s | operation=%s | tokens_in=%d | tokens_out=%d (stream)",
                     self.model_id, operation, tokens_in, tokens_out)
            from app.services import usage_log
            org_id, user_id = _caller_context()
            await usage_log.record(
                org_id=org_id,
                user_id=user_id,
                model_id=self.model_id,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                operation=operation,
            )
