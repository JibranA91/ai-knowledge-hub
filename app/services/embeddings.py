"""Optional semantic embedding via Amazon Bedrock.

If BEDROCK_EMBEDDING_MODEL_ID is set in config, embed_text() returns a float
vector. Otherwise it returns None and all callers fall back to BM25-only search.

Supported models (set via BEDROCK_EMBEDDING_MODEL_ID):
  amazon.titan-embed-text-v2:0          — 1536 dims (default, recommended)
  cohere.embed-english-v3               — 1024 dims (set EMBEDDING_DIMENSIONS=1024)
  cohere.embed-multilingual-v3          — 1024 dims
"""
import asyncio
import json
import logging

import boto3
from botocore.config import Config as BotocoreConfig
from botocore.exceptions import ClientError
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential, before_sleep_log

from app.config import settings
from app.logger import get_logger
from app.services.aws_auth import get_credentials

log = get_logger(__name__)


def is_enabled() -> bool:
    """Return True when an embedding model is configured."""
    return bool(settings.BEDROCK_EMBEDDING_MODEL_ID)


def _is_throttling(exc: BaseException) -> bool:
    return isinstance(exc, ClientError) and exc.response["Error"]["Code"] in (
        "ThrottlingException",
        "TooManyRequestsException",
    )


def _make_client():
    creds = get_credentials()
    kwargs: dict = {
        "region_name": creds.pop("region_name", settings.AWS_REGION),
        "config": BotocoreConfig(read_timeout=120, connect_timeout=10),
        "verify": settings.BEDROCK_SSL_VERIFY,
    }
    for k in ("aws_access_key_id", "aws_secret_access_key", "aws_session_token"):
        if creds.get(k):
            kwargs[k] = creds[k]
    return boto3.client("bedrock-runtime", **kwargs)


def _build_request_body(text: str) -> str:
    """Build the invoke_model request body for the configured embedding model."""
    model = settings.BEDROCK_EMBEDDING_MODEL_ID
    if "titan-embed" in model:
        return json.dumps({"inputText": text})
    if "cohere.embed" in model:
        return json.dumps({"texts": [text], "input_type": "search_document"})
    # Generic fallback (Titan-compatible)
    return json.dumps({"inputText": text})


def _parse_response_body(body_bytes: bytes) -> list[float]:
    """Extract the embedding vector from the model response."""
    result = json.loads(body_bytes)
    model = settings.BEDROCK_EMBEDDING_MODEL_ID
    if "titan-embed" in model:
        return result["embedding"]
    if "cohere.embed" in model:
        return result["embeddings"][0]
    # Generic fallback
    return result.get("embedding") or result.get("embeddings", [[]])[0]


@retry(
    retry=retry_if_exception(_is_throttling),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    before_sleep=before_sleep_log(log, logging.WARNING),
    reraise=True,
)
def _embed_sync(text: str) -> list[float]:
    log.debug("_embed_sync | model=%s | text_len=%d", settings.BEDROCK_EMBEDDING_MODEL_ID, len(text))
    client = _make_client()
    body = _build_request_body(text[:8000])
    response = client.invoke_model(
        modelId=settings.BEDROCK_EMBEDDING_MODEL_ID,
        body=body,
        contentType="application/json",
        accept="application/json",
    )
    vec = _parse_response_body(response["body"].read())
    log.debug("_embed_sync | dims=%d", len(vec))
    return vec


async def embed_text(text: str) -> list[float] | None:
    """Return an embedding vector for *text*, or None if embedding is disabled or fails."""
    if not is_enabled():
        return None
    log.debug("embed_text | text_len=%d", len(text))
    try:
        vec = await asyncio.to_thread(_embed_sync, text)
        log.debug("embed_text | ok | dims=%d", len(vec))
        return vec
    except Exception as e:
        log.warning("embed_text | failed (returning None): %s", e)
        return None


def vec_to_pg(vec: list[float]) -> str:
    """Format a float list as a pgvector literal string: '[0.1,0.2,...]'."""
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"
