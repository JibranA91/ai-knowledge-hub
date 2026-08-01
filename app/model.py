"""Central LLM access layer — the single entry point for every model call.

    RULE: no LLM connection is created anywhere outside `app/providers/`, and
    no application code imports a provider directly. Everything goes through
    this module. `tests/unit/test_no_direct_llm_clients.py` enforces it.

Call sites ask for a *role* (what the model is for), never a model ID. The
role → model-ID mapping lives here and resolves against config, so retargeting
a role at a different model — or the whole app at a different vendor — is a
config change, not a code change.

Usage:

    from app import model

    # LangChain runnable for the LangGraph agents (supports bind_tools)
    llm = model.get_chat(model.Role.INGEST_PLAN, max_tokens=4096)
    reply = await llm.ainvoke(messages)

    # Direct converse / streaming client
    client = model.get_converse(model.Role.QUERY)
    text = await client.converse(system_prompt, messages, operation="query")
    async for chunk in client.converse_stream(system_prompt, messages):
        ...

    # Embeddings
    if model.embedding_enabled():
        vec = await model.embed("some text")

Adding a provider: implement `app/providers/base.Provider`, register it in
`app/providers/__init__.py`, set LLM_PROVIDER. No call site changes.
"""
import asyncio
from enum import StrEnum

from app import providers
from app.config import settings
from app.logger import get_logger
from app.providers.usage import TrackedChat

log = get_logger(__name__)


class Role(StrEnum):
    """What a model is being used for. Call sites name a role, not a model."""

    INGEST_PLAN = "ingest_plan"        # ingest planner (tool-calling reasoner)
    INGEST_WRITE = "ingest_write"      # ingest page renderer (single-shot)
    QUERY = "query"                    # chat / Q&A over the wiki
    RECALIBRATE = "recalibrate"        # wiki-wide analysis + rewrite
    DRAFT_AGENT = "draft_agent"        # conversational AI Writer agent
    EDIT = "edit"                      # inline page/section editor
    EMBEDDING = "embedding"            # vector embeddings for semantic search


# Role → the Settings attribute holding its model ID. Resolved at call time so
# config changes (and test patches) are picked up without re-import.
_ROLE_SETTING: dict[Role, str] = {
    Role.INGEST_PLAN:  "BEDROCK_INGEST_MODEL_ID",
    Role.INGEST_WRITE: "BEDROCK_INGEST_WRITER_MODEL_ID",
    Role.QUERY:        "BEDROCK_QUERY_MODEL_ID",
    Role.RECALIBRATE:  "BEDROCK_RECALIBRATE_MODEL_ID",
    Role.DRAFT_AGENT:  "BEDROCK_DRAFT_AGENT_MODEL_ID",
    Role.EDIT:         "BEDROCK_EDIT_MODEL_ID",
    Role.EMBEDDING:    "BEDROCK_EMBEDDING_MODEL_ID",
}


def provider_name() -> str:
    """The active provider's name (LLM_PROVIDER, default 'bedrock')."""
    return (settings.LLM_PROVIDER or "bedrock").strip().lower()


def _provider():
    return providers.get(provider_name())


def model_id_for(role: Role) -> str:
    """Resolve *role* to a concrete model ID from config."""
    try:
        setting = _ROLE_SETTING[Role(role)]
    except (KeyError, ValueError):
        raise ValueError(f"Unknown model role: {role!r}. Known roles: "
                         f"{', '.join(r.value for r in Role)}") from None
    return getattr(settings, setting, "") or ""


# ── Chat ───────────────────────────────────────────────────────────────────

def get_chat(role: Role, max_tokens: int = 4096, operation: str | None = None) -> TrackedChat:
    """A LangChain chat runnable for *role*, with automatic usage tracking.

    *operation* labels the call in usage_log; defaults to the role name. Pass it
    when one role serves several distinct operations (e.g. recalibrate
    analyze vs. write).
    """
    model_id = model_id_for(role)
    log.debug("get_chat | role=%s | model=%s | provider=%s", role, model_id, provider_name())
    runnable = _provider().chat_model(model_id, max_tokens=max_tokens)
    return TrackedChat(runnable, model_id, operation or str(role))


def get_converse(role: Role):
    """A converse/streaming client for *role*.

    The per-call `operation` argument is supplied at converse() time.
    """
    model_id = model_id_for(role)
    log.debug("get_converse | role=%s | model=%s | provider=%s", role, model_id, provider_name())
    return _provider().converse_client(model_id)


# ── Embeddings ─────────────────────────────────────────────────────────────

def embedding_enabled() -> bool:
    """True when an embedding model is configured. When False, callers fall
    back to BM25-only search."""
    return bool(model_id_for(Role.EMBEDDING))


async def embed(text: str) -> list[float] | None:
    """Embed *text*, or return None if embeddings are disabled or the call fails.

    Never raises — semantic search degrades to keyword search instead.
    """
    if not embedding_enabled():
        return None
    model_id = model_id_for(Role.EMBEDDING)
    try:
        return await asyncio.to_thread(_provider().embed_sync, model_id, text)
    except Exception as exc:
        log.warning("embed | failed (returning None) | model=%s: %s", model_id, exc)
        return None
