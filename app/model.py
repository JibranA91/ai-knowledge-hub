"""Central LLM access layer — the single entry point for every model call.

    RULE: no LLM connection is created anywhere outside `app/providers/`, and
    no application code imports a provider directly. Everything goes through
    this module. `tests/unit/test_no_direct_llm_clients.py` enforces it.

Call sites ask for a *role* (what the model is for), never a model ID. Config
names a *model* per role ("haiku45", "sonnet45"), and the active provider maps
that name to its own concrete ID. So retargeting a role — or pointing the whole
app at a different vendor — is a config change, not a code change:

    role  ──►  config: a friendly model name  ──►  provider: the vendor's ID
    QUERY      MODEL_QUERY=haiku45                 us.anthropic.claude-haiku-4-5-20251001-v1:0

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
import json
import math
from enum import StrEnum

from app import providers
from app.config import settings
from app.logger import get_logger
from app.providers.base import UnknownModelError
from app.providers.usage import TrackedChat

log = get_logger(__name__)

# Matches wiki_pages.embedding vector(1536); changing settings cannot resize it.
EMBEDDING_STORAGE_DIMENSIONS = 1536


class Role(StrEnum):
    """What a model is being used for. Call sites name a role, not a model."""

    INGEST_PLAN = "ingest_plan"        # ingest planner (tool-calling reasoner)
    INGEST_WRITE = "ingest_write"      # ingest page renderer (single-shot)
    QUERY = "query"                    # chat / Q&A over the wiki
    RECALIBRATE = "recalibrate"        # wiki-wide analysis + rewrite
    DRAFT_AGENT = "draft_agent"        # conversational AI Writer agent
    EDIT = "edit"                      # inline page/section editor
    EMBEDDING = "embedding"            # vector embeddings for semantic search


# Role → the Settings attribute holding its model name. Read at call time so
# config changes (and test patches) are picked up without re-import. The older
# BEDROCK_*_MODEL_ID vars are folded onto these in Settings.model_post_init,
# so existing .env files keep working without a second lookup here.
_ROLE_SETTING: dict[Role, str] = {
    Role.INGEST_PLAN:  "MODEL_INGEST_PLAN",
    Role.INGEST_WRITE: "MODEL_INGEST_WRITE",
    Role.QUERY:        "MODEL_QUERY",
    Role.RECALIBRATE:  "MODEL_RECALIBRATE",
    Role.DRAFT_AGENT:  "MODEL_DRAFT_AGENT",
    Role.EDIT:         "MODEL_EDIT",
    Role.EMBEDDING:    "MODEL_EMBEDDING",
}


def validate_configuration() -> dict[Role, str]:
    """Resolve every configured role against the active provider.

    Called at startup so a typo'd model name fails the boot with a list of
    valid names, instead of surfacing later as an opaque Bedrock 400 on the
    first request that happens to use that role. Returns role → resolved ID for
    the roles that are configured.

    Raises `UnknownModelError` listing every bad role at once — fixing them one
    restart at a time is nobody's idea of a good time.
    """
    resolved: dict[Role, str] = {}
    problems: list[str] = []
    provider = _provider()
    if provider_name() == "bedrock" and (settings.LLM_API_KEY.get_secret_value() or settings.LLM_BASE_URL):
        problems.append("  bedrock: use AWS credentials and AWS_REGION; leave LLM_API_KEY and LLM_BASE_URL empty")
    for role in Role:
        name = model_name_for(role)
        if not name:
            if role != Role.EMBEDDING:
                problems.append(f"  {role.value}: {_ROLE_SETTING[role]} must not be empty")
            continue
        try:
            resolved[role] = provider.resolve_model(name)
            provider.validate_model(resolved[role], embedding=role == Role.EMBEDDING,
                                    dimensions=EMBEDDING_STORAGE_DIMENSIONS)
        except UnknownModelError as exc:
            problems.append(f"  {role.value}: {exc}")
    if embedding_enabled() and settings.EMBEDDING_DIMENSIONS != EMBEDDING_STORAGE_DIMENSIONS:
        problems.append("  embedding: the database requires 1536 dimensions; changing "
                        "EMBEDDING_DIMENSIONS alone does not migrate stored vectors")
    if problems:
        raise UnknownModelError(
            f"Invalid model configuration for provider {provider_name()!r}:\n"
            + "\n".join(problems)
        )
    return resolved


def provider_name() -> str:
    """The active provider's name (LLM_PROVIDER, default 'bedrock')."""
    return (settings.LLM_PROVIDER or "bedrock").strip().lower()


def _provider():
    return providers.get(provider_name())


def model_name_for(role: Role) -> str:
    """The model *name* configured for *role*, exactly as written in config."""
    try:
        setting = _ROLE_SETTING[Role(role)]
    except (KeyError, ValueError):
        raise ValueError(f"Unknown model role: {role!r}. Known roles: "
                         f"{', '.join(r.value for r in Role)}") from None
    return (getattr(settings, setting, "") or "").strip()


def model_id_for(role: Role) -> str:
    """Resolve *role* to the active provider's concrete model ID.

    Raises `UnknownModelError` if the configured name isn't one this provider
    can serve.
    """
    name = model_name_for(role)
    if not name:
        if Role(role) != Role.EMBEDDING:
            raise UnknownModelError(f"role {Role(role).value!r}: model must not be empty")
        return ""
    try:
        return _provider().resolve_model(name)
    except UnknownModelError as exc:
        raise UnknownModelError(f"role {Role(role).value!r}: {exc}") from None


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
    return TrackedChat(runnable, model_id, str(role) if operation is None else operation)


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
    return bool(model_name_for(Role.EMBEDDING))


def embedding_identity() -> str:
    """Identify comparable vectors by provider, resolved model and storage size."""
    if not embedding_enabled():
        return ""
    return json.dumps([provider_name(), model_id_for(Role.EMBEDDING),
                       EMBEDDING_STORAGE_DIMENSIONS], separators=(",", ":"))


def valid_embedding(vector) -> bool:
    """Only finite, nonzero vectors of the database's fixed size are usable."""
    return (
        isinstance(vector, list) and len(vector) == EMBEDDING_STORAGE_DIMENSIONS
        and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                and math.isfinite(v) and abs(v) <= 3.4028235e38 for v in vector)
        and any(abs(v) >= 1e-8 for v in vector)
    )


async def embed(text: str) -> list[float] | None:
    """Embed *text*, or return None if embeddings are disabled or the call fails.

    Never raises — semantic search degrades to keyword search instead.
    """
    if not embedding_enabled():
        return None
    try:
        model_id = model_id_for(Role.EMBEDDING)
        vec = await asyncio.to_thread(_provider().embed_sync, model_id, text)
        if not valid_embedding(vec):
            raise ValueError("Embedding must be a finite, nonzero 1536-dimensional vector")
        return vec
    except Exception as exc:
        log.warning("embed | failed (returning None) | model=%s: %s", model_name_for(Role.EMBEDDING), exc)
        return None
