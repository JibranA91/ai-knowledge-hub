"""Provider-neutral token accounting.

Lives below `app.model` (and is imported by providers) so that usage recording
does not depend on which vendor served the call, and so providers can import it
without a cycle back through the facade.
"""
from app.logger import get_logger

log = get_logger(__name__)

_FALLBACK_ORG_ID = "00000000-0000-0000-0000-000000000001"
_FALLBACK_USER_ID = ""


def caller_context() -> tuple[str, str]:
    """Return (org_id, user_id) from the active request context.

    Falls back to safe defaults for background/startup tasks that have no user
    context. asyncio.create_task inherits the ContextVar snapshot of the
    creator, so this correctly captures the request that triggered the LLM call.
    """
    from app.context import current_user
    ctx = current_user.get(None)
    if ctx is None:
        return (_FALLBACK_ORG_ID, _FALLBACK_USER_ID)
    return (ctx.org_id or _FALLBACK_ORG_ID, ctx.user_id or _FALLBACK_USER_ID)


async def record(model_id: str, tokens_in: int, tokens_out: int,
                 operation: str, suffix: str = "") -> None:
    """Log and persist one LLM call's token usage. Never raises."""
    log.info("llm_usage | model=%s | operation=%s | tokens_in=%d | tokens_out=%d%s",
             model_id, operation, tokens_in, tokens_out, suffix)
    if not operation or not (tokens_in or tokens_out):
        return
    from app.services import usage_log
    org_id, user_id = caller_context()
    await usage_log.record(
        org_id=org_id,
        user_id=user_id,
        model_id=model_id,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        operation=operation,
    )


class TrackedChat:
    """Wraps a LangChain chat runnable to record usage after each ainvoke.

    Backend-agnostic: it only relies on LangChain's `usage_metadata`, which
    every `BaseChatModel` populates.
    """

    def __init__(self, runnable, model_id: str, operation: str) -> None:
        self._runnable = runnable
        self._model_id = model_id
        self._operation = operation

    async def ainvoke(self, messages, **kwargs):
        log.debug("TrackedChat.ainvoke | model=%s | operation=%s", self._model_id, self._operation)
        response = await self._runnable.ainvoke(messages, **kwargs)
        usage = getattr(response, "usage_metadata", None) or {}
        await record(
            self._model_id,
            usage.get("input_tokens", 0),
            usage.get("output_tokens", 0),
            self._operation,
        )
        return response

    def bind_tools(self, tools, **kwargs):
        return TrackedChat(
            self._runnable.bind_tools(tools, **kwargs),
            self._model_id,
            self._operation,
        )

    def __getattr__(self, name: str):
        return getattr(self._runnable, name)
