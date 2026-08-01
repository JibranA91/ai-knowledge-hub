"""DEPRECATED — kept only so existing imports keep working.

LLM access now goes through `app.model`, which resolves a logical role to a
provider + model ID. Bedrock's client code lives in `app/providers/bedrock.py`.

    # old
    from app.services.bedrock import make_chat_llm, BedrockService
    llm = make_chat_llm(settings.BEDROCK_QUERY_MODEL_ID, operation="query")
    svc = BedrockService(settings.BEDROCK_QUERY_MODEL_ID)

    # new
    from app import model
    llm = model.get_chat(model.Role.QUERY)
    svc = model.get_converse(model.Role.QUERY)

This module will be removed; do not add to it.
"""
import warnings

from app.providers.bedrock import BedrockConverseClient
from app.providers.usage import TrackedChat

# Legacy aliases.
BedrockService = BedrockConverseClient
TrackedChatBedrock = TrackedChat


def make_chat_llm(model_id: str, max_tokens: int = 4096, operation: str = "unknown") -> TrackedChat:
    """Deprecated. Use `app.model.get_chat(role)`."""
    warnings.warn(
        "app.services.bedrock.make_chat_llm is deprecated; use app.model.get_chat(role)",
        DeprecationWarning,
        stacklevel=2,
    )
    from app.providers import bedrock as _bedrock
    return TrackedChat(_bedrock.BedrockProvider().chat_model(model_id, max_tokens), model_id, operation)


__all__ = ["BedrockService", "TrackedChatBedrock", "make_chat_llm"]
