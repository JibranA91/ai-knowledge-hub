"""LLM provider registry.

Every outbound connection to an LLM vendor is created inside this package and
nowhere else. Application code imports `app.model`, never a provider directly.

Adding a provider:
    _PROVIDER_MODULES["new_provider"] = ("app.providers.new_provider", "NewProvider")
…then set LLM_PROVIDER=new_provider. See `app/providers/base.py` for the contract.
"""
from functools import lru_cache

from app.providers.base import ChatModel, ConverseClient, Message, Provider

# name → provider class. Imported lazily inside get() so that adding a provider
# with heavy/optional SDK deps doesn't cost anything for users who don't use it.
_PROVIDER_MODULES: dict[str, tuple[str, str]] = {
    "bedrock": ("app.providers.bedrock", "BedrockProvider"),
    "anthropic": ("app.providers.anthropic", "AnthropicProvider"),
}


@lru_cache(maxsize=None)
def get(name: str) -> Provider:
    """Return the singleton provider instance registered under *name*."""
    key = (name or "").strip().lower()
    if key not in _PROVIDER_MODULES:
        raise ValueError(
            f"Unknown LLM provider {name!r}. "
            f"Available: {', '.join(sorted(_PROVIDER_MODULES))}. "
            "Register new providers in app/providers/__init__.py."
        )
    module_path, class_name = _PROVIDER_MODULES[key]
    module = __import__(module_path, fromlist=[class_name])
    return getattr(module, class_name)()


def available() -> list[str]:
    return sorted(_PROVIDER_MODULES)


__all__ = ["get", "available", "Provider", "ChatModel", "ConverseClient", "Message"]
