"""Provider protocols — the contract every LLM backend must satisfy.

A *provider* is the only thing in this codebase allowed to open a network
connection to an LLM vendor. Everything else goes through `app.model`.

To add a provider (OpenAI, Anthropic direct, Azure, Ollama, …):

  1. Create `app/providers/<name>.py` with a class implementing `Provider`.
  2. Register it in `app/providers/__init__.py::_PROVIDER_MODULES`.
  3. Add model bindings in `app/model_catalog.yaml`, select LLM_PROVIDER and friendly names.

API-key/endpoint adapters should use settings.LLM_API_KEY.get_secret_value()
and settings.LLM_BASE_URL rather than adding vendor-specific duplicates.
Only send the key to the configured service; never put it in logs or URLs.
Bedrock retains AWS authentication and rejects these generic connection settings.
MODEL_DEFAULT supplies unconfigured text roles; embeddings remain independent.
The connection-check CLI invokes clients with operation=""; this must not require
database access. Use usage.record, which already skips persistence in that case.

── Canonical message format ───────────────────────────────────────────────
`ConverseClient` speaks the Bedrock Converse message shape, which is the
internal wire format for this app:

    [{"role": "user"|"assistant", "content": [{"text": "..."}]}, ...]

The system prompt is passed separately, never as a message. A non-Bedrock
provider is responsible for translating this shape into its own SDK's format
(e.g. flattening `content` to a string for OpenAI). It is deliberately not
the caller's job.
"""
from typing import AsyncIterator, Protocol, runtime_checkable

# The canonical message type: Bedrock Converse shape (see module docstring).
Message = dict


class UnknownModelError(ValueError):
    """A configured model name isn't one the active provider can serve."""


@runtime_checkable
class ChatModel(Protocol):
    """A LangChain-compatible chat runnable, used by the LangGraph agents.

    Providers return something satisfying this; `app.model` wraps it in
    `TrackedChat` so token usage is recorded regardless of backend.
    """

    async def ainvoke(self, messages: list, **kwargs): ...

    def bind_tools(self, tools: list, **kwargs) -> "ChatModel": ...


@runtime_checkable
class ConverseClient(Protocol):
    """A direct request/response + streaming chat client.

    Implementations own their own retry, concurrency limiting and usage
    recording (see `app.providers.usage.record`).
    """

    model_id: str

    async def converse(
        self,
        system_prompt: str,
        messages: list[Message],
        max_tokens: int = 4096,
        operation: str = "",
        temperature: float = 0.3,
    ) -> str:
        """Return the full assistant reply as text."""
        ...

    def converse_stream(
        self,
        system_prompt: str,
        messages: list[Message],
        max_tokens: int = 4096,
        operation: str = "",
    ) -> AsyncIterator[str]:
        """Async-iterate text chunks as they arrive."""
        ...


@runtime_checkable
class Provider(Protocol):
    """Factory for a single vendor's clients."""

    name: str

    def validate_configuration(self) -> None:
        """Check provider connection settings locally, without constructing clients.

        Model resolution and capability checks belong to app.model_catalog.
        """
        ...

    def chat_model(self, model_id: str, max_tokens: int) -> ChatModel:
        """Build a LangChain chat runnable for *model_id*."""
        ...

    def converse_client(self, model_id: str) -> ConverseClient:
        """Build a converse/stream client for *model_id*."""
        ...

    def embed_sync(self, model_id: str, text: str) -> list[float]:
        """Return an embedding vector for *text*. Blocking — callers run it
        in a worker thread. Raise if the provider has no embedding support."""
        ...
