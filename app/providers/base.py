"""Provider protocols — the contract every LLM backend must satisfy.

A *provider* is the only thing in this codebase allowed to open a network
connection to an LLM vendor. Everything else goes through `app.model`.

To add a provider (OpenAI, Anthropic direct, Azure, Ollama, …):

  1. Create `app/providers/<name>.py` with a class implementing `Provider`.
  2. Register it in `app/providers/__init__.py::PROVIDERS`.
  3. Point `LLM_PROVIDER=<name>` at it and set the per-role model IDs.

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
