"""Semantic embedding service — a thin wrapper over `app.model`.

The actual vendor call (and each model's request/response shape) lives in the
provider; this module only exposes the app-facing helpers and the pgvector
formatting used by wiki_db.

If an embedding model is configured, embed_text() returns a float vector.
Otherwise it returns None and all callers fall back to BM25-only search.

Set MODEL_EMBEDDING to a model compatible with the database's 1536 dimensions
(e.g. titanembedv1). Changing EMBEDDING_DIMENSIONS alone cannot resize storage.
"""
from app import model
from app.logger import get_logger

log = get_logger(__name__)


def is_enabled() -> bool:
    """Return True when an embedding model is configured."""
    return model.embedding_enabled()


async def embed_text(text: str) -> list[float] | None:
    """Return an embedding vector for *text*, or None if embedding is disabled or fails."""
    if not is_enabled():
        return None
    log.debug("embed_text | text_len=%d", len(text))
    vec = await model.embed(text)
    if vec is not None:
        log.debug("embed_text | ok | dims=%d", len(vec))
    return vec


def vec_to_pg(vec: list[float]) -> str:
    """Format a float list as a pgvector literal string: '[0.1,0.2,...]'."""
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"
