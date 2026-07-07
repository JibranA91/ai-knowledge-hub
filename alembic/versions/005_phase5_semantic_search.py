"""Phase 5: Semantic Search + Knowledge Graph.

Adds pgvector extension (only when available), nullable embedding column on
wiki_pages, and the wiki_links table for structured link graph queries.

Revision ID: 005
Revises: 004
Create Date: 2026-05-06
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "005"
down_revision: Union[str, None] = "004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _pgvector_available(conn) -> bool:
    """Return True only when the 'vector' extension is listed in pg_available_extensions.

    This is a safe, read-only check that never aborts a transaction — as opposed to
    attempting CREATE EXTENSION and catching the error, which leaves the Postgres
    transaction in an aborted state and causes subsequent DDL to fail.
    """
    result = conn.execute(
        sa.text("SELECT COUNT(*) FROM pg_available_extensions WHERE name = 'vector'")
    )
    return (result.scalar() or 0) > 0


def upgrade() -> None:
    conn = op.get_bind()

    # ── pgvector extension (only when available in this Postgres install) ──
    if _pgvector_available(conn):
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")

        # Nullable so existing rows are unaffected; populated lazily on next write.
        op.execute("""
            ALTER TABLE wiki_pages
            ADD COLUMN IF NOT EXISTS embedding vector(1536)
        """)

        # IVFFlat index for approximate nearest-neighbour cosine search.
        # 'lists' is tuned for up to ~100k rows; adjust via migration when needed.
        op.execute("""
            CREATE INDEX IF NOT EXISTS wiki_pages_embedding_idx
            ON wiki_pages USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100)
        """)

    # ── wiki_links: structured link graph (supplement to .graph.json cache) ──
    # Always created regardless of pgvector availability.
    op.execute("""
        CREATE TABLE IF NOT EXISTS wiki_links (
            org_id    UUID NOT NULL,
            from_path TEXT NOT NULL,
            to_path   TEXT NOT NULL,
            PRIMARY KEY (org_id, from_path, to_path)
        )
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS wiki_links_from_idx
        ON wiki_links (org_id, from_path)
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS wiki_links_to_idx
        ON wiki_links (org_id, to_path)
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS wiki_links")
    op.execute("DROP INDEX IF EXISTS wiki_pages_embedding_idx")
    op.execute("ALTER TABLE wiki_pages DROP COLUMN IF EXISTS embedding")
    # Leave the vector extension in place — it may be used by other tables.
