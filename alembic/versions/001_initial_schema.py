"""Phase 1 initial schema: ingest_jobs, chat_sessions, auth_tokens,
recalibrate_jobs, wiki_pages, audit_log.

Revision ID: 001
Revises:
Create Date: 2025-05-05
"""
from typing import Sequence, Union

from alembic import op

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS ingest_jobs (
            id               SERIAL PRIMARY KEY,
            filename         TEXT NOT NULL UNIQUE,
            status           TEXT NOT NULL DEFAULT 'queued',
            message          TEXT NOT NULL DEFAULT '',
            plan             JSONB NOT NULL DEFAULT '[]',
            conflicts        JSONB NOT NULL DEFAULT '[]',
            index_additions  JSONB NOT NULL DEFAULT '[]',
            log_entry        TEXT NOT NULL DEFAULT '',
            doc_text         TEXT NOT NULL DEFAULT '',
            plan_chat_history JSONB NOT NULL DEFAULT '[]',
            pages_created    JSONB NOT NULL DEFAULT '[]',
            pages_updated    JSONB NOT NULL DEFAULT '[]',
            created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)

    op.execute("""
        CREATE TABLE IF NOT EXISTS chat_sessions (
            session_id     TEXT PRIMARY KEY,
            messages       JSONB NOT NULL DEFAULT '[]',
            summary        TEXT NOT NULL DEFAULT '',
            created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            last_active_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)

    op.execute("""
        CREATE TABLE IF NOT EXISTS auth_tokens (
            token_hash TEXT PRIMARY KEY,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            expires_at TIMESTAMPTZ NOT NULL
        )
    """)

    op.execute("""
        CREATE TABLE IF NOT EXISTS recalibrate_jobs (
            id               SERIAL PRIMARY KEY,
            status           TEXT NOT NULL DEFAULT 'idle',
            stage            TEXT NOT NULL DEFAULT '',
            progress         INT  NOT NULL DEFAULT 0,
            details          TEXT NOT NULL DEFAULT '',
            pages_improved   JSONB NOT NULL DEFAULT '[]',
            pages_deleted    JSONB NOT NULL DEFAULT '[]',
            pages_renamed    JSONB NOT NULL DEFAULT '[]',
            errors           JSONB NOT NULL DEFAULT '[]',
            started_at       TIMESTAMPTZ,
            finished_at      TIMESTAMPTZ,
            created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)

    op.execute("""
        CREATE TABLE IF NOT EXISTS wiki_pages (
            id            SERIAL PRIMARY KEY,
            path          TEXT NOT NULL UNIQUE,
            title         TEXT NOT NULL DEFAULT '',
            tags          JSONB NOT NULL DEFAULT '[]',
            summary       TEXT NOT NULL DEFAULT '',
            content       TEXT NOT NULL DEFAULT '',
            frontmatter   JSONB NOT NULL DEFAULT '{}',
            ingested_from TEXT NOT NULL DEFAULT '',
            s3_key        TEXT NOT NULL DEFAULT '',
            created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)

    # Generated tsvector column for full-text search (PostgreSQL 12+)
    op.execute("""
        ALTER TABLE wiki_pages
        ADD COLUMN IF NOT EXISTS search_vector tsvector
        GENERATED ALWAYS AS (
            to_tsvector('english',
                coalesce(title, '') || ' ' || coalesce(substr(content, 1, 50000), ''))
        ) STORED
    """)

    op.execute("""
        CREATE INDEX IF NOT EXISTS wiki_pages_search_idx ON wiki_pages USING GIN (search_vector)
    """)

    op.execute("""
        CREATE INDEX IF NOT EXISTS wiki_pages_path_idx ON wiki_pages (path)
    """)

    op.execute("""
        CREATE TABLE IF NOT EXISTS audit_log (
            id         BIGSERIAL PRIMARY KEY,
            operation  TEXT NOT NULL DEFAULT '',
            raw_text   TEXT NOT NULL DEFAULT '',
            details    JSONB NOT NULL DEFAULT '{}',
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS audit_log")
    op.execute("DROP INDEX IF EXISTS wiki_pages_path_idx")
    op.execute("DROP INDEX IF EXISTS wiki_pages_search_idx")
    op.execute("DROP TABLE IF EXISTS wiki_pages")
    op.execute("DROP TABLE IF EXISTS recalibrate_jobs")
    op.execute("DROP TABLE IF EXISTS auth_tokens")
    op.execute("DROP TABLE IF EXISTS chat_sessions")
    op.execute("DROP TABLE IF EXISTS ingest_jobs")
