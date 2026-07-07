"""Phase 6: Multi-tenancy — organizations, users, usage_log, org_id on all tables.

Revision ID: 004
Revises: 003
Create Date: 2026-05-06
"""
from typing import Sequence, Union
from alembic import op

revision: str = "004"
down_revision: Union[str, None] = "003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Deterministic UUID for the seed "default" org so migrations are idempotent.
_DEFAULT_ORG_ID = "00000000-0000-0000-0000-000000000001"


def upgrade() -> None:
    # ── New tables ────────────────────────────────────────────────────────

    op.execute("""
        CREATE TABLE IF NOT EXISTS organizations (
            id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            name       TEXT NOT NULL UNIQUE,
            slug       TEXT NOT NULL UNIQUE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)

    op.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            org_id         UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
            email          TEXT NOT NULL UNIQUE,
            password_hash  TEXT NOT NULL DEFAULT '',
            role           TEXT NOT NULL DEFAULT 'member'
                               CHECK (role IN ('admin', 'member')),
            created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            last_login_at  TIMESTAMPTZ
        )
    """)

    op.execute("""
        CREATE INDEX IF NOT EXISTS users_org_idx ON users (org_id)
    """)

    op.execute("""
        CREATE TABLE IF NOT EXISTS usage_log (
            id         BIGSERIAL PRIMARY KEY,
            org_id     UUID NOT NULL,
            user_id    UUID,
            model_id   TEXT NOT NULL DEFAULT '',
            tokens_in  INT  NOT NULL DEFAULT 0,
            tokens_out INT  NOT NULL DEFAULT 0,
            operation  TEXT NOT NULL DEFAULT '',
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)

    op.execute("""
        CREATE INDEX IF NOT EXISTS usage_log_org_idx ON usage_log (org_id, created_at DESC)
    """)

    # ── Seed the default org (idempotent) ─────────────────────────────────
    op.execute(f"""
        INSERT INTO organizations (id, name, slug)
        VALUES ('{_DEFAULT_ORG_ID}', 'Default Organization', 'default')
        ON CONFLICT (id) DO NOTHING
    """)

    # ── Add org_id to existing tables (nullable first, then fill, then NOT NULL) ──

    for table in ("wiki_pages", "ingest_jobs", "audit_log",
                  "recalibrate_jobs", "chat_sessions"):
        op.execute(f"""
            ALTER TABLE {table}
            ADD COLUMN IF NOT EXISTS org_id UUID
        """)
        op.execute(f"""
            UPDATE {table} SET org_id = '{_DEFAULT_ORG_ID}' WHERE org_id IS NULL
        """)
        op.execute(f"""
            ALTER TABLE {table} ALTER COLUMN org_id SET NOT NULL
        """)
        op.execute(f"""
            ALTER TABLE {table} ALTER COLUMN org_id SET DEFAULT '{_DEFAULT_ORG_ID}'
        """)

    # ── wiki_files: change PK from (key) → (org_id, key) ─────────────────

    op.execute("ALTER TABLE wiki_files ADD COLUMN IF NOT EXISTS org_id UUID")
    op.execute(f"UPDATE wiki_files SET org_id = '{_DEFAULT_ORG_ID}' WHERE org_id IS NULL")
    op.execute("ALTER TABLE wiki_files ALTER COLUMN org_id SET NOT NULL")
    op.execute(f"ALTER TABLE wiki_files ALTER COLUMN org_id SET DEFAULT '{_DEFAULT_ORG_ID}'")

    # Drop old PK and replace with composite
    op.execute("ALTER TABLE wiki_files DROP CONSTRAINT IF EXISTS wiki_files_pkey")
    op.execute("ALTER TABLE wiki_files ADD PRIMARY KEY (org_id, key)")

    # ── auth_tokens: add org_id + user_id for new-style tokens ───────────

    op.execute("ALTER TABLE auth_tokens ADD COLUMN IF NOT EXISTS org_id UUID")
    op.execute("ALTER TABLE auth_tokens ADD COLUMN IF NOT EXISTS user_id UUID")

    # ── Update unique constraints for tenant isolation ────────────────────

    # wiki_pages: (path) → (org_id, path)
    op.execute("ALTER TABLE wiki_pages DROP CONSTRAINT IF EXISTS wiki_pages_path_key")
    op.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS wiki_pages_org_path_idx
        ON wiki_pages (org_id, path)
    """)
    # Drop the old path-only unique index if present
    op.execute("DROP INDEX IF EXISTS wiki_pages_path_idx")
    op.execute("""
        CREATE INDEX IF NOT EXISTS wiki_pages_path_idx ON wiki_pages (org_id, path)
    """)

    # ingest_jobs: (filename) → (org_id, filename)
    op.execute("ALTER TABLE ingest_jobs DROP CONSTRAINT IF EXISTS ingest_jobs_filename_key")
    op.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS ingest_jobs_org_filename_idx
        ON ingest_jobs (org_id, filename)
    """)

    # ── Create indexes for org_id FTS on wiki_pages ───────────────────────
    op.execute("""
        CREATE INDEX IF NOT EXISTS wiki_pages_org_idx ON wiki_pages (org_id)
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS audit_log_org_idx ON audit_log (org_id)
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS chat_sessions_org_idx ON chat_sessions (org_id)
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS chat_sessions_org_idx")
    op.execute("DROP INDEX IF EXISTS audit_log_org_idx")
    op.execute("DROP INDEX IF EXISTS wiki_pages_org_idx")
    op.execute("DROP INDEX IF EXISTS ingest_jobs_org_filename_idx")
    op.execute("DROP INDEX IF EXISTS wiki_pages_org_path_idx")

    for table in ("wiki_pages", "ingest_jobs", "audit_log",
                  "recalibrate_jobs", "chat_sessions"):
        op.execute(f"ALTER TABLE {table} DROP COLUMN IF EXISTS org_id")

    op.execute("ALTER TABLE auth_tokens DROP COLUMN IF EXISTS user_id")
    op.execute("ALTER TABLE auth_tokens DROP COLUMN IF EXISTS org_id")

    # Restore wiki_files PK (best effort)
    op.execute("ALTER TABLE wiki_files DROP CONSTRAINT IF EXISTS wiki_files_pkey")
    op.execute("ALTER TABLE wiki_files DROP COLUMN IF EXISTS org_id")
    op.execute("ALTER TABLE wiki_files ADD PRIMARY KEY (key)")

    op.execute("DROP INDEX IF EXISTS usage_log_org_idx")
    op.execute("DROP TABLE IF EXISTS usage_log")
    op.execute("DROP INDEX IF EXISTS users_org_idx")
    op.execute("DROP TABLE IF EXISTS users")
    op.execute("DROP TABLE IF EXISTS organizations")
