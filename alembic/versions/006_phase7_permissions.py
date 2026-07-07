"""Phase 7: Admin dashboard + permission templates + workspaces.

Adds granular per-user access control via reusable permission templates,
workspace isolation, user suspension, and org-level usage limits.

Revision ID: 006
Revises: 005
Create Date: 2026-05-07
"""
from typing import Sequence, Union

from alembic import op

revision: str = "006"
down_revision: Union[str, None] = "005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_DEFAULT_ORG_ID = "00000000-0000-0000-0000-000000000001"

# Deterministic UUIDs for built-in templates (seeded for default org only;
# new orgs get their own copies via seed_builtin_templates()).
_TPL_READ_ONLY   = "00000000-0000-0000-0000-000000000010"
_TPL_CONTRIBUTOR = "00000000-0000-0000-0000-000000000011"
_TPL_POWER_USER  = "00000000-0000-0000-0000-000000000012"
_TPL_ADMIN       = "00000000-0000-0000-0000-000000000013"


def upgrade() -> None:
    # ── permission_templates ──────────────────────────────────────────────
    op.execute("""
        CREATE TABLE IF NOT EXISTS permission_templates (
            id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            org_id      UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
            name        TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            is_builtin  BOOL NOT NULL DEFAULT false,

            -- File operations
            can_upload          BOOL NOT NULL DEFAULT false,
            can_delete_files    BOOL NOT NULL DEFAULT false,
            can_download_files  BOOL NOT NULL DEFAULT true,
            max_upload_size_mb  INT,
            max_uploads_per_day  INT,
            max_uploads_per_week INT,

            -- Wiki operations
            can_view_wiki        BOOL NOT NULL DEFAULT true,
            can_edit_wiki        BOOL NOT NULL DEFAULT false,
            can_delete_wiki_pages BOOL NOT NULL DEFAULT false,

            -- LLM operations
            can_query                BOOL NOT NULL DEFAULT false,
            can_chat                 BOOL NOT NULL DEFAULT false,
            max_queries_per_day      INT,
            max_chat_messages_per_day INT,

            -- Token budget (NULL = unlimited)
            max_tokens_per_day  INT,
            max_tokens_per_week INT,

            -- Platform operations
            can_recalibrate      BOOL NOT NULL DEFAULT false,
            can_run_lint         BOOL NOT NULL DEFAULT false,
            can_manage_schema    BOOL NOT NULL DEFAULT false,
            can_view_audit_log   BOOL NOT NULL DEFAULT false,
            can_view_graph       BOOL NOT NULL DEFAULT true,
            can_rebuild_graph    BOOL NOT NULL DEFAULT false,
            can_manage_workspace BOOL NOT NULL DEFAULT false,
            can_approve_ingest   BOOL NOT NULL DEFAULT false,
            can_cancel_ingest    BOOL NOT NULL DEFAULT false,

            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

            UNIQUE (org_id, name)
        )
    """)

    op.execute("""
        CREATE INDEX IF NOT EXISTS permission_templates_org_idx
        ON permission_templates (org_id)
    """)

    # ── workspaces ────────────────────────────────────────────────────────
    op.execute("""
        CREATE TABLE IF NOT EXISTS workspaces (
            id                         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            org_id                     UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
            name                       TEXT NOT NULL,
            owner_user_id              UUID REFERENCES users(id) ON DELETE SET NULL,
            s3_prefix                  TEXT NOT NULL DEFAULT '',
            created_from_workspace_id  UUID REFERENCES workspaces(id) ON DELETE SET NULL,
            created_at                 TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE (org_id, name)
        )
    """)

    op.execute("""
        CREATE INDEX IF NOT EXISTS workspaces_org_idx ON workspaces (org_id)
    """)

    # ── Extend users table ────────────────────────────────────────────────
    op.execute("""
        ALTER TABLE users
        ADD COLUMN IF NOT EXISTS permission_template_id UUID
            REFERENCES permission_templates(id) ON DELETE SET NULL
    """)
    op.execute("""
        ALTER TABLE users
        ADD COLUMN IF NOT EXISTS workspace_id UUID
            REFERENCES workspaces(id) ON DELETE SET NULL
    """)
    op.execute("""
        ALTER TABLE users
        ADD COLUMN IF NOT EXISTS is_suspended BOOL NOT NULL DEFAULT false
    """)

    # ── Add user_id to ingest_jobs for per-user quota tracking ───────────
    op.execute("""
        ALTER TABLE ingest_jobs ADD COLUMN IF NOT EXISTS user_id UUID
    """)

    # ── Org-level limits on organizations ────────────────────────────────
    op.execute("""
        ALTER TABLE organizations
        ADD COLUMN IF NOT EXISTS max_uploads_per_day_org INT
    """)
    op.execute("""
        ALTER TABLE organizations
        ADD COLUMN IF NOT EXISTS max_tokens_per_day_org INT
    """)
    op.execute("""
        ALTER TABLE organizations
        ADD COLUMN IF NOT EXISTS max_members INT
    """)

    # ── Seed built-in permission templates for the default org ───────────

    # read_only: view wiki + graph only
    op.execute(f"""
        INSERT INTO permission_templates
            (id, org_id, name, description, is_builtin,
             can_upload, can_delete_files, can_download_files,
             can_view_wiki, can_edit_wiki, can_delete_wiki_pages,
             can_query, can_chat,
             can_recalibrate, can_run_lint, can_manage_schema,
             can_view_audit_log, can_view_graph, can_rebuild_graph,
             can_manage_workspace, can_approve_ingest, can_cancel_ingest)
        VALUES
            ('{_TPL_READ_ONLY}', '{_DEFAULT_ORG_ID}',
             'read_only', 'View-only access to wiki and graph', true,
             false, false, false,
             true, false, false,
             false, false,
             false, false, false,
             false, true, false,
             false, false, false)
        ON CONFLICT (org_id, name) DO NOTHING
    """)

    # contributor: upload, query, chat — no admin ops
    op.execute(f"""
        INSERT INTO permission_templates
            (id, org_id, name, description, is_builtin,
             can_upload, can_delete_files, can_download_files,
             can_view_wiki, can_edit_wiki, can_delete_wiki_pages,
             can_query, can_chat,
             max_uploads_per_day, max_chat_messages_per_day,
             can_recalibrate, can_run_lint, can_manage_schema,
             can_view_audit_log, can_view_graph, can_rebuild_graph,
             can_manage_workspace, can_approve_ingest, can_cancel_ingest)
        VALUES
            ('{_TPL_CONTRIBUTOR}', '{_DEFAULT_ORG_ID}',
             'contributor', 'Upload documents, query and chat with the wiki', true,
             true, false, true,
             true, false, false,
             true, true,
             20, 200,
             false, false, false,
             false, true, false,
             false, true, true)
        ON CONFLICT (org_id, name) DO NOTHING
    """)

    # power_user: everything except recalibrate
    op.execute(f"""
        INSERT INTO permission_templates
            (id, org_id, name, description, is_builtin,
             can_upload, can_delete_files, can_download_files,
             can_view_wiki, can_edit_wiki, can_delete_wiki_pages,
             can_query, can_chat,
             can_recalibrate, can_run_lint, can_manage_schema,
             can_view_audit_log, can_view_graph, can_rebuild_graph,
             can_manage_workspace, can_approve_ingest, can_cancel_ingest)
        VALUES
            ('{_TPL_POWER_USER}', '{_DEFAULT_ORG_ID}',
             'power_user', 'Full access except recalibration', true,
             true, true, true,
             true, true, true,
             true, true,
             false, true, true,
             true, true, true,
             true, true, true)
        ON CONFLICT (org_id, name) DO NOTHING
    """)

    # admin_template: full access, no quotas
    op.execute(f"""
        INSERT INTO permission_templates
            (id, org_id, name, description, is_builtin,
             can_upload, can_delete_files, can_download_files,
             can_view_wiki, can_edit_wiki, can_delete_wiki_pages,
             can_query, can_chat,
             can_recalibrate, can_run_lint, can_manage_schema,
             can_view_audit_log, can_view_graph, can_rebuild_graph,
             can_manage_workspace, can_approve_ingest, can_cancel_ingest)
        VALUES
            ('{_TPL_ADMIN}', '{_DEFAULT_ORG_ID}',
             'admin', 'Full platform access — all features enabled, no quotas', true,
             true, true, true,
             true, true, true,
             true, true,
             true, true, true,
             true, true, true,
             true, true, true)
        ON CONFLICT (org_id, name) DO NOTHING
    """)


def downgrade() -> None:
    op.execute("ALTER TABLE organizations DROP COLUMN IF EXISTS max_members")
    op.execute("ALTER TABLE organizations DROP COLUMN IF EXISTS max_tokens_per_day_org")
    op.execute("ALTER TABLE organizations DROP COLUMN IF EXISTS max_uploads_per_day_org")
    op.execute("ALTER TABLE ingest_jobs DROP COLUMN IF EXISTS user_id")
    op.execute("ALTER TABLE users DROP COLUMN IF EXISTS is_suspended")
    op.execute("ALTER TABLE users DROP COLUMN IF EXISTS workspace_id")
    op.execute("ALTER TABLE users DROP COLUMN IF EXISTS permission_template_id")
    op.execute("DROP INDEX IF EXISTS workspaces_org_idx")
    op.execute("DROP TABLE IF EXISTS workspaces")
    op.execute("DROP INDEX IF EXISTS permission_templates_org_idx")
    op.execute("DROP TABLE IF EXISTS permission_templates")
