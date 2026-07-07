"""Remove the "supervisor" built-in permission template.

Supervisor role bypasses all permission checks implicitly (same as admin),
so the template is redundant. Users assigned to it get
permission_template_id = NULL (ON DELETE SET NULL already in place).

Revision ID: 009
Revises: 008
Create Date: 2026-05-08
"""
from typing import Sequence, Union

from alembic import op

revision: str = "009"
down_revision: Union[str, None] = "008"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        DELETE FROM permission_templates
        WHERE is_builtin = true AND name = 'supervisor'
    """)


def downgrade() -> None:
    # Re-insert the supervisor template for each org that has none
    op.execute("""
        INSERT INTO permission_templates
            (org_id, name, description, is_builtin,
             can_upload, can_delete_files, can_download_files,
             can_view_wiki, can_edit_wiki, can_delete_wiki_pages,
             can_query, can_chat,
             can_recalibrate, can_run_lint, can_manage_schema,
             can_view_audit_log, can_view_graph, can_rebuild_graph,
             can_manage_workspace, can_approve_ingest, can_cancel_ingest)
        SELECT DISTINCT o.id,
            'supervisor',
            'Org-level full access — all features, manages users and workspaces',
            true,
            true, true, true, true, true, true, true, true,
            true, true, true, true, true, true, true, true, true
        FROM organizations o
        WHERE NOT EXISTS (
            SELECT 1 FROM permission_templates pt
            WHERE pt.org_id = o.id AND pt.name = 'supervisor'
        )
    """)
