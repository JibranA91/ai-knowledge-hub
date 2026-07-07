"""Backfill `type:` frontmatter field for existing wiki pages.

Path-prefix → type mapping (mirrors app.utils._TYPE_BY_PREFIX):
  sources/   → source_summary
  concepts/  → concept
  entities/  → entity
  rca/       → rca
  queries/   → query_result

For each existing wiki page that does NOT already have a `type:` line inside
its YAML frontmatter, inject one based on its path prefix. Pages outside the
five known folders are left untouched.

Revision ID: 014
Revises: 013
Create Date: 2026-05-16
"""
from typing import Sequence, Union

from alembic import op


revision: str = "014"
down_revision: Union[str, None] = "013"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# One UPDATE per (path_prefix, type) pair. Uses a regex check to only touch
# pages whose frontmatter block does not already declare a type, then injects
# the field right after the opening `---\n`.
_PREFIX_TO_TYPE = (
    ("sources/",  "source_summary"),
    ("concepts/", "concept"),
    ("entities/", "entity"),
    ("rca/",      "rca"),
    ("queries/",  "query_result"),
)


def upgrade() -> None:
    for prefix, type_name in _PREFIX_TO_TYPE:
        # Match pages whose path starts with the prefix AND whose existing
        # content does NOT already contain `type: <name>` within the leading
        # YAML frontmatter block. Inject the field right after the first `---`
        # line. Pages without any frontmatter block are skipped (regex_replace
        # only matches when the opening `---` is present).
        op.execute(
            f"""
            UPDATE wiki_pages
            SET content = regexp_replace(
                content,
                E'^---[ \\t]*\\n',
                E'---\\ntype: {type_name}\\n',
                'n'
            ),
            frontmatter = jsonb_set(
                COALESCE(frontmatter, '{{}}'::jsonb),
                '{{type}}',
                to_jsonb('{type_name}'::text),
                true
            ),
            updated_at = NOW()
            WHERE path LIKE '{prefix}%'
              AND content ~ E'^---[ \\t]*\\n'
              AND content !~ E'^---[ \\t]*\\n(?:.|\\n)*?\\ntype:\\s*'
            """
        )


def downgrade() -> None:
    # Reverse the injection: remove the `type:` line we added from frontmatter
    # blocks of pages under the five known prefixes. We can't tell which pages
    # had a manual `type:` vs an injected one, so this best-effort strips any
    # type line that matches our injected values from the relevant paths.
    for prefix, type_name in _PREFIX_TO_TYPE:
        op.execute(
            f"""
            UPDATE wiki_pages
            SET content = regexp_replace(
                content,
                E'\\ntype:\\s*{type_name}\\n',
                E'\\n',
                'n'
            ),
            frontmatter = frontmatter - 'type',
            updated_at = NOW()
            WHERE path LIKE '{prefix}%'
            """
        )
