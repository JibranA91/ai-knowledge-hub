"""Record embedding provenance without guessing the model of existing vectors."""
from alembic import op

revision = "028"
down_revision = "027"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Existing vectors remain untagged: their generating model cannot be inferred.
    op.execute("ALTER TABLE wiki_pages ADD COLUMN embedding_space TEXT")


def downgrade() -> None:
    op.execute("ALTER TABLE wiki_pages DROP COLUMN embedding_space")
