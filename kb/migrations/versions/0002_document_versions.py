"""Document versioning: doc_key groups the versions of one document; only one is current.

Re-uploading changed content under the same doc_key adds a new version and supersedes the old
one. valid_from / valid_until limit when a version is searchable; review_by is a reminder date.
Written with IF NOT EXISTS so schemas that already have these columns are adopted as-is.

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-06
"""

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS doc_key TEXT")
    op.execute("UPDATE documents SET doc_key = filename WHERE doc_key IS NULL")
    op.execute("ALTER TABLE documents ALTER COLUMN doc_key SET NOT NULL")
    op.execute("""
ALTER TABLE documents
    ADD COLUMN IF NOT EXISTS version       INT NOT NULL DEFAULT 1,
    ADD COLUMN IF NOT EXISTS is_current    BOOLEAN NOT NULL DEFAULT true,
    ADD COLUMN IF NOT EXISTS superseded_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS valid_from    DATE,
    ADD COLUMN IF NOT EXISTS valid_until   DATE,
    ADD COLUMN IF NOT EXISTS review_by     DATE""")
    # Before this migration the same filename could be uploaded twice; keep only the newest current.
    op.execute("""
UPDATE documents d SET is_current = false, superseded_at = now()
FROM (
    SELECT id, row_number() OVER (PARTITION BY doc_key ORDER BY created_at DESC) AS rn
    FROM documents WHERE is_current
) newest
WHERE d.id = newest.id AND newest.rn > 1""")
    op.execute("CREATE INDEX IF NOT EXISTS documents_doc_key_idx ON documents (doc_key)")
    op.execute("CREATE UNIQUE INDEX IF NOT EXISTS documents_current_key_idx ON documents (doc_key) WHERE is_current")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS documents_current_key_idx")
    op.execute("DROP INDEX IF EXISTS documents_doc_key_idx")
    op.execute("""
ALTER TABLE documents
    DROP COLUMN IF EXISTS review_by,
    DROP COLUMN IF EXISTS valid_until,
    DROP COLUMN IF EXISTS valid_from,
    DROP COLUMN IF EXISTS superseded_at,
    DROP COLUMN IF EXISTS is_current,
    DROP COLUMN IF EXISTS version,
    DROP COLUMN IF EXISTS doc_key""")
