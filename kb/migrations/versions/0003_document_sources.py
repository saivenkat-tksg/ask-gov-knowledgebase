"""Knowledge drift tracking: where each document version came from and when it was last checked.

source is a local file path or an http(s) URL. `kb refresh` re-reads it: changed content becomes
a new version, a vanished source is flagged (source_status = 'missing'). source_etag and
source_last_modified let URL checks skip unchanged files without downloading them.

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-07
"""

from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
ALTER TABLE documents
    ADD COLUMN IF NOT EXISTS source               TEXT,
    ADD COLUMN IF NOT EXISTS source_status        TEXT,
    ADD COLUMN IF NOT EXISTS source_etag          TEXT,
    ADD COLUMN IF NOT EXISTS source_last_modified TEXT,
    ADD COLUMN IF NOT EXISTS last_checked_at      TIMESTAMPTZ""")


def downgrade() -> None:
    op.execute("""
ALTER TABLE documents
    DROP COLUMN IF EXISTS last_checked_at,
    DROP COLUMN IF EXISTS source_last_modified,
    DROP COLUMN IF EXISTS source_etag,
    DROP COLUMN IF EXISTS source_status,
    DROP COLUMN IF EXISTS source""")
