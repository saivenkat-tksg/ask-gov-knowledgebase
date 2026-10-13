"""Readable chunk labels: chunks.labelchunk holds e.g. 'cash_grant_chunk_0'.

The label is the one `kb eval` matches `chunk_id` against (kb.ingestion.labels.chunk_label), stored so it
can be looked up in SQL. Existing chunks are backfilled from their document's filename.

Revision ID: 0004
Revises: 0003
Create Date: 2026-10-08
"""

import re
from pathlib import Path

from alembic import op
from sqlalchemy import text

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def chunk_label(filename: str, chunk_index: int) -> str:
    """Frozen copy of kb.ingestion.labels.chunk_label, so later code changes can't alter this backfill."""
    stem = re.sub(r"\(\d+\)", "", Path(filename).stem)
    key = re.sub(r"\W+", "_", stem).strip("_")
    return f"{key}_chunk_{chunk_index}"


def upgrade() -> None:
    op.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS labelchunk TEXT")
    conn = op.get_bind()
    rows = conn.execute(text(
        "SELECT c.id, d.filename, c.chunk_index FROM chunks c JOIN documents d ON d.id = c.document_id "
        "WHERE c.labelchunk IS NULL"
    )).fetchall()
    for chunk_id, filename, chunk_index in rows:
        conn.execute(text("UPDATE chunks SET labelchunk = :label WHERE id = :id"),
                     {"label": chunk_label(filename, chunk_index), "id": chunk_id})
    op.execute("CREATE INDEX IF NOT EXISTS chunks_labelchunk_idx ON chunks (labelchunk)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS chunks_labelchunk_idx")
    op.execute("ALTER TABLE chunks DROP COLUMN IF EXISTS labelchunk")
