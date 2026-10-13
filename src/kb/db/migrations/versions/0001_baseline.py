"""Baseline: documents and chunks tables with vector, JSONB and full-text indexes.

Written with IF NOT EXISTS so databases created by the old `kb init` are adopted as-is.

Revision ID: 0001
Revises:
Create Date: 2026-10-06
"""

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    dim = int(op.get_context().config.attributes["settings"].embedding_dim)
    op.execute("CREATE EXTENSION IF NOT EXISTS vector SCHEMA public")
    op.execute("""
CREATE TABLE IF NOT EXISTS documents (
    id            UUID PRIMARY KEY,
    filename      TEXT NOT NULL,
    file_type     TEXT NOT NULL,
    content_hash  TEXT NOT NULL UNIQUE,
    size_bytes    BIGINT NOT NULL,
    metadata      JSONB NOT NULL DEFAULT '{}'::jsonb,
    num_chunks    INT NOT NULL DEFAULT 0,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
)""")
    op.execute("CREATE INDEX IF NOT EXISTS documents_filename_idx ON documents (filename)")
    op.execute(f"""
CREATE TABLE IF NOT EXISTS chunks (
    id           UUID PRIMARY KEY,
    document_id  UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index  INT NOT NULL,
    content      TEXT NOT NULL,
    page         INT,
    char_start   INT,
    char_end     INT,
    metadata     JSONB NOT NULL DEFAULT '{{}}'::jsonb,
    embedding    vector({dim}) NOT NULL
)""")
    op.execute("CREATE INDEX IF NOT EXISTS chunks_document_idx ON chunks (document_id)")
    op.execute("CREATE INDEX IF NOT EXISTS chunks_metadata_idx ON chunks USING gin (metadata jsonb_path_ops)")
    op.execute("CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops)")
    # Keyword side of hybrid search. Generated, so existing rows are indexed without re-ingesting.
    op.execute("""
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS content_tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('english', content)) STORED""")
    op.execute("CREATE INDEX IF NOT EXISTS chunks_content_tsv_idx ON chunks USING gin (content_tsv)")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS chunks")
    op.execute("DROP TABLE IF EXISTS documents")
