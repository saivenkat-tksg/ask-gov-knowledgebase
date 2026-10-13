"""Knowledge graph sync: every chunk change is queued in graph_outbox for `kb graph worker`.

Triggers on chunks record inserts and content updates ('upsert'), deletes ('delete', including
rows removed by ON DELETE CASCADE from documents and deletes made by hand in SQL) and truncates
('truncate'); a trigger on documents records version changes ('document', when is_current flips).
The queue row is written in the same transaction as the change, so a rolled-back change queues
nothing, and pg_notify('kb_graph', <schema>) wakes the worker at once.

Nothing is queued until `kb graph init` sets graph_state.enabled (and back-fills existing chunks),
so schemas without a graph don't accumulate queue rows. The trigger functions qualify every table
with TG_TABLE_SCHEMA, so they work whatever search_path the deleting session uses.

Revision ID: 0005
Revises: 0004
Create Date: 2026-10-12
"""

from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
CREATE TABLE IF NOT EXISTS graph_state (
    id       BOOLEAN PRIMARY KEY DEFAULT true CHECK (id),
    enabled  BOOLEAN NOT NULL DEFAULT false
);
INSERT INTO graph_state (id, enabled) VALUES (true, false) ON CONFLICT (id) DO NOTHING;

CREATE TABLE IF NOT EXISTS graph_outbox (
    id            BIGSERIAL PRIMARY KEY,
    op            TEXT NOT NULL CHECK (op IN ('upsert', 'delete', 'document', 'truncate')),
    chunk_id      UUID,
    document_id   UUID,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    locked_until  TIMESTAMPTZ,
    done_at       TIMESTAMPTZ,
    attempts      INT NOT NULL DEFAULT 0,
    error         TEXT
);
CREATE INDEX IF NOT EXISTS graph_outbox_pending_idx ON graph_outbox (id) WHERE done_at IS NULL;

CREATE OR REPLACE FUNCTION kb_graph_queue() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE
    on_ boolean;
BEGIN
    EXECUTE format('SELECT enabled FROM %I.graph_state', TG_TABLE_SCHEMA) INTO on_;
    IF NOT coalesce(on_, false) THEN
        RETURN NULL;
    END IF;
    IF TG_LEVEL = 'STATEMENT' THEN  -- TRUNCATE chunks
        EXECUTE format('INSERT INTO %I.graph_outbox (op) VALUES (''truncate'')', TG_TABLE_SCHEMA);
    ELSIF TG_TABLE_NAME = 'documents' THEN
        EXECUTE format('INSERT INTO %I.graph_outbox (op, document_id) VALUES (''document'', $1)', TG_TABLE_SCHEMA)
            USING NEW.id;
    ELSIF TG_OP = 'DELETE' THEN
        EXECUTE format('INSERT INTO %I.graph_outbox (op, chunk_id, document_id) VALUES (''delete'', $1, $2)',
                       TG_TABLE_SCHEMA) USING OLD.id, OLD.document_id;
    ELSE  -- INSERT, or UPDATE of the chunk's content/metadata
        EXECUTE format('INSERT INTO %I.graph_outbox (op, chunk_id, document_id) VALUES (''upsert'', $1, $2)',
                       TG_TABLE_SCHEMA) USING NEW.id, NEW.document_id;
    END IF;
    PERFORM pg_notify('kb_graph', TG_TABLE_SCHEMA);
    RETURN NULL;
END $$;

DROP TRIGGER IF EXISTS chunks_graph_insert ON chunks;
CREATE TRIGGER chunks_graph_insert AFTER INSERT ON chunks
    FOR EACH ROW EXECUTE FUNCTION kb_graph_queue();
DROP TRIGGER IF EXISTS chunks_graph_update ON chunks;
CREATE TRIGGER chunks_graph_update AFTER UPDATE OF content, metadata ON chunks
    FOR EACH ROW EXECUTE FUNCTION kb_graph_queue();
DROP TRIGGER IF EXISTS chunks_graph_delete ON chunks;
CREATE TRIGGER chunks_graph_delete AFTER DELETE ON chunks
    FOR EACH ROW EXECUTE FUNCTION kb_graph_queue();
DROP TRIGGER IF EXISTS chunks_graph_truncate ON chunks;
CREATE TRIGGER chunks_graph_truncate AFTER TRUNCATE ON chunks
    FOR EACH STATEMENT EXECUTE FUNCTION kb_graph_queue();
DROP TRIGGER IF EXISTS documents_graph_current ON documents;
CREATE TRIGGER documents_graph_current AFTER UPDATE OF is_current ON documents
    FOR EACH ROW WHEN (OLD.is_current IS DISTINCT FROM NEW.is_current) EXECUTE FUNCTION kb_graph_queue();
""")


def downgrade() -> None:
    op.execute("""
DROP TRIGGER IF EXISTS documents_graph_current ON documents;
DROP TRIGGER IF EXISTS chunks_graph_truncate ON chunks;
DROP TRIGGER IF EXISTS chunks_graph_delete ON chunks;
DROP TRIGGER IF EXISTS chunks_graph_update ON chunks;
DROP TRIGGER IF EXISTS chunks_graph_insert ON chunks;
DROP FUNCTION IF EXISTS kb_graph_queue();
DROP TABLE IF EXISTS graph_outbox;
DROP TABLE IF EXISTS graph_state;
""")
