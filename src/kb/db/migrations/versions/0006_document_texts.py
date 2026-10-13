"""Document-level graph extraction: keep each uploaded document's full text and queue it for the LLM.

document_texts holds the text the loader read from the uploaded file (PDF pages with tables
rebuilt and headings marked, OCR for scanned pages), one row per document version, plus a page map
(character offset -> page). Inserting it queues an 'extract' row in graph_outbox, so `kb graph
worker` extracts entities and relations from the whole document rather than chunk by chunk.
Chunk inserts still queue 'upsert' rows, which now only write the document/section/chunk structure.

The row is deleted with its document (ON DELETE CASCADE).

Revision ID: 0006
Revises: 0005
Create Date: 2026-10-12
"""

from alembic import op

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
CREATE TABLE IF NOT EXISTS document_texts (
    document_id  UUID PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
    content      TEXT NOT NULL,
    page_map     JSONB NOT NULL DEFAULT '[]'::jsonb,   -- [[char_offset, page], ...]
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE graph_outbox DROP CONSTRAINT IF EXISTS graph_outbox_op_check;
ALTER TABLE graph_outbox ADD CONSTRAINT graph_outbox_op_check
    CHECK (op IN ('upsert', 'delete', 'document', 'truncate', 'extract'));

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
    ELSIF TG_TABLE_NAME = 'document_texts' THEN
        EXECUTE format('INSERT INTO %I.graph_outbox (op, document_id) VALUES (''extract'', $1)', TG_TABLE_SCHEMA)
            USING NEW.document_id;
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

DROP TRIGGER IF EXISTS document_texts_graph_extract ON document_texts;
CREATE TRIGGER document_texts_graph_extract AFTER INSERT OR UPDATE OF content ON document_texts
    FOR EACH ROW EXECUTE FUNCTION kb_graph_queue();
""")


def downgrade() -> None:
    op.execute("""
DROP TRIGGER IF EXISTS document_texts_graph_extract ON document_texts;
DROP TABLE IF EXISTS document_texts;
DELETE FROM graph_outbox WHERE op = 'extract';
ALTER TABLE graph_outbox DROP CONSTRAINT IF EXISTS graph_outbox_op_check;
ALTER TABLE graph_outbox ADD CONSTRAINT graph_outbox_op_check
    CHECK (op IN ('upsert', 'delete', 'document', 'truncate'));
""")
