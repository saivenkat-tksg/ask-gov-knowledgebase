"""The graph no longer stores chunks: stop queueing chunk inserts and updates.

Graph extraction is per document (0006), and Neo4j keeps chunk ids only as properties, so chunk
inserts and content updates have nothing to do in the graph. Their triggers are dropped and any
still-pending 'upsert' rows are cleared. Chunk deletes are still queued (the graph forgets the
deleted chunk ids and the facts traced to them), as are truncates, document texts and version changes.

Revision ID: 0007
Revises: 0006
Create Date: 2026-10-12
"""

from alembic import op

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
DROP TRIGGER IF EXISTS chunks_graph_insert ON chunks;
DROP TRIGGER IF EXISTS chunks_graph_update ON chunks;
UPDATE graph_outbox SET done_at = now(), error = 'chunk rows are no longer written to the graph (0007)'
WHERE op = 'upsert' AND done_at IS NULL;
""")


def downgrade() -> None:
    op.execute("""
CREATE TRIGGER chunks_graph_insert AFTER INSERT ON chunks
    FOR EACH ROW EXECUTE FUNCTION kb_graph_queue();
CREATE TRIGGER chunks_graph_update AFTER UPDATE OF content, metadata ON chunks
    FOR EACH ROW EXECUTE FUNCTION kb_graph_queue();
""")
