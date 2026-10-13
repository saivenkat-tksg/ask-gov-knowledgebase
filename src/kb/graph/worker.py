"""`kb graph worker`: apply the Postgres change queue (graph_outbox) to Neo4j.

Flow per batch:
  1. claim pending rows (a lease, so several workers or a crashed one never double-process)
  2. 'truncate'  -> wipe this knowledge base's graph
     'delete'    -> forget the deleted chunks (facts traced to them, ids in mention lists); a
                    document that no longer exists is removed with all of its facts
     'extract'   -> LLM-extract entities/relations from the WHOLE document text (kb.graph.document),
                    trace them to chunks, replace the document's facts. Chunks are never stored in
                    the graph as nodes; only their ids, as properties.
     'upsert'    -> rows from before migration 0007 (chunk-wise writes): marked done, nothing else
     'document'  -> copy the document's is_current to its graph nodes and facts
  3. mark rows done; a failed row records its error and is retried later (up to max_attempts)

The worker waits on LISTEN kb_graph (the triggers NOTIFY on every change) and also polls, so a
missed notification only delays work. `sync` reconciles the two stores in full (for anything the
queue missed, e.g. rows changed while the graph was disabled); `rebuild` starts the graph over.
"""

import logging
import time

import psycopg
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from kb.graph.document import ChunkText, extract_document
from kb.graph.extract import LLMExtractor, _normal_text
from kb.graph.resolve import Resolver
from kb.graph.store import DocumentRecord, GraphStore

log = logging.getLogger(__name__)

_CLAIM = """
UPDATE graph_outbox SET locked_until = now() + make_interval(secs => %(lease)s), attempts = attempts + 1
WHERE id IN (
    SELECT id FROM graph_outbox
    WHERE done_at IS NULL AND attempts < %(max_attempts)s AND (locked_until IS NULL OR locked_until < now())
    ORDER BY id LIMIT %(limit)s
    FOR UPDATE SKIP LOCKED)
RETURNING id, op, chunk_id::text, document_id::text"""

class GraphWorker:
    def __init__(self, pool: ConnectionPool, store: GraphStore, extractor: LLMExtractor, resolver: Resolver,
                 batch_size: int = 20, concurrency: int = 4, lease_seconds: int = 1800, max_attempts: int = 5,
                 window_tokens: int = 20000, length=None):
        self.pool = pool
        self.store = store
        self.extractor = extractor
        self.resolver = resolver
        self.batch_size = batch_size
        self.concurrency = concurrency
        self.lease_seconds = lease_seconds
        self.max_attempts = max_attempts
        self.window_tokens = window_tokens
        self.length = length or len

    # --- one batch ---------------------------------------------------------------------------------

    def process_once(self) -> int:
        """Process one batch of queued changes. Returns how many queue rows were claimed."""
        with self.pool.connection() as conn:
            rows = conn.execute(_CLAIM, {"lease": self.lease_seconds, "max_attempts": self.max_attempts,
                                         "limit": self.batch_size}).fetchall()
        if not rows:
            return 0
        rows.sort(key=lambda r: r[0])
        by_op: dict[str, list[tuple]] = {}
        for row in rows:
            by_op.setdefault(row[1], []).append(row)

        if "truncate" in by_op:
            self._run(by_op["truncate"], self.store.wipe)
        if "delete" in by_op:
            ids = [r[2] for r in by_op["delete"]]
            docs = sorted({r[3] for r in by_op["delete"] if r[3]})
            self._run(by_op["delete"], lambda: self._delete(ids, docs))
        if "upsert" in by_op:  # queued before 0007; chunks are no longer written one by one
            self._done([r[0] for r in by_op["upsert"]])
        for row in by_op.get("extract", []):
            self._run([row], lambda r=row: self._extract(r[3]))
        for row in by_op.get("document", []):
            self._run([row], lambda r=row: self._document(r[3]))
        return len(rows)

    def _run(self, rows: list[tuple], fn) -> None:
        try:
            fn()
        except Exception as exc:
            log.warning("graph %s failed: %s", rows[0][1], exc)
            self._failed([r[0] for r in rows], exc)
        else:
            self._done([r[0] for r in rows])

    def _delete(self, chunk_ids: list[str], document_ids: list[str]) -> None:
        self.store.delete_chunks(chunk_ids)
        with self.pool.connection() as conn:
            alive = {r[0] for r in conn.execute("SELECT id::text FROM documents WHERE id = ANY(%s::uuid[])",
                                                [document_ids]).fetchall()}
        gone = [d for d in document_ids if d not in alive]
        if gone:  # the document itself was deleted: its node and remaining facts go too
            self.store.delete_documents(gone)

    def _extract(self, document_id: str) -> None:
        with self.pool.connection() as conn:
            doc = conn.execute("SELECT t.content, t.page_map, d.filename, d.version, d.is_current, d.doc_key "
                               "FROM document_texts t JOIN documents d ON d.id = t.document_id "
                               "WHERE t.document_id = %s", [document_id]).fetchone()
            if doc is None:
                return  # document deleted since: its delete rows clean up
            chunks = conn.execute("SELECT id::text, content FROM chunks WHERE document_id = %s ORDER BY chunk_index",
                                  [document_id]).fetchall()
        text, page_map, filename, version, current, doc_key = doc
        started = time.monotonic()
        facts = extract_document(text, filename, page_map or [], [ChunkText(c, _normal_text(t)) for c, t in chunks],
                                 self.extractor.extract, self.resolver, self.window_tokens, self.length,
                                 self.concurrency)
        self.store.write_document_facts(DocumentRecord(document_id, doc_key, filename, version, current), facts,
                                        self.resolver)
        log.info("graph: %s -> %d part(s), %d entities, %d relations (%d dropped), %.0fs", filename, facts.parts,
                 len(facts.entities), len(facts.relations), facts.dropped, time.monotonic() - started)

    def _document(self, document_id: str) -> None:
        with self.pool.connection() as conn:
            row = conn.execute("SELECT is_current FROM documents WHERE id = %s", [document_id]).fetchone()
        if row is not None:  # a deleted document is cleaned up through its chunks' delete rows
            self.store.set_document_current(document_id, row[0])

    def _done(self, ids: list[int]) -> None:
        if ids:
            with self.pool.connection() as conn:
                conn.execute("UPDATE graph_outbox SET done_at = now(), locked_until = NULL, error = NULL "
                             "WHERE id = ANY(%s)", [ids])

    def _failed(self, ids: list[int], exc: Exception) -> None:
        with self.pool.connection() as conn:
            # Retry after a backoff that grows with the attempt count.
            conn.execute("UPDATE graph_outbox SET error = %s, "
                         "locked_until = now() + make_interval(secs => least(30 * attempts * attempts, 3600)) "
                         "WHERE id = ANY(%s)", [str(exc)[:1000], ids])

    # --- running continuously ----------------------------------------------------------------------

    def run(self, listen_conninfo: str, listen_kwargs: dict, schema: str, poll_seconds: float = 30.0,
            once: bool = False) -> None:
        """Drain the queue, then wait for NOTIFY kb_graph (or the poll interval) and repeat."""
        with psycopg.connect(listen_conninfo, autocommit=True, **listen_kwargs) as listener:
            listener.execute("LISTEN kb_graph")
            while True:
                while self.process_once():
                    pass
                if once:
                    return
                for note in listener.notifies(timeout=poll_seconds):
                    if note.payload == schema:
                        break


# --- maintenance -------------------------------------------------------------------------------------

def enable(pool: ConnectionPool) -> int:
    """Turn queueing on and queue every document that has its text for extraction. Returns the
    number queued; documents without stored text need backfill_texts()."""
    with pool.connection() as conn, conn.transaction():
        conn.execute("UPDATE graph_state SET enabled = true")
        return conn.execute("INSERT INTO graph_outbox (op, document_id) "
                            "SELECT 'extract', document_id FROM document_texts").rowcount


def document_text(sections) -> tuple[str, list[list[int]]]:
    """The loader's sections (PDF pages) joined as one text, with a [offset, page] map."""
    parts, page_map, offset = [], [], 0
    for section in sections:
        page = section.metadata.get("page")
        if page is not None:
            page_map.append([offset, page])
        parts.append(section.text)
        offset += len(section.text) + 2
    return "\n\n".join(parts), page_map


def backfill_texts(pool: ConnectionPool, settings, progress=print) -> int:
    """Store the full text of documents ingested before document_texts existed, re-read from the
    original file (documents.source) with the current loader. Without a readable source file, the
    chunks are joined instead (heading-path lines removed). With the graph enabled, each stored
    text queues its document for extraction. Returns the number of documents filled in."""
    from pathlib import Path

    from kb.ingestion.loaders import document_text, load_document

    with pool.connection() as conn:
        missing = conn.execute("SELECT d.id::text, d.filename, d.source FROM documents d "
                               "LEFT JOIN document_texts t ON t.document_id = d.id WHERE t.document_id IS NULL "
                               "ORDER BY d.created_at").fetchall()
    for doc_id, filename, source in missing:
        path = Path(source) if source and not source.lower().startswith(("http://", "https://")) else None
        if path is not None and path.is_file():
            progress(f"reading {path} for document-level extraction")
            loaded = load_document(filename, path.read_bytes(), structure=settings.chunk_strategy == "structure",
                                   tables=settings.table_extraction)
            text, page_map = document_text(loaded.sections)
        else:
            progress(f"{filename}: source file not available, using its chunks as the document text")
            with pool.connection() as conn:
                rows = conn.execute("SELECT content, page FROM chunks WHERE document_id = %s ORDER BY chunk_index",
                                    [doc_id]).fetchall()
            pieces, page_map, offset = [], [], 0
            for content, page in rows:
                body = content.split("\n", 1)[1] if content.startswith("[") and "\n" in content else content
                if page is not None and (not page_map or page_map[-1][1] != page):
                    page_map.append([offset, page])
                pieces.append(body)
                offset += len(body) + 2
            text = "\n\n".join(pieces)
        with pool.connection() as conn:
            conn.execute("INSERT INTO document_texts (document_id, content, page_map) VALUES (%s, %s, %s) "
                         "ON CONFLICT (document_id) DO NOTHING", [doc_id, text, Jsonb(page_map)])
    return len(missing)


def disable(pool: ConnectionPool) -> None:
    with pool.connection() as conn:
        conn.execute("UPDATE graph_state SET enabled = false")


def is_enabled(pool: ConnectionPool) -> bool:
    with pool.connection() as conn:
        return bool(conn.execute("SELECT enabled FROM graph_state").fetchone()[0])


def rebuild(pool: ConnectionPool, store: GraphStore) -> int:
    """Wipe this knowledge base's graph and queue every document again."""
    store.wipe()
    with pool.connection() as conn:
        conn.execute("UPDATE graph_outbox SET done_at = now(), error = 'superseded by rebuild' WHERE done_at IS NULL")
    return enable(pool)


def sync(pool: ConnectionPool, store: GraphStore) -> dict:
    """Make the graph match Postgres: remove documents and chunk ids Postgres no longer has, queue
    documents the graph lacks, refresh every document's current flag, purge week-old queue rows.
    Safe to run any time (e.g. nightly)."""
    with pool.connection() as conn:
        pg_docs = dict(conn.execute("SELECT id::text, is_current FROM documents").fetchall())
        with_text = {r[0] for r in conn.execute("SELECT document_id::text FROM document_texts").fetchall()}
        pg_chunks = {r[0] for r in conn.execute("SELECT id::text FROM chunks").fetchall()}
        pending = {r[0] for r in conn.execute("SELECT document_id::text FROM graph_outbox "
                                              "WHERE done_at IS NULL AND op = 'extract'").fetchall()}
    graph_docs = store.document_ids()
    extra_docs = sorted(graph_docs - set(pg_docs))
    extra_chunks = sorted(store.chunk_ids() - pg_chunks)
    missing = sorted((with_text - graph_docs) - pending)
    store.delete_documents(extra_docs)
    store.delete_chunks(extra_chunks)
    with pool.connection() as conn:
        if missing:
            conn.execute("INSERT INTO graph_outbox (op, document_id) SELECT 'extract', unnest(%s::uuid[])", [missing])
        purged = conn.execute("DELETE FROM graph_outbox WHERE done_at < now() - interval '7 days'").rowcount
    for doc_id in graph_docs & set(pg_docs):
        store.set_document_current(doc_id, pg_docs[doc_id])
    return {"documents_removed": len(extra_docs), "chunk_ids_removed": len(extra_chunks),
            "documents_queued": len(missing), "queue_rows_purged": purged}


def queue_status(pool: ConnectionPool) -> dict:
    with pool.connection() as conn:
        row = conn.execute("""
            SELECT count(*) FILTER (WHERE done_at IS NULL AND attempts < 5),
                   count(*) FILTER (WHERE done_at IS NULL AND attempts >= 5),
                   count(*) FILTER (WHERE done_at IS NOT NULL),
                   max(error) FILTER (WHERE done_at IS NULL)
            FROM graph_outbox""").fetchone()
    return {"pending": row[0], "failed": row[1], "done": row[2], "last_error": row[3]}


def wait_idle(pool: ConnectionPool, timeout: float = 600.0) -> bool:
    """For scripts and tests: block until the queue has no pending rows."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if queue_status(pool)["pending"] == 0:
            return True
        time.sleep(1)
    return False
