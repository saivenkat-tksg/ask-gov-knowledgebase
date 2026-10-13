"""Postgres -> graph sync against a real Postgres: triggers queue every chunk change and the worker
applies them. Neo4j and the LLM are replaced by in-memory fakes, so this checks the queue logic:
inserts, deletes made by hand in SQL, cascades from documents, version changes, truncate, rollback.

Runs when KB_TEST_DATABASE_URL is set; uses a throwaway schema (kb_graph_test) in that database
and drops it afterwards.
"""

import hashlib
import math
import os
import re
from pathlib import Path

import psycopg
import pytest

from kb.config import Settings
from kb.db.database import create_pool, init_db, session_options
from kb.graph import worker as gw
from kb.graph.extract import Entity, Extraction, Relation
from kb.graph.resolve import Resolver
from kb.ingestion.ingestor import Ingestor

URL = os.environ.get("KB_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not URL, reason="KB_TEST_DATABASE_URL not set")
SCHEMA = "kb_graph_test"
DIM = 64
SAMPLE = Path(__file__).resolve().parents[2] / "samples" / "flu-clinics.md"


class HashEmbedder:
    dim = DIM

    def embed(self, texts):
        out = []
        for t in texts:
            v = [0.0] * DIM
            for w in re.findall(r"\w+", t.lower()):
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % DIM] += 1
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out


class FakeExtractor:
    def __init__(self):
        self.calls = []

    def extract(self, text, context=None):
        self.calls.append(context)
        return Extraction([Entity("Service", "Flu vaccination", "flu"), Entity("Office", "Health Centre", "clinic")],
                          [Relation("HANDLED_BY", "Flu vaccination", "Health Centre", "flu")])


class FakeStore:
    """Records what the worker asks Neo4j to do (documents and the chunk ids their facts point at)."""

    def __init__(self):
        self.facts: dict[str, object] = {}  # document id -> DocumentFacts
        self.docs: dict[str, object] = {}  # document id -> DocumentRecord
        self.current: dict[str, bool] = {}
        self.wiped = 0

    def write_document_facts(self, doc, facts, resolver):
        self.docs[doc.document_id] = doc
        self.facts[doc.document_id] = facts

    def delete_chunks(self, chunk_ids):
        for facts in self.facts.values():
            facts.relations = [r for r in facts.relations if r[4] not in chunk_ids]
            for entity in facts.entities:
                entity[3].difference_update(chunk_ids)

    def delete_documents(self, document_ids):
        for did in document_ids:
            self.facts.pop(did, None)
            self.docs.pop(did, None)

    def set_document_current(self, document_id, current):
        self.current[document_id] = current

    def wipe(self):
        self.facts.clear()
        self.docs.clear()
        self.wiped += 1

    def document_ids(self):
        return set(self.docs)

    def chunk_ids(self):
        ids = set()
        for facts in self.facts.values():
            ids |= {r[4] for r in facts.relations if r[4]}
            for entity in facts.entities:
                ids |= entity[3]
        return ids


@pytest.fixture(scope="module")
def env():
    settings = Settings(_env_file=None, database_url=URL, db_schema=SCHEMA, embedding_dim=DIM,
                        chunk_length_unit="chars", chunk_size=400, chunk_overlap=40, chunk_min_tokens=50)
    with psycopg.connect(URL, autocommit=True, connect_timeout=10) as conn:
        conn.execute(f'DROP SCHEMA IF EXISTS "{SCHEMA}" CASCADE')
    init_db(settings)
    pool = create_pool(settings)
    yield settings, pool
    pool.close()
    with psycopg.connect(URL, autocommit=True, connect_timeout=10) as conn:
        conn.execute("SET lock_timeout = '10s'")
        conn.execute(f'DROP SCHEMA IF EXISTS "{SCHEMA}" CASCADE')


def drain(worker):
    while worker.process_once():
        pass


def pending(pool):
    with pool.connection() as conn:
        return conn.execute("SELECT op, count(*) FROM graph_outbox WHERE done_at IS NULL GROUP BY op").fetchall()


def chunk_ids(pool, doc_id=None):
    with pool.connection() as conn:
        sql = "SELECT id::text FROM chunks" + (" WHERE document_id = %s" if doc_id else "")
        return {r[0] for r in conn.execute(sql, [doc_id] if doc_id else []).fetchall()}


def test_queue_and_worker_follow_every_change(env):
    settings, pool = env
    ingestor = Ingestor(pool, HashEmbedder(), settings)
    store = FakeStore()
    extractor = FakeExtractor()
    worker = gw.GraphWorker(pool, store, extractor, Resolver(), batch_size=7, concurrency=2)

    # Nothing is queued while the graph is disabled.
    first = ingestor.ingest_bytes("flu-clinics.md", SAMPLE.read_bytes())
    assert first.status == "ingested" and pending(pool) == []

    # Enabling queues the document once (whole-document extraction from its stored text).
    assert gw.enable(pool) == 1
    drain(worker)
    assert len(extractor.calls) == 1 and "Document: flu-clinics.md" in extractor.calls[0]
    doc = store.docs[str(first.document_id)]
    assert doc.doc_key == "flu-clinics.md" and doc.current
    facts = store.facts[str(first.document_id)]
    first_chunks = chunk_ids(pool, str(first.document_id))
    assert facts.relations and facts.relations[0][4] in first_chunks  # traced to a chunk (an id, not a node)
    assert store.chunk_ids() <= first_chunks
    assert pending(pool) == []

    # A new upload queues ONE extraction for the document and nothing per chunk.
    second = ingestor.ingest_bytes("notes.md", b"# Notes\n\n## A\n\nalpha text here\n\n## B\n\nbeta text\n\n## C\n\ngamma")
    assert dict(pending(pool)) == {"extract": 1}
    drain(worker)
    assert str(second.document_id) in store.facts

    # A delete made by hand in SQL, with a different search_path, reaches the graph.
    victim = facts.relations[0][4]
    with psycopg.connect(URL, autocommit=True, connect_timeout=10) as conn:
        conn.execute(f'DELETE FROM "{SCHEMA}".chunks WHERE id = %s', [victim])
    assert dict(pending(pool)) == {"delete": 1}
    drain(worker)
    assert victim not in store.chunk_ids()
    assert str(first.document_id) in store.docs  # the document itself is still there

    # Deleting a document cascades to its chunks; the document and all of its facts leave the graph.
    with psycopg.connect(URL, autocommit=True, connect_timeout=10) as conn:
        conn.execute(f'DELETE FROM "{SCHEMA}".documents WHERE id = %s', [str(second.document_id)])
    drain(worker)
    assert str(second.document_id) not in store.facts and str(second.document_id) not in store.docs

    # A rolled-back delete queues nothing.
    keep = sorted(chunk_ids(pool, str(first.document_id)))[0]
    with psycopg.connect(URL, connect_timeout=10, **session_options(settings)) as conn:
        conn.execute("DELETE FROM chunks WHERE id = %s", [keep])
        conn.rollback()
    assert pending(pool) == []

    # A new version supersedes the old one: the old document's flag flips; the new one is extracted.
    v2 = ingestor.ingest_bytes("flu-clinics.md", SAMPLE.read_bytes() + b"\n\nUpdated hours for 2026.")
    assert v2.status == "ingested" and v2.version == 2
    drain(worker)
    assert store.current[str(first.document_id)] is False
    assert str(v2.document_id) in store.facts and store.docs[str(v2.document_id)].current

    # sync repairs drift: a document the graph has but Postgres doesn't is removed, a missing one queued.
    store.docs["00000000-0000-0000-0000-000000000000"] = doc
    store.docs.pop(str(v2.document_id))
    result = gw.sync(pool, store)
    assert result["documents_removed"] == 1 and result["documents_queued"] == 1
    drain(worker)
    assert str(v2.document_id) in store.docs

    # TRUNCATE clears the whole graph for this schema.
    with pool.connection() as conn:
        conn.execute("TRUNCATE chunks CASCADE")
    drain(worker)
    assert store.wiped == 1 and store.chunk_ids() == set()
    assert gw.queue_status(pool)["failed"] == 0
