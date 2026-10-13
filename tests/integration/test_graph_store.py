"""GraphStore against a real Neo4j: document-level facts, entity merging, chunk and document deletes,
orphan cleanup, version flags. The graph holds no chunk nodes, only chunk ids as properties.

Runs when KB_TEST_NEO4J_URI is set (KB_TEST_NEO4J_PASSWORD, optionally KB_TEST_NEO4J_DATABASE);
uses its own `kb` namespace and wipes it afterwards, so other data in the same Neo4j is safe.

    KB_TEST_NEO4J_URI=neo4j://127.0.0.1:7687 KB_TEST_NEO4J_PASSWORD=... pytest tests/integration
"""

import os

import pytest

from kb.graph.document import DocumentFacts
from kb.graph.resolve import Resolver
from kb.graph.store import DocumentRecord, GraphStore

URI = os.environ.get("KB_TEST_NEO4J_URI")
DATABASE = os.environ.get("KB_TEST_NEO4J_DATABASE", "neo4j")
pytestmark = pytest.mark.skipif(not URI, reason="KB_TEST_NEO4J_URI not set")
KB = "kb_graph_store_test"


def make_store(kb: str) -> GraphStore:
    return GraphStore(URI, os.environ.get("KB_TEST_NEO4J_USER", "neo4j"), os.environ.get("KB_TEST_NEO4J_PASSWORD"),
                      DATABASE, kb)


@pytest.fixture
def store():
    s = make_store(KB)
    s.init()
    s.wipe()
    yield s
    s.wipe()
    s.close()


def doc(document_id: str = "doc-1", current: bool = True) -> DocumentRecord:
    return DocumentRecord(document_id, "booklet.pdf", "booklet.pdf", 1, current)


def facts(office_name: str = "General Register Office") -> DocumentFacts:
    return DocumentFacts(
        entities=[("Office", office_name, "the General Register Office", {"c1", "c2"}),
                  ("Service", "Birth registration", "Registrations for Births", {"c1"}),
                  ("Fee", "Free", "free of charge", {"c2"}),
                  ("Law", "Act 44:01", "Act 44:01", set())],
        relations=[("HANDLED_BY", "Birth registration", office_name, "to the GRO", "c1"),
                   ("COSTS", "Birth registration", "Free", "free of charge", "c2"),
                   ("GOVERNED_BY", "Birth registration", "Act 44:01", "Act 44:01", None)])  # quote not in a chunk


def count(store, query, **params) -> int:
    return store._read(query, **params)[0]["n"]


def test_document_facts_without_chunk_nodes(store):
    resolver = Resolver()
    store.write_document_facts(doc(), facts(), resolver)
    store.write_document_facts(doc(), facts(), resolver)  # re-extraction replaces, never duplicates

    assert set(store.stats()["nodes"]) == {"Document", "Entity", "Office", "Service", "Fee", "Law"}  # no Chunk/Section
    mention = store._read("MATCH (:Document {kb: $kb})-[m:MENTIONS]->(:Office) RETURN m.chunk_ids AS ids")[0]
    assert mention["ids"] == ["c1", "c2"]
    assert count(store, "MATCH ()-[r {kb: $kb, document_id: 'doc-1'}]->() RETURN count(r) AS n") == 3
    assert store.chunk_ids() == {"c1", "c2"} and store.document_ids() == {"doc-1"}
    hits = store._read("CALL db.index.fulltext.queryNodes('kb_entity_search', 'register') YIELD node "
                       "WHERE node.kb = $kb RETURN node.name AS name")
    assert [h["name"] for h in hits] == ["General Register Office"]
    assert any(r["relation"] == "HANDLED_BY" and r["chunk_id"] == "c1" for r in store.show("register office"))

    # Superseding the document marks it and its facts as not current.
    store.set_document_current("doc-1", False)
    assert count(store, "MATCH ()-[r:COSTS {kb: $kb}]->() WHERE r.current RETURN count(r) AS n") == 0
    store.set_document_current("doc-1", True)

    # Deleting a chunk: facts traced to it go, its id leaves the mention lists, the rest stays.
    store.delete_chunks(["c2"])
    assert count(store, "MATCH ()-[r:COSTS {kb: $kb}]->() RETURN count(r) AS n") == 0
    assert count(store, "MATCH ()-[r:GOVERNED_BY {kb: $kb}]->() RETURN count(r) AS n") == 1
    assert store.chunk_ids() == {"c1"}

    # Deleting the document removes its node and every remaining fact; orphan entities go.
    store.delete_documents(["doc-1"])
    assert store.stats() == {"nodes": {}, "relationships": {}}


def test_shared_entities_and_namespaces(store):
    resolver = Resolver()
    store.write_document_facts(doc("doc-1"), facts("General Register Office"), resolver)
    store.write_document_facts(doc("doc-2"), facts("GRO"), resolver)  # same office, other name, other document
    assert count(store, "MATCH (o:Office {kb: $kb}) RETURN count(o) AS n") == 1
    office = store._read("MATCH (o:Office {kb: $kb}) RETURN o.name AS name, o.aliases AS aliases")[0]
    assert office["name"] == "General Register Office" and "GRO" in office["aliases"]

    store.delete_documents(["doc-1"])  # the office stays: doc-2 still mentions it
    assert count(store, "MATCH (o:Office {kb: $kb}) RETURN count(o) AS n") == 1

    other = make_store(KB + "_other")
    try:
        other.write_document_facts(doc("x"), facts(), resolver)
        store.wipe()
        assert store.document_ids() == set() and other.document_ids() == {"x"}
    finally:
        other.wipe()
        other.close()
