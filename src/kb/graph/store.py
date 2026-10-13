"""Neo4j access: a document's extracted entities and facts, deletes, and version flags.

The graph holds documents, entities and facts only. Chunks are not stored as nodes; where the
graph needs to point back at pgvector chunks it keeps their ids as properties:

    (:Document {document_id, doc_key, filename, version, current})
      -[:MENTIONS {chunk_ids: [...], quote}]-> (:Entity:<Type> {type, name, aliases, search_text})
    (:Entity)-[:<RELATION> {document_id, source_chunk, version, current, quote}]->(:Entity)

`chunk_ids` lists the document's chunks that state or name the entity; `source_chunk` is the chunk
holding the fact's quote (null when the quote couldn't be located). Deleting chunks removes their
ids from those properties and the facts traced to them; deleting a document removes its node and all
of its facts; entities nothing refers to any more are removed.
"""

from dataclasses import dataclass

from kb.graph.document import DocumentFacts
from kb.graph.resolve import Resolver
from kb.graph.schema import ENTITY_TYPES, EXTRACTION_VERSION, RELATION_TYPES

# Labels from the earlier chunk-level graph; wiped with everything else.
_LEGACY_LABELS = ("Chunk", "Section")


@dataclass
class DocumentRecord:
    document_id: str
    doc_key: str
    filename: str
    version: int
    current: bool


class GraphStore:
    def __init__(self, uri: str, user: str, password: str | None, database: str, kb: str,
                 timeout: float | None = None):
        """`timeout` (seconds) bounds connecting and retrying, for use inside a search request."""
        from neo4j import GraphDatabase

        self.kb = kb
        self.database = database
        options = {"connection_timeout": timeout, "max_transaction_retry_time": timeout} if timeout else {}
        # Server notifications (e.g. "property `kb` does not exist" on an empty database) are
        # informational and would otherwise be logged as warnings on every command.
        self.driver = GraphDatabase.driver(uri, auth=(user, password or ""), notifications_min_severity="OFF",
                                           **options)

    def close(self) -> None:
        self.driver.close()

    def _write(self, fn, *args):
        with self.driver.session(database=self.database) as session:
            return session.execute_write(fn, *args)

    def _read(self, query: str, **params):
        with self.driver.session(database=self.database) as session:
            return session.execute_read(lambda tx: [r.data() for r in tx.run(query, kb=self.kb, **params)])

    # --- setup ------------------------------------------------------------------------------------

    def init(self) -> None:
        """Uniqueness constraints and indexes (idempotent)."""
        statements = [f"CREATE CONSTRAINT kb_{label.lower()}_key IF NOT EXISTS FOR (n:{label}) REQUIRE n.key IS UNIQUE"
                      for label in ("Document", "Entity")]
        statements += [f"CREATE INDEX kb_{label.lower()}_kb IF NOT EXISTS FOR (n:{label}) ON (n.kb)"
                       for label in ("Document", "Entity")]
        for rel in RELATION_TYPES:
            statements.append(f"CREATE INDEX kb_rel_{rel.lower()}_source IF NOT EXISTS "
                              f"FOR ()-[r:{rel}]-() ON (r.source_chunk)")
            statements.append(f"CREATE INDEX kb_rel_{rel.lower()}_document IF NOT EXISTS "
                              f"FOR ()-[r:{rel}]-() ON (r.document_id)")
        # Entity linking for graph search (kb.graph.rank): names and aliases, full-text.
        statements.append("CREATE FULLTEXT INDEX kb_entity_search IF NOT EXISTS FOR (n:Entity) ON EACH [n.search_text]")
        with self.driver.session(database=self.database) as session:
            for statement in statements:
                session.run(statement).consume()

    def ping(self) -> str:
        with self.driver.session(database=self.database) as session:
            return session.run("CALL dbms.components() YIELD versions RETURN versions[0] AS v").single()["v"]

    # --- writing a document's facts ----------------------------------------------------------------

    def write_document_facts(self, doc: DocumentRecord, facts: DocumentFacts, resolver: Resolver) -> None:
        """Replace a document's entities and facts with a new extraction."""
        self._write(self._write_document_tx, doc, facts, resolver)

    def _write_document_tx(self, tx, doc: DocumentRecord, facts: DocumentFacts, resolver: Resolver) -> None:
        kb, did = self.kb, doc.document_id
        doc_node = f"{kb}|{did}"
        for rel in RELATION_TYPES:
            tx.run(f"MATCH ()-[r:{rel} {{document_id: $did}}]->() WHERE r.kb = $kb DELETE r", did=did, kb=kb).consume()
        tx.run("MATCH (d:Document {key: $key})-[m:MENTIONS]->() DELETE m", key=doc_node).consume()
        tx.run("""
            MERGE (d:Document {key: $key})
            SET d.kb = $kb, d.document_id = $did, d.doc_key = $doc_key, d.filename = $filename,
                d.version = $version, d.current = $current""",
               key=doc_node, kb=kb, did=did, doc_key=doc.doc_key, filename=doc.filename, version=doc.version,
               current=doc.current).consume()

        keys: dict[tuple[str, str], str] = {}
        by_type: dict[str, list[dict]] = {}
        for entity_type, name, quote, chunk_ids in facts.entities:
            key = resolver.key(kb, entity_type, name)
            keys[(entity_type, name)] = key
            by_type.setdefault(entity_type, []).append(
                {"key": key, "name": name, "quote": quote, "chunks": sorted(chunk_ids)})
        for entity_type, items in by_type.items():
            assert entity_type in ENTITY_TYPES  # the label goes into the query text
            tx.run(f"""
                UNWIND $items AS e
                MERGE (n:Entity:{entity_type} {{key: e.key}})
                ON CREATE SET n.name = e.name, n.aliases = []
                SET n.kb = $kb, n.type = $type,
                    n.aliases = CASE WHEN e.name = n.name OR e.name IN n.aliases THEN n.aliases ELSE n.aliases + e.name END
                WITH n, e
                SET n.search_text = reduce(t = n.name, a IN n.aliases | t + ' ' + a)
                WITH n, e
                MATCH (d:Document {{key: $doc}})
                MERGE (d)-[m:MENTIONS]->(n)
                SET m.chunk_ids = e.chunks, m.quote = e.quote""",
                   items=items, kb=kb, type=entity_type, doc=doc_node).consume()

        names = {name: entity_type for entity_type, name, _, _ in facts.entities}
        rel_items: dict[str, list[dict]] = {}
        for rel_type, source, target, quote, chunk_id in facts.relations:
            a, b = keys.get((names.get(source), source)), keys.get((names.get(target), target))
            if a and b:
                rel_items.setdefault(rel_type, []).append({"a": a, "b": b, "quote": quote, "chunk": chunk_id})
        for rel_type, items in rel_items.items():
            assert rel_type in RELATION_TYPES
            tx.run(f"""
                UNWIND $items AS r
                MATCH (a:Entity {{key: r.a}}) MATCH (b:Entity {{key: r.b}})
                MERGE (a)-[x:{rel_type} {{document_id: $did}}]->(b)
                SET x.kb = $kb, x.source_chunk = r.chunk, x.version = $version, x.current = $current,
                    x.quote = r.quote, x.extraction_version = $ev""",
                   items=items, did=did, kb=kb, version=doc.version, current=doc.current,
                   ev=EXTRACTION_VERSION).consume()

    # --- deleting and versions ---------------------------------------------------------------------

    def delete_chunks(self, chunk_ids: list[str]) -> None:
        """Forget deleted chunks: facts traced to them go, their ids leave entity mention lists."""
        def tx_fn(tx):
            for rel in RELATION_TYPES:
                tx.run(f"UNWIND $ids AS id MATCH ()-[r:{rel} {{source_chunk: id}}]->() WHERE r.kb = $kb DELETE r",
                       ids=chunk_ids, kb=self.kb).consume()
            tx.run("""
                MATCH (d:Document {kb: $kb})-[m:MENTIONS]->()
                WHERE any(c IN m.chunk_ids WHERE c IN $ids)
                SET m.chunk_ids = [c IN m.chunk_ids WHERE NOT c IN $ids]""", ids=chunk_ids, kb=self.kb).consume()
        if chunk_ids:
            self._write(tx_fn)
        self.cleanup()

    def delete_documents(self, document_ids: list[str]) -> None:
        """Remove documents that no longer exist in Postgres, with all of their facts."""
        def tx_fn(tx):
            for rel in RELATION_TYPES:
                tx.run(f"UNWIND $ids AS did MATCH ()-[r:{rel} {{document_id: did}}]->() WHERE r.kb = $kb DELETE r",
                       ids=document_ids, kb=self.kb).consume()
            tx.run("UNWIND $keys AS k MATCH (d:Document {key: k}) DETACH DELETE d",
                   keys=[f"{self.kb}|{d}" for d in document_ids]).consume()
        if document_ids:
            self._write(tx_fn)
        self.cleanup()

    def cleanup(self) -> None:
        """Delete entities nothing refers to any more (no document mentions them, no facts)."""
        self._write(lambda tx: tx.run("MATCH (e:Entity {kb: $kb}) WHERE NOT (e)--() DELETE e", kb=self.kb).consume())

    def set_document_current(self, document_id: str, current: bool) -> None:
        def tx_fn(tx):
            tx.run("MATCH (d:Document {key: $key}) SET d.current = $cur",
                   key=f"{self.kb}|{document_id}", cur=current).consume()
            for rel in RELATION_TYPES:
                tx.run(f"MATCH ()-[r:{rel} {{document_id: $did}}]->() WHERE r.kb = $kb SET r.current = $cur",
                       did=document_id, kb=self.kb, cur=current).consume()
        self._write(tx_fn)

    def wipe(self) -> None:
        """Remove this knowledge base's whole graph (other schemas' graphs are untouched)."""
        def tx_fn(tx):
            for label in (*_LEGACY_LABELS, "Document", "Entity"):
                tx.run(f"MATCH (n:{label} {{kb: $kb}}) DETACH DELETE n", kb=self.kb).consume()
        self._write(tx_fn)

    # --- reading ---------------------------------------------------------------------------------

    def document_ids(self) -> set[str]:
        return {r["id"] for r in self._read("MATCH (d:Document {kb: $kb}) RETURN d.document_id AS id")}

    def chunk_ids(self) -> set[str]:
        """Every chunk id the graph refers to (mention lists and fact sources)."""
        rows = self._read("""
            MATCH (:Document {kb: $kb})-[m:MENTIONS]->() UNWIND m.chunk_ids AS id RETURN DISTINCT id
            UNION
            MATCH ()-[r]->() WHERE r.kb = $kb AND r.source_chunk IS NOT NULL RETURN DISTINCT r.source_chunk AS id""")
        return {r["id"] for r in rows}

    def stats(self) -> dict:
        nodes = self._read("MATCH (n {kb: $kb}) UNWIND labels(n) AS label RETURN label, count(*) AS n ORDER BY label")
        rels = self._read("MATCH (a {kb: $kb})-[r]->() RETURN type(r) AS type, count(*) AS n ORDER BY type")
        return {"nodes": {r["label"]: r["n"] for r in nodes}, "relationships": {r["type"]: r["n"] for r in rels}}

    def show(self, name: str, limit: int = 50) -> list[dict]:
        """Facts about entities whose name or alias contains `name`, with quotes and source chunk ids."""
        return self._read("""
            MATCH (e:Entity {kb: $kb})
            WHERE toLower(e.name) CONTAINS toLower($name) OR any(a IN e.aliases WHERE toLower(a) CONTAINS toLower($name))
            OPTIONAL MATCH (e)-[r]-(o:Entity) WHERE r.kb = $kb
            RETURN e.type AS type, e.name AS entity, type(r) AS relation,
                   CASE WHEN startNode(r) = e THEN 'out' ELSE 'in' END AS direction,
                   o.type AS other_type, o.name AS other, r.quote AS quote, r.current AS current,
                   r.source_chunk AS chunk_id
            LIMIT $limit""", name=name, limit=limit)
