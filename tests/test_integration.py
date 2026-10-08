"""End-to-end ingest + retrieve against a real pgvector, with fake embeddings/rerank.

Runs only when KB_TEST_DATABASE_URL points at an admin-capable Postgres, e.g.
    KB_TEST_DATABASE_URL=postgresql://kb:kb@localhost:5433/kb
A throwaway `kb_test` database is created and dropped.
"""

import hashlib
import math
import os
import re

import psycopg
import pytest

from kb.config import Settings
from kb.db import create_pool, init_db, supports_iterative_scan
from kb.documents import delete_document, list_documents
from kb.ingest import Ingestor
from kb.lc import KnowledgeBaseRetriever
from kb.retrieve import Retriever
from kb.schemas import QueryRequest

ADMIN_URL = os.environ.get("KB_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not ADMIN_URL, reason="KB_TEST_DATABASE_URL not set")
DIM = 64


class HashEmbedder:
    """Bag-of-words hashed into DIM buckets: similar words -> similar vectors."""

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


class KeywordReranker:
    enabled = True

    def rerank(self, query, documents, top_n):
        # Deterministic stand-in: prefer documents containing the most query words.
        words = set(query.lower().split())
        scores = [sum(w in d.lower() for w in words) / max(len(words), 1) for d in documents]
        order = sorted(range(len(documents)), key=lambda i: -scores[i])[:top_n]
        return [(i, scores[i]) for i in order]


class FailingReranker:
    enabled = True

    def rerank(self, query, documents, top_n):
        raise RuntimeError("429 Too Many Requests")


@pytest.fixture(scope="module")
def services():
    with psycopg.connect(ADMIN_URL, autocommit=True) as conn:
        conn.execute("DROP DATABASE IF EXISTS kb_test WITH (FORCE)")
        conn.execute("CREATE DATABASE kb_test")
    url = re.sub(r"/[^/]+$", "/kb_test", ADMIN_URL)
    settings = Settings(database_url=url, embedding_dim=DIM, chunk_size=200, chunk_overlap=20,
                        chunk_length_unit="chars", cohere_api_key=None, openai_api_key=None,
                        db_schema="kb_it")
    version = init_db(settings)
    pool = create_pool(settings)
    embedder = HashEmbedder()
    yield (
        Ingestor(pool, embedder, settings),
        Retriever(pool, embedder, KeywordReranker(), supports_iterative_scan(version)),
        pool,
    )
    pool.close()
    with psycopg.connect(ADMIN_URL, autocommit=True) as conn:
        conn.execute("DROP DATABASE IF EXISTS kb_test WITH (FORCE)")


def test_end_to_end(services):
    ingestor, retriever, pool = services
    health = "# Vaccines\n\n" + "Flu vaccines are free at county clinics every autumn. " * 5
    tax = "Property tax payments are due on the first of April each year. " * 5

    r1 = ingestor.ingest_bytes("health.md", health.encode(), {"department": "health", "year": 2024, "tags": ["flu"]})
    r2 = ingestor.ingest_bytes("tax.txt", tax.encode(), {"department": "finance", "year": 2021})
    assert r1.status == r2.status == "ingested" and r1.num_chunks > 1

    dup = ingestor.ingest_bytes("copy.md", health.encode(), {})
    assert dup.status == "duplicate" and dup.document_id == r1.document_id

    resp = retriever.search(QueryRequest(query="when are flu vaccines free", top_k=3))
    assert resp.reranked and resp.results[0].citation.filename == "health.md"
    assert resp.context.startswith("[1] health.md")

    only_finance = retriever.search(QueryRequest(query="flu vaccines", filters={"department": "finance"}))
    assert {r.citation.filename for r in only_finance.results} == {"tax.txt"}

    recent = retriever.search(QueryRequest(query="payments", filters={"year": {"$gte": 2023}}, rerank=False))
    assert not recent.reranked and {r.metadata["department"] for r in recent.results} == {"health"}

    tagged = retriever.search(QueryRequest(query="x", filters={"tags": {"$contains": "flu"}}))
    assert tagged.results and all(r.citation.filename == "health.md" for r in tagged.results)

    r = tagged.results[0]
    assert r.metadata["document_id"] == str(r1.document_id)
    assert r.citation.char_start is not None and r.text in health

    lc_docs = KnowledgeBaseRetriever(retriever=retriever, top_k=2).invoke(
        "flu vaccines", filters={"department": "health"}
    )
    assert lc_docs and all(d.metadata["filename"] == "health.md" for d in lc_docs)
    assert lc_docs[0].metadata["citation"]["label"] == "[1]"

    # Hybrid: an exact code the embedding barely registers is found by keyword search.
    forms = "Office notes. " * 20 + "To appeal, submit form XJ-4471 at the counter."
    ingestor.ingest_bytes("forms.txt", forms.encode(), {"department": "legal"})
    hybrid = retriever.search(QueryRequest(query="XJ-4471", top_k=3))
    assert hybrid.hybrid and hybrid.results[0].citation.filename == "forms.txt"
    assert hybrid.results[0].found_by in ("keyword", "both") and hybrid.results[0].keyword_score > 0
    vector_only = retriever.search(QueryRequest(query="XJ-4471", top_k=3, hybrid=False))
    assert not vector_only.hybrid and all(r.found_by == "vector" for r in vector_only.results)
    # Filters apply to the keyword side too, and stopword-only questions don't break it.
    legal_blocked = retriever.search(QueryRequest(query="XJ-4471", filters={"department": "finance"}))
    assert all(r.citation.filename == "tax.txt" for r in legal_blocked.results)
    assert retriever.search(QueryRequest(query="what is it", top_k=2)).status in ("ok", "no_relevant_context")

    # Relevance cut: an unrelated question returns nothing instead of the nearest noise.
    strict = Retriever(pool, retriever.embedder, KeywordReranker(), retriever.iterative_scan,
                       min_rerank_score=0.3, min_vector_score=0.2)
    unrelated = strict.search(QueryRequest(query="cricket scores tonight", top_k=3))
    assert unrelated.status == "no_relevant_context" and unrelated.results == []
    assert unrelated.message and unrelated.context == "" and unrelated.min_score_applied == 0.3
    related = strict.search(QueryRequest(query="flu vaccines", top_k=3))
    assert related.status == "ok" and related.results
    assert strict.search(QueryRequest(query="cricket scores tonight", min_score=0)).results  # 0 disables

    # Reranker outage degrades to vector order with a warning instead of failing the query.
    degraded = Retriever(pool, retriever.embedder, FailingReranker(), retriever.iterative_scan,
                         min_rerank_score=0.3, min_vector_score=0.2)
    resp = degraded.search(QueryRequest(query="flu vaccines free clinics", top_k=2))
    assert not resp.reranked and resp.warnings and resp.min_score_applied == 0.2
    assert resp.results and resp.results[0].citation.filename == "health.md"

    assert delete_document(pool, r1.document_id)
    assert sorted(d.filename for d in list_documents(pool)) == ["forms.txt", "tax.txt"]



def test_versioning(services):
    from datetime import date, timedelta

    from kb.documents import document_versions

    ingestor, retriever, pool = services
    v1_text = "Birth certificate fee is 500 dollars at the General Register Office. " * 4
    v2_text = "Birth certificate fee is 1000 dollars at the General Register Office. " * 4

    v1 = ingestor.ingest_bytes("fees_2024.txt", v1_text.encode(), {}, doc_key="gro-fees")
    v2 = ingestor.ingest_bytes("fees_2025.txt", v2_text.encode(), {}, doc_key="gro-fees",
                               review_by=date.today() + timedelta(days=180))
    assert (v1.version, v2.version) == (1, 2) and v2.superseded_id == v1.document_id

    # Only the current version is searchable.
    def fee_files(resp):
        return {r.citation.filename for r in resp.results if r.metadata.get("doc_key") == "gro-fees"}

    found = retriever.search(QueryRequest(query="birth certificate fee", top_k=5, min_score=0))
    assert fee_files(found) == {"fees_2025.txt"}
    assert all(r.metadata["version"] == 2 for r in found.results if r.metadata.get("doc_key") == "gro-fees")

    versions = document_versions(pool, "gro-fees")
    assert [(d.version, d.is_current) for d in versions] == [(2, True), (1, False)]
    assert versions[1].superseded_at is not None
    assert "fees_2024.txt" not in {d.filename for d in list_documents(pool)}
    assert "fees_2024.txt" in {d.filename for d in list_documents(pool, include_history=True)}

    # Deleting the current version restores the previous one.
    assert delete_document(pool, v2.document_id)
    restored = retriever.search(QueryRequest(query="birth certificate fee", top_k=5, min_score=0))
    assert fee_files(restored) == {"fees_2024.txt"}

    # Expired versions drop out of search; replace=True discards history.
    expired = ingestor.ingest_bytes("fees_old.txt", b"Expired notice about birth certificate fees. " * 4, {},
                                    doc_key="gro-fees", valid_until=date.today() - timedelta(days=1))
    assert expired.version == 2  # v2 was deleted, so its number is free again
    gone = retriever.search(QueryRequest(query="birth certificate fee", top_k=5, min_score=0))
    assert fee_files(gone) == set()
    fresh = ingestor.ingest_bytes("fees_new.txt", v2_text.encode() + b" Updated.", {}, doc_key="gro-fees",
                                  replace=True)
    assert fresh.version == 1 and [d.version for d in document_versions(pool, "gro-fees")] == [1]

    # Identical bytes: a plain re-upload is a duplicate; replace=True re-processes it in place.
    again = ingestor.ingest_bytes("fees_new.txt", v2_text.encode() + b" Updated.", {}, doc_key="gro-fees")
    assert again.status == "duplicate"
    redo = ingestor.ingest_bytes("fees_new.txt", v2_text.encode() + b" Updated.", {}, doc_key="gro-fees",
                                 replace=True)
    assert redo.status == "ingested" and redo.document_id != fresh.document_id
    assert [d.id for d in document_versions(pool, "gro-fees")] == [redo.document_id]


def test_drift_refresh(services, tmp_path):
    from datetime import date, timedelta

    from kb.documents import document_versions, stale_documents
    from kb.drift import refresh

    ingestor, retriever, pool = services
    source = tmp_path / "grant.txt"
    source.write_text("Cash grant applicants need a +592 mobile number to register. " * 4)
    first = ingestor.ingest_bytes("grant.txt", source.read_bytes(), {"department": "finance"},
                                  source=str(source))

    def grant(results):
        return [r for r in results if r.doc_key == "grant.txt"]

    [r] = grant(refresh(ingestor, pool)[0])
    assert r.outcome == "unchanged" and r.document_id == first.document_id

    # Changed file -> new version, user metadata carried over, old version out of search.
    source.write_text("Cash grant applicants may now register with a foreign mobile number. " * 4)
    [dry] = grant(refresh(ingestor, pool, dry_run=True)[0])
    assert dry.outcome == "updated" and len(document_versions(pool, "grant.txt")) == 1
    [r] = grant(refresh(ingestor, pool)[0])
    assert r.outcome == "updated" and r.version == 2
    current = document_versions(pool, "grant.txt")[0]
    assert current.is_current and current.source == str(source) and current.metadata["department"] == "finance"
    found = retriever.search(QueryRequest(query="foreign mobile number cash grant", top_k=5, min_score=0))
    texts = [x.text for x in found.results if x.metadata.get("doc_key") == "grant.txt"]
    assert texts and all("foreign" in t for t in texts)

    # Vanished source -> flagged, still searchable, reported as stale and in query warnings.
    source.unlink()
    [r] = grant(refresh(ingestor, pool)[0])
    assert r.outcome == "missing"
    assert any(s.document.doc_key == "grant.txt" and "no longer exists" in s.reasons[0]
               for s in stale_documents(pool, 365))
    warned = retriever.search(QueryRequest(query="foreign mobile number cash grant", top_k=5, min_score=0))
    assert any("grant.txt (v2) may be out of date" in w for w in warned.warnings)

    # Re-ingesting an unchanged file records its source without a new version.
    other = tmp_path / "other.txt"
    other.write_text("Registry office hours are eight to four on weekdays. " * 4)
    plain = ingestor.ingest_bytes("other.txt", other.read_bytes(), {},
                                  review_by=date.today() - timedelta(days=1))
    assert ingestor.ingest_bytes("other.txt", other.read_bytes(), {}, source=str(other)).status == "duplicate"
    attached = document_versions(pool, "other.txt")
    assert len(attached) == 1 and attached[0].source == str(other) and attached[0].id == plain.document_id
    assert any("review date" in s.reasons[0] for s in stale_documents(pool, 365) if s.document.doc_key == "other.txt")
