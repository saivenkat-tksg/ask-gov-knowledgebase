from uuid import UUID

from kb.generation import build_messages
from kb.graph.rank import GraphHit, lucene_query
from kb.retrieval.retriever import Retriever, found_by_label, reciprocal_rank_fusion
from kb.schemas import QueryRequest

A, B, C, D = (UUID(int=i) for i in range(1, 5))


def row(chunk_id: UUID, text: str, vector_score: float = 0.5) -> dict:
    return {"id": chunk_id, "document_id": UUID(int=99), "chunk_index": chunk_id.int, "content": text, "page": 1,
            "char_start": 0, "char_end": len(text), "metadata": {}, "filename": "guide.pdf", "file_type": "pdf",
            "labelchunk": f"guide_chunk_{chunk_id.int}", "vector_score": vector_score}


class FakeGraph:
    def __init__(self, hits=None, error=None):
        self.hits, self.error, self.calls = hits or [], error, []

    def search(self, question, limit):
        self.calls.append(question)
        if self.error:
            raise self.error
        return self.hits


class OrderReranker:
    """Keeps the incoming order, scoring 0.9, 0.8, ... so the test sees what fusion produced."""
    enabled = True

    def rerank(self, query, documents, top_n):
        return [(i, round(0.9 - 0.1 * i, 2)) for i in range(min(top_n, len(documents)))]


class StubEmbedder:
    dim = 4

    def embed(self, texts):
        return [[0.0] * 4 for _ in texts]


def retriever(monkeypatch, graph, vector, keyword, by_id) -> Retriever:
    r = Retriever(pool=None, embedder=StubEmbedder(), reranker=OrderReranker(), iterative_scan=False,
                  min_rerank_score=0.5, min_vector_score=0.2, graph=graph)
    monkeypatch.setattr(r, "_vector_search", lambda *a: [dict(x) for x in vector])
    monkeypatch.setattr(r, "_keyword_search", lambda *a: [dict(x, keyword_score=0.1) for x in keyword])
    monkeypatch.setattr(r, "_rows_by_id", lambda ids, *a: [dict(by_id[UUID(i)]) for i in ids if UUID(i) in by_id])
    monkeypatch.setattr(r, "_drift_warnings", lambda results: [])
    return r


def test_weighted_rrf_and_labels():
    fused = reciprocal_rank_fusion([["a", "b"], ["b"], ["c"]], weights=[1.0, 1.0, 2.0])
    assert fused["b"] > fused["a"] and fused["c"] > fused["a"]  # double weight beats rank 1 of one list
    assert found_by_label({"vector", "keyword"}) == "both"
    assert found_by_label({"graph"}) == "graph"
    assert found_by_label({"keyword", "graph", "vector"}) == "vector+keyword+graph"


def test_graph_hits_join_the_candidates_and_get_reranked(monkeypatch):
    fact = {"subject": "Birth registration", "relation": "COSTS", "object": "free of charge",
            "quote": "free of charge", "chunk_id": str(C)}
    graph = FakeGraph([GraphHit(str(C), 2.0, [fact]), GraphHit(str(A), 1.0), GraphHit(str(D), 0.5)])
    rows = {C: row(C, "Registrations are free of charge", 0.3)}  # D is filtered out in Postgres
    r = retriever(monkeypatch, graph, vector=[row(A, "vector hit", 0.6)], keyword=[row(B, "keyword hit")], by_id=rows)

    resp = r.search(QueryRequest(query="is birth registration free"))
    by = {res.chunk_id: res for res in resp.results}
    assert resp.graph and graph.calls == ["is birth registration free"]
    assert set(by) == {A, B, C}  # D never reaches the results: Postgres filters decide
    assert by[C].found_by == "graph" and by[C].graph_score == 2.0
    assert by[A].found_by == "vector+graph"
    assert by[B].found_by == "keyword" and by[B].graph_score is None
    assert resp.results[0].chunk_id == A  # in two lists: fused first, then reranked
    assert [(f.subject, f.relation, f.object, f.source) for f in resp.facts] == [
        ("Birth registration", "COSTS", "free of charge", by[C].citation.label)]


def test_graph_off_per_request_and_failure_degrades(monkeypatch):
    graph = FakeGraph([GraphHit(str(C), 1.0)])
    r = retriever(monkeypatch, graph, vector=[row(A, "vector hit")], keyword=[], by_id={C: row(C, "graph only")})
    resp = r.search(QueryRequest(query="anything", graph=False))
    assert not resp.graph and graph.calls == [] and [x.chunk_id for x in resp.results] == [A]

    broken = retriever(monkeypatch, FakeGraph(error=ConnectionError("neo4j down")), vector=[row(A, "vector hit")],
                       keyword=[], by_id={})
    resp = broken.search(QueryRequest(query="anything"))
    assert [x.chunk_id for x in resp.results] == [A]
    assert any("Knowledge graph unavailable" in w for w in resp.warnings)


def test_lucene_query_drops_stopwords_and_allows_typos():
    assert lucene_query("How much is a birth certificate?") == "birth OR certificate OR certificate~1"
    assert lucene_query("what is the") is None
    assert "registar~1" in lucene_query("registar office")


def test_answer_prompt_includes_cited_facts(monkeypatch):
    fact = {"subject": "Birth registration", "relation": "COSTS", "object": "free of charge", "quote": "free",
            "chunk_id": str(A)}
    r = retriever(monkeypatch, FakeGraph([GraphHit(str(A), 1.0, [fact])]), vector=[row(A, "Registrations are free")],
                  keyword=[], by_id={})
    content = build_messages(r.search(QueryRequest(query="is it free")))[1]["content"]
    assert "Facts from the knowledge graph" in content and "- Birth registration costs free of charge [1]" in content
