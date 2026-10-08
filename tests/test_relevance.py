"""Relevance cut in Retriever._search: rerank score plus the vector-score floor. No database needed."""

import uuid

from kb.retrieve import Retriever
from kb.schemas import QueryRequest


def _row(text: str, vector_score: float) -> dict:
    return {"id": uuid.uuid4(), "document_id": uuid.uuid4(), "content": text, "vector_score": vector_score,
            "metadata": {}, "filename": "kb.pdf", "file_type": "pdf", "page": 1, "chunk_index": 0,
            "char_start": 0, "char_end": len(text)}


class FixedReranker:
    """Scores every candidate 0.7, the way Cohere scores "hii" against short generic chunks."""

    enabled = True

    def rerank(self, query, texts, top_n):
        return [(i, 0.7) for i in range(min(top_n, len(texts)))]


class FailingReranker(FixedReranker):
    def rerank(self, query, texts, top_n):
        raise RuntimeError("rate limited")


def _retriever(monkeypatch, reranker, rows) -> Retriever:
    r = Retriever(pool=None, embedder=None, reranker=reranker, iterative_scan=False,
                  min_rerank_score=0.5, min_vector_score=0.245)
    monkeypatch.setattr(r, "_candidates", lambda *a: rows)
    monkeypatch.setattr(r, "_drift_warnings", lambda results: [])
    return r


def test_reranked_results_must_clear_vector_floor(monkeypatch):
    rows = [_row("MINISTRY OF HOME AFFAIRS", 0.232), _row("Applicants must be 18 or older", 0.41)]
    resp = _retriever(monkeypatch, FixedReranker(), rows)._search(QueryRequest(query="q"), None, None)
    assert [r.text for r in resp.results] == ["Applicants must be 18 or older"]
    assert resp.min_score_applied == 0.5


def test_contentless_query_is_refused(monkeypatch):
    rows = [_row("MINISTRY OF HOME AFFAIRS", 0.232), _row("Visa steps", 0.19)]
    resp = _retriever(monkeypatch, FixedReranker(), rows)._search(QueryRequest(query="hii"), None, None)
    assert resp.status == "no_relevant_context" and resp.results == []


def test_explicit_min_score_skips_the_floor(monkeypatch):
    rows = [_row("MINISTRY OF HOME AFFAIRS", 0.232)]
    resp = _retriever(monkeypatch, FixedReranker(), rows)._search(QueryRequest(query="q", min_score=0), None, None)
    assert len(resp.results) == 1


def test_without_rerank_the_floor_is_the_cut(monkeypatch):
    rows = [_row("low", 0.2), _row("high", 0.3)]
    resp = _retriever(monkeypatch, FailingReranker(), rows)._search(QueryRequest(query="q"), None, None)
    assert not resp.reranked and [r.text for r in resp.results] == ["high"] and resp.min_score_applied == 0.245
