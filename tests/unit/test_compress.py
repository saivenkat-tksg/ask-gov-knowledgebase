from uuid import uuid4

import pytest

from kb.retrieval.compress import compress, split_units
from kb.retrieval.retriever import Retriever
from kb.schemas import Citation, QueryRequest, QueryResponse, RetrievedChunk


class KeywordReranker:
    """Scores a unit by the first keyword it holds (keyword -> score), else 0.1."""

    enabled = True

    def __init__(self, scores: dict[str, float]):
        self.scores = scores
        self.calls: list[list[str]] = []

    def rerank(self, query, documents, top_n):
        self.calls.append(documents)
        scored = [(i, next((s for k, s in self.scores.items() if k in d), 0.1)) for i, d in enumerate(documents)]
        return sorted(scored, key=lambda x: -x[1])[:top_n]


def _chunk(text: str, rank: int = 1) -> RetrievedChunk:
    return RetrievedChunk(rank=rank, chunk_id=uuid4(), text=text, vector_score=0.5, metadata={},
                          citation=Citation(label=f"[{rank}]", document_id=uuid4(), filename="a.pdf",
                                            file_type="pdf", chunk_index=0))


def test_split_units_joins_wrapped_lines_and_keeps_rows_headings_bullets():
    text = ("[Chapter 3: POLICE > Overview]\nCurrently there are approximately 340 CPGs with a \n"
            "total membership of over 7300 volunteers. The NCP employs 25 officers.\n"
            "- first bullet\n- second bullet\n\n| A | B |\n|---|---|\n| 1 | 2 |")
    assert split_units(text) == [
        "[Chapter 3: POLICE > Overview]",
        "Currently there are approximately 340 CPGs with a total membership of over 7300 volunteers.",
        "The NCP employs 25 officers.", "- first bullet", "- second bullet", "| A | B |", "|---|---|", "| 1 | 2 |"]
    assert split_units("Fee is $1.50 per copy. Pay at G.R.O.") == ["Fee is $1.50 per copy.", "Pay at G.R.O."]


def test_keeps_units_near_the_best_with_heading_and_gaps_in_one_rerank_call():
    reranker = KeywordReranker({"65%": 0.95, "sticker": 0.85, "Brakes": 0.7})
    chunks = [_chunk("[Traffic > Tint]\nBrakes are checked. The tint can be no darker than 65%. Forms are signed. "
                     "The tint sticker is issued."), _chunk("Opening hours are 8 to 4. Closed on holidays.", 2)]
    out = compress("tint?", chunks, reranker, min_score=0.5, margin=0.15)
    # 0.95 - 0.15 = 0.8: "Brakes" (0.7) is cut although it clears min_score
    assert out[0].text == "[Traffic > Tint]\n…\nThe tint can be no darker than 65%.\n…\nThe tint sticker is issued."
    assert out[1].text == "Opening hours are 8 to 4."  # nothing qualifies: its best (first) unit only
    assert [(e.text, e.score) for e in out[0].evidence] == [("The tint can be no darker than 65%.", 0.95),
                                                            ("The tint sticker is issued.", 0.85)]
    assert out[0].compressed and not chunks[0].compressed and chunks[0].evidence == []
    assert len(reranker.calls) == 1 and "[Traffic > Tint]" not in reranker.calls[0]  # headings are not scored
    assert chunks[0].text.startswith("[Traffic")  # inputs untouched


def test_window_table_header_and_separator():
    reranker = KeywordReranker({"65%": 0.9})
    text = "Intro line.\n\nBefore.\n\nThe limit is 65%.\n\nAfter.\n\nOutro.\n| Item | Limit |\n|---|---|\n| Other | 10% |\n| Tint | 65% |"
    out = compress("q", [_chunk(text)], reranker, min_score=0.5, margin=0.15, window=1)
    assert out[0].text == "Before.\nThe limit is 65%.\nAfter.\n…\n| Item | Limit |\n|---|---|\n| Other | 10% |\n| Tint | 65% |"
    assert "|---|---|" not in reranker.calls[0]


def test_reranker_errors_propagate():
    class Broken:
        def rerank(self, *a):
            raise RuntimeError("down")

    with pytest.raises(RuntimeError):
        compress("q", [_chunk("A one. B two.")], Broken(), 0.5, 0.15)


def _retriever(reranker, **kw) -> Retriever:
    return Retriever(pool=None, embedder=None, reranker=reranker, iterative_scan=False, compress_context=True, **kw)


def _response(query_type: str | None) -> QueryResponse:
    chunks = [_chunk("Brakes are checked. The tint can be no darker than 65%.")]
    return QueryResponse(query="q", filters=None, status="ok", reranked=True, min_score_applied=0.5,
                         query_type=query_type, results=chunks, context="", took_ms=0)


def test_retriever_compresses_except_list_and_process_and_survives_rerank_errors():
    retriever = _retriever(KeywordReranker({"65%": 0.9}))
    out = retriever._compressed(QueryRequest(query="q"), _response("normal"))
    assert out.results[0].text == "The tint can be no darker than 65%."
    assert out.context.endswith("\nThe tint can be no darker than 65%.")
    for kind in ("list", "process"):
        assert retriever._compressed(QueryRequest(query="q"), _response(kind)).results[0].text.startswith("Brakes")

    class Broken:
        enabled = True

        def rerank(self, *a):
            raise RuntimeError("429")

    out = _retriever(Broken())._compressed(QueryRequest(query="q"), _response("normal"))
    assert out.results[0].text.startswith("Brakes")
