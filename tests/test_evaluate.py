import uuid

import pytest

from kb.evaluate import Case, failures, load_cases, run
from kb.schemas import NO_RELEVANT_CONTEXT_MESSAGE, Citation, QueryRequest, QueryResponse, RetrievedChunk


def _chunk(rank: int, filename: str, text: str, page: int | None = None) -> RetrievedChunk:
    return RetrievedChunk(
        rank=rank, chunk_id=uuid.uuid4(), text=text, vector_score=0.5, rerank_score=0.9, metadata={},
        citation=Citation(label=f"[{rank}]", document_id=uuid.uuid4(), filename=filename,
                          file_type="pdf", page=page, chunk_index=rank - 1),
    )


class CannedRetriever:
    """Returns fixed results per question; unknown questions get no_relevant_context."""

    def __init__(self, answers: dict[str, list[RetrievedChunk]]):
        self.answers = answers

    def search(self, req: QueryRequest) -> QueryResponse:
        results = self.answers.get(req.query, [])
        return QueryResponse(
            query=req.query, filters=req.filters, status="ok" if results else "no_relevant_context",
            message=None if results else NO_RELEVANT_CONTEXT_MESSAGE, reranked=True,
            min_score_applied=0.3, results=results, context="", took_ms=10.0,
        )


RETRIEVER = CannedRetriever({
    "fees": [_chunk(1, "other.pdf", "unrelated"), _chunk(2, "fees.pdf", "The fee is $25", page=2)],
    "hours": [_chunk(1, "clinics.md", "Open 8 AM to 4 PM")],
    "missed": [_chunk(1, "other.pdf", "nothing useful")],
    "passport": [_chunk(1, "other.pdf", "forms")],
})


def test_metrics():
    cases = [
        Case("fees", "fees", [{"filename": "fees.pdf", "page": 2, "contains": "$25"}]),  # hit at rank 2
        Case("hours", "hours", [{"filename": "clinics.md"}, {"contains": "weekends"}]),  # rank 1, 1 of 2
        Case("missed", "missed", [{"filename": "fees.pdf"}]),                             # miss
        Case("refused", "no results", [{"filename": "fees.pdf"}]),                        # false refusal
        Case("tax", "property tax", []),                                                  # correct refusal
        Case("passport", "passport", []),                                                 # should have refused
    ]
    results, s = run(RETRIEVER, cases)

    assert [r.first_hit_rank for r in results[:4]] == [2, 1, None, None]
    assert s["hit@1"] == 0.25
    assert s["hit@k"] == 0.5
    assert s["mrr"] == round((0.5 + 1) / 4, 3)
    assert s["recall"] == round((1 + 0.5) / 4, 3)
    assert s["false_refusal"] == 0.25
    assert s["correct_refusal"] == 0.5

    lines = failures(results)
    assert [line.split()[0] for line in lines] == ["PARTIAL", "MISS", "MISS", "NO-REFUSE"]


def test_rerank_fallbacks_counted():
    class FallbackRetriever(CannedRetriever):
        def search(self, req):
            return super().search(req).model_copy(update={"reranked": False})

    cases = [Case("hours", "hours", [{"filename": "clinics.md"}]), Case("tax", "property tax", [])]
    _, with_rerank = run(FallbackRetriever(RETRIEVER.answers), cases, rerank=True)
    _, without = run(FallbackRetriever(RETRIEVER.answers), cases, rerank=False)
    assert with_rerank["rerank_fallbacks"] == 2
    assert without["rerank_fallbacks"] == 0
    _, ok = run(RETRIEVER, cases, rerank=True)
    assert ok["rerank_fallbacks"] == 0


def test_delay_between_searches(monkeypatch):
    slept = []
    monkeypatch.setattr("kb.evaluate.time.sleep", slept.append)
    run(RETRIEVER, [Case(str(i), "hours", []) for i in range(3)], delay=6.5)
    assert slept == [6.5, 6.5]  # none before the first search


def test_load_cases(tmp_path):
    path = tmp_path / "q.jsonl"
    path.write_text('# comment\n\n{"id": "a", "question": "q?", "expected": [{"filename": "x.md"}]}\n'
                    '{"question": "unanswerable", "filters": {"year": 2025}}\n', encoding="utf-8")
    cases = load_cases(path)
    assert [(c.id, c.expected, c.filters) for c in cases] == [
        ("a", [{"filename": "x.md"}], None), ("4", [], {"year": 2025}),
    ]


@pytest.mark.parametrize("line", ['{"expected": []}', '{"question": "q", "expected": [{"file": "x"}]}', "not json"])
def test_load_cases_rejects_bad_lines(tmp_path, line):
    path = tmp_path / "q.jsonl"
    path.write_text(line, encoding="utf-8")
    with pytest.raises(ValueError):
        load_cases(path)
