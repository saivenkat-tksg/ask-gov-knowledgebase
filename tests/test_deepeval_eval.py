import uuid

import pytest

pytest.importorskip("deepeval")

from kb.deepeval_eval import PIPELINE_ERROR, from_json, lowest, run, to_json  # noqa: E402
from kb.evaluate import Case  # noqa: E402
from kb.schemas import NO_RELEVANT_CONTEXT_MESSAGE, Citation, QueryRequest, QueryResponse, RetrievedChunk  # noqa: E402


def _chunk(rank: int, filename: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        rank=rank, chunk_id=uuid.uuid4(), text=text, vector_score=0.5, rerank_score=0.9, metadata={},
        citation=Citation(label=f"[{rank}]", document_id=uuid.uuid4(), filename=filename,
                          file_type="pdf", page=2, chunk_index=rank - 1),
    )


class CannedRetriever:
    def __init__(self, answers):
        self.answers = answers

    def search(self, req: QueryRequest) -> QueryResponse:
        results = self.answers.get(req.query, [])
        return QueryResponse(
            query=req.query, filters=req.filters, status="ok" if results else "no_relevant_context",
            message=None if results else NO_RELEVANT_CONTEXT_MESSAGE, reranked=True,
            min_score_applied=0.3, results=results, context="", took_ms=1.0,
        )


class FakeMetric:
    """Scores 1.0 when any retrieved context contains the reference's first word, else 0.2."""

    def __init__(self, name, threshold, model):
        self.name, self.threshold, self.model = name, threshold, model
        self.score, self.reason = None, None

    def measure(self, test_case, **_):
        if self.name == "contextual_relevancy" and "boom" in test_case.input:
            raise RuntimeError("judge unavailable")
        key = test_case.expected_output.split()[0].lower()
        hit = any(key in c.lower() for c in test_case.retrieval_context)
        self.score, self.reason = (1.0, "relevant") if hit else (0.2, "not relevant")
        assert test_case.retrieval_context[0].startswith("Source: ")  # judge sees the source file
        return self.score


RETRIEVER = CannedRetriever({
    "fee?": [_chunk(1, "birth.pdf", "The fee is GYD $300.")],
    "hours?": [_chunk(1, "other.pdf", "Unrelated text")],
    "boom?": [_chunk(1, "birth.pdf", "Fee details")],
})


def test_scores_skips_and_errors():
    cases = [
        Case("fee", "fee?", [], reference="Fee is GYD $300"),
        Case("hours", "hours?", [], reference="Hours are 8 to 4"),
        Case("empty", "nothing?", [], reference="Something"),   # nothing retrieved
        Case("boom", "boom?", [], reference="Fee details"),     # judge error on one metric
        Case("refuse", "tax?", []),                              # no reference: skipped
    ]
    results, summary = run(RETRIEVER, cases, metric_factory=FakeMetric, backoff=(0,))

    assert summary["questions_judged"] == 4
    assert summary["skipped_no_reference"] == 1
    by_id = {r.id: r for r in results}
    assert by_id["fee"].scores["contextual_recall"].score == 1.0
    assert by_id["fee"].scores["contextual_recall"].passed
    assert not by_id["hours"].scores["contextual_recall"].passed
    assert by_id["empty"].scores["contextual_precision"].score == 0.0
    assert by_id["empty"].scores["contextual_precision"].reason.startswith("Nothing retrieved")
    assert by_id["boom"].scores["contextual_relevancy"].error.startswith("RuntimeError")
    assert summary["contextual_relevancy"]["errors"] == 1
    # recall: fee 1.0, hours 0.2, empty 0.0, boom 1.0
    assert summary["contextual_recall"]["mean"] == round((1.0 + 0.2 + 0.0 + 1.0) / 4, 3)
    assert summary["contextual_recall"]["pass_rate"] == 0.5
    assert [r.id for r in lowest(results, "contextual_recall", 2)] == ["empty", "hours"]


class FlakyRetriever(CannedRetriever):
    """Fails `fail_times` times for every question in `flaky`, like a dropped network connection."""

    def __init__(self, answers, flaky, fail_times):
        super().__init__(answers)
        self.flaky, self.fail_times, self.calls = flaky, fail_times, {}

    def search(self, req):
        self.calls[req.query] = self.calls.get(req.query, 0) + 1
        if req.query in self.flaky and self.calls[req.query] <= self.fail_times:
            raise ConnectionError("getaddrinfo failed")
        return super().search(req)


def test_retries_then_recovers():
    retriever = FlakyRetriever(RETRIEVER.answers, {"fee?"}, fail_times=2)
    results, summary = run(retriever, [Case("fee", "fee?", [], reference="Fee is GYD $300")],
                           metric_factory=FakeMetric, backoff=(0, 0, 0))
    assert retriever.calls["fee?"] == 3
    assert results[0].scores["contextual_recall"].score == 1.0
    assert summary["pipeline_errors"] == 0


def test_failed_search_does_not_stop_run_and_resume_retries_it():
    cases = [Case("fee", "fee?", [], reference="Fee is GYD $300"),
             Case("hours", "hours?", [], reference="Hours are 8 to 4")]
    saved = []
    down = FlakyRetriever(RETRIEVER.answers, {"fee?"}, fail_times=99)
    results, summary = run(down, cases, metric_factory=FakeMetric, backoff=(0,),
                           on_result=lambda partial: saved.append(len(partial)))
    assert [r.status for r in results] == [PIPELINE_ERROR, "ok"]   # second question still judged
    assert summary["pipeline_errors"] == 1
    assert summary["contextual_recall"]["errors"] == 1
    assert saved == [1, 2]                                        # progress written after each question

    done = from_json(to_json(results, summary))
    retriever = CannedRetriever(RETRIEVER.answers)
    calls = []
    retriever.search = lambda req, _s=retriever.search: calls.append(req.query) or _s(req)
    resumed, summary = run(retriever, cases, metric_factory=FakeMetric, done=done)
    assert calls == ["fee?"]                                      # only the failed question is searched again
    assert [r.status for r in resumed] == ["ok", "ok"]
    assert summary["pipeline_errors"] == 0


class FlakyJudge(FakeMetric):
    """Times out on the first `fail_times` judge calls of the run, like an overloaded judge API."""

    calls = 0
    fail_times = 0

    def measure(self, test_case, **kwargs):
        FlakyJudge.calls += 1
        if FlakyJudge.calls <= FlakyJudge.fail_times:
            raise TimeoutError("judge timed out")
        return super().measure(test_case, **kwargs)


def test_judge_errors_are_retried():
    FlakyJudge.calls, FlakyJudge.fail_times = 0, 2
    results, summary = run(RETRIEVER, [Case("fee", "fee?", [], reference="Fee is GYD $300")],
                           metric_factory=FlakyJudge, backoff=(0, 0))
    assert results[0].scores["contextual_precision"].score == 1.0  # failed twice, then succeeded
    assert summary["contextual_precision"]["errors"] == 0


def test_resume_rejudges_questions_with_judge_errors():
    cases = [Case("fee", "fee?", [], reference="Fee is GYD $300"),
             Case("hours", "hours?", [], reference="Hours are 8 to 4")]
    FlakyJudge.calls, FlakyJudge.fail_times = 0, 1
    results, summary = run(RETRIEVER, cases, metric_factory=FlakyJudge, backoff=())
    assert results[0].scores["contextual_precision"].error.startswith("TimeoutError")

    calls = []
    retriever = CannedRetriever(RETRIEVER.answers)
    retriever.search = lambda req, _s=retriever.search: calls.append(req.query) or _s(req)
    resumed, summary = run(retriever, cases, metric_factory=FakeMetric, done=from_json(to_json(results, summary)))
    assert calls == ["fee?"]                                      # only the question with a judge error
    assert resumed[0].scores["contextual_precision"].score == 1.0
    assert summary["contextual_precision"]["errors"] == 0


class FakeAnswerMetric(FakeMetric):
    """Answer metrics: pass when the answer mentions the reference's first word."""

    def measure(self, test_case, **_):
        if self.name in ("correctness", "faithfulness", "answer_relevancy"):
            assert test_case.actual_output is not None
            key = test_case.expected_output.split()[0].lower()
            hit = key in test_case.actual_output.lower()
            self.score, self.reason = (1.0, "ok") if hit else (0.0, "wrong")
            return self.score
        return super().measure(test_case)


def test_answer_metrics_use_generated_answer():
    thresholds = {"contextual_recall": 0.7, "correctness": 0.7, "faithfulness": 0.8, "answer_relevancy": 0.7}
    answers = {"fee?": "The fee is GYD $300 [1].", "hours?": "Unrelated reply."}
    cases = [Case("fee", "fee?", [], reference="Fee is GYD $300"),
             Case("hours", "hours?", [], reference="Hours are 8 to 4"),
             Case("empty", "nothing?", [], reference="Something")]
    results, summary = run(RETRIEVER, cases, metric_factory=FakeAnswerMetric, thresholds=thresholds,
                           answer_fn=lambda resp: answers.get(resp.query, "I don't know."))
    by_id = {r.id: r for r in results}
    assert by_id["fee"].answer == "The fee is GYD $300 [1]."
    assert by_id["fee"].scores["correctness"].passed
    assert not by_id["hours"].scores["correctness"].passed
    # nothing retrieved: retrieval scores 0, faithfulness trivially 1 (refusal), correctness still judged
    assert by_id["empty"].scores["contextual_recall"].score == 0.0
    assert by_id["empty"].scores["faithfulness"].score == 1.0
    assert by_id["empty"].scores["correctness"].score == 0.0
    assert summary["refused"] == 1
    assert summary["correctness"]["mean"] == round(1 / 3, 3)


def test_answer_metrics_require_answer_fn():
    with pytest.raises(ValueError):
        run(RETRIEVER, [Case("fee", "fee?", [], reference="x")], thresholds={"correctness": 0.7},
            metric_factory=FakeAnswerMetric)


def test_generate_answer_refuses_without_llm_call():
    from kb.generation import generate_answer
    from kb.schemas import NO_RELEVANT_CONTEXT_MESSAGE

    empty = RETRIEVER.search(QueryRequest(query="nothing?"))
    assert generate_answer(empty, model="unused-model") == NO_RELEVANT_CONTEXT_MESSAGE
