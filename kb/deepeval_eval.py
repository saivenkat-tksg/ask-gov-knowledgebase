"""LLM-judged evaluation with DeepEval: retrieval quality and answer quality.

For every question that has a `reference` answer, the normal search runs, an answer is
generated from the retrieved chunks (kb/generation.py), and a judge model scores:

Retrieval (the chunks):
    contextual_precision  relevant chunks are ranked above irrelevant ones
    contextual_recall     the chunks contain everything the reference answer needs
    contextual_relevancy  share of the retrieved text that is relevant to the question
Answer (the generated reply):
    correctness           the answer agrees with the reference answer (G-Eval)
    faithfulness          every claim in the answer is supported by the retrieved chunks
    answer_relevancy      the answer addresses the question

Unlike kb/evaluate.py, this accepts correct answers worded differently from the
`contains` label, and every score comes with the judge's reason. It costs judge-model
calls (a few per metric per question) plus one generation call per question when answer
metrics are on. Questions without a `reference` (the off-topic ones) are skipped;
`kb eval` covers refusals.
"""

import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable

from .evaluate import Case
from .schemas import QueryRequest, QueryResponse

RETRIEVAL_METRICS = ("contextual_precision", "contextual_recall", "contextual_relevancy")
ANSWER_METRICS = ("correctness", "faithfulness", "answer_relevancy")
METRIC_SETS = {"retrieval": RETRIEVAL_METRICS, "answer": ANSWER_METRICS, "all": RETRIEVAL_METRICS + ANSWER_METRICS}
DEFAULT_THRESHOLDS = {
    "contextual_precision": 0.7, "contextual_recall": 0.7, "contextual_relevancy": 0.5,
    "correctness": 0.7, "faithfulness": 0.8, "answer_relevancy": 0.7,
}
# gpt-4o-mini misread sources as a judge (e.g. "you need at least one ID" scored as contradicting
# "a National ID or a passport"), which made faithfulness and correctness numbers unreliable.
DEFAULT_JUDGE_MODEL = "gpt-4.1"
# DeepEval's own per-call limits (default: 2 attempts within 180 s) timed out on faithfulness, which
# reads every claim in the context; give each judge call more time and attempts.
JUDGE_ENV = {"DEEPEVAL_RETRY_MAX_ATTEMPTS": "3", "DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE": "120"}

# G-Eval steps for correctness: key facts must match; extra correct detail and other wording are fine.
CORRECTNESS_STEPS = [
    "Compare the facts in the actual output with the facts in the expected output.",
    "Heavily penalise any fact in the actual output that contradicts the expected output "
    "(fees, numbers, dates, deadlines, emails, phone numbers, office names, eligibility rules).",
    "Penalise omitting a key fact from the expected output that the question asks for.",
    "Do not penalise different wording, citation markers such as [1], or extra details that do not contradict "
    "the expected output.",
    "If the expected output says the information is not available and the actual output also says so, "
    "treat it as correct.",
]

PIPELINE_ERROR = "pipeline_error"
_FAILED_STATUSES = {PIPELINE_ERROR, "search_error"}  # "search_error": reports written by earlier versions
RETRY_BACKOFF = (5.0, 15.0, 30.0)  # seconds before each retry of a failed search or generation call

MetricFactory = Callable[[str, float, str], Any]
AnswerFn = Callable[[QueryResponse], str]


@dataclass
class MetricScore:
    score: float | None
    passed: bool
    reason: str | None = None
    error: str | None = None


@dataclass
class JudgedCase:
    id: str
    question: str
    status: str
    reranked: bool
    returned: list[str]
    scores: dict[str, MetricScore]
    answer: str | None = None


def configure_env(openai_api_key: str | None) -> None:
    """DeepEval reads its judge key and limits from the environment; keep its telemetry off.
    Call before importing deepeval."""
    os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
    for name, value in JUDGE_ENV.items():
        os.environ.setdefault(name, value)
    if openai_api_key and not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = openai_api_key


def deepeval_metric(name: str, threshold: float, model: str):
    from deepeval.metrics import (AnswerRelevancyMetric, ContextualPrecisionMetric, ContextualRecallMetric,
                                  ContextualRelevancyMetric, FaithfulnessMetric, GEval)
    from deepeval.test_case import SingleTurnParams

    if name == "correctness":
        return GEval(
            name="Correctness", evaluation_steps=CORRECTNESS_STEPS, model=model, threshold=threshold,
            evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT,
                               SingleTurnParams.EXPECTED_OUTPUT],
            async_mode=False,
        )
    cls = {
        "contextual_precision": ContextualPrecisionMetric,
        "contextual_recall": ContextualRecallMetric,
        "contextual_relevancy": ContextualRelevancyMetric,
        "faithfulness": FaithfulnessMetric,
        "answer_relevancy": AnswerRelevancyMetric,
    }[name]
    return cls(threshold=threshold, model=model, include_reason=True, async_mode=False)


def _contexts(response: QueryResponse) -> list[str]:
    # The source line lets the judge tell the near-identical GRO documents apart.
    return [
        f"Source: {r.citation.filename}" + (f", p. {r.citation.page}" if r.citation.page else "") + f"\n{r.text}"
        for r in response.results
    ]


def _returned(response: QueryResponse) -> list[str]:
    return [f"{r.citation.label} {r.citation.filename}" + (f", p. {r.citation.page}" if r.citation.page else "")
            for r in response.results]


def judge_case(case: Case, response: QueryResponse, thresholds: dict[str, float], model: str,
               metric_factory: MetricFactory = deepeval_metric, answer: str | None = None,
               backoff: tuple[float, ...] = RETRY_BACKOFF) -> JudgedCase:
    from deepeval.test_case import LLMTestCase

    contexts = _contexts(response)
    test_case = LLMTestCase(input=case.question, actual_output=answer, expected_output=case.reference,
                            retrieval_context=contexts or None)
    scores: dict[str, MetricScore] = {}
    for name, threshold in thresholds.items():
        if not contexts and name in RETRIEVAL_METRICS:
            scores[name] = MetricScore(0.0, False, f"Nothing retrieved ({response.status})")
            continue
        if not contexts and name == "faithfulness":
            # With no context the pipeline refuses without calling the LLM, so no claim can be unsupported.
            scores[name] = MetricScore(1.0, True, "Nothing retrieved; the answer is a refusal and makes no claims")
            continue
        metric = metric_factory(name, threshold, model)
        try:
            with_retry(lambda: metric.measure(test_case, _show_indicator=False), backoff)
        except Exception as exc:  # judge API errors, rate limits, malformed judge output, after retries
            scores[name] = MetricScore(None, False, error=f"{type(exc).__name__}: {exc}")
            continue
        score = float(metric.score)
        scores[name] = MetricScore(round(score, 3), score >= threshold, getattr(metric, "reason", None))
    return JudgedCase(case.id, case.question, response.status, response.reranked, _returned(response), scores, answer)


def failed_case(case: Case, error: str, thresholds: dict[str, float]) -> JudgedCase:
    return JudgedCase(case.id, case.question, PIPELINE_ERROR, False, [],
                      {name: MetricScore(None, False, error=error) for name in thresholds})


def with_retry(fn: Callable[[], Any], backoff: tuple[float, ...] = RETRY_BACKOFF) -> Any:
    """Network drops (DNS, timeouts, 5xx) shouldn't end a 25-minute run: retry, then give up on this question."""
    for wait in (*backoff, None):
        try:
            return fn()
        except Exception:
            if wait is None:
                raise
            time.sleep(wait)
    raise AssertionError("unreachable")


def summarize(results: list[JudgedCase], thresholds: dict[str, float], skipped: int, rerank: bool) -> dict[str, Any]:
    ran = [r for r in results if r.status not in _FAILED_STATUSES]
    summary: dict[str, Any] = {
        "questions_judged": len(results),
        "skipped_no_reference": skipped,
        "pipeline_errors": len(results) - len(ran),
        "rerank_fallbacks": sum(not r.reranked for r in ran) if rerank else 0,
        "refused": sum(r.status == "no_relevant_context" for r in ran),
    }
    for name in thresholds:
        scored = [r.scores[name] for r in results if name in r.scores and r.scores[name].score is not None]
        summary[name] = {
            "mean": round(sum(s.score for s in scored) / len(scored), 3) if scored else None,
            "pass_rate": round(sum(s.passed for s in scored) / len(scored), 3) if scored else None,
            "errors": sum(name not in r.scores or r.scores[name].score is None for r in results),
        }
    return summary


def run(retriever, cases: list[Case], *, top_k: int = 5, hybrid: bool = True, rerank: bool = True,
        min_score: float | None = None, delay: float = 0.0, model: str = DEFAULT_JUDGE_MODEL,
        thresholds: dict[str, float] | None = None, metric_factory: MetricFactory = deepeval_metric,
        answer_fn: AnswerFn | None = None, progress: Callable[[str], None] | None = None,
        done: dict[str, JudgedCase] | None = None, on_result: Callable[[list[JudgedCase]], None] | None = None,
        backoff: tuple[float, ...] = RETRY_BACKOFF) -> tuple[list[JudgedCase], dict[str, Any]]:
    """`delay` seconds are slept between searches (6.5 keeps a 10 calls/minute Cohere trial key under its limit).

    `answer_fn` turns a search response into an answer; it is required when any answer metric is selected.
    `done` holds results from an earlier, interrupted run; those questions are reused instead of re-judged
    (failed ones, and ones where a judge call failed, are run again). `on_result` is called with all results so far after every question, so a
    crash loses at most the question in progress.
    """
    thresholds = thresholds or {name: DEFAULT_THRESHOLDS[name] for name in RETRIEVAL_METRICS}
    needs_answer = any(name in ANSWER_METRICS for name in thresholds)
    if needs_answer and answer_fn is None:
        raise ValueError("answer metrics need answer_fn (see kb.generation.generate_answer)")
    done = {k: v for k, v in (done or {}).items()
            if v.status not in _FAILED_STATUSES and set(thresholds) <= set(v.scores)
            and not any(v.scores[name].error for name in thresholds)}
    judged = [c for c in cases if c.reference]
    results: list[JudgedCase] = []
    searched = 0
    for i, case in enumerate(judged):
        if case.id in done:
            results.append(done[case.id])
            if progress:
                progress(f"[{i + 1}/{len(judged)}] {case.id} (from previous run)")
            continue
        if delay and searched:
            time.sleep(delay)
        searched += 1
        req = QueryRequest(query=case.question, top_k=top_k, filters=case.filters,
                           hybrid=hybrid, rerank=rerank, min_score=min_score)
        stage = "search"
        try:
            response = with_retry(lambda: retriever.search(req), backoff)
            stage = "answer generation"
            answer = with_retry(lambda: answer_fn(response), backoff) if needs_answer else None
        except Exception as exc:
            results.append(failed_case(case, f"{stage} failed: {type(exc).__name__}: {exc}", thresholds))
            if progress:
                progress(f"[{i + 1}/{len(judged)}] {case.id} {stage.upper()} FAILED after retries: "
                         f"{type(exc).__name__}")
        else:
            results.append(judge_case(case, response, thresholds, model, metric_factory, answer, backoff))
            if progress:
                progress(f"[{i + 1}/{len(judged)}] {case.id}")
        if on_result:
            on_result(results)
    return results, summarize(results, thresholds, len(cases) - len(judged), rerank)


def lowest(results: list[JudgedCase], metric: str, n: int = 10) -> list[JudgedCase]:
    scored = [r for r in results if metric in r.scores and r.scores[metric].score is not None]
    return sorted(scored, key=lambda r: r.scores[metric].score)[:n]


def to_json(results: list[JudgedCase], summary: dict[str, Any]) -> str:
    return json.dumps({"summary": summary, "cases": [asdict(r) for r in results]}, indent=2, ensure_ascii=False)


def from_json(text: str) -> dict[str, JudgedCase]:
    """Results of an earlier run (as written by to_json), keyed by question id, for --resume."""
    out = {}
    for c in json.loads(text).get("cases", []):
        scores = {name: MetricScore(**s) for name, s in c["scores"].items()}
        out[c["id"]] = JudgedCase(c["id"], c["question"], c["status"], c["reranked"], c["returned"], scores,
                                  c.get("answer"))
    return out
