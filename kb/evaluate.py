"""Retrieval evaluation: run labelled questions through the pipeline and score the results.

Each line of the questions file is a JSON object:

    {"id": "hours", "question": "when are flu clinics open?",
     "expected": [{"chunk_id": "flu_clinics_chunk_3"}]}

`expected` lists the chunks that answer the question. A result matches an expected
item when every key given matches: `chunk_id` (the chunk's label, see chunk_label(),
or its UUID; a list means any one of them), `filename` (equal), `page` (equal) and `contains` (case-insensitive
substring of the chunk text). Chunk labels follow chunk_index, so they must be
regenerated (scripts/expected_to_chunk_ids.py) after re-chunking. Leave `expected` empty
for questions the knowledge base cannot answer; those should come back as
no_relevant_context. An optional `filters` object is passed through to the query.
An optional `reference` string holds the correct answer as plain text; it is
used by the LLM-judged evaluation in kb/deepeval_eval.py, not by these metrics.

Metrics (answerable questions):
    hit@1 / hit@k   share of questions whose first matching chunk is at rank 1 / in the top k
    mrr             mean of 1 / rank of the first matching chunk (0 when missed)
    recall          share of expected items found in the top k
    false_refusal   share answered with no_relevant_context although an answer exists
Unanswerable questions:
    correct_refusal share answered with no_relevant_context
"""

import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .schemas import QueryRequest, QueryResponse, RetrievedChunk

_MATCH_KEYS = {"chunk_id", "filename", "page", "contains"}

# (name, hybrid, rerank) for `kb eval --compare`
CONFIGS = [("vector", False, False), ("hybrid", True, False), ("hybrid+rerank", True, True)]


@dataclass
class Case:
    id: str
    question: str
    expected: list[dict[str, Any]]
    filters: dict[str, Any] | None = None
    reference: str | None = None


@dataclass
class CaseResult:
    id: str
    question: str
    answerable: bool
    status: str
    first_hit_rank: int | None
    found: int
    expected: int
    returned: list[str]
    took_ms: float
    reranked: bool = False


def load_cases(path: str | Path) -> list[Case]:
    cases: list[Case] = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{n}: invalid JSON: {exc}") from exc
        if not isinstance(raw, dict) or not raw.get("question"):
            raise ValueError(f"{path}:{n}: each line needs a 'question'")
        expected = raw.get("expected", [])
        for item in expected:
            if not isinstance(item, dict) or not item.keys() & _MATCH_KEYS or item.keys() - _MATCH_KEYS:
                raise ValueError(f"{path}:{n}: expected items use only {sorted(_MATCH_KEYS)}")
        reference = raw.get("reference")
        if reference is not None and (not isinstance(reference, str) or not reference.strip()):
            raise ValueError(f"{path}:{n}: 'reference' must be a non-empty string")
        cases.append(Case(str(raw.get("id", n)), raw["question"], expected, raw.get("filters"), reference))
    if not cases:
        raise ValueError(f"{path}: no questions found")
    return cases


def doc_key(filename: str) -> str:
    """'cash_grant (1)(1).pdf' -> 'cash_grant': the stem without download-copy suffixes, as an identifier."""
    stem = re.sub(r"\(\d+\)", "", Path(filename).stem)
    return re.sub(r"\W+", "_", stem).strip("_")


def chunk_label(filename: str, chunk_index: int) -> str:
    """Stable, readable chunk id used in questions files, e.g. 'cash_grant_chunk_3'."""
    return f"{doc_key(filename)}_chunk_{chunk_index}"


def matches(result: RetrievedChunk, item: dict[str, Any]) -> bool:
    if "chunk_id" in item:
        ids = item["chunk_id"] if isinstance(item["chunk_id"], list) else [item["chunk_id"]]
        mine = {chunk_label(result.citation.filename, result.citation.chunk_index).lower(), str(result.chunk_id)}
        if not mine & {str(i).lower() for i in ids}:
            return False
    if "filename" in item and result.citation.filename != item["filename"]:
        return False
    if "page" in item and result.citation.page != item["page"]:
        return False
    if "contains" in item and item["contains"].lower() not in result.text.lower():
        return False
    return True


def score_case(case: Case, response: QueryResponse) -> CaseResult:
    ranks = [next((r.rank for r in response.results if matches(r, item)), None) for item in case.expected]
    hits = [r for r in ranks if r is not None]
    return CaseResult(
        id=case.id,
        question=case.question,
        answerable=bool(case.expected),
        status=response.status,
        first_hit_rank=min(hits, default=None),
        found=len(hits),
        expected=len(case.expected),
        # Filename rather than locator: PDF titles are often junk like "(anonymous)".
        returned=[f"{r.citation.label} {chunk_label(r.citation.filename, r.citation.chunk_index)}"
                  + (f", p. {r.citation.page}" if r.citation.page else "")
                  for r in response.results],
        took_ms=response.took_ms,
        reranked=response.reranked,
    )


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def summarize(results: list[CaseResult], rerank: bool = False) -> dict[str, Any]:
    answerable = [r for r in results if r.answerable]
    unanswerable = [r for r in results if not r.answerable]
    refused = lambda r: r.status == "no_relevant_context"  # noqa: E731
    return {
        # Searches that asked for rerank but fell back to retrieval order (rate limit, network, no key).
        "rerank_fallbacks": sum(not r.reranked for r in results) if rerank else 0,
        "questions": len(results),
        "answerable": len(answerable),
        "unanswerable": len(unanswerable),
        "hit@1": _mean([r.first_hit_rank == 1 for r in answerable]),
        "hit@k": _mean([r.first_hit_rank is not None for r in answerable]),
        "mrr": _mean([1 / r.first_hit_rank if r.first_hit_rank else 0.0 for r in answerable]),
        "recall": _mean([r.found / r.expected for r in answerable]),
        "false_refusal": _mean([refused(r) for r in answerable]),
        "correct_refusal": _mean([refused(r) for r in unanswerable]),
        "avg_ms": _mean([r.took_ms for r in results]),
    }


def run(retriever, cases: list[Case], top_k: int = 5, hybrid: bool = True, rerank: bool = True,
        min_score: float | None = None, delay: float = 0.0) -> tuple[list[CaseResult], dict[str, Any]]:
    """`delay` seconds are slept between searches, e.g. 6.5 to stay under a 10 calls/minute rerank key."""
    results = []
    for i, case in enumerate(cases):
        if delay and i:
            time.sleep(delay)
        req = QueryRequest(query=case.question, top_k=top_k, filters=case.filters,
                           hybrid=hybrid, rerank=rerank, min_score=min_score)
        results.append(score_case(case, retriever.search(req)))
    return results, summarize(results, rerank)


def failures(results: list[CaseResult]) -> list[str]:
    lines = []
    for r in results:
        if r.answerable and r.first_hit_rank is None:
            got = "; ".join(r.returned) or r.status
            lines.append(f"MISS      {r.id}: {r.question!r}  got: {got}")
        elif r.answerable and r.found < r.expected:
            lines.append(f"PARTIAL   {r.id}: {r.question!r}  found {r.found}/{r.expected} expected chunks")
        elif not r.answerable and r.status != "no_relevant_context":
            lines.append(f"NO-REFUSE {r.id}: {r.question!r}  returned: {'; '.join(r.returned)}")
    return lines


def to_json(reports: dict[str, tuple[list[CaseResult], dict[str, Any]]]) -> str:
    return json.dumps(
        {name: {"summary": summary, "cases": [asdict(r) for r in results]}
         for name, (results, summary) in reports.items()},
        indent=2,
    )
