"""Context compression: keep only the sentences of each result that bear on the question.

A 512-token chunk that holds the answer also holds a dozen sentences about other things; the LLM reads
them all and DeepEval's contextual relevancy (relevant statements / all statements) counts them all.
compress() splits every result into units (sentences, table rows, the section heading), scores all
units against the question in one rerank call, and keeps a unit when it scores at least min_score and
within `margin` of the best unit of the whole query (rerank scores are not calibrated across queries:
fragments like "Start" score 0.5, so a fixed cut alone keeps too much). Each kept unit brings `window`
neighbours, a kept table row its header row, and every result keeps its section heading ("[Chapter 4:
GENERAL REGISTRAR OFFICE > ...]", which tells the near-identical agencies apart). A result where nothing
qualifies keeps its single best unit, so no result is emptied.

List and process questions are not compressed by the retriever: their answer is the whole section.
"""

import re

from kb.schemas import Evidence, RetrievedChunk

# A sentence ends at . ! or ? followed by space and an upper-case letter, digit, quote or bracket.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
# A line that starts its own unit rather than continuing the previous line's sentence.
_BLOCK_START = re.compile(r"(?:[-*•▪◦]|\d{1,2}[.)]|[a-z][.)]|\()\s")
GAP = "…"


def _is_heading(line: str) -> bool:
    return line.startswith("[") and line.endswith("]")


def split_units(text: str) -> list[str]:
    """Units of a chunk: the heading line, table rows, and sentences. PDF text breaks lines mid-
    sentence, so lines are joined into paragraphs first (a blank line, heading, table row or bullet
    starts a new one) and the paragraphs split at sentence ends."""
    units: list[str] = []
    paragraph: list[str] = []

    def flush():
        if paragraph:
            joined = re.sub(r"\s+", " ", " ".join(paragraph)).strip()
            units.extend(s.strip() for s in _SENTENCE_END.split(joined) if s.strip())
            paragraph.clear()

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            flush()
        elif line.startswith("|") or _is_heading(line):
            flush()
            units.append(line)
        else:
            if _BLOCK_START.match(line):
                flush()
            paragraph.append(line)
    flush()
    return units


def _is_table_separator(unit: str) -> bool:
    return bool(re.fullmatch(r"\|[\s:|-]+\|?", unit))


def _table_headers(units: list[str], index: int) -> list[int]:
    """Header and separator row indexes of the markdown table holding row `index` (empty if none)."""
    if not units[index].startswith("|"):
        return []
    start = index
    while start > 0 and units[start - 1].startswith("|"):
        start -= 1
    headers = [start]
    if start + 1 < len(units) and _is_table_separator(units[start + 1]):
        headers.append(start + 1)
    return [h for h in headers if h != index]


def _kept_text(units: list[str], keep: set[int]) -> str:
    """Kept units in document order; a gap marker where units were dropped between them."""
    parts: list[str] = []
    previous = None
    for i in sorted(keep):
        if previous is not None and i != previous + 1:
            parts.append(GAP)
        parts.append(units[i])
        previous = i
    return "\n".join(parts)


def compress(query: str, results: list[RetrievedChunk], reranker, min_score: float, margin: float,
             window: int = 0) -> list[RetrievedChunk]:
    """`results` with each text cut to its relevant units (copies; the order and labels are unchanged).
    Raises whatever the reranker raises: the caller decides how to degrade."""
    split = [split_units(r.text) for r in results]
    flat = [(n, i) for n, units in enumerate(split) for i in range(len(units))
            if not _is_heading(units[i]) and not _is_table_separator(units[i])]
    if not flat:
        return results
    order = reranker.rerank(query, [split[n][i] for n, i in flat], len(flat))
    scores: list[dict[int, float]] = [{} for _ in results]
    for position, score in order:
        n, i = flat[position]
        scores[n][i] = score
    cut = max(min_score, max((s for _, s in order), default=0.0) - margin)

    out: list[RetrievedChunk] = []
    for n, result in enumerate(results):
        units, chunk_scores = split[n], scores[n]
        if not chunk_scores:
            out.append(result)
            continue
        hits = [i for i, s in chunk_scores.items() if s >= cut] or [max(chunk_scores, key=chunk_scores.get)]
        keep: set[int] = set()
        for i in hits:
            keep.update(j for j in range(max(0, i - window), min(len(units), i + window + 1))
                        if not _is_heading(units[j]))
        for i in list(keep):
            keep.update(_table_headers(units, i))
        keep.update(i for i, u in enumerate(units) if _is_heading(u))
        evidence = [Evidence(text=units[i], score=round(chunk_scores[i], 4))
                    for i in sorted(hits, key=lambda i: -chunk_scores[i])]
        out.append(result.model_copy(update={"text": _kept_text(units, keep), "compressed": True,
                                             "evidence": evidence}))
    return out
