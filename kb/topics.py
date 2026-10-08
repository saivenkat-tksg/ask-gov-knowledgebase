"""Topic routing: keep a question about one service from being answered with another's documents.

Each document about a single service is tagged at ingest with `topic` metadata (detected from its
text, or set explicitly with `--metadata '{"topic": "death"}'`). A question that names a topic is
searched only in documents tagged with it; when that finds nothing relevant, the search runs
again over everything. Documents that cover several services (the MoHA booklet) get no tag, so
they only answer questions that name no topic, or that the tagged documents cannot answer.

Detection is keyword-based: cheap, predictable, and easy to extend for a new service.
"""

import re

TOPICS: dict[str, list[str]] = {
    "cash_grant": [r"\bcash\b", r"\bgrants?\b", r"\b100,?000\b"],
    "birth": [r"\bbirths?\b", r"\bborn\b", r"\bbab(y|ies)\b", r"\bnewborns?\b", r"\b(son|daughter|child)\b"],
    "death": [r"\bdeaths?\b", r"\bdied\b", r"\bdies\b", r"\bdeceased\b", r"\bdead\b", r"\bfuneral\b",
              r"\bbur(y|ied|ial)\b", r"\bcremation\b", r"\bpass(ed away|ing)\b"],
    "marriage": [r"\bmarr(y|ied|ies|ying|iages?)\b", r"\bwedding\b", r"\bbride\b", r"\bgroom\b",
                 r"\bdivorced?\b"],
}
_PATTERNS = {topic: [re.compile(p, re.IGNORECASE) for p in patterns] for topic, patterns in TOPICS.items()}

# A document is about one topic when that topic has at least this many keyword hits and this share of all hits.
DOC_MIN_HITS = 10
DOC_MIN_SHARE = 0.6


def topic_hits(text: str) -> dict[str, int]:
    return {topic: sum(len(p.findall(text)) for p in patterns) for topic, patterns in _PATTERNS.items()}


def query_topics(query: str) -> list[str]:
    """Every topic the question mentions, e.g. ["death", "marriage"] for "my husband died, can I remarry"."""
    return [topic for topic, hits in topic_hits(query).items() if hits]


def document_topic(text: str) -> str | None:
    """The topic a document is mainly about, or None for a document about several (or none)."""
    hits = topic_hits(text)
    topic, best = max(hits.items(), key=lambda kv: kv[1])
    total = sum(hits.values())
    return topic if best >= DOC_MIN_HITS and best >= DOC_MIN_SHARE * total else None
