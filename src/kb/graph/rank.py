"""Graph search: rank chunks by how they connect to the entities a question names.

1. Entity linking: the question's words are looked up in a full-text index over entity names and
   aliases ("birth certificate fee" -> Service "Birth registration", Fee "free of charge", ...).
   Seeds are weighted by their match score, relative to the best match.
2. Scoring (current documents and facts only; superseded versions are ignored). The graph has no
   chunk nodes: chunk ids come from the document->entity MENTIONS links (`chunk_ids`) and from
   facts (`source_chunk`). Facts are followed up to `hops` steps out from each seed (multi-hop):
     chunk that states or names a seed                              weight x 1.0
     chunk that is the source of a fact on a path from a seed       weight x 0.8 x 0.5^(step-1)
     chunk that names an entity reached by a path                   weight x 0.4 x 0.5^(steps-1)
   so a fact two steps away counts half as much as one step away.
3. The facts on those paths come back too (with their step number), keyed by their source chunk,
   so the answer can follow a chain such as "Print team -PERFORMED_BY- Births Certificates
   -GOVERNED_BY-> Act 44:01" and cite each link.

The ranked chunk ids are a third candidate list for the retriever (next to vector and keyword
search): the retriever loads them from Postgres with the same filters, fuses all three lists by
reciprocal rank fusion and reranks the result, so the graph widens recall without bypassing the
rerank and relevance thresholds.
"""

import re
from dataclasses import dataclass, field

from kb.graph.schema import RELATION_TYPES
from kb.graph.store import GraphStore

FULLTEXT_INDEX = "kb_entity_search"
_STOPWORDS = set("""
a an and are as at be been but by can could did do does for from get got had has have how i if in into is it
its me my no not of on or our please should so than that the their them then there these they this to too
was we were what when where which who whom why will with would you your yours about any also just much many
need needs want wants tell know
""".split())


@dataclass
class GraphHit:
    chunk_id: str
    score: float
    facts: list[dict] = field(default_factory=list)


def lucene_query(question: str) -> str | None:
    """Question words as an OR query; longer words also match with one typo ("registar")."""
    words = [w for w in re.findall(r"[A-Za-z0-9]+", question.lower()) if len(w) > 2 and w not in _STOPWORDS]
    terms = []
    for w in dict.fromkeys(words):
        terms.append(w)
        if len(w) >= 6 and not w.isdigit():
            terms.append(f"{w}~1")
    return " OR ".join(terms) or None


_SEARCH = """
CALL db.index.fulltext.queryNodes($index, $q) YIELD node, score
WHERE node.kb = $kb
WITH node, score ORDER BY score DESC LIMIT $seeds
WITH collect({e: node, s: score}) AS found, max(score) AS top
UNWIND found AS seed
WITH seed.e AS e, seed.s / top AS w
WHERE w >= $min_weight
CALL (e, w) {
    MATCH (:Document {kb: $kb, current: true})-[m:MENTIONS]->(e)
    UNWIND m.chunk_ids AS cid
    RETURN cid, w * 1.0 AS s, null AS fact
  UNION ALL
    MATCH path = (e)-[rels:%(types)s*1..%(hops)d]-(:Entity)
    WHERE all(r IN rels WHERE r.kb = $kb AND r.current = true)
    WITH path, rels LIMIT $max_paths
    UNWIND range(0, size(rels) - 1) AS i
    WITH rels[i] AS r, min(i) AS i
    RETURN r.source_chunk AS cid, w * 0.8 * (0.5 ^ i) AS s,
           {subject: startNode(r).name, relation: type(r), object: endNode(r).name,
            quote: r.quote, chunk_id: r.source_chunk, hop: i + 1} AS fact
  UNION ALL
    MATCH path = (e)-[rels:%(types)s*1..%(hops)d]-(o:Entity)
    WHERE all(r IN rels WHERE r.kb = $kb AND r.current = true)
    WITH o, rels LIMIT $max_paths
    WITH o, min(size(rels)) AS d
    MATCH (:Document {kb: $kb, current: true})-[m:MENTIONS]->(o)
    UNWIND m.chunk_ids AS cid
    RETURN cid, w * 0.4 * (0.5 ^ (d - 1)) AS s, null AS fact
}
WITH cid, sum(s) AS score, collect(DISTINCT fact) AS facts
WHERE cid IS NOT NULL
RETURN cid, score, facts ORDER BY score DESC LIMIT $limit
"""


class GraphSearcher:
    def __init__(self, store: GraphStore, max_seeds: int = 8, min_seed_weight: float = 0.3, hops: int = 2,
                 max_paths: int = 300):
        """`hops`: how many facts to follow from each entity the question names (1-3)."""
        self.store = store
        self.max_seeds = max_seeds
        self.min_seed_weight = min_seed_weight
        self.hops = max(1, min(int(hops), 3))
        self.max_paths = max_paths
        # Relationship types and the hop limit can't be query parameters; both come from fixed values.
        self._query = _SEARCH % {"types": "|".join(RELATION_TYPES), "hops": self.hops}

    def search(self, question: str, limit: int = 40) -> list[GraphHit]:
        q = lucene_query(question)
        if q is None:
            return []
        rows = self.store._read(self._query, index=FULLTEXT_INDEX, q=q, seeds=self.max_seeds,
                                min_weight=self.min_seed_weight, max_paths=self.max_paths, limit=limit)
        return [GraphHit(r["cid"], round(float(r["score"]), 4), [f for f in r["facts"] if f]) for r in rows
                if r["cid"]]
