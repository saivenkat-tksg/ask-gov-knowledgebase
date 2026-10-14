import logging
import time
from collections.abc import Hashable, Sequence
from concurrent.futures import ThreadPoolExecutor
from uuid import UUID

import numpy as np
from psycopg import sql
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from kb.documents.repository import get_documents, stale_reasons
from kb.providers.embeddings import Embedder
from kb.providers.rerank import Reranker
from kb.retrieval.compress import compress
from kb.retrieval.filters import compile_filters
from kb.schemas import (
    NO_RELEVANT_CONTEXT_MESSAGE,
    Citation,
    GraphFact,
    QueryRequest,
    QueryResponse,
    RetrievedChunk,
)

log = logging.getLogger(__name__)

_COLUMNS = """c.id, c.labelchunk, c.document_id, c.chunk_index, c.content, c.page, c.char_start, c.char_end,
       c.metadata, d.filename, d.file_type,
       1 - (c.embedding <=> %s) AS vector_score"""

# Only the current version of each document, and only while it is in force.
_LIVE = """d.is_current
  AND (d.valid_from IS NULL OR d.valid_from <= current_date)
  AND (d.valid_until IS NULL OR d.valid_until >= current_date)"""

_VECTOR_SEARCH = f"""
SELECT {_COLUMNS}
FROM chunks c
JOIN documents d ON d.id = c.document_id
WHERE {_LIVE} AND {{where}}
ORDER BY c.embedding <=> %s
LIMIT %s
"""

# Postgres full-text search as the keyword (BM25-style) side. plainto_tsquery ANDs every
# term, which is too strict for natural questions, so its terms are OR'ed; ts_rank_cd then
# favours chunks matching more terms close together. Stopword-only questions match nothing.
_KEYWORD_SEARCH = f"""
SELECT {_COLUMNS}, ts_rank_cd(c.content_tsv, q.tsq, 1) AS keyword_score
FROM chunks c
JOIN documents d ON d.id = c.document_id
CROSS JOIN (
    SELECT NULLIF(replace(plainto_tsquery('english', %s)::text, ' & ', ' | '), '')::tsquery AS tsq
) q
WHERE c.content_tsv @@ q.tsq AND {_LIVE} AND {{where}}
ORDER BY keyword_score DESC
LIMIT %s
"""

# The chunks of one process (same `service` metadata) in one document, for process questions.
_SECTION = f"""
SELECT {_COLUMNS}
FROM chunks c
JOIN documents d ON d.id = c.document_id
WHERE c.document_id = %s AND c.metadata->>'service' = %s AND {_LIVE} AND {{where}}
ORDER BY c.chunk_index
"""

# Chunks the knowledge graph points at, loaded with the same filters as the other searches.
_BY_ID = f"""
SELECT {_COLUMNS}
FROM chunks c
JOIN documents d ON d.id = c.document_id
WHERE c.id = ANY(%s::uuid[]) AND {_LIVE} AND {{where}}
"""

RRF_K = 60


def reciprocal_rank_fusion(rankings: Sequence[Sequence[Hashable]], k: int = RRF_K,
                           weights: Sequence[float] | None = None) -> dict[Hashable, float]:
    """score(id) = sum over lists of weight / (k + rank). Items found by several lists rise to the top."""
    scores: dict[Hashable, float] = {}
    for n, ranking in enumerate(rankings):
        weight = weights[n] if weights else 1.0
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + weight / (k + rank)
    return scores


def found_by_label(sources: set[str]) -> str:
    if sources == {"vector", "keyword"}:
        return "both"
    return "+".join(s for s in ("vector", "keyword", "graph") if s in sources)


class Retriever:
    def __init__(
        self,
        pool: ConnectionPool,
        embedder: Embedder,
        reranker: Reranker,
        iterative_scan: bool,
        min_rerank_score: float = 0.0,
        min_vector_score: float = 0.0,
        max_document_age_days: int = 0,
        graph=None,
        graph_weight: float = 1.0,
        max_merged_results: int = 10,
        router=None,
        list_top_k: int = 10,
        process_max_chunks: int = 8,
        normal_top_k: int | None = None,
        compress_context: bool = False,
        compress_min_score: float = 0.5,
        compress_margin: float = 0.15,
        compress_window: int = 0,
    ):
        self.pool = pool
        self.embedder = embedder
        self.reranker = reranker
        self.iterative_scan = iterative_scan
        # Default relevance cuts; below these a chunk is treated as unrelated to the question.
        self.min_rerank_score = min_rerank_score
        self.min_vector_score = min_vector_score
        self.max_document_age_days = max_document_age_days
        # Knowledge-graph searcher (kb.graph.rank.GraphSearcher) or None; its list's weight in RRF.
        self.graph = graph
        self.graph_weight = graph_weight
        # Evidence cap for questions searched in several parts (multi_intent, comparison, multi_hop...).
        self.max_merged_results = max_merged_results
        # Query routing: kb.retrieval.router.Router or None. Result sizes for list and process questions.
        self.router = router
        self.list_top_k = list_top_k
        self.process_max_chunks = process_max_chunks
        # Results for a routed `normal` question (one fact from one place): fewer chunks, less noise.
        self.normal_top_k = normal_top_k
        # Context compression (kb.retrieval.compress): cut results to their relevant sentences.
        self.compress_context = compress_context
        self.compress_min_score = compress_min_score
        self.compress_margin = compress_margin
        self.compress_window = compress_window

    def search(self, req: QueryRequest) -> QueryResponse:
        """Search. With a router, the question is classified and searched by the plan for its type
        (kb.retrieval.router); without one (or with it turned off per request), one plain search."""
        started = time.perf_counter()
        if self.router is not None and (req.router if req.router is not None else True):
            response = self._search_routed(req)
        else:
            response = self._search_question(req)
        if req.compress if req.compress is not None else self.compress_context:
            response = self._compressed(req, response)
        response.took_ms = round((time.perf_counter() - started) * 1000, 1)
        return response

    def _search_routed(self, req: QueryRequest) -> QueryResponse:
        plan = self.router.classify(req.query)
        if plan.type in ("multi_intent", "comparison") and plan.sub_questions:
            response = self._search_multihop(req, plan.sub_questions)
        elif plan.type == "multi_hop" and plan.sub_questions:
            response = self._search_chain(req, plan.sub_questions)
        elif plan.type == "conditional" and plan.rewritten:
            response = self._search_multihop(req, [plan.rewritten])
        elif plan.type == "list":
            response = self._search_list(req)
        elif plan.type == "process":
            response = self._search_process(req)
        else:
            top_k = min(req.top_k, self.normal_top_k) if self.normal_top_k else req.top_k
            response = self._search_question(req.model_copy(update={"top_k": top_k}))
        response.query_type = plan.type
        return response

    def _compressed(self, req: QueryRequest, response: QueryResponse) -> QueryResponse:
        """`response` with each result cut to the sentences relevant to the question. List and process
        answers are whole sections, so they are left as they are; a rerank failure leaves all as is."""
        if (response.query_type in ("list", "process") or not response.results or not req.rerank
                or not self.reranker.enabled):
            return response
        try:
            results = compress(req.query, response.results, self.reranker, self.compress_min_score,
                               self.compress_margin, self.compress_window)
        except Exception as exc:
            log.warning("Context compression failed, returning whole chunks: %s", exc)
            return response
        return response.model_copy(update={"results": results, "context": format_context(results)})

    def _search_multihop(self, req: QueryRequest, subs: list[str]) -> QueryResponse:
        """Search the question and each sub-question in parallel, then interleave their best results so
        every part of the question is represented, up to max_merged_results."""
        queries = list(dict.fromkeys([req.query, *subs]))
        per_query = max(3, req.top_k // 2)
        with ThreadPoolExecutor(max_workers=len(queries)) as pool:
            parts = list(pool.map(lambda q: self._search_question(req.model_copy(update={"query": q,
                                                                                            "top_k": per_query})),
                                  queries))
        return self._merge(req, queries, parts, subs, per_query)

    def _search_chain(self, req: QueryRequest, steps: list[str]) -> QueryResponse:
        """Multi-hop: search the steps in order, writing each later step from what the previous search
        found ("What is the name of {answer}?" -> "What is the name of the Minister of Home Affairs?").
        The original question is searched alongside, and the evidence merged as for sub-questions."""
        per_query = max(3, req.top_k // 2)
        sub = lambda q: req.model_copy(update={"query": q, "top_k": per_query})  # noqa: E731
        with ThreadPoolExecutor(max_workers=1) as pool:
            original = pool.submit(self._search_question, sub(req.query))
            written: list[str] = []
            parts: list[QueryResponse] = []
            for step in steps:
                if written:
                    step = self.router.next_step(req.query, written[-1], step, parts[-1].context)
                written.append(step)
                parts.append(self._search_question(sub(step)))
            parts.insert(0, original.result())
        return self._merge(req, [req.query, *written], parts, written, per_query)

    def _merge(self, req: QueryRequest, queries: list[str], parts: list[QueryResponse], subs: list[str],
               per_query: int) -> QueryResponse:
        cap = max(req.top_k, min(self.max_merged_results, req.top_k + 2 * len(subs)))
        merged: list[RetrievedChunk] = []
        seen: set = set()
        for position in range(per_query):
            for query, part in zip(queries, parts):
                if position < len(part.results) and part.results[position].chunk_id not in seen and len(merged) < cap:
                    chunk = part.results[position]
                    seen.add(chunk.chunk_id)
                    merged.append(chunk.model_copy(update={"sub_question": None if query == req.query else query},
                                                    deep=True))
        for rank, chunk in enumerate(merged, start=1):
            chunk.rank = rank
            chunk.citation.label = f"[{rank}]"
        label = {str(c.chunk_id): c.citation.label for c in merged}
        facts, fact_keys = [], set()
        for part in parts:
            for f in part.facts:
                key = (f.subject, f.relation, f.object, str(f.chunk_id))
                if str(f.chunk_id) in label and key not in fact_keys:
                    fact_keys.add(key)
                    facts.append(f.model_copy(update={"source": label[str(f.chunk_id)]}))
        first = parts[0]
        return QueryResponse(
            query=req.query,
            filters=req.filters,
            status="ok" if merged else "no_relevant_context",
            message=None if merged else NO_RELEVANT_CONTEXT_MESSAGE,
            hybrid=req.hybrid,
            reranked=any(p.reranked for p in parts),
            min_score_applied=first.min_score_applied,
            warnings=list(dict.fromkeys(w for p in parts for w in p.warnings)),
            graph=any(p.graph for p in parts),
            sub_questions=subs,
            results=merged,
            facts=facts,
            context=format_context(merged),
            took_ms=0.0,
        )

    def _search_list(self, req: QueryRequest) -> QueryResponse:
        """List questions: rerank a wider pool and keep the best result of each process or section
        before second-best ones, so items from several places fit in the results."""
        wide = max(self.list_top_k, req.top_k)
        response = self._search_question(req.model_copy(update={"top_k": wide * 2,
                                                                 "candidate_k": max(req.candidate_k, wide * 8)}))
        firsts, rest, groups = [], [], set()
        for chunk in response.results:
            path = chunk.metadata.get("section_path") or []
            group = chunk.metadata.get("service") or (path[0] if path else chunk.citation.filename)
            (rest if group in groups else firsts).append(chunk)
            groups.add(group)
        return self._renumbered(response, (firsts + rest)[:wide])

    def _search_process(self, req: QueryRequest) -> QueryResponse:
        """Process questions: find the process, then return its section's chunks in document order
        (steps spread over several chunks), followed by the other results, up to process_max_chunks."""
        qvec = np.asarray(self.embedder.embed([req.query])[0], dtype=np.float32)
        response = self._search_question(req, qvec)
        if not response.results:
            return response
        top = response.results[0]
        service = top.metadata.get("service")
        if not service:
            return response
        rows = self._section_rows(qvec, req.filters, top.citation.document_id, service)
        # A window of the section around the best match, so a long section keeps the part asked about.
        cap = max(req.top_k, self.process_max_chunks)
        at = next((i for i, r in enumerate(rows) if r["id"] == top.chunk_id), 0)
        start = max(0, min(at - cap // 2, len(rows) - cap))
        found = {r.chunk_id: r for r in response.results}
        section = [found.get(r["id"]) or self._to_result(0, r, None) for r in rows[start:start + cap]]
        ids = {c.chunk_id for c in section}
        others = [c for c in response.results if c.chunk_id not in ids]
        return self._renumbered(response, (section + others)[:cap])

    def _section_rows(self, qvec, filters: dict | None, document_id, service: str) -> list[dict]:
        where, filter_params = compile_filters(filters)
        with self.pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            return cur.execute(sql.SQL(_SECTION).format(where=where),
                               [qvec, document_id, service, *filter_params]).fetchall()

    @staticmethod
    def _renumbered(response: QueryResponse, chunks: list[RetrievedChunk]) -> QueryResponse:
        """`response` with `chunks` as its results, ranked and labelled in this order; graph facts are
        kept for the chunks that remain, under their new labels."""
        chunks = [c.model_copy(deep=True) for c in chunks]
        for rank, chunk in enumerate(chunks, start=1):
            chunk.rank = rank
            chunk.citation.label = f"[{rank}]"
        label = {str(c.chunk_id): c.citation.label for c in chunks}
        facts = [f.model_copy(update={"source": label[str(f.chunk_id)]})
                 for f in response.facts if str(f.chunk_id) in label]
        return response.model_copy(update={"results": chunks, "facts": facts, "context": format_context(chunks)})

    def _search_question(self, req: QueryRequest, qvec=None) -> QueryResponse:
        """One search for req.query: vector + keyword + graph, rerank, thresholds."""
        if qvec is None:
            qvec = np.asarray(self.embedder.embed([req.query])[0], dtype=np.float32)
        return self._search(req, qvec, req.filters)

    def _search(self, req: QueryRequest, qvec, filters: dict | None) -> QueryResponse:
        where, filter_params = compile_filters(filters)
        candidate_k = max(req.candidate_k, req.top_k)
        warnings: list[str] = []
        candidates = self._candidates(req, qvec, where, filter_params, candidate_k, warnings)

        picked: list[tuple[dict, float | None]] | None = None
        if req.rerank and self.reranker.enabled and candidates:
            try:
                order = self.reranker.rerank(req.query, [c["content"] for c in candidates], req.top_k)
                picked = [(candidates[i], score) for i, score in order]
            except Exception as exc:  # rate limit, network, auth: degrade instead of failing the query
                log.warning("Rerank failed, falling back to retrieval order: %s", exc)
                warnings.append(
                    f"Rerank unavailable ({type(exc).__name__}); results use retrieval order and vector similarity, "
                    "which separate relevant from unrelated text less reliably."
                )
        reranked = picked is not None
        if picked is None:
            picked = [(c, None) for c in candidates[: req.top_k]]

        if req.min_score is not None:
            threshold, floor = req.min_score, 0.0
        else:
            threshold = self.min_rerank_score if reranked else self.min_vector_score
            # Rerank scores are unreliable for contentless queries ("hii", "bye" score 0.7 against
            # short generic chunks), so reranked results must also clear the vector floor.
            floor = self.min_vector_score
        picked = [(c, s) for c, s in picked
                  if (s if s is not None else c["vector_score"]) >= threshold and c["vector_score"] >= floor]

        results = [self._to_result(rank, row, score) for rank, (row, score) in enumerate(picked, start=1)]
        facts = _cited_facts(results, picked)
        warnings.extend(self._drift_warnings(results))
        return QueryResponse(
            query=req.query,
            filters=req.filters,
            status="ok" if results else "no_relevant_context",
            message=None if results else NO_RELEVANT_CONTEXT_MESSAGE,
            hybrid=req.hybrid,
            reranked=reranked,
            min_score_applied=threshold,
            warnings=warnings,
            graph=any(c.get("graph_score") is not None for c in candidates),
            results=results,
            facts=facts,
            context=format_context(results),
            took_ms=0.0,
        )

    def _candidates(self, req: QueryRequest, qvec, where, filter_params, k: int,
                    warnings: list[str] | None = None) -> list[dict]:
        """Vector hits, plus keyword hits when hybrid and knowledge-graph hits when enabled, merged
        best-first by reciprocal rank fusion. Every list obeys the same filters."""
        vector_rows = self._vector_search(qvec, where, filter_params, k)
        vector_rows.sort(key=lambda r: r["vector_score"], reverse=True)
        sources: dict = {}
        by_id: dict = {}
        for row in vector_rows:
            row["keyword_score"], row["graph_score"], row["graph_facts"] = None, None, []
            by_id[row["id"]] = row
            sources[row["id"]] = {"vector"}
        rankings = [[r["id"] for r in vector_rows]]
        weights = [1.0]

        if req.hybrid:
            keyword_rows = self._keyword_search(req.query, qvec, where, filter_params, k)
            for row in keyword_rows:
                if row["id"] in by_id:
                    by_id[row["id"]]["keyword_score"] = row["keyword_score"]
                else:
                    row["graph_score"], row["graph_facts"] = None, []
                    by_id[row["id"]] = row
                sources.setdefault(row["id"], set()).add("keyword")
            rankings.append([r["id"] for r in keyword_rows])
            weights.append(1.0)

        use_graph = self.graph is not None and (req.graph if req.graph is not None else True)
        if use_graph:
            graph_ids = self._graph_candidates(req.query, qvec, where, filter_params, k, by_id, warnings)
            for chunk_id in graph_ids:
                sources.setdefault(chunk_id, set()).add("graph")
            rankings.append(graph_ids)
            weights.append(self.graph_weight)

        for chunk_id, row in by_id.items():
            row["found_by"] = found_by_label(sources[chunk_id])
        if len(rankings) == 1:
            return vector_rows
        fused = reciprocal_rank_fusion(rankings, weights=weights)
        return sorted(by_id.values(), key=lambda r: fused[r["id"]], reverse=True)

    def _graph_candidates(self, query: str, qvec, where, filter_params, k: int, by_id: dict,
                          warnings: list[str] | None) -> list:
        """Chunk ids from the knowledge graph, best first, that also pass this search's filters
        (current version, valid dates, topic routing, user filters). Rows not yet in `by_id` are
        loaded from Postgres. A graph failure degrades to the other searches with a warning."""
        try:
            hits = self.graph.search(query, k)
        except Exception as exc:
            log.warning("Graph search failed, continuing without it: %s", exc)
            if warnings is not None:
                warnings.append(f"Knowledge graph unavailable ({type(exc).__name__}); "
                                "results come from vector and keyword search only.")
            return []
        if not hits:
            return []
        missing = [h.chunk_id for h in hits if UUID(h.chunk_id) not in by_id]
        for row in self._rows_by_id(missing, qvec, where, filter_params):
            row["keyword_score"], row["graph_facts"] = None, []
            by_id[row["id"]] = row
        ranked = []
        for hit in hits:
            row = by_id.get(UUID(hit.chunk_id))
            if row is None:
                continue  # filtered out, superseded or deleted in Postgres: Postgres decides
            row["graph_score"] = hit.score
            row["graph_facts"] = hit.facts
            ranked.append(row["id"])
        return ranked

    def _rows_by_id(self, ids: list[str], qvec, where, filter_params) -> list[dict]:
        if not ids:
            return []
        query = sql.SQL(_BY_ID).format(where=where)
        with self.pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            return cur.execute(query, [qvec, ids, *filter_params]).fetchall()

    def _vector_search(self, qvec, where, filter_params, k: int) -> list[dict]:
        query = sql.SQL(_VECTOR_SEARCH).format(where=where)
        params = [qvec, *filter_params, qvec, k]
        with self.pool.connection() as conn, conn.transaction():
            # HNSW returns at most ef_search rows; raise it to cover k.
            conn.execute("SELECT set_config('hnsw.ef_search', %s, true)", [str(min(max(k, 40), 1000))])
            if self.iterative_scan:
                # pgvector >= 0.8: keep scanning the index when filters discard candidates.
                conn.execute("SELECT set_config('hnsw.iterative_scan', 'relaxed_order', true)")
            with conn.cursor(row_factory=dict_row) as cur:
                return cur.execute(query, params).fetchall()

    def _keyword_search(self, text: str, qvec, where, filter_params, k: int) -> list[dict]:
        query = sql.SQL(_KEYWORD_SEARCH).format(where=where)
        params = [qvec, text, *filter_params, k]
        with self.pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            return cur.execute(query, params).fetchall()

    def _drift_warnings(self, results: list[RetrievedChunk]) -> list[str]:
        """One warning per cited document that may be out of date (review overdue, source gone)."""
        doc_ids = list(dict.fromkeys(r.citation.document_id for r in results))
        if not doc_ids:
            return []
        warnings = []
        for doc in get_documents(self.pool, doc_ids):
            reasons = stale_reasons(doc, self.max_document_age_days)
            if reasons:
                warnings.append(f"{doc.filename} (v{doc.version}) may be out of date: {'; '.join(reasons)}.")
        return warnings

    @staticmethod
    def _to_result(rank: int, row: dict, rerank_score: float | None) -> RetrievedChunk:
        meta = row["metadata"] or {}
        keyword_score = row.get("keyword_score")
        graph_score = row.get("graph_score")
        return RetrievedChunk(
            rank=rank,
            chunk_id=row["id"],
            chunk_label=row.get("labelchunk"),
            text=row["content"],
            found_by=row.get("found_by", "vector"),
            vector_score=round(float(row["vector_score"]), 6),
            keyword_score=None if keyword_score is None else round(float(keyword_score), 6),
            graph_score=None if graph_score is None else round(float(graph_score), 6),
            rerank_score=None if rerank_score is None else round(float(rerank_score), 6),
            metadata=meta,
            citation=Citation(
                label=f"[{rank}]",
                document_id=row["document_id"],
                filename=row["filename"],
                file_type=row["file_type"],
                page=row["page"],
                chunk_index=row["chunk_index"],
                char_start=row["char_start"],
                char_end=row["char_end"],
                title=meta.get("title"),
            ),
        )


def _cited_facts(results: list[RetrievedChunk], picked: list[tuple[dict, float | None]]) -> list[GraphFact]:
    """Graph facts whose source chunk is in the results, each cited with that chunk's label."""
    label = {str(r.chunk_id): r.citation.label for r in results}
    facts, seen = [], set()
    for row, _ in picked:
        for f in row.get("graph_facts") or []:
            key = (f["subject"], f["relation"], f["object"], f["chunk_id"])
            if f["chunk_id"] in label and key not in seen:
                seen.add(key)
                facts.append(GraphFact(subject=f["subject"], relation=f["relation"], object=f["object"],
                                       quote=f.get("quote"), source=label[f["chunk_id"]], chunk_id=f["chunk_id"]))
    return facts


def format_context(results: list[RetrievedChunk]) -> str:
    """Numbered context block, e.g. '[1] report.pdf, p. 4\\n<text>'.

    Uses the filename rather than the PDF title: titles are often missing or junk ("(anonymous)"), and
    the model needs to see which document each source comes from to keep services apart."""
    return "\n\n".join(
        f"{r.citation.label} {r.citation.filename}" + (f", p. {r.citation.page}" if r.citation.page is not None else "")
        + f"\n{r.text}"
        for r in results
    )
