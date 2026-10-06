import logging
import time
from collections.abc import Hashable, Sequence

import numpy as np
from psycopg import sql
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from .embeddings import Embedder
from .filters import compile_filters
from .rerank import Reranker
from .schemas import NO_RELEVANT_CONTEXT_MESSAGE, Citation, QueryRequest, QueryResponse, RetrievedChunk

log = logging.getLogger(__name__)

_COLUMNS = """c.id, c.document_id, c.chunk_index, c.content, c.page, c.char_start, c.char_end,
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

RRF_K = 60


def reciprocal_rank_fusion(rankings: Sequence[Sequence[Hashable]], k: int = RRF_K) -> dict[Hashable, float]:
    """score(id) = sum over lists of 1 / (k + rank). Items found by several lists rise to the top."""
    scores: dict[Hashable, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return scores


class Retriever:
    def __init__(
        self,
        pool: ConnectionPool,
        embedder: Embedder,
        reranker: Reranker,
        iterative_scan: bool,
        min_rerank_score: float = 0.0,
        min_vector_score: float = 0.0,
    ):
        self.pool = pool
        self.embedder = embedder
        self.reranker = reranker
        self.iterative_scan = iterative_scan
        # Default relevance cuts; below these a chunk is treated as unrelated to the question.
        self.min_rerank_score = min_rerank_score
        self.min_vector_score = min_vector_score

    def search(self, req: QueryRequest) -> QueryResponse:
        started = time.perf_counter()
        where, filter_params = compile_filters(req.filters)
        candidate_k = max(req.candidate_k, req.top_k)
        
        qvec = np.asarray(self.embedder.embed([req.query])[0], dtype=np.float32)

        candidates = self._candidates(req, qvec, where, filter_params, candidate_k)

        warnings: list[str] = []
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
            threshold = req.min_score
        else:
            threshold = self.min_rerank_score if reranked else self.min_vector_score
        picked = [(c, s) for c, s in picked if (s if s is not None else c["vector_score"]) >= threshold]

        results = [self._to_result(rank, row, score) for rank, (row, score) in enumerate(picked, start=1)]
        return QueryResponse(
            query=req.query,
            filters=req.filters,
            status="ok" if results else "no_relevant_context",
            message=None if results else NO_RELEVANT_CONTEXT_MESSAGE,
            hybrid=req.hybrid,
            reranked=reranked,
            min_score_applied=threshold,
            warnings=warnings,
            results=results,
            context=format_context(results),
            took_ms=round((time.perf_counter() - started) * 1000, 1),
        )

    def _candidates(self, req: QueryRequest, qvec, where, filter_params, k: int) -> list[dict]:
        """Vector hits, plus keyword hits when hybrid, merged best-first by reciprocal rank fusion."""
        vector_rows = self._vector_search(qvec, where, filter_params, k)
        vector_rows.sort(key=lambda r: r["vector_score"], reverse=True)
        for row in vector_rows:
            row["found_by"], row["keyword_score"] = "vector", None
        if not req.hybrid:
            return vector_rows

        keyword_rows = self._keyword_search(req.query, qvec, where, filter_params, k)
        by_id = {row["id"]: row for row in vector_rows}
        for row in keyword_rows:
            if row["id"] in by_id:
                by_id[row["id"]].update(found_by="both", keyword_score=row["keyword_score"])
            else:
                row["found_by"] = "keyword"
                by_id[row["id"]] = row

        fused = reciprocal_rank_fusion([[r["id"] for r in vector_rows], [r["id"] for r in keyword_rows]])
        return sorted(by_id.values(), key=lambda r: fused[r["id"]], reverse=True)

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

    @staticmethod
    def _to_result(rank: int, row: dict, rerank_score: float | None) -> RetrievedChunk:
        meta = row["metadata"] or {}
        keyword_score = row.get("keyword_score")
        return RetrievedChunk(
            rank=rank,
            chunk_id=row["id"],
            text=row["content"],
            found_by=row.get("found_by", "vector"),
            vector_score=round(float(row["vector_score"]), 6),
            keyword_score=None if keyword_score is None else round(float(keyword_score), 6),
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


def format_context(results: list[RetrievedChunk]) -> str:
    """Numbered context block, e.g. '[1] report.pdf, p. 4\\n<text>'."""
    return "\n\n".join(f"{r.citation.label} {r.citation.locator}\n{r.text}" for r in results)
