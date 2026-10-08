"""LangChain adapter: expose the pipeline as a `BaseRetriever` returning `Document`s.

    from kb.deps import get_lc_retriever
    docs = get_lc_retriever().invoke("flu clinic hours", filters={"department": "health"})
    docs[0].metadata["citation"]["label"]   # "[1]"

Search stays in our own pgvector + Cohere code (HNSW tuning, JSONB filters,
exact citations); this class only makes it composable with LCEL chains.
"""

from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from .schemas import QueryRequest, QueryResponse

_OVERRIDES = {"top_k", "candidate_k", "filters", "hybrid", "rerank", "min_score"}


def to_documents(response: QueryResponse) -> list[Document]:
    return [
        Document(
            id=str(r.chunk_id),
            page_content=r.text,
            metadata={
                **r.metadata,
                "rank": r.rank,
                "found_by": r.found_by,
                "vector_score": r.vector_score,
                "keyword_score": r.keyword_score,
                "rerank_score": r.rerank_score,
                "citation": r.citation.model_dump(mode="json"),
            },
        )
        for r in response.results
    ]


def format_docs(docs: list[Document]) -> str:
    """Same numbered, cited block as QueryResponse.context."""
    def source(c: dict) -> str:
        return f"{c['label']} {c['filename']}" + (f", p. {c['page']}" if c.get("page") is not None else "")

    return "\n\n".join(f"{source(d.metadata['citation'])}\n{d.page_content}" for d in docs)


class KnowledgeBaseRetriever(BaseRetriever):
    """Per-call overrides: `.invoke(query, filters=..., top_k=..., rerank=...)`."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    retriever: Any  # kb.retrieve.Retriever, or anything with .search(QueryRequest)
    top_k: int = 5
    candidate_k: int = 40
    filters: dict[str, Any] | None = None
    hybrid: bool = True
    rerank: bool = True
    min_score: float | None = None

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun, **kwargs: Any
    ) -> list[Document]:
        unknown = kwargs.keys() - _OVERRIDES
        if unknown:
            raise TypeError(f"Unexpected retriever arguments: {', '.join(sorted(unknown))}")
        params = {name: getattr(self, name) for name in _OVERRIDES} | kwargs
        return to_documents(self.retriever.search(QueryRequest(query=query, **params)))
