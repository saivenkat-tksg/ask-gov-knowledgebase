from datetime import date, datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field, computed_field


class IngestResult(BaseModel):
    filename: str
    status: Literal["ingested", "duplicate", "error"]
    document_id: UUID | None = None
    file_type: str | None = None
    num_chunks: int = 0
    doc_key: str | None = None
    version: int | None = None
    superseded_id: UUID | None = Field(default=None, description="Previous current version, now superseded")
    error: str | None = None


class DocumentInfo(BaseModel):
    id: UUID
    filename: str
    file_type: str
    size_bytes: int
    num_chunks: int
    metadata: dict[str, Any]
    created_at: datetime
    doc_key: str
    version: int
    is_current: bool
    superseded_at: datetime | None = None
    valid_from: date | None = None
    valid_until: date | None = None
    review_by: date | None = None
    source: str | None = Field(default=None, description="File path or URL that `kb refresh` re-checks")
    source_status: Literal["ok", "missing", "error"] | None = None
    last_checked_at: datetime | None = None


class StaleDocument(BaseModel):
    document: DocumentInfo
    reasons: list[str] = Field(description="Why the document may be out of date")


class RefreshResult(BaseModel):
    doc_key: str
    filename: str
    source: str
    outcome: Literal["unchanged", "updated", "missing", "error", "duplicate"]
    document_id: UUID = Field(description="The checked version, or the new version when updated")
    version: int | None = None
    detail: str | None = None


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=100)
    candidate_k: int = Field(default=40, ge=1, le=1000, description="Candidates taken from each search method (vector, keyword) before reranking")
    filters: dict[str, Any] | None = Field(default=None, description="Metadata filter, see kb/filters.py")
    hybrid: bool = Field(default=True, description="Also run keyword (full-text) search and fuse it with vector search")
    rerank: bool = True
    min_score: float | None = Field(
        default=None,
        description="Drop results below this score (rerank score if reranked, else cosine similarity). "
        "Omit to use the server default (KB_MIN_RERANK_SCORE / KB_MIN_VECTOR_SCORE); 0 disables the cut.",
    )
    topic_routing: bool | None = Field(
        default=None,
        description="Search only documents tagged with the topics the question names (birth, death, ...), "
        "falling back to all documents when they hold nothing relevant. Omit for the server default "
        "(KB_TOPIC_ROUTING).",
    )


class Citation(BaseModel):
    label: str  # "[1]" — the marker an LLM should cite
    document_id: UUID
    filename: str
    file_type: str
    page: int | None = None
    chunk_index: int
    char_start: int | None = None
    char_end: int | None = None
    title: str | None = None

    @computed_field
    @property
    def locator(self) -> str:
        where = f", p. {self.page}" if self.page is not None else ""
        return f"{self.title or self.filename}{where}"


class RetrievedChunk(BaseModel):
    rank: int
    chunk_id: UUID
    text: str
    found_by: Literal["vector", "keyword", "both"] = "vector"
    vector_score: float
    keyword_score: float | None = Field(default=None, description="Full-text rank; set when keyword search matched")
    rerank_score: float | None = None
    citation: Citation
    metadata: dict[str, Any]


NO_RELEVANT_CONTEXT_MESSAGE = "No relevant information was found in the knowledge base for this question."


class QueryResponse(BaseModel):
    query: str
    filters: dict[str, Any] | None
    status: Literal["ok", "no_relevant_context"]
    message: str | None = Field(default=None, description="Set when status is no_relevant_context")
    hybrid: bool = False
    reranked: bool
    min_score_applied: float = Field(description="Relevance cut used; compared to rerank_score if reranked, else vector_score")
    topics: list[str] = Field(
        default_factory=list,
        description="Topics the search was restricted to; empty when it searched all documents",
    )
    warnings: list[str] = Field(default_factory=list)
    results: list[RetrievedChunk]
    context: str = Field(description="Numbered, cited context block ready to drop into an LLM prompt")
    took_ms: float
