import re
from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="KB_", extra="ignore", populate_by_name=True)

    database_url: str = "postgresql://kb:kb@localhost:5433/kb"
    db_pool_size: int = 10
    # Postgres schema holding documents/chunks. Use a dedicated one (e.g. "kb") when the
    # database is shared with another application.
    db_schema: str = "public"

    # Embeddings via LiteLLM
    embedding_model: str = "text-embedding-3-large"
    embedding_dim: int = 1536
    embedding_batch_size: int = 128
    embedding_api_base: str | None = None
    # text-embedding-3-large returns 3072 dimensions; send `dimensions` so the API shortens them to
    # embedding_dim, matching the vector column.
    embedding_send_dimensions: bool = True
    openai_api_key: str | None = Field(
        default=None, validation_alias=AliasChoices("OPENAI_API_KEY", "KB_OPENAI_API_KEY")
    )

    # Rerank via Cohere
    cohere_api_key: str | None = Field(
        default=None, validation_alias=AliasChoices("COHERE_API_KEY", "KB_COHERE_API_KEY")
    )
    rerank_model: str = "rerank-v4.0-pro"

    # Answer generation (kb ask, kb deepeval) via LiteLLM
    generation_model: str = "gpt-4o-mini"

    # Bearer token the MCP server (kb.mcp_server) requires. Unset = no auth, so localhost only.
    mcp_token: str | None = None

    # Knowledge graph (kb.graph): `kb graph worker` LLM-extracts entities and relations from chunks
    # into Neo4j, and removes them again when chunks are deleted in Postgres (queue: graph_outbox).
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str | None = None
    neo4j_database: str = "neo4j"
    # Reads whole documents (in chapter-sized parts when long): a long-context model is the default.
    graph_extraction_model: str = "gpt-4.1-mini"
    graph_window_tokens: int = 20000  # largest document part per extraction call
    graph_max_output_tokens: int = 16000  # reply limit per call; a cut-off reply splits the part and retries
    graph_concurrency: int = 4  # extraction calls in parallel
    graph_batch_size: int = 20  # queue rows per worker batch
    graph_poll_seconds: float = 30.0  # worker re-checks the queue at least this often
    graph_aliases_file: str | None = None  # JSON {"alias": "canonical name"} for entity resolution
    # Search: add knowledge-graph chunks as a third list next to vector and keyword search; all
    # candidates are then reranked. A Neo4j outage only adds a warning (kb.graph.rank).
    graph_search: bool = True  # used whenever Neo4j is configured (KB_NEO4J_PASSWORD set)
    graph_weight: float = 1.0  # the graph list's weight in reciprocal rank fusion
    graph_timeout_s: float = 5.0  # connect/retry limit for graph search inside a request
    graph_hops: int = 2  # facts followed from each entity a question names (1-3)

    # Query routing (kb.retrieval.router): an LLM classifies each question (normal, multi_intent,
    # comparison, multi_hop, conditional, list, process) and the retriever searches by the plan for
    # that type. Questions shorter than router_min_words skip the call. Off: one plain search.
    query_router: bool = True
    router_model: str = "gpt-4.1-mini"
    router_min_words: int = 5
    router_max_subquestions: int = 3  # sub-questions, sides or multi-hop steps
    max_merged_results: int = 10  # evidence passed on for a multi-part question (vs top_k otherwise)
    list_top_k: int = 10  # results for a list question, at most one per process/section first
    process_max_chunks: int = 8  # chunks of the matched process returned in document order

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 64
    chunk_length_unit: Literal["tokens", "chars"] = "tokens"
    tokenizer_encoding: str = "cl100k_base"
    # structure: split at headings (PDF headings found from fonts), recursive only inside sections
    # longer than chunk_size; documents with fewer than structure_min_headings headings fall back to
    # recursive. recursive: the previous behaviour, size-based splitting only (kb.ingestion.structure).
    chunk_strategy: Literal["structure", "recursive"] = "structure"
    chunk_min_tokens: int = 150  # sections below this are merged with their siblings
    # With structure chunking: rebuild PDF tables (pdfplumber), DOCX and HTML tables as markdown tables
    # that are split only between rows, with the header repeated (kb.ingestion.tables).
    table_extraction: bool = True
    structure_min_headings: int = 3
    # Topics (kb.ingestion.topics): at ingest an LLM tags each section with 1 to topic_max_per_section
    # subject names, reusing the knowledge base's existing topics, stored as `topics` on every chunk
    # for filtering ({"topics": {"$contains": "Leave management"}}).
    topic_tagging: bool = True
    topic_model: str = "gpt-4.1-mini"
    topic_max_per_section: int = 3

    # Retrieval defaults
    default_top_k: int = 5
    default_candidate_k: int = 40
    # Relevance cuts: results below these are dropped and the query returns no_relevant_context.
    # Rerank scores (Cohere, 0-1) separate relevant from unrelated text well, except for contentless
    # queries ("hii" scores 0.7), so every result must also clear min_vector_score. 0.245 sits below
    # the lowest answer chunk in eval/questions_contains.jsonl (0.249) and above greetings (~0.23-0.24).
    min_rerank_score: float = 0.5
    min_vector_score: float = 0.245

    # OCR for PDF pages without a text layer (scanned documents).
    # auto: OCR only pages whose extracted text has fewer than ocr_min_chars letters/digits;
    # force: OCR every page (for PDFs with a broken text layer); off: never.
    ocr_mode: Literal["auto", "force", "off"] = "auto"
    # rapidocr needs only pip packages; tesseract needs the Tesseract binary on PATH.
    ocr_engine: Literal["rapidocr", "tesseract"] = "rapidocr"
    ocr_dpi: int = 300
    ocr_min_chars: int = 20
    ocr_min_confidence: float = 0.5  # rapidocr: drop text boxes scored below this
    ocr_lang: str = "eng"  # tesseract language codes, e.g. "eng+fra"

    max_upload_mb: int = 50

    # Knowledge drift: a current document without a review_by date counts as stale (kb stale,
    # query warnings) once it is older than this. 0 disables the age check.
    max_document_age_days: int = 365
    source_timeout_s: float = 30.0  # kb refresh: timeout for downloading a source URL

    @field_validator("db_schema")
    @classmethod
    def _valid_schema(cls, value: str) -> str:
        if not re.fullmatch(r"[a-z_][a-z0-9_]*", value):
            raise ValueError("KB_DB_SCHEMA must be lowercase letters, digits or underscores")
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
