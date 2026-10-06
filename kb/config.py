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
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    embedding_batch_size: int = 128
    embedding_api_base: str | None = None
    embedding_send_dimensions: bool = False
    openai_api_key: str | None = Field(
        default=None, validation_alias=AliasChoices("OPENAI_API_KEY", "KB_OPENAI_API_KEY")
    )

    # Rerank via Cohere
    cohere_api_key: str | None = Field(
        default=None, validation_alias=AliasChoices("COHERE_API_KEY", "KB_COHERE_API_KEY")
    )
    rerank_model: str = "rerank-v3.5"

    # Answer generation (kb ask, kb deepeval) via LiteLLM
    generation_model: str = "gpt-4o-mini"

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 64
    chunk_length_unit: Literal["tokens", "chars"] = "tokens"
    tokenizer_encoding: str = "cl100k_base"

    # Retrieval defaults
    default_top_k: int = 3
    default_candidate_k: int = 40
    # Relevance cuts: results below these are dropped and the query returns no_relevant_context.
    # Rerank scores (Cohere, 0-1) separate relevant from unrelated text well; cosine similarity
    # does not, so the vector fallback only blocks clearly unrelated text.
    min_rerank_score: float = 0.8
    min_vector_score: float = 0.2

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

    @field_validator("db_schema")
    @classmethod
    def _valid_schema(cls, value: str) -> str:
        if not re.fullmatch(r"[a-z_][a-z0-9_]*", value):
            raise ValueError("KB_DB_SCHEMA must be lowercase letters, digits or underscores")
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
