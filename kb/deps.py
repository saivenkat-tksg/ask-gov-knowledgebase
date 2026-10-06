"""Process-wide singletons shared by the API and CLI."""

from functools import lru_cache

from psycopg_pool import ConnectionPool

from .config import get_settings
from .db import check_db, create_pool, supports_iterative_scan
from .embeddings import LiteLLMEmbedder
from .ingest import Ingestor
from .lc import KnowledgeBaseRetriever
from .rerank import CohereReranker
from .retrieve import Retriever


@lru_cache
def pgvector_version() -> str:
    return check_db(get_settings())


@lru_cache
def get_pool() -> ConnectionPool:
    pgvector_version()  # schema must be migrated before the pool registers the vector type
    return create_pool(get_settings())


@lru_cache
def get_embedder() -> LiteLLMEmbedder:
    return LiteLLMEmbedder(get_settings())


@lru_cache
def get_ingestor() -> Ingestor:
    return Ingestor(get_pool(), get_embedder(), get_settings())


@lru_cache
def get_retriever() -> Retriever:
    return Retriever(
        get_pool(),
        get_embedder(),
        CohereReranker(get_settings()),
        iterative_scan=supports_iterative_scan(pgvector_version()),
        min_rerank_score=get_settings().min_rerank_score,
        min_vector_score=get_settings().min_vector_score,
    )


def get_lc_retriever(**defaults) -> KnowledgeBaseRetriever:
    """LangChain retriever over the same pipeline; `defaults` sets top_k, filters, etc."""
    return KnowledgeBaseRetriever(retriever=get_retriever(), **defaults)


def close() -> None:
    if get_pool.cache_info().currsize:
        get_pool().close()
