"""Process-wide singletons shared by the API and CLI."""

from functools import lru_cache

from psycopg_pool import ConnectionPool

from kb.config import get_settings
from kb.db.database import check_db, create_pool, supports_iterative_scan
from kb.ingestion.ingestor import Ingestor
from kb.providers.embeddings import LiteLLMEmbedder
from kb.providers.rerank import CohereReranker
from kb.retrieval.langchain_retriever import KnowledgeBaseRetriever
from kb.retrieval.retriever import Retriever


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
def get_topic_tagger():
    """Topic tagger for ingestion, or None when KB_TOPIC_TAGGING is off."""
    s = get_settings()
    if not s.topic_tagging:
        return None
    from kb.ingestion.topics import TopicTagger

    return TopicTagger(s.topic_model, s.openai_api_key, s.topic_max_per_section)


@lru_cache
def get_ingestor() -> Ingestor:
    return Ingestor(get_pool(), get_embedder(), get_settings(), get_topic_tagger())


@lru_cache
def get_search_graph():
    """GraphSearcher for query time (kept open; short timeouts), or None when KB_GRAPH_SEARCH is off or
    Neo4j isn't configured."""
    s = get_settings()
    if not s.graph_search or not s.neo4j_password:
        return None
    from kb.graph.rank import GraphSearcher
    from kb.graph.store import GraphStore

    return GraphSearcher(GraphStore(s.neo4j_uri, s.neo4j_user, s.neo4j_password, s.neo4j_database, s.db_schema,
                                    timeout=s.graph_timeout_s), hops=s.graph_hops)


@lru_cache
def get_router():
    """Query router, or None when KB_QUERY_ROUTER is off."""
    s = get_settings()
    if not s.query_router:
        return None
    from kb.retrieval.router import Router

    return Router(s.router_model, s.openai_api_key, s.router_max_subquestions, s.router_min_words)


@lru_cache
def get_retriever() -> Retriever:
    return Retriever(
        get_pool(),
        get_embedder(),
        CohereReranker(get_settings()),
        iterative_scan=supports_iterative_scan(pgvector_version()),
        min_rerank_score=get_settings().min_rerank_score,
        min_vector_score=get_settings().min_vector_score,
        max_document_age_days=get_settings().max_document_age_days,
        graph=get_search_graph(),
        graph_weight=get_settings().graph_weight,
        max_merged_results=get_settings().max_merged_results,
        router=get_router(),
        list_top_k=get_settings().list_top_k,
        process_max_chunks=get_settings().process_max_chunks,
        normal_top_k=get_settings().normal_top_k or None,
        compress_context=get_settings().compress_context,
        compress_min_score=get_settings().compress_min_score,
        compress_margin=get_settings().compress_margin,
        compress_window=get_settings().compress_window,
    )


def get_graph_store():
    """Neo4j store for this knowledge base (KB_DB_SCHEMA). Not cached: callers close it."""
    from kb.graph.store import GraphStore

    s = get_settings()
    return GraphStore(s.neo4j_uri, s.neo4j_user, s.neo4j_password, s.neo4j_database, s.db_schema)


def get_lc_retriever(**defaults) -> KnowledgeBaseRetriever:
    """LangChain retriever over the same pipeline; `defaults` sets top_k, filters, etc."""
    return KnowledgeBaseRetriever(retriever=get_retriever(), **defaults)


def close() -> None:
    if get_pool.cache_info().currsize:
        get_pool().close()
    if get_search_graph.cache_info().currsize and get_search_graph() is not None:
        get_search_graph().store.close()
