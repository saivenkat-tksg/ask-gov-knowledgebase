from kb.retrieve import Retriever
from kb.schemas import NO_RELEVANT_CONTEXT_MESSAGE, QueryRequest, QueryResponse
from kb.topics import document_topic, query_topics


def test_query_topics():
    assert query_topics("how much is a death certificate") == ["death"]
    assert query_topics("my husband died, what do I need to marry again") == ["death", "marriage"]
    assert query_topics("we are not married, can we still register our son") == ["birth", "marriage"]
    assert query_topics("do I have to pay the grant back later") == ["cash_grant"]
    assert query_topics("what time does the office open") == []


def test_document_topic_needs_a_dominant_topic():
    assert document_topic("Death registration. " * 20 + "Bring a birth certificate.") == "death"
    assert document_topic("Birth, death and marriage services. " * 20) is None  # a booklet on everything
    assert document_topic("Register a death.") is None  # too little to tell


class StubEmbedder:
    def embed(self, texts):
        return [[0.0] * 3 for _ in texts]


class RoutingRetriever(Retriever):
    """Records the filters of each search; the topic-filtered search finds `topic_hits` results."""

    def __init__(self, topic_hits: int, routing: bool = True):
        super().__init__(None, StubEmbedder(), None, iterative_scan=False, topic_routing=routing)
        self.topic_hits, self.searched = topic_hits, []

    def _search(self, req, qvec, filters):
        self.searched.append(filters)
        n = self.topic_hits if filters and "topic" in str(filters) else 1
        return QueryResponse.model_construct(query=req.query, filters=filters, status="ok" if n else "no_relevant_context",
                             message=None if n else NO_RELEVANT_CONTEXT_MESSAGE, reranked=True,
                             min_score_applied=0.5, results=[object()] * n, context="", took_ms=0.0)


def test_question_naming_a_topic_searches_only_that_topic():
    r = RoutingRetriever(topic_hits=2)
    resp = r.search(QueryRequest(query="death certificate fee", filters={"year": 2026}))
    assert r.searched == [{"$and": [{"year": 2026}, {"topic": {"$in": ["death"]}}]}]
    assert resp.topics == ["death"]


def test_falls_back_to_all_documents_when_topic_has_nothing():
    r = RoutingRetriever(topic_hits=0)
    resp = r.search(QueryRequest(query="death certificate fee"))
    assert r.searched == [{"topic": {"$in": ["death"]}}, None]
    assert resp.topics == [] and len(resp.results) == 1


def test_no_routing_without_topic_or_when_disabled():
    r = RoutingRetriever(topic_hits=2)
    r.search(QueryRequest(query="what time does the office open"))
    r.search(QueryRequest(query="death certificate fee", topic_routing=False))
    assert r.searched == [None, None]
    assert RoutingRetriever(topic_hits=2, routing=False).search(QueryRequest(query="death fee")).topics == []
