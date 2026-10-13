import json
from uuid import UUID

import litellm

from kb.retrieval.retriever import Retriever
from kb.retrieval.router import QueryPlan, Router
from kb.schemas import Citation, GraphFact, QueryRequest, QueryResponse, RetrievedChunk


def _reply(payload):
    class Message:
        content = json.dumps(payload)

    class Choice:
        message = Message()

    class Response:
        choices = [Choice()]

    return Response()


def _llm(monkeypatch, payload):
    calls = []
    monkeypatch.setattr(litellm, "completion", lambda **kw: calls.append(kw) or _reply(payload))
    return calls


LONG = "What database does the agency that issues firearm licences use for its records?"


def test_short_questions_are_normal_without_a_call(monkeypatch):
    calls = _llm(monkeypatch, {"type": "list"})
    assert Router("m").classify("GRO opening hours?") == QueryPlan()
    assert calls == []


def test_classify_reads_each_plan(monkeypatch):
    _llm(monkeypatch, {"type": "multi_hop", "sub_questions": ["Which agency issues firearm licences?",
                                                              "which agency issues firearm licences?",
                                                              "What database does {answer} use?"]})
    plan = Router("m").classify(LONG)
    assert plan == QueryPlan("multi_hop", ["Which agency issues firearm licences?", "What database does {answer} use?"])

    _llm(monkeypatch, {"type": "conditional", "rewritten": "  How does a  student extend a visa? ",
                       "sub_questions": ["ignored"]})
    assert Router("m").classify(LONG) == QueryPlan("conditional", [], "How does a student extend a visa?")

    calls = _llm(monkeypatch, {"type": "process"})
    assert Router("m").classify(LONG) == QueryPlan("process")
    assert calls[0]["temperature"] == 0 and calls[0]["response_format"] == {"type": "json_object"}


def test_incomplete_or_unknown_plans_fall_back(monkeypatch):
    _llm(monkeypatch, {"type": "comparison", "sub_questions": ["only one side?"]})
    assert Router("m").classify(LONG) == QueryPlan("comparison")  # labelled, searched as a normal question
    _llm(monkeypatch, {"type": "conditional"})
    assert Router("m").classify(LONG) == QueryPlan("conditional")
    _llm(monkeypatch, {"type": "something_else"})
    assert Router("m").classify(LONG) == QueryPlan()
    monkeypatch.setattr(litellm, "completion", lambda **kw: (_ for _ in ()).throw(TimeoutError("slow")))
    assert Router("m").classify(LONG) == QueryPlan()


def test_next_step_fills_the_answer(monkeypatch):
    calls = _llm(monkeypatch, {"answer": "the Ministry", "next": "What database does the Ministry use?"})
    router = Router("m")
    assert router.next_step(LONG, "Which agency?", "What database does {answer} use?", "[1] b.pdf\nThe Ministry "
                            "issues licences") == "What database does the Ministry use?"
    assert "The Ministry issues licences" in calls[0]["messages"][0]["content"]
    assert router.next_step(LONG, "a", "No slot here?", "context") == "No slot here?"  # nothing to fill: no call
    assert len(calls) == 1
    _llm(monkeypatch, {"next": "Still {answer}?"})
    assert router.next_step(LONG, "a", "What does {answer} use?", "ctx") == "What does it use?"
    assert router.next_step(LONG, "a", "What does {answer} use?", "") == "What does it use?"


# --- the retriever's strategies -----------------------------------------------------------------------

def chunk(n: int, service: str | None = None, rank: int = 1) -> RetrievedChunk:
    return RetrievedChunk(rank=rank, chunk_id=UUID(int=n), text=f"text {n}", vector_score=0.5, rerank_score=0.9,
                          metadata={"service": service} if service else {},
                          citation=Citation(label=f"[{rank}]", document_id=UUID(int=99), filename="b.pdf",
                                            file_type="pdf", chunk_index=n))


def response(query: str, chunks: list[RetrievedChunk], facts=()) -> QueryResponse:
    results = [c.model_copy(update={"rank": r}) for r, c in enumerate(chunks, start=1)]
    return QueryResponse(query=query, filters=None, status="ok" if chunks else "no_relevant_context", reranked=True,
                         min_score_applied=0.5, results=results, facts=list(facts), context=f"context for {query}",
                         took_ms=0.0)


class FakeRouter:
    def __init__(self, plan: QueryPlan):
        self.plan, self.steps = plan, []

    def classify(self, question):
        return self.plan

    def next_step(self, question, step, next_step, context):
        self.steps.append((step, next_step, context))
        return next_step.replace("{answer}", "the Ministry")


class FakeEmbedder:
    def embed(self, texts):
        return [[0.0] * 3 for _ in texts]


def make(monkeypatch, plan: QueryPlan, answers: dict) -> Retriever:
    r = Retriever(pool=None, embedder=FakeEmbedder(), reranker=None, iterative_scan=False,
                  router=FakeRouter(plan), list_top_k=4, process_max_chunks=4)
    r.seen = []
    monkeypatch.setattr(r, "_search_question",
                        lambda req, qvec=None: r.seen.append((req.query, req.top_k)) or answers[req.query])
    return r


Q = "What database does the agency that issues firearm licences use for its records?"


def test_multi_intent_searches_each_part(monkeypatch):
    plan = QueryPlan("multi_intent", ["How many staff?", "What fee?"])
    answers = {Q: response(Q, [chunk(1)]), "How many staff?": response("a", [chunk(2)]),
               "What fee?": response("b", [chunk(3)])}
    r = make(monkeypatch, plan, answers)
    resp = r.search(QueryRequest(query=Q))
    assert resp.query_type == "multi_intent" and resp.sub_questions == ["How many staff?", "What fee?"]
    assert [c.chunk_id.int for c in resp.results] == [1, 2, 3]


def test_multi_hop_writes_each_step_from_the_previous_results(monkeypatch):
    plan = QueryPlan("multi_hop", ["Which agency issues firearm licences?", "What database does {answer} use?"])
    answers = {Q: response(Q, [chunk(1)]),
               "Which agency issues firearm licences?": response("Which agency issues firearm licences?", [chunk(2)]),
               "What database does the Ministry use?": response("x", [chunk(3)])}
    r = make(monkeypatch, plan, answers)
    resp = r.search(QueryRequest(query=Q))
    assert r.router.steps == [("Which agency issues firearm licences?", "What database does {answer} use?",
                               "context for Which agency issues firearm licences?")]
    assert resp.query_type == "multi_hop"
    assert resp.sub_questions == ["Which agency issues firearm licences?", "What database does the Ministry use?"]
    assert {c.chunk_id.int for c in resp.results} == {1, 2, 3}  # the original question's evidence is kept too


def test_conditional_also_searches_the_rule(monkeypatch):
    plan = QueryPlan("conditional", rewritten="How does a student extend a visa?")
    answers = {Q: response(Q, [chunk(1)]), "How does a student extend a visa?": response("r", [chunk(2)])}
    resp = make(monkeypatch, plan, answers).search(QueryRequest(query=Q))
    assert resp.query_type == "conditional" and [c.chunk_id.int for c in resp.results] == [1, 2]


def test_list_puts_one_result_per_process_first(monkeypatch):
    found = [chunk(1, "A"), chunk(2, "A"), chunk(3, "B"), chunk(4, "A"), chunk(5, "C"), chunk(6, "B")]
    fact = GraphFact(subject="s", relation="R", object="o", source="[5]", chunk_id=UUID(int=5))
    r = make(monkeypatch, QueryPlan("list"), {Q: response(Q, found, [fact])})
    resp = r.search(QueryRequest(query=Q, top_k=3))
    assert r.seen == [(Q, 8)]  # a wider rerank: twice list_top_k
    assert [c.chunk_id.int for c in resp.results] == [1, 3, 5, 2]  # A, B, C first, then the next best
    assert [c.citation.label for c in resp.results] == ["[1]", "[2]", "[3]", "[4]"]
    assert [(f.chunk_id.int, f.source) for f in resp.facts] == [(5, "[3]")]
    assert resp.context.startswith("[1] b.pdf")


def _row(n: int) -> dict:
    return {"id": UUID(int=n), "labelchunk": None, "document_id": UUID(int=99), "chunk_index": n,
            "content": f"text {n}", "page": 1, "char_start": 0, "char_end": 5, "metadata": {"service": "P"},
            "filename": "b.pdf", "file_type": "pdf", "vector_score": 0.4}


def test_process_returns_its_section_in_document_order(monkeypatch):
    found = [chunk(12, "P"), chunk(30, "Q"), chunk(10, "P")]
    r = make(monkeypatch, QueryPlan("process"), {Q: response(Q, found)})
    asked = []
    monkeypatch.setattr(r, "_section_rows", lambda qvec, filters, doc, service: asked.append((doc, service)) or
                        [_row(n) for n in range(9, 16)])
    resp = r.search(QueryRequest(query=Q, top_k=3))
    assert asked == [(UUID(int=99), "P")]
    # A window of process_max_chunks around the best match (12), in document order, then the other results.
    assert [c.chunk_id.int for c in resp.results] == [10, 11, 12, 13]
    assert resp.results[0].rerank_score == 0.9 and resp.results[1].rerank_score is None


def test_process_without_a_section_keeps_the_results(monkeypatch):
    r = make(monkeypatch, QueryPlan("process"), {Q: response(Q, [chunk(1), chunk(2)])})
    assert [c.chunk_id.int for c in r.search(QueryRequest(query=Q)).results] == [1, 2]


def test_router_can_be_turned_off_per_request(monkeypatch):
    r = make(monkeypatch, QueryPlan("list"), {Q: response(Q, [chunk(1)])})
    resp = r.search(QueryRequest(query=Q, router=False))
    assert resp.query_type is None and r.seen == [(Q, 5)]
    assert len(resp.results) == 1
