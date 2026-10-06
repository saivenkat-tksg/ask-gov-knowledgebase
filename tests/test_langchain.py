import uuid

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from kb.generation import build_rag_chain
from kb.lc import KnowledgeBaseRetriever, format_docs
from kb.retrieve import format_context
from kb.schemas import Citation, QueryRequest, QueryResponse, RetrievedChunk


class FakeRetriever:
    def __init__(self):
        self.requests: list[QueryRequest] = []

    def search(self, req: QueryRequest) -> QueryResponse:
        self.requests.append(req)
        doc_id = uuid.uuid4()
        results = [
            RetrievedChunk(
                rank=i,
                chunk_id=uuid.uuid4(),
                text=f"chunk {i} about {req.query}",
                vector_score=0.9 - i / 10,
                rerank_score=0.8,
                metadata={"department": "health", "page": i},
                citation=Citation(label=f"[{i}]", document_id=doc_id, filename="guide.pdf",
                                  file_type="pdf", page=i, chunk_index=i - 1),
            )
            for i in range(1, req.top_k + 1)
        ]
        return QueryResponse(query=req.query, filters=req.filters, status="ok", reranked=True,
                             min_score_applied=0.3, results=results,
                             context=format_context(results), took_ms=1.0)


def test_retriever_returns_cited_documents():
    fake = FakeRetriever()
    docs = KnowledgeBaseRetriever(retriever=fake, top_k=2).invoke("flu")
    assert len(docs) == 2
    assert docs[0].page_content == "chunk 1 about flu"
    assert docs[0].metadata["citation"]["label"] == "[1]"
    assert docs[0].metadata["citation"]["locator"] == "guide.pdf, p. 1"
    assert docs[0].metadata["department"] == "health"
    assert format_docs(docs) == fake.search(QueryRequest(query="flu", top_k=2)).context


def test_invoke_overrides_defaults():
    fake = FakeRetriever()
    retriever = KnowledgeBaseRetriever(retriever=fake, filters={"department": "health"})
    retriever.invoke("tax", top_k=3, filters={"year": {"$gte": 2024}}, rerank=False)
    req = fake.requests[-1]
    assert (req.top_k, req.filters, req.rerank) == (3, {"year": {"$gte": 2024}}, False)


def test_invoke_rejects_unknown_arguments():
    with pytest.raises(TypeError):
        KnowledgeBaseRetriever(retriever=FakeRetriever()).invoke("x", bogus=1)


def test_rag_chain_passes_cited_context_to_llm():
    seen = {}

    def fake_llm(prompt_value):
        seen["messages"] = prompt_value.to_messages()
        return AIMessage(content="Clinics are open weekdays [1].")

    chain = build_rag_chain(KnowledgeBaseRetriever(retriever=FakeRetriever(), top_k=1), RunnableLambda(fake_llm))
    out = chain.invoke("clinic hours")

    assert out["answer"] == "Clinics are open weekdays [1]."
    assert out["question"] == "clinic hours"
    assert out["sources"][0].metadata["citation"]["label"] == "[1]"
    human = seen["messages"][-1].content
    assert "[1] guide.pdf, p. 1\nchunk 1 about clinic hours" in human
    assert human.endswith("Question: clinic hours")
