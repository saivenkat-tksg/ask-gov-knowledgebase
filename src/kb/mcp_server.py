"""Remote MCP server: exposes read-only knowledge-base tools to agents over Streamable HTTP.

    kb mcp                      # http://127.0.0.1:8001/mcp
    kb mcp --host 0.0.0.0       # reachable from the LAN (needs KB_MCP_TOKEN)

Agents send `Authorization: Bearer <KB_MCP_TOKEN>` when the token is set. Without a token
the server only listens on localhost.
"""

import secrets
from typing import Literal

import anyio
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import BaseModel

from kb import deps
from kb.config import get_settings
from kb.documents.repository import list_documents as _list_documents
from kb.generation import generate_answer
from kb.schemas import Evidence, QueryRequest, QueryResponse

mcp = FastMCP("askgov-kb")


class Passage(BaseModel):
    rank: int
    label: str  # "[1]", the marker to cite
    chunk_id: str
    chunk_label: str | None  # e.g. 'cash_grant_chunk_3'
    text: str  # cut to the relevant sentences when `compressed`
    filename: str
    page: int | None
    rerank_score: float  # rerank score when reranked, else vector_score
    vector_score: float  # cosine similarity to the question
    found_by: str  # vector, keyword, both, graph, or a mix such as vector+graph
    compressed: bool  # text cut to the sentences relevant to the question; chunk_id holds the whole chunk
    evidence: list[Evidence]  # with compression: the sentences that qualified, with scores, best first


class Fact(BaseModel):
    """A knowledge-graph fact (Neo4j) from one of the returned passages."""

    subject: str
    relation: str
    object: str
    quote: str | None  # the passage text the fact was extracted from
    source: str  # label of the passage it comes from, e.g. "[2]"


class SearchResult(BaseModel):
    status: Literal["ok", "no_relevant_context"]
    message: str | None
    query_type: str | None  # how the router classified the question (normal, list, process...)
    graph: bool  # whether the knowledge graph contributed passages
    warnings: list[str]  # e.g. knowledge graph or rerank unavailable, document out of date
    results: list[Passage]
    facts: list[Fact]


class Source(BaseModel):
    label: str  # "[1]", as cited in the answer
    chunk_id: str
    chunk_label: str | None
    filename: str
    page: int | None
    found_by: str  # vector, keyword, both, graph, or a mix such as vector+graph
    evidence: list[Evidence]  # with compression: the sentences the answer could draw on, with scores


class Answer(BaseModel):
    status: Literal["ok", "no_relevant_context"]
    answer: str
    query_type: str | None
    graph: bool
    sources: list[Source]
    facts: list[Fact]


class Document(BaseModel):
    filename: str
    chunks: int
    version: int


def _retrieve(query: str, top_k: int) -> QueryResponse:
    return deps.get_retriever().search(QueryRequest(query=query, top_k=max(1, min(top_k, 20))))


def _facts(resp: QueryResponse) -> list[Fact]:
    return [Fact(subject=f.subject, relation=f.relation, object=f.object, quote=f.quote, source=f.source)
            for f in resp.facts]


def _search(query: str, top_k: int) -> SearchResult:
    resp = _retrieve(query, top_k)
    return SearchResult(
        status=resp.status,
        message=resp.message,
        query_type=resp.query_type,
        graph=resp.graph,
        warnings=resp.warnings,
        results=[
            Passage(rank=r.rank, label=r.citation.label, chunk_id=str(r.chunk_id), chunk_label=r.chunk_label,
                    text=r.text, filename=r.citation.filename, page=r.citation.page,
                    rerank_score=r.rerank_score if r.rerank_score is not None else r.vector_score,
                    vector_score=r.vector_score, found_by=r.found_by, compressed=r.compressed,
                    evidence=r.evidence)
            for r in resp.results
        ],
        facts=_facts(resp),
    )


@mcp.tool()
async def search_knowledge_base(query: str, top_k: int = 5) -> SearchResult:
    """Search official Guyana government process documents: the Ministry of Home Affairs and its
    agencies (General Register Office, Guyana Police Force, Prison Service, immigration and
    passports, Fire and Rescue Service, forensic science lab, CANU, juvenile justice, community
    policing) and related services such as birth, marriage and death registration and the National
    Cash Grant. Returns the most relevant passages with their source file and page, how each was
    found (found_by: vector, keyword, graph), the exact sentences that matched (evidence, with
    scores) and knowledge-graph facts citing their passage. Call this
    before answering any question on these topics, and cite the sources. status
    "no_relevant_context" means the knowledge base has no answer. Call `list_documents` to see
    exactly which documents are loaded."""
    # The retriever is blocking (DB + HTTP calls), so run it off the event loop.
    return await anyio.to_thread.run_sync(_search, query, top_k)


def _ask(question: str, top_k: int) -> Answer:
    resp = _retrieve(question, top_k)
    settings = get_settings()
    return Answer(
        status=resp.status,
        answer=generate_answer(resp, settings.generation_model, settings.openai_api_key),
        query_type=resp.query_type,
        graph=resp.graph,
        sources=[Source(label=r.citation.label, chunk_id=str(r.chunk_id), chunk_label=r.chunk_label,
                        filename=r.citation.filename, page=r.citation.page, found_by=r.found_by,
                        evidence=r.evidence)
                 for r in resp.results],
        facts=_facts(resp),
    )


@mcp.tool()
async def ask(question: str, top_k: int = 5) -> Answer:
    """Answer a question about Guyana government processes: the Ministry of Home Affairs and its
    agencies (General Register Office, police, prisons, immigration and passports, fire service,
    forensics, CANU, juvenile justice, community policing) and related services such as civil
    registration and the National Cash Grant. Searches the knowledge base and writes an answer
    grounded only in what it finds, with citation markers like [1] that refer to `sources`.
    status "no_relevant_context" means the knowledge base has no answer; do not make one up."""
    return await anyio.to_thread.run_sync(_ask, question, top_k)


@mcp.tool()
async def list_documents() -> list[Document]:
    """List the documents currently in the knowledge base."""
    docs = await anyio.to_thread.run_sync(lambda: _list_documents(deps.get_pool()))
    return [Document(filename=d.filename, chunks=d.num_chunks, version=d.version) for d in docs]


class BearerAuth:
    """ASGI middleware: reject HTTP requests without the right bearer token."""

    def __init__(self, app, token: str):
        self.app, self.expected = app, f"Bearer {token}".encode()

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            given = dict(scope["headers"]).get(b"authorization", b"")
            if not secrets.compare_digest(given, self.expected):
                await send({"type": "http.response.start", "status": 401,
                            "headers": [(b"content-type", b"text/plain")]})
                await send({"type": "http.response.body", "body": b"unauthorized"})
                return
        await self.app(scope, receive, send)


def create_app():
    token = get_settings().mcp_token
    if token:
        # FastMCP only accepts Host: localhost by default (DNS-rebinding protection), which rejects
        # LAN IPs and tunnel URLs with 421. With a token, a rebound browser request can't authenticate.
        mcp.settings.transport_security = TransportSecuritySettings(enable_dns_rebinding_protection=False)
    app = mcp.streamable_http_app()  # serves the MCP endpoint at /mcp
    return BearerAuth(app, token) if token else app


def run(host: str = "127.0.0.1", port: int = 8001) -> None:
    import uvicorn

    if host not in ("127.0.0.1", "localhost") and not get_settings().mcp_token:
        raise SystemExit("Set KB_MCP_TOKEN before exposing the MCP server beyond localhost.")
    uvicorn.run(create_app(), host=host, port=port)


if __name__ == "__main__":
    run()
