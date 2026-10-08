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

from . import deps
from .config import get_settings
from .documents import list_documents as _list_documents
from .generation import generate_answer
from .schemas import QueryRequest, QueryResponse

mcp = FastMCP("askgov-kb")


class Passage(BaseModel):
    rank: int
    text: str
    filename: str
    page: int | None
    score: float


class SearchResult(BaseModel):
    status: Literal["ok", "no_relevant_context"]
    message: str | None
    results: list[Passage]


class Source(BaseModel):
    label: str  # "[1]", as cited in the answer
    filename: str
    page: int | None


class Answer(BaseModel):
    status: Literal["ok", "no_relevant_context"]
    answer: str
    sources: list[Source]


class Document(BaseModel):
    filename: str
    chunks: int
    version: int


def _retrieve(query: str, top_k: int) -> QueryResponse:
    return deps.get_retriever().search(QueryRequest(query=query, top_k=max(1, min(top_k, 20))))


def _search(query: str, top_k: int) -> SearchResult:
    resp = _retrieve(query, top_k)
    return SearchResult(
        status=resp.status,
        message=resp.message,
        results=[
            Passage(rank=r.rank, text=r.text, filename=r.citation.filename, page=r.citation.page,
                    score=r.rerank_score if r.rerank_score is not None else r.vector_score)
            for r in resp.results
        ],
    )


@mcp.tool()
async def search_knowledge_base(query: str, top_k: int = 5) -> SearchResult:
    """Search official Guyana information: General Register Office (GRO) birth, marriage and
    death services, and the National Cash Grant. Returns the most relevant passages with their
    source file and page. Call this before answering any question on these topics, and cite the
    sources. status "no_relevant_context" means the knowledge base has no answer."""
    # The retriever is blocking (DB + HTTP calls), so run it off the event loop.
    return await anyio.to_thread.run_sync(_search, query, top_k)


def _ask(question: str, top_k: int) -> Answer:
    resp = _retrieve(question, top_k)
    settings = get_settings()
    return Answer(
        status=resp.status,
        answer=generate_answer(resp, settings.generation_model, settings.openai_api_key),
        sources=[Source(label=r.citation.label, filename=r.citation.filename, page=r.citation.page)
                 for r in resp.results],
    )


@mcp.tool()
async def ask(question: str, top_k: int = 5) -> Answer:
    """Answer a question about Guyana GRO birth, marriage and death services or the National Cash
    Grant. Searches the knowledge base and writes an answer grounded only in what it finds, with
    citation markers like [1] that refer to `sources`. status "no_relevant_context" means the
    knowledge base has no answer; do not make one up."""
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
