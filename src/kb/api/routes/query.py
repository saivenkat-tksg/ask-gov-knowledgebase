from fastapi import APIRouter, HTTPException

from kb import deps
from kb.providers.embeddings import EmbeddingError
from kb.retrieval.filters import FilterError
from kb.schemas import QueryRequest, QueryResponse

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    try:
        return deps.get_retriever().search(req)
    except FilterError as exc:
        raise HTTPException(400, f"Invalid filters: {exc}") from exc
    except EmbeddingError as exc:
        raise HTTPException(502, str(exc)) from exc
