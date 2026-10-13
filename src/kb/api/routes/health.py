from fastapi import APIRouter

from kb import deps
from kb.ingestion.loaders import SUPPORTED_EXTENSIONS

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict:
    with deps.get_pool().connection() as conn:
        conn.execute("SELECT 1")
    return {"status": "ok", "pgvector": deps.pgvector_version(), "supported_types": SUPPORTED_EXTENSIONS}
