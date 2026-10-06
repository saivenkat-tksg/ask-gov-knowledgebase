import json
from contextlib import asynccontextmanager
from datetime import date
from uuid import UUID

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile

from . import deps
from .config import get_settings
from .documents import delete_document, document_versions, get_document, list_documents
from .embeddings import EmbeddingError
from .filters import FilterError
from .loaders import SUPPORTED_EXTENSIONS
from .schemas import DocumentInfo, IngestResult, QueryRequest, QueryResponse


@asynccontextmanager
async def lifespan(_: FastAPI):
    deps.get_pool()
    yield
    deps.close()


app = FastAPI(title="AskGov Knowledge Base", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    with deps.get_pool().connection() as conn:
        conn.execute("SELECT 1")
    return {"status": "ok", "pgvector": deps.pgvector_version(), "supported_types": SUPPORTED_EXTENSIONS}


@app.post("/documents", response_model=list[IngestResult])
def upload_documents(
    files: list[UploadFile] = File(...),
    metadata: str = Form("{}", description='JSON object applied to every file, e.g. {"department": "health"}'),
    replace: bool = Form(False, description="Delete all earlier versions instead of keeping them as history"),
    doc_key: str | None = Form(
        None, description="Stable document identity across versions (default: the filename). One file only."
    ),
    valid_from: date | None = Form(None, description="Not searchable before this date (YYYY-MM-DD)"),
    valid_until: date | None = Form(None, description="Not searchable after this date (YYYY-MM-DD)"),
    review_by: date | None = Form(None, description="Date by which the content should be reviewed"),
) -> list[IngestResult]:
    """Upload files. A file whose doc_key already exists becomes its new current version."""
    if doc_key and len(files) > 1:
        raise HTTPException(400, "doc_key can only be set when uploading a single file")
    try:
        meta = json.loads(metadata or "{}")
    except json.JSONDecodeError as exc:
        raise HTTPException(400, f"metadata is not valid JSON: {exc}") from exc

    max_bytes = get_settings().max_upload_mb * 1024 * 1024
    ingestor = deps.get_ingestor()
    results = []
    for upload in files:
        name = upload.filename or "upload"
        data = upload.file.read(max_bytes + 1)
        if len(data) > max_bytes:
            results.append(IngestResult(filename=name, status="error",
                                        error=f"File exceeds {get_settings().max_upload_mb} MB"))
            continue
        try:
            results.append(ingestor.ingest_bytes(name, data, meta, replace=replace, doc_key=doc_key,
                                                 valid_from=valid_from, valid_until=valid_until,
                                                 review_by=review_by))
        except (ValueError, EmbeddingError) as exc:  # bad file/metadata, or embedding provider problem
            results.append(IngestResult(filename=name, status="error", error=str(exc)))
    return results


@app.get("/documents", response_model=list[DocumentInfo])
def get_documents(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    include_history: bool = Query(False, description="Also list superseded versions"),
):
    return list_documents(deps.get_pool(), limit, offset, include_history)


@app.get("/documents/{doc_id}", response_model=DocumentInfo)
def get_one_document(doc_id: UUID):
    doc = get_document(deps.get_pool(), doc_id)
    if doc is None:
        raise HTTPException(404, "Document not found")
    return doc


@app.get("/documents/{doc_id}/versions", response_model=list[DocumentInfo])
def get_document_versions(doc_id: UUID):
    """Every version sharing this document's doc_key, newest first."""
    doc = get_document(deps.get_pool(), doc_id)
    if doc is None:
        raise HTTPException(404, "Document not found")
    return document_versions(deps.get_pool(), doc.doc_key)


@app.delete("/documents/{doc_id}", status_code=204)
def remove_document(doc_id: UUID):
    if not delete_document(deps.get_pool(), doc_id):
        raise HTTPException(404, "Document not found")


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    try:
        return deps.get_retriever().search(req)
    except FilterError as exc:
        raise HTTPException(400, f"Invalid filters: {exc}") from exc
    except EmbeddingError as exc:
        raise HTTPException(502, str(exc)) from exc
 




 