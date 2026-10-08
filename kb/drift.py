"""Knowledge drift: re-check each current document's source and pick up changed content.

`refresh` reads every recorded source (a local file path or an http(s) URL). Unchanged content
only updates last_checked_at; changed content is ingested as a new version of the same doc_key
(the old version is superseded, as with a manual upload); a vanished source is flagged as
'missing' but the document stays searchable until someone replaces, expires or deletes it.
"""

import hashlib
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from psycopg_pool import ConnectionPool

from .documents import list_documents
from .embeddings import EmbeddingError
from .ingest import Ingestor
from .schemas import DocumentInfo, RefreshResult

log = logging.getLogger(__name__)

# Metadata the loader reads from the file itself; a new version gets it from the new file.
_FILE_METADATA_KEYS = {"title"}


@dataclass
class SourceRead:
    status: Literal["ok", "not_modified", "missing", "error"]
    data: bytes | None = None
    etag: str | None = None
    last_modified: str | None = None
    error: str | None = None


def is_url(source: str) -> bool:
    return source.lower().startswith(("http://", "https://"))


def read_source(source: str, etag: str | None = None, last_modified: str | None = None,
                timeout: float = 30.0) -> SourceRead:
    if not is_url(source):
        path = Path(source)
        if not path.is_file():
            return SourceRead("missing")
        try:
            return SourceRead("ok", data=path.read_bytes())
        except OSError as exc:
            return SourceRead("error", error=str(exc))

    request = urllib.request.Request(source, headers={"User-Agent": "askgov-kb-refresh"})
    # Conditional GET: an unchanged file comes back as 304 without a body.
    if etag:
        request.add_header("If-None-Match", etag)
    if last_modified:
        request.add_header("If-Modified-Since", last_modified)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            return SourceRead("ok", data=resp.read(), etag=resp.headers.get("ETag"),
                              last_modified=resp.headers.get("Last-Modified"))
    except urllib.error.HTTPError as exc:
        if exc.code == 304:
            return SourceRead("not_modified", etag=etag, last_modified=last_modified)
        if exc.code in (404, 410):
            return SourceRead("missing", error=f"HTTP {exc.code}")
        return SourceRead("error", error=f"HTTP {exc.code}")
    except (urllib.error.URLError, OSError) as exc:
        return SourceRead("error", error=str(getattr(exc, "reason", exc)))


def _record_check(pool: ConnectionPool, doc_id, status: str, read: SourceRead) -> None:
    with pool.connection() as conn:
        conn.execute(
            "UPDATE documents SET source_status = %s, last_checked_at = now(), "
            "source_etag = COALESCE(%s, source_etag), source_last_modified = COALESCE(%s, source_last_modified) "
            "WHERE id = %s",
            [status, read.etag, read.last_modified, doc_id],
        )


def _content_hash(pool: ConnectionPool, doc_id) -> str:
    with pool.connection() as conn:
        return conn.execute("SELECT content_hash FROM documents WHERE id = %s", [doc_id]).fetchone()[0]


def _check_one(ingestor: Ingestor, pool: ConnectionPool, doc: DocumentInfo, timeout: float,
               dry_run: bool) -> RefreshResult:
    with pool.connection() as conn:
        etag, last_modified = conn.execute(
            "SELECT source_etag, source_last_modified FROM documents WHERE id = %s", [doc.id]
        ).fetchone()
    read = read_source(doc.source, etag, last_modified, timeout)
    result = RefreshResult(doc_key=doc.doc_key, filename=doc.filename, source=doc.source,
                           outcome="unchanged", document_id=doc.id, version=doc.version)

    if read.status in ("missing", "error"):
        result.outcome, result.detail = read.status, read.error
        if not dry_run:
            _record_check(pool, doc.id, read.status, read)
        return result
    if read.status == "not_modified" or hashlib.sha256(read.data).hexdigest() == _content_hash(pool, doc.id):
        if not dry_run:
            _record_check(pool, doc.id, "ok", read)
        return result

    result.outcome = "updated"
    if dry_run:
        result.detail = "content changed (dry run: not ingested)"
        return result
    metadata = {k: v for k, v in doc.metadata.items() if k not in _FILE_METADATA_KEYS}
    try:
        ingested = ingestor.ingest_bytes(doc.filename, read.data, metadata, doc_key=doc.doc_key, source=doc.source)
    except (ValueError, EmbeddingError) as exc:
        result.outcome, result.detail = "error", f"changed, but ingest failed: {exc}"
        _record_check(pool, doc.id, "error", read)
        return result
    if ingested.status == "duplicate":
        # Same bytes as another stored version (e.g. the source was rolled back); nothing new to add.
        result.outcome, result.detail = "duplicate", f"content matches stored document {ingested.document_id}"
        _record_check(pool, doc.id, "ok", read)
        return result
    _record_check(pool, ingested.document_id, "ok", read)
    result.document_id, result.version = ingested.document_id, ingested.version
    result.detail = f"new version v{ingested.version} ({ingested.num_chunks} chunks) supersedes v{doc.version}"
    return result


def refresh(ingestor: Ingestor, pool: ConnectionPool, timeout: float = 30.0,
            dry_run: bool = False) -> tuple[list[RefreshResult], int]:
    """Check every current document that has a source. Returns (results, documents without a source)."""
    docs = list_documents(pool, limit=100_000)
    results = []
    for doc in docs:
        if doc.source:
            results.append(_check_one(ingestor, pool, doc, timeout, dry_run))
            log.info("%s %s", results[-1].outcome, doc.source)
    return results, sum(not d.source for d in docs)
