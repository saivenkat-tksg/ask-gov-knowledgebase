import hashlib
import json
import uuid
from datetime import date, datetime, timezone
from typing import Any

import numpy as np
from psycopg.errors import UniqueViolation
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from .chunking import DEFAULT_SEPARATORS, MARKDOWN_SEPARATORS, RecursiveChunker, token_length_function
from .config import Settings
from .embeddings import Embedder
from .loaders import load_document
from .schemas import IngestResult
from .topics import document_topic

# Set by the pipeline on every chunk; user metadata may not override these.
RESERVED_KEYS = {
    "document_id", "filename", "file_type", "chunk_index", "page",
    "char_start", "char_end", "uploaded_at", "ocr", "doc_key", "version",
}


def validate_metadata(metadata: Any) -> dict[str, Any]:
    if metadata is None:
        return {}
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a JSON object")
    clash = RESERVED_KEYS & metadata.keys()
    if clash:
        raise ValueError(f"metadata keys are reserved: {', '.join(sorted(clash))}")
    try:
        json.dumps(metadata)
    except TypeError as exc:
        raise ValueError(f"metadata must be JSON-serialisable: {exc}") from exc
    return metadata


class Ingestor:
    def __init__(self, pool: ConnectionPool, embedder: Embedder, settings: Settings):
        self.pool = pool
        self.embedder = embedder
        self._schema = settings.db_schema
        length_fn = (
            token_length_function(settings.tokenizer_encoding)
            if settings.chunk_length_unit == "tokens"
            else len
        )
        self._chunkers = {
            structured: RecursiveChunker(
                settings.chunk_size,
                settings.chunk_overlap,
                MARKDOWN_SEPARATORS if structured else DEFAULT_SEPARATORS,
                length_fn,
            )
            for structured in (True, False)
        }
        self._topic_routing = settings.topic_routing

    def ingest_bytes(
        self,
        filename: str,
        data: bytes,
        metadata: dict[str, Any] | None = None,
        replace: bool = False,
        doc_key: str | None = None,
        valid_from: date | None = None,
        valid_until: date | None = None,
        review_by: date | None = None,
        source: str | None = None,
    ) -> IngestResult:
        """Ingest one file as a new version of `doc_key` (default: the filename).

        The previous current version of the same doc_key is superseded and drops out of search;
        with replace=True all earlier versions are deleted instead. `source` (a file path or URL)
        is where `kb refresh` looks for newer content.
        """
        user_meta = validate_metadata(metadata)
        doc_key = (doc_key or filename).strip()
        if not doc_key:
            raise ValueError("doc_key must not be empty")
        if valid_from and valid_until and valid_until < valid_from:
            raise ValueError("valid_until is before valid_from")
        content_hash = hashlib.sha256(data).hexdigest()

        existing = self._find_by_hash(content_hash)
        # replace=True re-processes identical bytes stored under the same doc_key, e.g. to pick up
        # pipeline improvements such as OCR; the old row is deleted in the transaction below.
        if existing and not (replace and existing[3] == doc_key):
            if source and existing[3] == doc_key:
                # Unchanged file: record where it lives so `kb refresh` can watch it from now on.
                self._attach_source(existing[0], source)
            return IngestResult(filename=filename, status="duplicate", document_id=existing[0],
                                file_type=existing[1], num_chunks=existing[2])

        loaded = load_document(filename, data)
        doc_id = uuid.uuid4()
        uploaded_at = datetime.now(timezone.utc).isoformat()
        doc_meta = {**loaded.metadata, **user_meta}
        if self._topic_routing and "topic" not in doc_meta:
            topic = document_topic("\n".join(s.text for s in loaded.sections))
            if topic:
                doc_meta["topic"] = topic
        chunker = self._chunkers[loaded.structured]

        rows: list[dict[str, Any]] = []
        for section in loaded.sections:
            for piece in chunker.split(section.text):
                page = section.metadata.get("page")
                rows.append({
                    "text": piece.text,
                    "page": page,
                    "char_start": piece.start,
                    "char_end": piece.end,
                    "metadata": {
                        **doc_meta,
                        **{k: v for k, v in section.metadata.items() if k != "page"},
                        "document_id": str(doc_id),
                        "filename": filename,
                        "file_type": loaded.file_type,
                        "chunk_index": len(rows),
                        "page": page,
                        "char_start": piece.start,
                        "char_end": piece.end,
                        "uploaded_at": uploaded_at,
                        "doc_key": doc_key,
                    },
                })
        if not rows:
            raise ValueError("No extractable text found (empty file, or a scanned PDF with OCR off or unreadable?)")

        vectors = self.embedder.embed([r["text"] for r in rows])

        try:
            with self.pool.connection() as conn, conn.transaction():
                # Serialise uploads of the same doc_key so version numbers can't collide.
                conn.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                             [f"{self._schema}:{doc_key}"])
                if replace:
                    conn.execute("DELETE FROM documents WHERE doc_key = %s", [doc_key])
                latest = conn.execute(
                    "SELECT max(version) FROM documents WHERE doc_key = %s", [doc_key]
                ).fetchone()[0]
                version = (latest or 0) + 1
                superseded = conn.execute(
                    "UPDATE documents SET is_current = false, superseded_at = now() "
                    "WHERE doc_key = %s AND is_current RETURNING id",
                    [doc_key],
                ).fetchone()
                conn.execute(
                    "INSERT INTO documents (id, filename, file_type, content_hash, size_bytes, metadata, num_chunks, "
                    "doc_key, version, valid_from, valid_until, review_by, source, source_status, last_checked_at) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    [doc_id, filename, loaded.file_type, content_hash, len(data), Jsonb(doc_meta), len(rows),
                     doc_key, version, valid_from, valid_until, review_by, source,
                     "ok" if source else None, datetime.now(timezone.utc) if source else None],
                )
                with conn.cursor() as cur:
                    cur.executemany(
                        "INSERT INTO chunks (id, document_id, chunk_index, content, page, char_start, char_end, "
                        "metadata, embedding) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                        [
                            (uuid.uuid4(), doc_id, i, r["text"], r["page"], r["char_start"], r["char_end"],
                             Jsonb({**r["metadata"], "version": version}), np.asarray(v, dtype=np.float32))
                            for i, (r, v) in enumerate(zip(rows, vectors))
                        ],
                    )
        except UniqueViolation:
            existing = self._find_by_hash(content_hash)
            return IngestResult(filename=filename, status="duplicate",
                                document_id=existing[0] if existing else None, file_type=loaded.file_type)

        return IngestResult(filename=filename, status="ingested", document_id=doc_id,
                            file_type=loaded.file_type, num_chunks=len(rows), doc_key=doc_key, version=version,
                            superseded_id=superseded[0] if superseded else None)

    def _attach_source(self, doc_id, source: str) -> None:
        with self.pool.connection() as conn:
            conn.execute(
                "UPDATE documents SET source = %s, source_status = 'ok', last_checked_at = now() "
                "WHERE id = %s AND source IS DISTINCT FROM %s",
                [source, doc_id, source],
            )

    def _find_by_hash(self, content_hash: str):
        with self.pool.connection() as conn:
            return conn.execute(
                "SELECT id, file_type, num_chunks, doc_key FROM documents WHERE content_hash = %s", [content_hash]
            ).fetchone()
