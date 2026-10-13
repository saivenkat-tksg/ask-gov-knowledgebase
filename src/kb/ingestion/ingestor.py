import bisect
import hashlib
import json
import uuid
from datetime import UTC, date, datetime
from typing import Any

import numpy as np
from psycopg.errors import UniqueViolation
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from kb.config import Settings
from kb.ingestion.chunking import DEFAULT_SEPARATORS, MARKDOWN_SEPARATORS, RecursiveChunker, token_length_function
from kb.ingestion.labels import chunk_label
from kb.ingestion.loaders import LoadedDocument, document_text, load_document
from kb.ingestion.structure import StructureChunker, service_of
from kb.ingestion.tables import is_row
from kb.ingestion.topics import TopicTagger, group_sections, known_topics
from kb.providers.embeddings import Embedder
from kb.schemas import IngestResult

# Set by the pipeline on every chunk; user metadata may not override these.
RESERVED_KEYS = {
    "document_id", "filename", "file_type", "chunk_index", "page",
    "char_start", "char_end", "uploaded_at", "ocr", "doc_key", "version",
    "chunk_strategy", "section_path", "heading", "service", "page_end", "has_table", "tables", "topics",
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
    def __init__(self, pool: ConnectionPool, embedder: Embedder, settings: Settings,
                 tagger: TopicTagger | None = None):
        self.pool = pool
        self.embedder = embedder
        # Names each section's topics at ingest (kb.ingestion.topics); None = no topics.
        self.tagger = tagger
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
        self._strategy = settings.chunk_strategy
        self._tables = settings.table_extraction
        self._structure = StructureChunker(settings.chunk_size, settings.chunk_overlap, length_fn,
                                           settings.chunk_min_tokens, settings.structure_min_headings)

    def build_chunks(self, filename: str, data: bytes,
                     metadata: dict[str, Any] | None = None) -> tuple[LoadedDocument, dict[str, Any], list[dict]]:
        """Load and chunk a file without storing it: (loaded document, document metadata, chunk rows).

        Each row has text, page, char_start, char_end and `extra` chunk metadata. The document
        metadata records which chunk_strategy was actually used (structure falls back to recursive
        for documents without enough headings)."""
        loaded = load_document(filename, data, structure=self._strategy == "structure", tables=self._tables)
        doc_meta = {**loaded.metadata, **validate_metadata(metadata)}
        rows = self._structure_rows(loaded) if self._strategy == "structure" else None
        doc_meta["chunk_strategy"] = "structure" if rows is not None else "recursive"
        if rows is None:
            rows = self._recursive_rows(loaded)
        return loaded, doc_meta, rows

    def _recursive_rows(self, loaded: LoadedDocument) -> list[dict]:
        chunker = self._chunkers[loaded.structured]
        rows = []
        for section in loaded.sections:
            for piece in chunker.split(section.text):
                rows.append({"text": piece.text, "page": section.metadata.get("page"),
                             "char_start": piece.start, "char_end": piece.end,
                             "extra": {k: v for k, v in section.metadata.items() if k != "page"}})
        return rows

    def _structure_rows(self, loaded: LoadedDocument) -> list[dict] | None:
        """Chunk the whole document at once so sections can run across page breaks; pages are
        mapped back from character offsets. None when the document has too few headings."""
        starts, texts, offset = [], [], 0
        for section in loaded.sections:
            starts.append(offset)
            texts.append(section.text)
            offset += len(section.text) + 2
        chunks = self._structure.split("\n\n".join(texts))
        if chunks is None:
            return None

        def page_index(pos: int) -> int:
            return max(bisect.bisect_right(starts, pos) - 1, 0)

        rows = []
        for chunk in chunks:
            first, last = page_index(chunk.start), page_index(max(chunk.end - 1, chunk.start))
            extra: dict[str, Any] = {
                "section_path": list(chunk.path),
                "heading": chunk.path[-1] if chunk.path else None,
                "service": service_of(chunk.path),
            }
            if any(is_row(line) for line in chunk.text.splitlines()):
                extra["has_table"] = True
            page = loaded.sections[first].metadata.get("page")
            if page is not None:
                extra["page_end"] = loaded.sections[last].metadata.get("page")
            if any(s.metadata.get("ocr") for s in loaded.sections[first:last + 1]):
                extra["ocr"] = True
            char_start = chunk.start - starts[first]
            rows.append({"text": chunk.text, "page": page, "char_start": char_start,
                         "char_end": char_start + (chunk.end - chunk.start), "extra": extra})
        return rows

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
        validate_metadata(metadata)
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

        loaded, doc_meta, rows = self.build_chunks(filename, data, metadata)
        if self.tagger is not None and rows:
            self._tag_topics(rows)
        doc_id = uuid.uuid4()
        uploaded_at = datetime.now(UTC).isoformat()
        for i, row in enumerate(rows):
            row["metadata"] = {
                **doc_meta,
                **row.pop("extra"),
                "document_id": str(doc_id),
                "filename": filename,
                "file_type": loaded.file_type,
                "chunk_index": i,
                "page": row["page"],
                "char_start": row["char_start"],
                "char_end": row["char_end"],
                "uploaded_at": uploaded_at,
                "doc_key": doc_key,
            }
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
                     "ok" if source else None, datetime.now(UTC) if source else None],
                )
                # The uploaded document's full text, for document-level graph extraction (0006).
                full_text, page_map = document_text(loaded.sections)
                with conn.cursor() as cur:
                    cur.executemany(
                        "INSERT INTO chunks (id, document_id, chunk_index, content, page, char_start, char_end, "
                        "metadata, embedding, labelchunk) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                        [
                            (uuid.uuid4(), doc_id, i, r["text"], r["page"], r["char_start"], r["char_end"],
                             Jsonb({**r["metadata"], "version": version}), np.asarray(v, dtype=np.float32),
                             chunk_label(filename, i))
                            for i, (r, v) in enumerate(zip(rows, vectors))
                        ],
                    )
                conn.execute("INSERT INTO document_texts (document_id, content, page_map) VALUES (%s, %s, %s)",
                             [doc_id, full_text, Jsonb(page_map)])
        except UniqueViolation:
            existing = self._find_by_hash(content_hash)
            return IngestResult(filename=filename, status="duplicate",
                                document_id=existing[0] if existing else None, file_type=loaded.file_type)

        return IngestResult(filename=filename, status="ingested", document_id=doc_id,
                            file_type=loaded.file_type, num_chunks=len(rows), doc_key=doc_key, version=version,
                            superseded_id=superseded[0] if superseded else None)

    def _tag_topics(self, rows: list[dict]) -> None:
        """Set `topics` on each row's chunk metadata, per section, reusing the knowledge base's topics."""
        sections = group_sections(rows)
        with self.pool.connection() as conn:
            known = known_topics(conn)
        for section, topics in zip(sections, self.tagger.tag(sections, known)):
            if topics:
                for i in section.rows:
                    rows[i]["extra"]["topics"] = topics

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
