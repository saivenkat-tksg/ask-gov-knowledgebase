from datetime import date
from uuid import UUID

from psycopg.rows import class_row
from psycopg_pool import ConnectionPool

from .schemas import DocumentInfo, StaleDocument

_COLUMNS = (
    "id, filename, file_type, size_bytes, num_chunks, metadata, created_at, "
    "doc_key, version, is_current, superseded_at, valid_from, valid_until, review_by, "
    "source, source_status, last_checked_at"
)


def list_documents(
    pool: ConnectionPool, limit: int = 100, offset: int = 0, include_history: bool = False
) -> list[DocumentInfo]:
    where = "" if include_history else "WHERE is_current"
    with pool.connection() as conn, conn.cursor(row_factory=class_row(DocumentInfo)) as cur:
        return cur.execute(
            f"SELECT {_COLUMNS} FROM documents {where} ORDER BY created_at DESC LIMIT %s OFFSET %s", [limit, offset]
        ).fetchall()


def get_document(pool: ConnectionPool, doc_id: UUID) -> DocumentInfo | None:
    with pool.connection() as conn, conn.cursor(row_factory=class_row(DocumentInfo)) as cur:
        return cur.execute(f"SELECT {_COLUMNS} FROM documents WHERE id = %s", [doc_id]).fetchone()


def get_documents(pool: ConnectionPool, doc_ids: list[UUID]) -> list[DocumentInfo]:
    with pool.connection() as conn, conn.cursor(row_factory=class_row(DocumentInfo)) as cur:
        return cur.execute(f"SELECT {_COLUMNS} FROM documents WHERE id = ANY(%s)", [doc_ids]).fetchall()


def document_versions(pool: ConnectionPool, doc_key: str) -> list[DocumentInfo]:
    with pool.connection() as conn, conn.cursor(row_factory=class_row(DocumentInfo)) as cur:
        return cur.execute(
            f"SELECT {_COLUMNS} FROM documents WHERE doc_key = %s ORDER BY version DESC", [doc_key]
        ).fetchall()


def delete_document(pool: ConnectionPool, doc_id: UUID) -> bool:
    """Delete one version. If it was the current one, the newest remaining version becomes current."""
    with pool.connection() as conn, conn.transaction():
        deleted = conn.execute(
            "DELETE FROM documents WHERE id = %s RETURNING doc_key, is_current", [doc_id]
        ).fetchone()
        if deleted is None:
            return False
        doc_key, was_current = deleted
        if was_current:
            conn.execute(
                "UPDATE documents SET is_current = true, superseded_at = NULL WHERE id = ("
                "SELECT id FROM documents WHERE doc_key = %s ORDER BY version DESC LIMIT 1)",
                [doc_key],
            )
        return True


def stale_reasons(doc: DocumentInfo, max_age_days: int, today: date | None = None) -> list[str]:
    """Why a current document may no longer match reality (knowledge drift); empty if it looks fresh."""
    today = today or date.today()
    reasons = []
    if doc.review_by and doc.review_by < today:
        reasons.append(f"review date {doc.review_by} has passed")
    elif not doc.review_by and max_age_days and (today - doc.created_at.date()).days > max_age_days:
        reasons.append(f"ingested {doc.created_at.date()}, over {max_age_days} days ago, with no review date")
    if doc.valid_until and doc.valid_until < today:
        reasons.append(f"expired on {doc.valid_until} (no longer searchable)")
    if doc.source_status == "missing":
        reasons.append(f"source no longer exists: {doc.source}")
    elif doc.source_status == "error":
        reasons.append(f"source could not be checked: {doc.source}")
    return reasons


def stale_documents(pool: ConnectionPool, max_age_days: int) -> list[StaleDocument]:
    """Current documents that need a human to check or replace them."""
    docs = list_documents(pool, limit=100_000)
    return [StaleDocument(document=d, reasons=r) for d in docs if (r := stale_reasons(d, max_age_days))]
