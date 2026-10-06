from uuid import UUID

from psycopg.rows import class_row
from psycopg_pool import ConnectionPool

from .schemas import DocumentInfo

_COLUMNS = (
    "id, filename, file_type, size_bytes, num_chunks, metadata, created_at, "
    "doc_key, version, is_current, superseded_at, valid_from, valid_until, review_by"
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
