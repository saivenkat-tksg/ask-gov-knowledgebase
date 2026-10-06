from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

from .config import Settings


def session_options(settings: Settings) -> dict:
    """Connection kwargs that resolve unqualified table names in KB_DB_SCHEMA (vector type stays in public)."""
    if settings.db_schema == "public":
        return {}
    return {"options": f"-c search_path={settings.db_schema},public"}


MIGRATIONS_DIR = Path(__file__).parent / "migrations"


def alembic_config(settings: Settings):
    from alembic.config import Config

    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS_DIR))
    cfg.attributes["settings"] = settings
    return cfg


def migrate(settings: Settings, revision: str = "head") -> None:
    """Apply migrations up (or down) to `revision` in KB_DB_SCHEMA."""
    from alembic import command

    if revision.startswith("-") or revision == "base":
        command.downgrade(alembic_config(settings), revision)
    else:
        command.upgrade(alembic_config(settings), revision)


def schema_revision(settings: Settings) -> tuple[str | None, str]:
    """(revision the database is at, latest revision in the code)."""
    from alembic.script import ScriptDirectory

    head = ScriptDirectory.from_config(alembic_config(settings)).get_current_head()
    with psycopg.connect(settings.database_url, **session_options(settings)) as conn:
        exists = conn.execute(
            "SELECT 1 FROM information_schema.tables WHERE table_schema = %s AND table_name = 'alembic_version'",
            [settings.db_schema],
        ).fetchone()
        current = conn.execute("SELECT version_num FROM alembic_version").fetchone() if exists else None
    return (current[0] if current else None), head


def check_db(settings: Settings) -> str:
    """Verify the schema is migrated and matches KB_EMBEDDING_DIM. Returns the pgvector version.

    Runs no DDL: apply migrations with `kb migrate` (or `kb init`) as a deploy step.
    """
    current, head = schema_revision(settings)
    if current != head:
        raise RuntimeError(
            f"Database schema '{settings.db_schema}' is at revision {current or 'none'}, code expects {head}. "
            "Run `kb migrate`."
        )
    with psycopg.connect(settings.database_url, **session_options(settings)) as conn:
        dim = conn.execute(
            "SELECT atttypmod FROM pg_attribute "
            "WHERE attrelid = 'chunks'::regclass AND attname = 'embedding'"
        ).fetchone()[0]
        if dim != settings.embedding_dim:
            raise RuntimeError(
                f"chunks.embedding is vector({dim}) but KB_EMBEDDING_DIM={settings.embedding_dim}. "
                "Re-embed into a fresh database or set the matching dimension."
            )
        return conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()[0]


def init_db(settings: Settings) -> str:
    """Apply all migrations, then verify. Returns the pgvector version."""
    migrate(settings)
    return check_db(settings)


def create_pool(settings: Settings) -> ConnectionPool:
    pool = ConnectionPool(
        settings.database_url,
        min_size=1,
        max_size=settings.db_pool_size,
        kwargs={"autocommit": True, **session_options(settings)},
        configure=register_vector,
        open=True,
    )
    pool.wait()
    return pool


def supports_iterative_scan(pgvector_version: str) -> bool:
    try:
        major, minor = (int(x) for x in pgvector_version.split(".")[:2])
    except ValueError:
        return False
    return (major, minor) >= (0, 8)
