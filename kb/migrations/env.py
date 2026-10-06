"""Alembic environment: migrations are plain SQL run inside KB_DB_SCHEMA.

Each schema keeps its own alembic_version table, so several knowledge bases (e.g. kb and kb_eval)
can share one database, alongside another application's tables in public.
"""

from alembic import context
from sqlalchemy import create_engine, pool, text
from sqlalchemy.engine import make_url

from kb.config import get_settings

config = context.config
# kb.db passes Settings in programmatically; the alembic CLI falls back to .env.
settings = config.attributes.setdefault("settings", get_settings())
schema = settings.db_schema


def _url():
    return make_url(settings.database_url).set(drivername="postgresql+psycopg")


def run_migrations_offline() -> None:
    context.configure(url=_url(), literal_binds=True, version_table_schema=schema)
    with context.begin_transaction():
        context.execute(f"SET search_path TO {schema}, public")
        context.run_migrations()


def run_migrations_online() -> None:
    engine = create_engine(
        _url(), poolclass=pool.NullPool, connect_args={"options": f"-c search_path={schema},public"}
    )
    with engine.connect() as conn:
        # The version table lives in the schema, so the schema must exist first.
        conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema}"'))
        conn.commit()
        context.configure(connection=conn, version_table_schema=schema)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
