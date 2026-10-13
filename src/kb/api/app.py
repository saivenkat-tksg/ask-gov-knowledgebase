"""FastAPI application: `uvicorn kb.api.app:app` (or `kb serve`)."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from kb import deps
from kb.api.routes import documents, health, query


@asynccontextmanager
async def lifespan(_: FastAPI):
    deps.get_pool()
    yield
    deps.close()


def create_app() -> FastAPI:
    app = FastAPI(title="AskGov Knowledge Base", version="0.1.0", lifespan=lifespan)
    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(query.router)
    return app


app = create_app()
