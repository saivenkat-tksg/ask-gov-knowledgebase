# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# OpenCV (pulled in by rapidocr) needs these shared libraries at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md alembic.ini ./
COPY src ./src
RUN pip install ".[graph]"

RUN useradd --create-home --uid 10001 kb
USER kb

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

# Migrations are a separate deploy step: `docker compose run --rm api kb migrate`
CMD ["uvicorn", "kb.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
