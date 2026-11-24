# syntax=docker/dockerfile:1

## ---- Builder ----------------------------------------------------------
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

COPY pyproject.toml uv.lock README.md LICENSE CHANGELOG.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

COPY src/ src/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

## ---- Runtime ------------------------------------------------------------
FROM python:3.12-slim-bookworm AS runtime

LABEL org.opencontainers.image.title="bedrock-rag-platform" \
      org.opencontainers.image.description="Local-first RAG API (Ollama + pgvector today)" \
      org.opencontainers.image.source="https://github.com/thomm-011/bedrock-rag-platform" \
      org.opencontainers.image.licenses="MIT"

RUN groupadd --system app && useradd --system --gid app --create-home --home-dir /home/app app

COPY --from=builder --chown=app:app /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER app
WORKDIR /home/app

EXPOSE 8000
ENTRYPOINT ["rag"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]
