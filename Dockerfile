FROM python:3.12.13-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.9.7 /uv /bin/

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

COPY src/app ./app

# Bake embedding weights into the image so startup never fetches them on the first request.
ENV HF_HOME=/app/.cache/huggingface
RUN /app/.venv/bin/python -c "from app.rag.store import build_embeddings; build_embeddings()"


FROM python:3.12.13-slim AS runtime

RUN groupadd --system app && useradd --system --gid app --create-home --home-dir /home/app app

WORKDIR /app
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface \
    HF_HUB_OFFLINE=1

COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --from=builder --chown=app:app /app/.cache /app/.cache
COPY --chown=app:app src/app ./app
COPY --chown=app:app src/rules_config ./rules_config
COPY --chown=app:app .docs/sources ./.docs/sources
RUN mkdir -p /app/logs && chown app:app /app/logs

USER app
EXPOSE 8000 8501

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
