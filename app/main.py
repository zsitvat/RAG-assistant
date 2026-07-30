import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import redis
from fastapi import FastAPI

from app.api.router import router
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.core.observability import RequestContextMiddleware
from app.integrations.llm import build_chat_model
from app.integrations.redis import build_redis_client
from app.rag.ingest import run_ingest
from app.rag.store import build_embeddings, build_vector_store
from app.rules.loader import get_rule_catalogue

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    settings = get_settings()
    configure_logging(service="api", log_level=settings.log_level)
    app.state.settings = settings
    app.state.chat_model = build_chat_model(settings)
    app.state.rule_catalogue = get_rule_catalogue()

    app.state.redis_client = None
    app.state.vector_store = None
    try:
        redis_client = build_redis_client(settings.redis_url)
        redis_client.ping()
    except redis.RedisError:
        logger.warning("Redis unavailable at startup; RAG features are disabled")
    else:
        vector_store = build_vector_store(settings.redis_url, build_embeddings())
        run_ingest(redis_client, vector_store, rule_catalogue=app.state.rule_catalogue)
        app.state.redis_client = redis_client
        app.state.vector_store = vector_store
        logger.info("Redis index ready")

    logger.info("API startup complete (backend=%s)", settings.llm_backend)
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="RAG Assistant API", lifespan=lifespan)
    app.add_middleware(RequestContextMiddleware)
    app.include_router(router)
    return app


app = create_app()
