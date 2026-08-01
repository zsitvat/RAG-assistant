import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.router import router
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.core.observability import RequestContextMiddleware
from app.dependencies import ApplicationDependencies

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Configures logging and builds application dependencies for the app's lifetime."""
    settings = get_settings()
    configure_logging(service="api", log_level=settings.log_level)
    app.state.dependencies = ApplicationDependencies.build(settings)

    logger.info("API startup complete (backend=%s)", settings.llm_backend)
    yield


def create_app() -> FastAPI:
    """Builds the FastAPI application with its middleware and routes."""
    app = FastAPI(title="RAG Assistant API", lifespan=lifespan)
    app.add_middleware(RequestContextMiddleware)
    app.include_router(router)
    return app


app = create_app()
