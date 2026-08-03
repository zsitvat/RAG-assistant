import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.router import router
from app.dependencies import ApplicationDependencies
from app.logging.config import configure_logging
from app.settings import get_settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Configures logging and builds application dependencies for the app's lifetime."""
    settings = get_settings()
    configure_logging(service="api", log_level=settings.log_level)
    app.state.dependencies = await ApplicationDependencies.build(settings)

    logger.info(f"API startup complete (backend={settings.llm_backend})")
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="RAG Assistant API", lifespan=lifespan)
    app.include_router(router)
    return app


app = create_app()
