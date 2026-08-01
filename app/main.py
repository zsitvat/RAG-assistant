import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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
    app.state.dependencies = ApplicationDependencies.build(settings)

    logger.info("API startup complete (backend=%s)", settings.llm_backend)
    yield


def create_app() -> FastAPI:
    """Builds the FastAPI application with its middleware and routes."""
    app = FastAPI(title="RAG Assistant API", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[get_settings().ui_origin],
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*"],
    )
    app.include_router(router)
    return app


app = create_app()
