import pytest

from app.main import create_app
from app.settings import get_settings


async def test_app_startup_fails_without_redis(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "dummy")
    monkeypatch.setenv("REDIS_URL", "redis://127.0.0.1:1/0")
    # Never let a full-app test boot against a developer's real .env Langfuse credentials.
    monkeypatch.setenv("LANGFUSE_ENABLED", "false")
    get_settings.cache_clear()
    app = create_app()

    try:
        with pytest.raises(RuntimeError, match="Redis is required but unavailable at startup"):
            async with app.router.lifespan_context(app):
                pass
    finally:
        get_settings.cache_clear()
