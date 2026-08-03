import pytest
from httpx2 import ASGITransport, AsyncClient

from app.main import app
from app.settings import get_settings
from app.tests.redis_availability import TEST_REDIS_URL


@pytest.fixture
async def client(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "dummy")
    monkeypatch.setenv("REDIS_URL", TEST_REDIS_URL)
    # Never let a full-app test boot against a developer's real .env Langfuse credentials.
    monkeypatch.setenv("LANGFUSE_ENABLED", "false")
    get_settings.cache_clear()
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as test_client:
            yield test_client
    get_settings.cache_clear()
