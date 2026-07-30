import pytest
from httpx2 import ASGITransport, AsyncClient

from app.core.config import get_settings
from app.main import app


@pytest.fixture
async def client(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "dummy")
    monkeypatch.setenv("REDIS_URL", "redis://127.0.0.1:1/0")
    get_settings.cache_clear()
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as test_client:
            yield test_client
    get_settings.cache_clear()


async def test_admin_ingest_returns_503_when_redis_unavailable(client):
    response = await client.post("/admin/ingest")
    assert response.status_code == 503
    assert "Redis" in response.json()["detail"]


async def test_admin_stats_returns_503_when_redis_unavailable(client):
    response = await client.get("/admin/stats")
    assert response.status_code == 503
    assert "Redis" in response.json()["detail"]
