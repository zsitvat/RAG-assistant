import os

import pytest
import redis as redis_lib
from httpx2 import ASGITransport, AsyncClient

from app.core.config import get_settings
from app.main import app

TEST_REDIS_URL = os.environ.get("TEST_REDIS_URL", "redis://127.0.0.1:6379/0")


def _redis_available() -> bool:
    try:
        redis_lib.Redis.from_url(TEST_REDIS_URL).ping()
    except redis_lib.RedisError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _redis_available(), reason="Redis 8 not reachable")


@pytest.fixture
async def client(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "dummy")
    monkeypatch.setenv("REDIS_URL", TEST_REDIS_URL)
    get_settings.cache_clear()
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as test_client:
            yield test_client
    get_settings.cache_clear()


async def test_admin_ingest_returns_the_ingest_result(client):
    response = await client.post("/admin/ingest")
    assert response.status_code == 200
    body = response.json()
    assert body["action"] in ("built", "rebuilt", "reused")
    assert isinstance(body["chunk_count"], int)


async def test_admin_stats_returns_the_index_stats(client):
    response = await client.get("/admin/stats")
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["total_chunks"], int)
    assert isinstance(body["category_counts"], dict)
