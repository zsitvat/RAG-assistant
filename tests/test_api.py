import os

import pytest
import redis as redis_lib
from httpx2 import ASGITransport, AsyncClient

from app.main import app, create_app
from app.settings import get_settings

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
    # Never let a full-app test boot against a developer's real .env Langfuse credentials.
    monkeypatch.setenv("LANGFUSE_ENABLED", "false")
    get_settings.cache_clear()
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as test_client:
            yield test_client
    get_settings.cache_clear()


async def test_health_reports_ok(client):
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_ready_reports_dummy_llm_ok_and_redis_ok(client):
    response = await client.get("/ready")
    assert response.status_code == 200

    body = response.json()
    assert body["ready"] is True
    checks = {check["name"]: check for check in body["checks"]}
    assert checks["llm"]["status"] == "ok"
    assert checks["redis"]["status"] == "ok"


async def test_unknown_route_returns_fastapi_default_404(client):
    response = await client.get("/does-not-exist")
    assert response.status_code == 404
    assert response.json() == {"detail": "Not Found"}


async def test_openapi_includes_shell_endpoints(client):
    schema = (await client.get("/openapi.json")).json()
    assert "/health" in schema["paths"]
    assert "/ready" in schema["paths"]


async def test_chat_returns_a_typed_response_even_without_a_real_llm(client):
    response = await client.post("/chat", json={"thread_id": "t1", "message": "hello"})

    assert response.status_code == 200
    body = response.json()
    assert body["thread_id"] == "t1"
    assert isinstance(body["answer"], str) and body["answer"]
    assert isinstance(body["response_time_ms"], int)
    assert body["decision"] is None
    assert body["sources"] == []
    assert "Request understood" in body["steps"]
    assert "Answer prepared" in body["steps"]


async def test_thread_reset_deletes_the_conversation_state(client):
    await client.post("/chat", json={"thread_id": "reset-me", "message": "hello"})

    response = await client.delete("/threads/reset-me")

    assert response.status_code == 200
    assert response.json() == {"thread_id": "reset-me", "reset": True}


async def test_unhandled_exception_returns_500_without_leaking_a_traceback(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "dummy")
    # Never let a full-app test boot against a developer's real .env Langfuse credentials.
    monkeypatch.setenv("LANGFUSE_ENABLED", "false")
    get_settings.cache_clear()

    test_app = create_app()

    @test_app.get("/test/boom")
    async def _boom():
        raise RuntimeError("boom")

    async with test_app.router.lifespan_context(test_app):
        transport = ASGITransport(app=test_app, raise_app_exceptions=False)
        async with AsyncClient(transport=transport, base_url="http://testserver") as test_client:
            response = await test_client.get("/test/boom")
            assert response.status_code == 500
            assert "RuntimeError" not in response.text
            assert "Traceback" not in response.text

    get_settings.cache_clear()
