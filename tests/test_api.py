import pytest
from httpx2 import ASGITransport, AsyncClient

from app.core.config import get_settings
from app.main import app, create_app


@pytest.fixture
async def client(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "dummy")
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


async def test_ready_reports_dummy_llm_ok_and_redis_unavailable_without_a_server(client):
    response = await client.get("/ready")
    assert response.status_code == 200

    body = response.json()
    assert body["ready"] is False
    checks = {check["name"]: check for check in body["checks"]}
    assert checks["llm"]["status"] == "ok"
    assert checks["redis"]["status"] == "unavailable"


async def test_unknown_route_returns_fastapi_default_404(client):
    response = await client.get("/does-not-exist")
    assert response.status_code == 404
    assert response.json() == {"detail": "Not Found"}


async def test_request_id_header_is_echoed(client):
    response = await client.get("/health", headers={"X-Request-ID": "test-request-id"})
    assert response.headers["x-request-id"] == "test-request-id"


async def test_openapi_includes_shell_endpoints(client):
    schema = (await client.get("/openapi.json")).json()
    assert "/health" in schema["paths"]
    assert "/ready" in schema["paths"]


async def test_unhandled_exception_returns_500_without_leaking_a_traceback(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "dummy")
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
