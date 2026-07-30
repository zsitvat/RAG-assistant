import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import app, create_app


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "dummy")
    get_settings.cache_clear()
    with TestClient(app) as test_client:
        yield test_client
    get_settings.cache_clear()


def test_health_reports_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_reports_dummy_backend_ok_and_redis_not_configured(client):
    response = client.get("/ready")
    assert response.status_code == 200

    body = response.json()
    assert body["ready"] is True
    checks = {check["name"]: check for check in body["checks"]}
    assert checks["llm"]["status"] == "ok"
    assert checks["redis"]["status"] == "not_configured"


def test_unknown_route_returns_fastapi_default_404(client):
    response = client.get("/does-not-exist")
    assert response.status_code == 404
    assert response.json() == {"detail": "Not Found"}


def test_request_id_header_is_echoed(client):
    response = client.get("/health", headers={"X-Request-ID": "test-request-id"})
    assert response.headers["x-request-id"] == "test-request-id"


def test_openapi_includes_shell_endpoints(client):
    schema = client.get("/openapi.json").json()
    assert "/health" in schema["paths"]
    assert "/ready" in schema["paths"]


def test_unhandled_exception_returns_500_without_leaking_a_traceback(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "dummy")
    get_settings.cache_clear()

    test_app = create_app()

    @test_app.get("/test/boom")
    async def _boom():
        raise RuntimeError("boom")

    with TestClient(test_app, raise_server_exceptions=False) as test_client:
        response = test_client.get("/test/boom")
        assert response.status_code == 500
        assert "RuntimeError" not in response.text
        assert "Traceback" not in response.text

    get_settings.cache_clear()
