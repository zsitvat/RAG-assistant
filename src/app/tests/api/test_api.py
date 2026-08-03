import pytest
from httpx2 import ASGITransport, AsyncClient

from app.main import create_app
from app.settings import get_settings
from app.tests.redis_availability import redis_available

pytestmark = pytest.mark.skipif(not redis_available(), reason="Redis 8 not reachable")


async def test_health_reports_ok(client):
    # Act
    response = await client.get("/health")

    # Assert
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_ready_reports_dummy_llm_ok_and_redis_ok(client):
    # Act
    response = await client.get("/ready")

    # Assert
    assert response.status_code == 200

    body = response.json()
    assert body["ready"] is True
    checks = {check["name"]: check for check in body["checks"]}
    assert checks["llm"]["status"] == "ok"
    assert checks["redis"]["status"] == "ok"


async def test_unknown_route_returns_fastapi_default_404(client):
    # Act
    response = await client.get("/does-not-exist")

    # Assert
    assert response.status_code == 404
    assert response.json() == {"detail": "Not Found"}


async def test_openapi_includes_shell_endpoints(client):
    # Act
    schema = (await client.get("/openapi.json")).json()

    # Assert
    assert "/health" in schema["paths"]
    assert "/ready" in schema["paths"]


async def test_chat_returns_a_typed_response_even_without_a_real_llm(client):
    # Act
    response = await client.post("/chat", json={"thread_id": "t1", "message": "hello"})

    # Assert
    assert response.status_code == 200
    body = response.json()
    assert body["thread_id"] == "t1"
    assert isinstance(body["answer"], str) and body["answer"]
    assert isinstance(body["response_time_ms"], int)
    assert body["decision"] is None
    assert body["sources"] == []
    assert "Intent classified — policy_question" in body["steps"]
    assert "Answer generated" in body["steps"]
    assert body["degraded"] is True


async def test_chat_rejects_an_empty_message(client):
    # Act
    response = await client.post("/chat", json={"thread_id": "t1", "message": ""})

    # Assert
    assert response.status_code == 422


async def test_chat_rejects_a_message_over_the_length_limit(client):
    # Act
    response = await client.post("/chat", json={"thread_id": "t1", "message": "a" * 501})

    # Assert
    assert response.status_code == 422


async def test_chat_rejects_a_thread_id_with_disallowed_characters(client):
    # Act
    response = await client.post("/chat", json={"thread_id": "not a valid id!", "message": "hi"})

    # Assert
    assert response.status_code == 422


async def test_admin_eval_returns_the_typed_evaluation_projection(client):
    # Act
    response = await client.post(
        "/admin/eval",
        json={"thread_id": "eval-t1", "message": "hello", "reference_date": "2026-08-02"},
    )

    # Assert
    assert response.status_code == 200
    body = response.json()
    assert body["thread_id"] == "eval-t1"
    assert body["intent"] == "policy_question"
    assert body["decision"] is None
    assert body["tool_calls"] == []
    assert body["retrieved_doc_ids"] == []
    assert body["cited_doc_ids"] == []
    assert body["degraded"] is True


async def test_thread_reset_deletes_the_conversation_state(client):
    # Arrange
    await client.post("/chat", json={"thread_id": "reset-me", "message": "hello"})

    # Act
    response = await client.delete("/threads/reset-me")

    # Assert
    assert response.status_code == 200
    assert response.json() == {"thread_id": "reset-me", "reset": True}


async def test_unhandled_exception_returns_500_without_leaking_a_traceback(monkeypatch):
    # Arrange
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
            # Act
            response = await test_client.get("/test/boom")

            # Assert
            assert response.status_code == 500
            assert "RuntimeError" not in response.text
            assert "Traceback" not in response.text

    get_settings.cache_clear()
