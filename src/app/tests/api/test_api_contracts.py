import os

import pytest
import redis as redis_lib
from httpx2 import ASGITransport, AsyncClient

from app.api.schemas import ChatResponse, EvaluationResponse
from app.main import app
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


CHAT_RESPONSE_FIELDS = {
    "thread_id",
    "answer",
    "generated_at",
    "response_time_ms",
    "decision",
    "sources",
    "steps",
}
EVALUATION_RESPONSE_FIELDS = {
    "thread_id",
    "intent",
    "category",
    "decision",
    "claim",
    "missing_slots",
    "tool_calls",
    "calculation",
    "findings",
    "retrieved_doc_ids",
    "cited_doc_ids",
    "degraded",
    "answer",
}


def test_chat_response_contract_is_the_minimal_public_shape():
    assert set(ChatResponse.model_fields) == CHAT_RESPONSE_FIELDS


def test_evaluation_response_contract_carries_typed_diagnostics_only():
    assert set(EvaluationResponse.model_fields) == EVALUATION_RESPONSE_FIELDS


def test_evaluation_only_fields_never_leak_into_the_public_chat_contract():
    eval_only_fields = EVALUATION_RESPONSE_FIELDS - CHAT_RESPONSE_FIELDS
    assert eval_only_fields & set(ChatResponse.model_fields) == set()
    assert {"intent", "claim", "tool_calls", "findings", "degraded"} <= eval_only_fields


async def test_openapi_exposes_chat_and_eval_as_distinct_response_schemas(client):
    schema = (await client.get("/openapi.json")).json()

    assert "/chat" in schema["paths"]
    assert "/admin/eval" in schema["paths"]

    chat_response_ref = schema["paths"]["/chat"]["post"]["responses"]["200"]["content"][
        "application/json"
    ]["schema"]["$ref"]
    eval_response_ref = schema["paths"]["/admin/eval"]["post"]["responses"]["200"]["content"][
        "application/json"
    ]["schema"]["$ref"]

    assert chat_response_ref != eval_response_ref
    assert chat_response_ref.endswith("ChatResponse")
    assert eval_response_ref.endswith("EvaluationResponse")
