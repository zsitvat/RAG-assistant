import pytest

from app.api.schemas import ChatResponse, EvaluationResponse
from app.tests.redis_availability import redis_available

pytestmark = pytest.mark.skipif(not redis_available(), reason="Redis 8 not reachable")


CHAT_RESPONSE_FIELDS = {
    "thread_id",
    "answer",
    "generated_at",
    "response_time_ms",
    "decision",
    "sources",
    "steps",
    "degraded",
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
    # Arrange
    eval_only_fields = EVALUATION_RESPONSE_FIELDS - CHAT_RESPONSE_FIELDS

    # Assert
    assert eval_only_fields & set(ChatResponse.model_fields) == set()
    assert {"intent", "claim", "tool_calls", "findings"} <= eval_only_fields


async def test_openapi_exposes_chat_and_eval_as_distinct_response_schemas(client):
    # Act
    schema = (await client.get("/openapi.json")).json()

    # Assert
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
