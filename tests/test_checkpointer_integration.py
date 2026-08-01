import os

import pytest
import redis as redis_lib
from langchain_core.messages import AIMessage

from app.agent.calculator import ReimbursementCalculator
from app.agent.graph import build_agent_graph
from app.agent.model import ExpenseClaim, IntentClassification
from app.agent.nodes import AgentNodes
from app.integrations.checkpointer import CHECKPOINT_TTL_MINUTES, build_checkpointer
from app.rules.loader import load_rule_catalogue
from tests.fakes import ScriptedChatModel, build_agent_tools, policy_document

TEST_REDIS_URL = os.environ.get("TEST_REDIS_URL", "redis://127.0.0.1:6379/0")
CATALOGUE = load_rule_catalogue()
CALCULATOR = ReimbursementCalculator(CATALOGUE)

COMMUTING_DOCUMENT = policy_document(
    "03",
    "4. Commuting by personal vehicle",
    "Reimbursement rate HUF 30 per kilometre travelled, monthly maximum HUF 40,000.",
    ["R-COMM-02"],
    ["commuting"],
)


def _redis_available() -> bool:
    try:
        redis_lib.Redis.from_url(TEST_REDIS_URL).ping()
    except redis_lib.RedisError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _redis_available(), reason="Redis 8 not reachable")


@pytest.fixture
def redis_client() -> redis_lib.Redis:
    return redis_lib.Redis.from_url(TEST_REDIS_URL)


@pytest.fixture
def checkpointer():
    return build_checkpointer(TEST_REDIS_URL)


def _clarifying_model() -> ScriptedChatModel:
    return ScriptedChatModel(
        chat_responses=iter([]),
        structured_responses=iter(
            [
                IntentClassification(intent="calculation", category="commuting"),
                ExpenseClaim(
                    category="commuting",
                    expense_type="own_car",
                    distance_km=18,
                    commute_days_per_month=10,
                ),
            ]
        ),
    )


def _graph(model: ScriptedChatModel, checkpointer):
    tools = build_agent_tools(COMMUTING_DOCUMENT, CATALOGUE)
    return build_agent_graph(AgentNodes(model, model, tools, CALCULATOR), checkpointer)


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": 12}


def test_pending_claim_survives_a_new_graph_instance_for_the_same_thread(checkpointer):
    config = _config("checkpoint-restart")
    checkpointer.delete_thread("checkpoint-restart")
    _graph(_clarifying_model(), checkpointer).invoke(
        {"messages": [("human", "I drive 18 km, 10 days a month.")]}, config=config
    )

    resume_model = ScriptedChatModel(
        chat_responses=iter([AIMessage(content="")]),
        structured_responses=iter(
            [
                IntentClassification(intent="calculation", category="commuting"),
                ExpenseClaim(distance_is_one_way=True),
            ]
        ),
    )
    restarted = _graph(resume_model, build_checkpointer(TEST_REDIS_URL))

    state = restarted.get_state(config)

    assert ExpenseClaim.from_state(state.values["claim"]).distance_km == 18
    assert state.values["decision"] == "needs_info"


def test_checkpoints_use_their_own_namespace_and_expire_after_24_hours(checkpointer, redis_client):
    config = _config("checkpoint-ttl")
    checkpointer.delete_thread("checkpoint-ttl")
    _graph(_clarifying_model(), checkpointer).invoke(
        {"messages": [("human", "I drive 18 km, 10 days a month.")]}, config=config
    )

    keys = [key.decode() for key in redis_client.keys("checkpoint*checkpoint-ttl*")]

    assert keys
    assert all(not key.startswith("chunk:") for key in keys)
    assert {redis_client.ttl(key) for key in keys} == {CHECKPOINT_TTL_MINUTES * 60}


def test_deleting_a_thread_makes_the_next_message_start_a_new_conversation(
    checkpointer, redis_client
):
    config = _config("checkpoint-reset")
    checkpointer.delete_thread("checkpoint-reset")
    _graph(_clarifying_model(), checkpointer).invoke(
        {"messages": [("human", "I drive 18 km, 10 days a month.")]}, config=config
    )
    assert redis_client.keys("checkpoint*checkpoint-reset*")

    checkpointer.delete_thread("checkpoint-reset")

    assert redis_client.keys("checkpoint*checkpoint-reset*") == []
    assert (
        build_agent_graph(
            AgentNodes(_clarifying_model(), _clarifying_model(), [], CALCULATOR), checkpointer
        )
        .get_state(config)
        .values
        == {}
    )


def test_separate_workers_share_one_thread_through_redis(checkpointer):
    config = _config("checkpoint-shared")
    checkpointer.delete_thread("checkpoint-shared")
    worker_one = _graph(_clarifying_model(), build_checkpointer(TEST_REDIS_URL))
    worker_two = _graph(_clarifying_model(), build_checkpointer(TEST_REDIS_URL))

    worker_one.invoke({"messages": [("human", "I drive 18 km, 10 days a month.")]}, config=config)

    restored = ExpenseClaim.from_state(worker_two.get_state(config).values["claim"])
    assert restored.distance_km == 18
