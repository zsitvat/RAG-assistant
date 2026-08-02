import os

import pytest
import redis as redis_lib
from langchain_core.messages import AIMessage

from app.agent.calculator import ReimbursementCalculator
from app.agent.graph import build_agent_graph
from app.agent.model import ExpenseClaim, IntentClassification
from app.agent.nodes import AgentNodes
from app.agent.service import AgentService
from app.integrations.checkpointer import CHECKPOINT_TTL_MINUTES, build_checkpointer
from app.rules.loader import load_rule_catalogue
from app.tests.fakes import ScriptedChatModel, build_agent_tools, policy_document

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


async def _invoke(graph, payload: dict, config: dict) -> dict:
    return await graph.ainvoke(payload, config=config)


async def _get_state(graph, config: dict):
    return await graph.aget_state(config)


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


async def test_pending_claim_survives_a_new_graph_instance_for_the_same_thread():
    checkpointer = await build_checkpointer(TEST_REDIS_URL)
    config = _config("checkpoint-restart")
    await checkpointer.adelete_thread("checkpoint-restart")
    await _invoke(
        _graph(_clarifying_model(), checkpointer),
        {"messages": [("human", "I drive 18 km, 10 days a month.")]},
        config,
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
    restarted = _graph(resume_model, await build_checkpointer(TEST_REDIS_URL))

    state = await _get_state(restarted, config)

    assert ExpenseClaim.from_state(state.values["claim"]).distance_km == 18
    assert state.values["decision"] == "needs_info"


async def test_checkpoints_use_their_own_namespace_and_expire_after_24_hours(redis_client):
    checkpointer = await build_checkpointer(TEST_REDIS_URL)
    config = _config("checkpoint-ttl")
    await checkpointer.adelete_thread("checkpoint-ttl")
    await _invoke(
        _graph(_clarifying_model(), checkpointer),
        {"messages": [("human", "I drive 18 km, 10 days a month.")]},
        config,
    )

    keys = [key.decode() for key in redis_client.keys("checkpoint*checkpoint-ttl*")]

    assert keys
    assert all(not key.startswith("chunk:") for key in keys)
    assert {redis_client.ttl(key) for key in keys} == {CHECKPOINT_TTL_MINUTES * 60}


async def test_deleting_a_thread_makes_the_next_message_start_a_new_conversation(redis_client):
    checkpointer = await build_checkpointer(TEST_REDIS_URL)
    config = _config("checkpoint-reset")
    await checkpointer.adelete_thread("checkpoint-reset")
    await _invoke(
        _graph(_clarifying_model(), checkpointer),
        {"messages": [("human", "I drive 18 km, 10 days a month.")]},
        config,
    )
    assert redis_client.keys("checkpoint*checkpoint-reset*")

    await checkpointer.adelete_thread("checkpoint-reset")

    assert redis_client.keys("checkpoint*checkpoint-reset*") == []
    fresh_graph = build_agent_graph(
        AgentNodes(_clarifying_model(), _clarifying_model(), [], CALCULATOR), checkpointer
    )
    state = await _get_state(fresh_graph, config)
    assert state.values == {}


async def test_separate_workers_share_one_thread_through_redis():
    config = _config("checkpoint-shared")
    setup_checkpointer = await build_checkpointer(TEST_REDIS_URL)
    await setup_checkpointer.adelete_thread("checkpoint-shared")

    worker_one = _graph(_clarifying_model(), await build_checkpointer(TEST_REDIS_URL))
    worker_two = _graph(_clarifying_model(), await build_checkpointer(TEST_REDIS_URL))

    await _invoke(worker_one, {"messages": [("human", "I drive 18 km, 10 days a month.")]}, config)

    state = await _get_state(worker_two, config)
    restored = ExpenseClaim.from_state(state.values["claim"])
    assert restored.distance_km == 18


async def test_agent_service_stream_completes_against_the_real_redis_checkpointer():
    """Regression test: AsyncRedisSaver's sync methods raise InvalidStateError when called
    directly on the event-loop thread. AgentService.astream() must use aget_state(), not
    get_state(), to build its final result event — this only reproduces against a real
    async checkpointer, never against the default in-memory one used by other graph tests."""
    checkpointer = await build_checkpointer(TEST_REDIS_URL)
    await checkpointer.adelete_thread("checkpoint-stream")
    service = AgentService(_graph(_clarifying_model(), checkpointer))

    events = [
        event
        async for event in service.astream("checkpoint-stream", "I drive 18 km, 10 days a month.")
    ]

    assert events[-1].event == "result"
    assert events[-1].data.decision == "needs_info"
