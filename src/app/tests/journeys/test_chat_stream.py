import json

import pytest
from langchain_core.messages import AIMessage

from app.agent.calculator import ReimbursementCalculator
from app.agent.graph import build_agent_graph
from app.agent.model import ExpenseClaim, IntentClassification
from app.agent.nodes import AgentNodes
from app.agent.service import AgentService
from app.rules.loader import load_rule_catalogue
from app.tests.fakes import ScriptedChatModel, build_agent_tools, policy_document

CATALOGUE = load_rule_catalogue()
CALCULATOR = ReimbursementCalculator(CATALOGUE)
MEAL_DOCUMENT = policy_document(
    "01",
    "4. Business meals",
    "Business meals are reimbursed up to 15,000 HUF per person.",
    ["R-MEAL-01"],
    ["meal"],
)
ANSWER = "The business meal limit is 15,000 HUF per person [S1]."


def _service(model: ScriptedChatModel) -> AgentService:
    tools = build_agent_tools(MEAL_DOCUMENT, CATALOGUE)
    return AgentService(build_agent_graph(AgentNodes(model, model, tools, CALCULATOR)))


def _grounded_model() -> ScriptedChatModel:
    return ScriptedChatModel(
        chat_responses=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "search_policies",
                            "args": {"question": "meal limit", "category": "meal"},
                            "id": "1",
                        }
                    ],
                ),
                AIMessage(content=""),
                AIMessage(content=ANSWER),
            ]
        ),
        structured_responses=iter(
            [IntentClassification(intent="policy_question", category="meal"), ExpenseClaim()]
        ),
    )


async def _collect(service: AgentService, thread_id: str, message: str) -> list:
    return [event async for event in service.stream(thread_id, message)]


async def test_stream_emits_only_the_documented_event_types_and_ends_with_one_result():
    events = await _collect(_service(_grounded_model()), "stream-1", "What is the meal limit?")

    assert {event.event for event in events} <= {"step", "source", "token", "result"}
    assert events[-1].event == "result"
    assert [event.event for event in events].count("result") == 1


async def test_step_events_are_allow_listed_and_deduplicated():
    events = await _collect(_service(_grounded_model()), "stream-2", "What is the meal limit?")

    steps = [event.data for event in events if event.event == "step"]

    assert steps == list(dict.fromkeys(steps))
    assert steps == [
        "Request understood",
        "Information extracted",
        "Policies searched",
        "Answer prepared",
    ]


async def test_source_events_carry_citation_metadata_and_are_deduplicated():
    events = await _collect(_service(_grounded_model()), "stream-3", "What is the meal limit?")

    sources = [event.data for event in events if event.event == "source"]

    assert len(sources) == 1
    assert sources[0].source_id == "S1"
    assert sources[0].doc_id == "01"
    assert set(sources[0].model_dump()) == {"source_id", "doc_id", "title", "section"}


async def test_token_events_contain_only_final_answer_text():
    events = await _collect(_service(_grounded_model()), "stream-4", "What is the meal limit?")

    streamed = "".join(event.data for event in events if event.event == "token")

    assert streamed == ANSWER
    assert "tool_call" not in streamed
    assert "intent" not in streamed


async def test_step_and_source_events_arrive_before_the_final_result():
    events = await _collect(_service(_grounded_model()), "stream-5", "What is the meal limit?")

    kinds = [event.event for event in events]

    assert kinds.index("step") < kinds.index("result")
    assert kinds.index("source") < kinds.index("result")


async def test_streamed_result_matches_the_blocking_endpoint():
    streamed_events = await _collect(_service(_grounded_model()), "parity-a", "meal limit?")
    streamed = streamed_events[-1].data
    blocking = _service(_grounded_model()).respond("parity-b", "meal limit?")

    assert streamed.answer == blocking.answer
    assert streamed.steps == blocking.steps
    assert [s.model_dump() for s in streamed.sources] == [s.model_dump() for s in blocking.sources]
    assert streamed.decision == blocking.decision


async def test_clarification_completes_without_any_token_event():
    model = ScriptedChatModel(
        chat_responses=iter([]),
        structured_responses=iter(
            [
                IntentClassification(intent="expense_check", category="meal"),
                ExpenseClaim(category="meal", amount_huf=1000),
            ]
        ),
    )

    events = await _collect(_service(model), "stream-clarify", "I spent 1000 HUF on a meal.")

    assert not [event for event in events if event.event == "token"]
    result = events[-1].data
    assert events[-1].event == "result"
    assert result.answer
    assert result.decision == "needs_info"


async def test_out_of_scope_completes_without_any_token_event():
    model = ScriptedChatModel(
        chat_responses=iter([]),
        structured_responses=iter([IntentClassification(intent="unsupported"), ExpenseClaim()]),
    )

    events = await _collect(_service(model), "stream-refuse", "What tax rate applies to me?")

    assert not [event for event in events if event.event == "token"]
    assert events[-1].data.decision == "out_of_scope"


@pytest.mark.parametrize("event_type", ["step", "source", "token", "result"])
def test_sse_rendering_uses_the_event_and_data_wire_format(event_type):
    from app.api.schemas import StreamEvent

    event = StreamEvent(event=event_type, data="x") if event_type != "source" else None
    if event is None:
        from app.api.schemas import ChatSource

        event = StreamEvent(
            event="source",
            data=ChatSource(source_id="S1", doc_id="01", title="Doc", section="4"),
        )

    rendered = event.to_sse()

    assert rendered.startswith(f"event: {event_type}\ndata: ")
    assert rendered.endswith("\n\n")
    assert "data" in json.loads(rendered.split("data: ", 1)[1].strip())


async def test_streaming_endpoint_serves_sse_over_http():
    from types import SimpleNamespace

    from fastapi import FastAPI
    from httpx2 import ASGITransport, AsyncClient

    from app.api.routes.chat import router

    api = FastAPI()
    api.include_router(router)
    api.state.dependencies = SimpleNamespace(
        agent_service=_service(_grounded_model()), checkpointer=None
    )

    async with (
        AsyncClient(transport=ASGITransport(app=api), base_url="http://t") as http,
        http.stream(
            "POST", "/chat/stream", json={"thread_id": "http-1", "message": "meal limit?"}
        ) as response,
    ):
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        names = [
            line.removeprefix("event: ").strip()
            async for line in response.aiter_lines()
            if line.startswith("event: ")
        ]

    assert names[0] == "step"
    assert names[-1] == "result"
    assert set(names) <= {"step", "source", "token", "result"}
