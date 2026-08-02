from langchain_core.messages import AIMessage

from app.agent.calculator import ReimbursementCalculator
from app.agent.graph import build_agent_graph
from app.agent.model import ExpenseClaim, IntentClassification
from app.agent.nodes import AgentNodes
from app.agent.static_texts import CLARIFICATION_QUESTIONS
from app.rules.loader import load_rule_catalogue
from app.tests.fakes import ScriptedChatModel, build_agent_tools, policy_document, tool_message

CATALOGUE = load_rule_catalogue()
CALCULATOR = ReimbursementCalculator(CATALOGUE)

COMMUTING_DOCUMENT = policy_document(
    "03",
    "4. Commuting by personal vehicle",
    "Reimbursement rate HUF 30 per kilometre travelled, monthly maximum HUF 40,000.",
    ["R-COMM-02"],
    ["commuting"],
)


def _graph(model: ScriptedChatModel):
    tools = build_agent_tools(COMMUTING_DOCUMENT, CATALOGUE)
    return build_agent_graph(AgentNodes(model, model, tools, CALCULATOR))


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": 12}


def _search_then_calculate_then_answer(answer: str) -> list[AIMessage]:
    return [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_policies",
                    "args": {"question": "commuting support", "category": "commuting"},
                    "id": "1",
                }
            ],
        ),
        AIMessage(content="", tool_calls=[{"name": "calculate", "args": {}, "id": "2"}]),
        AIMessage(content="", tool_calls=[{"name": "check_rules", "args": {}, "id": "3"}]),
        AIMessage(content=""),
        AIMessage(content=answer),
    ]


async def test_missing_distance_direction_ends_the_turn_with_a_focused_question():
    model = ScriptedChatModel(
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

    result = await _graph(model).ainvoke(
        {"messages": [("human", "I drive 18 km to the office on 10 days a month.")]},
        config=_config("commute-clarify"),
    )

    assert result["decision"] == "needs_info"
    assert result["messages"][-1].content == CLARIFICATION_QUESTIONS["distance_is_one_way"]
    assert result["claim"].distance_km == 18


async def test_clarification_answer_resumes_the_same_thread_and_completes_the_claim():
    model = ScriptedChatModel(
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
    graph = _graph(model)
    config = _config("commute-resume")
    await graph.ainvoke(
        {"messages": [("human", "I drive 18 km to the office on 10 days a month.")]},
        config=config,
    )

    resume_model = ScriptedChatModel(
        chat_responses=iter(_search_then_calculate_then_answer("You may claim 10,800 HUF [S1].")),
        structured_responses=iter(
            [
                IntentClassification(intent="calculation", category="commuting"),
                ExpenseClaim(distance_is_one_way=True),
            ]
        ),
    )
    resumed = build_agent_graph(
        AgentNodes(
            resume_model,
            resume_model,
            build_agent_tools(COMMUTING_DOCUMENT, CATALOGUE),
            CALCULATOR,
        ),
        graph.checkpointer,
    )

    result = await resumed.ainvoke({"messages": [("human", "one-way")]}, config=config)

    assert result["claim"].distance_km == 18
    assert result["claim"].commute_days_per_month == 10
    assert result["claim"].distance_is_one_way is True
    assert tool_message(result, "calculate").artifact.amount_huf == 10800


async def test_a_new_expense_in_the_same_thread_does_not_inherit_the_previous_claim():
    model = ScriptedChatModel(
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
    graph = _graph(model)
    config = _config("commute-switch")
    await graph.ainvoke({"messages": [("human", "I drive 18 km, 10 days a month.")]}, config=config)

    follow_up_model = ScriptedChatModel(
        chat_responses=iter(_search_then_calculate_then_answer("That trip is 22,500 HUF [S1].")),
        structured_responses=iter(
            [
                IntentClassification(intent="calculation", category="mileage"),
                ExpenseClaim(category="mileage", distance_km=250, distance_is_one_way=False),
            ]
        ),
    )
    follow_up = build_agent_graph(
        AgentNodes(
            follow_up_model,
            follow_up_model,
            build_agent_tools(COMMUTING_DOCUMENT, CATALOGUE),
            CALCULATOR,
        ),
        graph.checkpointer,
    )

    result = await follow_up.ainvoke(
        {"messages": [("human", "Separately, I drove 250 km round trip to a client.")]},
        config=config,
    )

    assert result["claim"].category == "mileage"
    assert result["claim"].commute_days_per_month is None
    assert tool_message(result, "calculate").artifact.amount_huf == 11250


async def test_refusing_to_disambiguate_the_distance_returns_both_conditional_outcomes():
    model = ScriptedChatModel(
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
    graph = _graph(model)
    config = _config("commute-refuse")
    await graph.ainvoke({"messages": [("human", "I drive 18 km, 10 days a month.")]}, config=config)

    refusal_model = ScriptedChatModel(
        chat_responses=iter([]),
        structured_responses=iter(
            [
                IntentClassification(intent="calculation", category="commuting"),
                ExpenseClaim(),
            ]
        ),
    )
    refused = build_agent_graph(
        AgentNodes(
            refusal_model,
            refusal_model,
            build_agent_tools(COMMUTING_DOCUMENT, CATALOGUE),
            CALCULATOR,
        ),
        graph.checkpointer,
    )

    result = await refused.ainvoke({"messages": [("human", "I do not know.")]}, config=config)

    answer = result["messages"][-1].content
    assert "10800" in answer
    assert "5400" in answer
    assert answer != CLARIFICATION_QUESTIONS["distance_is_one_way"]
