from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from app.agent.calculator import ReimbursementCalculator
from app.agent.graph import build_agent_graph
from app.agent.graph_state import MAX_AGENT_STEPS
from app.agent.model import ExpenseClaim, IntentClassification
from app.agent.nodes import AgentNodes
from app.agent.static_texts import LLM_UNAVAILABLE_MESSAGE, OUT_OF_SCOPE_MESSAGE
from app.rules.loader import load_rule_catalogue
from app.tests.fakes import ScriptedChatModel

CALCULATOR = ReimbursementCalculator(load_rule_catalogue())


def _tool_calls_past_the_step_limit():
    """Yields MAX_AGENT_STEPS tool calls, then a real final answer for generate_response."""
    for i in range(1, MAX_AGENT_STEPS + 1):
        yield AIMessage(
            content="", tool_calls=[{"name": "search_policies", "args": {"n": i}, "id": str(i)}]
        )
    while True:
        yield AIMessage(content="Here is what the policy says so far.")


class _AlwaysFailingChatModel(ScriptedChatModel):
    def _generate(self, *args, **kwargs):
        raise ConnectionError("Ollama is unreachable")


def _make_counting_tool(name: str = "counting_tool"):
    calls: list[dict] = []

    @tool(name, response_format="content_and_artifact")
    def counting_tool(n: int = 0) -> tuple[str, dict]:
        """A trivial tool used to test loop guardrails."""
        calls.append({"n": n})
        return f"called with n={n}", {"n": n, "call_count": len(calls)}

    return counting_tool, calls


def _make_failing_tool(name: str = "failing_tool"):
    @tool(name, response_format="content_and_artifact")
    def failing_tool() -> tuple[str, dict]:
        """A tool that always raises, used to test the arg-error disable guardrail."""
        raise ValueError("invalid arguments")

    return failing_tool


async def _invoke(model, tools, messages, config=None):
    nodes = AgentNodes(model, model, tools, CALCULATOR)
    graph = build_agent_graph(nodes)
    return await graph.ainvoke(
        {"messages": messages},
        config=config or {"configurable": {"thread_id": "t1"}, "recursion_limit": 20},
    )


async def test_general_policy_question_completes_with_a_grounded_answer():
    # Arrange
    tool_, calls = _make_counting_tool("search_policies")
    model = ScriptedChatModel(
        chat_responses=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[{"name": "search_policies", "args": {"n": 1}, "id": "1"}],
                ),
                AIMessage(content=""),
                AIMessage(content="The business meal limit is 15,000 HUF per person [S1]."),
            ]
        ),
        structured_responses=iter(
            [IntentClassification(intent="policy_question", category="meal"), ExpenseClaim()]
        ),
    )

    # Act
    result = await _invoke(model, [tool_], [("human", "What is the business meal limit?")])

    # Assert
    assert calls == [{"n": 1}]
    assert (
        result["messages"][-1].content == "The business meal limit is 15,000 HUF per person [S1]."
    )
    assert result["intent"] == "policy_question"


async def test_unsupported_request_never_calls_a_tool_and_is_refused():
    # Arrange
    tool_, calls = _make_counting_tool()
    model = ScriptedChatModel(
        chat_responses=iter([]),
        structured_responses=iter([IntentClassification(intent="unsupported"), ExpenseClaim()]),
    )

    # Act
    result = await _invoke(model, [tool_], [("human", "What tax rate applies to my salary?")])

    # Assert
    assert calls == []
    assert result["decision"] == "out_of_scope"
    assert result["messages"][-1].content == OUT_OF_SCOPE_MESSAGE


async def test_missing_required_slot_triggers_clarification():
    # Arrange
    model = ScriptedChatModel(
        chat_responses=iter([]),
        structured_responses=iter(
            [
                IntentClassification(intent="expense_check", category="meal"),
                ExpenseClaim(category="meal", amount_huf=1000),
            ]
        ),
    )

    # Act
    result = await _invoke(model, [], [("human", "I spent 1000 HUF on a business meal.")])

    # Assert
    assert result["decision"] == "needs_info"
    assert "how many people" in result["messages"][-1].content.lower()


async def test_clarification_answer_merges_into_the_pending_claim():
    # Arrange
    model = ScriptedChatModel(
        chat_responses=iter([AIMessage(content="")]),
        structured_responses=iter(
            [
                IntentClassification(intent="expense_check", category="meal"),
                ExpenseClaim(
                    headcount=2,
                    is_business_related=True,
                    non_reimbursable_amount=0,
                ),
            ]
        ),
    )
    nodes = AgentNodes(model, model, [], CALCULATOR)
    graph = build_agent_graph(nodes)
    config = {"configurable": {"thread_id": "t2"}, "recursion_limit": 20}

    # Act
    result = await graph.ainvoke(
        {
            "messages": [
                (
                    "human",
                    "It was for 2 people at a client meeting, with no excluded items.",
                )
            ],
            "claim": ExpenseClaim(category="meal", amount_huf=1000),
            "decision": "needs_info",
        },
        config=config,
    )

    # Assert
    assert result["claim"].amount_huf == 1000
    assert result["claim"].headcount == 2
    assert result["decision"] != "needs_info"


async def test_loop_stops_after_max_agent_steps_without_calling_the_model_again():
    # Arrange
    tool_, calls = _make_counting_tool("search_policies")
    responses = [
        AIMessage(
            content="", tool_calls=[{"name": "search_policies", "args": {"n": i}, "id": str(i)}]
        )
        for i in range(MAX_AGENT_STEPS)
    ]
    responses.append(AIMessage(content="Here is what I found so far, evidence is incomplete."))
    model = ScriptedChatModel(
        chat_responses=iter(responses),
        structured_responses=iter([IntentClassification(intent="policy_question"), ExpenseClaim()]),
    )

    # Act
    result = await _invoke(model, [tool_], [("human", "Tell me everything about every policy.")])

    # Assert
    assert len(calls) == MAX_AGENT_STEPS
    final_content = result["messages"][-1].content
    assert final_content.startswith("Here is what I found so far, evidence is incomplete.")
    assert "incomplete information" in final_content


async def test_recursion_limit_is_generous_enough_for_the_worst_case_step_budget():
    # Arrange
    from app.agent.graph_state import RECURSION_LIMIT

    tool_, calls = _make_counting_tool("search_policies")
    model = ScriptedChatModel(
        chat_responses=_tool_calls_past_the_step_limit(),
        structured_responses=iter([IntentClassification(intent="policy_question"), ExpenseClaim()]),
    )
    config = {"configurable": {"thread_id": "t-recursion"}, "recursion_limit": RECURSION_LIMIT}

    # Act
    result = await _invoke(
        model, [tool_], [("human", "Tell me everything about every policy.")], config
    )

    # Assert
    assert len(calls) == MAX_AGENT_STEPS
    assert "incomplete information" in result["messages"][-1].content


async def test_duplicate_identical_tool_call_is_reused_without_re_executing():
    # Arrange
    tool_, calls = _make_counting_tool("search_policies")
    model = ScriptedChatModel(
        chat_responses=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[{"name": "search_policies", "args": {"n": 1}, "id": "1"}],
                ),
                AIMessage(
                    content="",
                    tool_calls=[{"name": "search_policies", "args": {"n": 1}, "id": "2"}],
                ),
                AIMessage(content=""),
                AIMessage(content="Final answer."),
            ]
        ),
        structured_responses=iter([IntentClassification(intent="policy_question"), ExpenseClaim()]),
    )

    # Act
    result = await _invoke(model, [tool_], [("human", "Ask the same thing twice.")])

    # Assert
    assert len(calls) == 1
    assert result["messages"][-1].content == "Final answer."


async def test_tool_disabled_after_repeated_invalid_arguments():
    # Arrange
    failing = _make_failing_tool("failing_tool")
    model = ScriptedChatModel(
        chat_responses=iter(
            [
                AIMessage(content="", tool_calls=[{"name": "failing_tool", "args": {}, "id": "1"}]),
                AIMessage(content="", tool_calls=[{"name": "failing_tool", "args": {}, "id": "2"}]),
                AIMessage(content=""),
                AIMessage(content="Evidence was incomplete after repeated tool failures."),
            ]
        ),
        structured_responses=iter([IntentClassification(intent="policy_question"), ExpenseClaim()]),
    )

    # Act
    result = await _invoke(model, [failing], [("human", "Trigger a failing tool twice.")])

    # Assert
    assert result["messages"][-1].content == "Evidence was incomplete after repeated tool failures."


async def test_generate_response_refuses_without_a_supporting_tool_artifact():
    # Arrange
    model = ScriptedChatModel(
        chat_responses=iter([AIMessage(content="")]),
        structured_responses=iter([IntentClassification(intent="policy_question"), ExpenseClaim()]),
    )

    # Act
    result = await _invoke(model, [], [("human", "What is the policy on X?")])

    # Assert
    assert result["decision"] is None
    assert "do not have enough verified evidence" in result["messages"][-1].content


async def test_llm_unavailable_after_retries_returns_a_clear_failure_not_a_fabricated_answer():
    # Arrange
    model = _AlwaysFailingChatModel(chat_responses=iter([]), structured_responses=iter([]))

    # Act
    result = await _invoke(model, [], [("human", "What is the policy on X?")])

    # Assert
    assert result["messages"][-1].content == LLM_UNAVAILABLE_MESSAGE
