from datetime import date
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from app.agent.calculator import ReimbursementCalculator
from app.agent.deadline import DeadlineChecker
from app.agent.rule_checker import RuleChecker
from app.agent.tools import build_calculate_tool, build_check_rules_tool
from app.rules.loader import load_rule_catalogue

CATALOGUE = load_rule_catalogue()


class _ToolTestState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    claim: dict


def _run_tool_via_node(tool_, claim: dict, tool_name: str, args: dict | None = None):
    graph = StateGraph(_ToolTestState)
    graph.add_node("execute", ToolNode([tool_], handle_tool_errors=True))
    graph.add_edge(START, "execute")
    graph.add_edge("execute", END)
    compiled = graph.compile()

    state = {
        "claim": claim,
        "messages": [
            AIMessage(content="", tool_calls=[{"name": tool_name, "args": args or {}, "id": "1"}])
        ],
    }
    result = compiled.invoke(state, config={"configurable": {"thread_id": "x"}})
    return result["messages"][-1]


def test_calculate_tool_reads_the_claim_from_state_and_returns_an_artifact():
    calculator = ReimbursementCalculator(CATALOGUE)
    tool_ = build_calculate_tool(calculator)

    message = _run_tool_via_node(tool_, {"category": "equipment", "amount_huf": 50000}, "calculate")

    assert message.status == "success"
    assert message.artifact.amount_huf == 50000
    assert "reimbursable" in message.content


def test_calculate_tool_returns_an_error_tool_message_for_an_incomplete_claim():
    calculator = ReimbursementCalculator(CATALOGUE)
    tool_ = build_calculate_tool(calculator)

    message = _run_tool_via_node(tool_, {"category": "meal"}, "calculate")

    assert message.status == "error"


def test_check_rules_tool_reads_the_claim_from_state_and_returns_findings():
    rule_checker = RuleChecker(CATALOGUE, DeadlineChecker(CATALOGUE.submission.deadline_days))
    tool_ = build_check_rules_tool(rule_checker, lambda: date(2026, 8, 1))

    message = _run_tool_via_node(
        tool_, {"category": "equipment", "amount_huf": 20000}, "check_rules"
    )

    assert message.status == "success"
    assert isinstance(message.artifact, list)
    assert message.artifact[0].rule_id == "SUBMISSION-APPROVAL"
