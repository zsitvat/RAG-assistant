from datetime import date

from langchain_core.messages import AIMessage

from app.agent.calculator import ReimbursementCalculator
from app.agent.graph import build_agent_graph
from app.agent.model import ExpenseClaim, IntentClassification
from app.agent.nodes import AgentNodes
from app.rules.loader import load_rule_catalogue
from app.tests.fakes import (
    ScriptedChatModel,
    build_agent_tools,
    policy_document,
    tool_calls,
    tool_message,
)

CATALOGUE = load_rule_catalogue()
CALCULATOR = ReimbursementCalculator(CATALOGUE)


def test_recreational_benefit_claim_produces_the_expected_reimbursement_and_decision():
    document = policy_document(
        "05",
        "3. Annual benefit allowances",
        "The recreational benefit allowance is HUF 120,000 per year, reimbursed at 100%.",
        ["R-BEN-01"],
        ["benefits"],
    )
    tools = build_agent_tools(document, CATALOGUE)
    model = ScriptedChatModel(
        chat_responses=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "search_policies",
                            "args": {
                                "question": "recreational benefit allowance",
                                "category": "benefits",
                            },
                            "id": "1",
                        }
                    ],
                ),
                AIMessage(content="", tool_calls=[{"name": "calculate", "args": {}, "id": "2"}]),
                AIMessage(content="", tool_calls=[{"name": "check_rules", "args": {}, "id": "3"}]),
                AIMessage(content=""),
                AIMessage(
                    content=(
                        "Your recreational booking is reimbursable in full, 90,000 HUF, within "
                        "your remaining allowance [S1]."
                    )
                ),
            ]
        ),
        structured_responses=iter(
            [
                IntentClassification(intent="expense_check", category="benefits"),
                ExpenseClaim(
                    category="benefits",
                    expense_type="recreational",
                    amount_huf=90000,
                    annual_budget_used_huf=0,
                    tenure_months=12,
                    approval_obtained=True,
                    has_receipt=True,
                    provided_documents=["invoice_or_booking_document"],
                ),
            ]
        ),
    )
    nodes = AgentNodes(model, model, tools, CALCULATOR)
    graph = build_agent_graph(nodes)

    result = graph.invoke(
        {
            "messages": [
                (
                    "human",
                    "I booked a 90000 HUF holiday accommodation, I have been employed for a "
                    "year, approval was obtained, and I have the invoice.",
                )
            ]
        },
        config={"configurable": {"thread_id": "benefits-1"}, "recursion_limit": 20},
    )

    calculate_message = tool_message(result, "calculate")
    assert calculate_message.artifact.amount_huf == 90000
    assert calculate_message.artifact.cap_huf == 120000

    check_rules_message = tool_message(result, "check_rules")
    findings_by_id = {f.rule_id: f for f in check_rules_message.artifact}
    assert findings_by_id["R-BEN-01"].status == "pass"
    assert findings_by_id["R-BEN-TENURE"].status == "pass"
    assert findings_by_id["R-BEN-CARRY-OVER"].status == "pass"

    assert result["decision"] == "eligible"
    assert "[S1]" in result["messages"][-1].content


def test_deadline_question_uses_only_search_and_check_rules_no_calculation():
    document = policy_document(
        "01",
        "7. Submission deadline and late claims",
        "A claim must be submitted within 30 calendar days of the expense date.",
        [],
        ["general", "meal"],
    )
    tools = build_agent_tools(document, CATALOGUE)
    model = ScriptedChatModel(
        chat_responses=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "search_policies",
                            "args": {"question": "submission deadline", "category": "meal"},
                            "id": "1",
                        }
                    ],
                ),
                AIMessage(content="", tool_calls=[{"name": "check_rules", "args": {}, "id": "2"}]),
                AIMessage(content=""),
                AIMessage(
                    content=(
                        "Your claim is past the 30-day deadline; please submit a late-claim "
                        "exception request to Finance with a written justification [S1]."
                    )
                ),
            ]
        ),
        structured_responses=iter(
            [
                IntentClassification(intent="deadline_check", category="meal"),
                ExpenseClaim(category="meal", expense_date=date(2026, 6, 1)),
            ]
        ),
    )
    nodes = AgentNodes(model, model, tools, CALCULATOR)
    graph = build_agent_graph(nodes)

    result = graph.invoke(
        {"messages": [("human", "Can I still submit a meal receipt from 2026-06-01?")]},
        config={"configurable": {"thread_id": "deadline-1"}, "recursion_limit": 20},
    )

    assert "calculate" not in tool_calls(result)

    check_rules_message = tool_message(result, "check_rules")
    deadline_finding = next(
        f for f in check_rules_message.artifact if f.rule_id == "SUBMISSION-DEADLINE"
    )
    assert deadline_finding.status == "fail"
    assert "Finance" in deadline_finding.message

    assert "[S1]" in result["messages"][-1].content


def test_document_question_uses_only_search_and_check_rules():
    document = policy_document(
        "02",
        "7. Required documents",
        "Attach an approved travel request, booking confirmation, and hotel invoice.",
        ["TRAVEL-REQUIRED-DOCUMENTS"],
        ["travel"],
    )
    tools = build_agent_tools(document, CATALOGUE)
    model = ScriptedChatModel(
        chat_responses=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "search_policies",
                            "args": {"question": "travel claim documents", "category": "travel"},
                            "id": "1",
                        }
                    ],
                ),
                AIMessage(content="", tool_calls=[{"name": "check_rules", "args": {}, "id": "2"}]),
                AIMessage(content=""),
                AIMessage(
                    content=(
                        "Attach the approved travel request, booking confirmation, and hotel "
                        "invoice [S1]."
                    )
                ),
            ]
        ),
        structured_responses=iter(
            [
                IntentClassification(intent="document_requirements", category="travel"),
                ExpenseClaim(category="travel"),
            ]
        ),
    )
    graph = build_agent_graph(AgentNodes(model, model, tools, CALCULATOR))

    result = graph.invoke(
        {"messages": [("human", "Which documents do I need for a travel claim?")]},
        config={"configurable": {"thread_id": "documents-1"}, "recursion_limit": 20},
    )

    assert tool_calls(result) == ["search_policies", "check_rules"]
    requirement = tool_message(result, "check_rules").artifact[0]
    assert requirement.rule_id == "TRAVEL-REQUIRED-DOCUMENTS"
    assert requirement.doc_ref == "02#travel-required-documents"
    assert "[S1]" in result["messages"][-1].content
