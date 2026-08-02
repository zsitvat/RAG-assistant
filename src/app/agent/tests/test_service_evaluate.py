from datetime import date

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
    "Business meals are reimbursed up to 15,000 HUF per person per occasion.",
    ["R-MEAL-01"],
    ["meal"],
)


def _service(model: ScriptedChatModel) -> AgentService:
    tools = build_agent_tools(MEAL_DOCUMENT, CATALOGUE)
    nodes = AgentNodes(model, model, tools, CALCULATOR)
    return AgentService(build_agent_graph(nodes))


def _meal_turn_model() -> ScriptedChatModel:
    return ScriptedChatModel(
        chat_responses=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "search_policies",
                            "args": {"question": "business meal limit", "category": "meal"},
                            "id": "1",
                        }
                    ],
                ),
                AIMessage(content="", tool_calls=[{"name": "calculate", "args": {}, "id": "2"}]),
                AIMessage(content="", tool_calls=[{"name": "check_rules", "args": {}, "id": "3"}]),
                AIMessage(content=""),
                AIMessage(content="The business dinner is reimbursable up to 45,000 HUF [S1]."),
            ]
        ),
        structured_responses=iter(
            [
                IntentClassification(intent="expense_check", category="meal"),
                ExpenseClaim(
                    category="meal",
                    amount_huf=50000,
                    headcount=3,
                    is_business_related=True,
                    non_reimbursable_amount=0,
                    has_receipt=True,
                    provided_documents=["invoice", "business_purpose_note", "participant_list"],
                    approval_obtained=True,
                ),
            ]
        ),
    )


async def test_evaluate_projects_tool_calls_calculation_findings_and_citations():
    service = _service(_meal_turn_model())

    result = await service.evaluate(
        "eval-meal-1",
        "We had a business dinner for 3 people, total bill was 50000 HUF, I have the receipt.",
        reference_date=date(2026, 8, 1),
    )

    assert result.thread_id == "eval-meal-1"
    assert result.intent == "expense_check"
    assert result.category == "meal"
    assert result.decision == "partially_eligible"
    assert result.missing_slots == []
    assert result.tool_calls == ["search_policies", "calculate", "check_rules"]
    assert result.calculation.amount_huf == 45000
    assert result.calculation.excess_huf == 5000
    assert {finding.rule_id for finding in result.findings} >= {"R-MEAL-02", "SUBMISSION-APPROVAL"}
    assert result.retrieved_doc_ids == ["01"]
    assert result.cited_doc_ids == ["01"]
    assert result.degraded is False


async def test_evaluate_reports_degraded_when_classification_falls_back_to_a_default():
    model = ScriptedChatModel(
        chat_responses=iter([AIMessage(content="")]), structured_responses=iter([])
    )
    tools = build_agent_tools(MEAL_DOCUMENT, CATALOGUE)
    nodes = AgentNodes(model, model, tools, CALCULATOR)
    service = AgentService(build_agent_graph(nodes))

    result = await service.evaluate("eval-degraded-1", "hello", reference_date=date(2026, 8, 1))

    assert result.degraded is True
    assert result.intent == "policy_question"


async def test_evaluate_pins_the_reference_date_used_by_the_deadline_check():
    model = ScriptedChatModel(
        chat_responses=iter(
            [
                AIMessage(content="", tool_calls=[{"name": "check_rules", "args": {}, "id": "1"}]),
                AIMessage(content=""),
                AIMessage(content="Your deadline status is above."),
            ]
        ),
        structured_responses=iter(
            [
                IntentClassification(intent="deadline_check", category=None),
                ExpenseClaim(expense_date=date(2026, 7, 10)),
            ]
        ),
    )
    tools = build_agent_tools(MEAL_DOCUMENT, CATALOGUE)
    nodes = AgentNodes(model, model, tools, CALCULATOR)
    service = AgentService(build_agent_graph(nodes))

    result = await service.evaluate(
        "eval-deadline-1",
        "Is my expense still within the deadline?",
        reference_date=date(2026, 8, 2),
    )

    deadline_finding = next(f for f in result.findings if f.rule_id == "SUBMISSION-DEADLINE")
    assert deadline_finding.status == "pass"
    assert "7 days remain" in deadline_finding.message
