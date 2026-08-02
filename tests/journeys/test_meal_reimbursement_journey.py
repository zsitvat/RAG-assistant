from langchain_core.messages import AIMessage

from app.agent.calculator import ReimbursementCalculator
from app.agent.graph import build_agent_graph
from app.agent.model import ExpenseClaim, IntentClassification
from app.agent.nodes import AgentNodes
from app.rules.loader import load_rule_catalogue
from tests.fakes import ScriptedChatModel, build_agent_tools, policy_document, tool_message

CATALOGUE = load_rule_catalogue()
CALCULATOR = ReimbursementCalculator(CATALOGUE)


def test_reference_dinner_example_produces_the_expected_reimbursement_and_citations():
    document = policy_document(
        "01",
        "4. Business meals",
        "Business meals are reimbursed up to 15,000 HUF per person per occasion.",
        ["R-MEAL-01"],
        ["meal"],
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
                            "args": {"question": "business meal limit", "category": "meal"},
                            "id": "1",
                        }
                    ],
                ),
                AIMessage(content="", tool_calls=[{"name": "calculate", "args": {}, "id": "2"}]),
                AIMessage(content="", tool_calls=[{"name": "check_rules", "args": {}, "id": "3"}]),
                AIMessage(content=""),
                AIMessage(
                    content=(
                        "The business dinner is reimbursable up to 45,000 HUF (cap 45,000 HUF, "
                        "5,000 HUF over the cap) [S1]."
                    )
                ),
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
    nodes = AgentNodes(model, model, tools, CALCULATOR)
    graph = build_agent_graph(nodes)

    result = graph.invoke(
        {
            "messages": [
                (
                    "human",
                    "We had a business dinner for 3 people, total bill was 50000 HUF, "
                    "I have the receipt.",
                )
            ]
        },
        config={"configurable": {"thread_id": "meal-1"}, "recursion_limit": 20},
    )

    calculate_message = tool_message(result, "calculate")
    assert calculate_message.artifact.amount_huf == 45000
    assert calculate_message.artifact.cap_huf == 45000
    assert calculate_message.artifact.excess_huf == 5000

    check_rules_message = tool_message(result, "check_rules")
    rule_ids = {finding.rule_id for finding in check_rules_message.artifact}
    assert "R-MEAL-02" in rule_ids
    assert "SUBMISSION-APPROVAL" in rule_ids
    assert "SUBMISSION-DOCUMENTS" in rule_ids
    assert "MEAL-REQUIRED-DOCUMENTS" in rule_ids

    search_message = tool_message(result, "search_policies")
    assert search_message.artifact.citations[0].marker == "S1"
    assert search_message.artifact.citations[0].doc_id == "01"

    assert result["decision"] == "partially_eligible"
    assert "45,000" in result["messages"][-1].content
    assert "[S1]" in result["messages"][-1].content
