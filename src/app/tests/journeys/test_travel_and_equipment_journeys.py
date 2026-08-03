from types import SimpleNamespace

from fastapi import FastAPI
from httpx2 import ASGITransport, AsyncClient
from langchain_core.messages import AIMessage

from app.agent.calculator import ReimbursementCalculator
from app.agent.graph import build_agent_graph
from app.agent.model import ExpenseClaim, IntentClassification
from app.agent.nodes import AgentNodes
from app.agent.service import AgentService
from app.api.router import router
from app.dependencies import get_agent_service
from app.rules.loader import load_rule_catalogue
from app.tests.fakes import ScriptedChatModel, build_agent_tools, policy_document, tool_message

CATALOGUE = load_rule_catalogue()
CALCULATOR = ReimbursementCalculator(CATALOGUE)


async def test_domestic_accommodation_within_threshold_is_deterministically_approved():
    # Arrange
    document = policy_document(
        "02",
        "3. Accommodation",
        "Domestic accommodation is reimbursed up to 45,000 HUF per night.",
        ["R-TRAVEL-02"],
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
                            "args": {"question": "accommodation limit", "category": "travel"},
                            "id": "1",
                        }
                    ],
                ),
                AIMessage(content="", tool_calls=[{"name": "calculate", "args": {}, "id": "2"}]),
                AIMessage(content="", tool_calls=[{"name": "check_rules", "args": {}, "id": "3"}]),
                AIMessage(content=""),
                AIMessage(
                    content=(
                        "Your domestic accommodation is reimbursable up to 45,000 HUF per night "
                        "[S1]."
                    )
                ),
            ]
        ),
        structured_responses=iter(
            [
                IntentClassification(intent="expense_check", category="travel"),
                ExpenseClaim(
                    category="travel",
                    expense_type="accommodation_domestic",
                    amount_huf=60000,
                    is_business_related=True,
                    is_international_trip=False,
                    approval_obtained=True,
                    has_receipt=True,
                    provided_documents=[
                        "travel_request",
                        "ticket_or_booking_confirmation",
                        "hotel_invoice",
                    ],
                ),
            ]
        ),
    )
    nodes = AgentNodes(model, model, tools, CALCULATOR)
    graph = build_agent_graph(nodes)

    # Act
    result = await graph.ainvoke(
        {
            "messages": [
                (
                    "human",
                    "I booked a domestic hotel for 60000 HUF on a business trip, receipt attached.",
                )
            ]
        },
        config={"configurable": {"thread_id": "travel-1"}, "recursion_limit": 20},
    )

    # Assert
    calculate_message = tool_message(result, "calculate")
    assert calculate_message.artifact.amount_huf == 45000
    assert calculate_message.artifact.cap_huf == 45000
    assert calculate_message.artifact.excess_huf == 15000

    check_rules_message = tool_message(result, "check_rules")
    rule_ids = {finding.rule_id for finding in check_rules_message.artifact}
    assert "R-TRAVEL-01" in rule_ids
    business_use = next(f for f in check_rules_message.artifact if f.rule_id == "R-TRAVEL-04")
    assert business_use.status == "pass"
    travel_approval = next(f for f in check_rules_message.artifact if f.rule_id == "R-TRAVEL-01")
    assert travel_approval.status == "pass"

    search_message = tool_message(result, "search_policies")
    assert search_message.artifact.citations[0].marker == "S1"

    assert result["decision"] == "partially_eligible"
    assert "[S1]" in result["messages"][-1].content


async def test_equipment_above_threshold_without_approval_is_rejected():
    # Arrange
    document = policy_document(
        "01",
        "5. Work equipment and minor purchases",
        "Purchases above HUF 50,000 require department head approval.",
        ["R-EQUIP-01"],
        ["equipment"],
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
                            "args": {"question": "equipment approval threshold"},
                            "id": "1",
                        }
                    ],
                ),
                AIMessage(content="", tool_calls=[{"name": "calculate", "args": {}, "id": "2"}]),
                AIMessage(content="", tool_calls=[{"name": "check_rules", "args": {}, "id": "3"}]),
                AIMessage(content=""),
                AIMessage(
                    content=(
                        "This purchase requires department head approval, which was not "
                        "obtained, so it cannot be reimbursed yet [S1]."
                    )
                ),
            ]
        ),
        structured_responses=iter(
            [
                IntentClassification(intent="expense_check", category="equipment"),
                ExpenseClaim(
                    category="equipment",
                    amount_huf=80000,
                    is_business_related=True,
                    approval_obtained=False,
                    has_receipt=True,
                    provided_documents=["invoice"],
                ),
            ]
        ),
    )
    nodes = AgentNodes(model, model, tools, CALCULATOR)
    graph = build_agent_graph(nodes)

    # Act
    result = await graph.ainvoke(
        {"messages": [("human", "I bought an 80000 HUF monitor without prior approval.")]},
        config={"configurable": {"thread_id": "equipment-1"}, "recursion_limit": 20},
    )

    # Assert
    calculate_message = tool_message(result, "calculate")
    assert calculate_message.artifact.amount_huf == 80000

    check_rules_message = tool_message(result, "check_rules")
    approval_finding = next(f for f in check_rules_message.artifact if f.rule_id == "R-EQUIP-01")
    assert approval_finding.status == "fail"

    assert result["decision"] == "not_eligible"


async def test_travel_journey_is_exposed_through_the_chat_endpoint():
    # Arrange
    document = policy_document(
        "02",
        "3. Accommodation",
        "Domestic accommodation is reimbursed up to 45,000 HUF per night.",
        ["R-TRAVEL-02"],
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
                            "args": {"question": "domestic hotel", "category": "travel"},
                            "id": "1",
                        }
                    ],
                ),
                AIMessage(content="", tool_calls=[{"name": "calculate", "args": {}, "id": "2"}]),
                AIMessage(content="", tool_calls=[{"name": "check_rules", "args": {}, "id": "3"}]),
                AIMessage(content=""),
                AIMessage(content="45,000 HUF is reimbursable [S1]."),
            ]
        ),
        structured_responses=iter(
            [
                IntentClassification(intent="expense_check", category="travel"),
                ExpenseClaim(
                    category="travel",
                    expense_type="accommodation_domestic",
                    amount_huf=60000,
                    is_business_related=True,
                    is_international_trip=False,
                    approval_obtained=True,
                    has_receipt=True,
                    provided_documents=[
                        "travel_request",
                        "ticket_or_booking_confirmation",
                        "hotel_invoice",
                    ],
                ),
            ]
        ),
    )
    graph = build_agent_graph(AgentNodes(model, model, tools, CALCULATOR))
    test_app = FastAPI()
    test_app.include_router(router)
    service = AgentService(graph)
    test_app.state.dependencies = SimpleNamespace(agent_service=service)

    async def provide_agent_service():
        """Provides the real journey service."""

        return service

    test_app.dependency_overrides[get_agent_service] = provide_agent_service

    transport = ASGITransport(app=test_app)

    # Act
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/chat",
            json={"thread_id": "travel-api-1", "message": "Can I claim my domestic hotel?"},
        )

    # Assert
    assert response.status_code == 200
    body = response.json()
    assert body["decision"] == "partially_eligible"
    assert body["sources"][0]["source_id"] == "S1"
    assert body["steps"] == [
        "Intent classified — expense_check, travel",
        "Details extracted — category=travel, expense_type=accommodation_domestic, "
        "amount_huf=60000, has_receipt=True, approval_obtained=True, is_business_related=True, "
        "is_international_trip=False, provided_documents=travel_request / "
        "ticket_or_booking_confirmation / hotel_invoice",
        "Policies searched — 1 passage from Doc 02",
        "Amount calculated — reimbursable 45000 HUF, cap 45000, excess 15000",
        "Rules checked — 4 pass",
        "Answer generated",
    ]
