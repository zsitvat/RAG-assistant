from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agent.calculator import ReimbursementCalculator
from app.agent.model import ExpenseClaim, IntentClassification
from app.agent.nodes import AgentNodes
from app.rules.loader import load_rule_catalogue
from app.tests.fakes import ScriptedChatModel

CALCULATOR = ReimbursementCalculator(load_rule_catalogue())


def _old_request():
    old_human = HumanMessage(content="old question")
    old_tool_call = AIMessage(
        content="", tool_calls=[{"name": "search_policies", "args": {}, "id": "1"}]
    )
    old_tool_message = ToolMessage(content="found", tool_call_id="1", name="search_policies")
    old_final_answer = AIMessage(content="old final answer")
    return old_human, old_tool_call, old_tool_message, old_final_answer


async def test_classify_intent_sends_the_model_only_the_filtered_context():
    # Arrange
    old_human, old_tool_call, old_tool_message, old_final_answer = _old_request()
    current_human = HumanMessage(content="new question")
    model = ScriptedChatModel(
        chat_responses=iter([]),
        structured_responses=iter([IntentClassification(intent="policy_question")]),
    )
    nodes = AgentNodes(model, model, [], CALCULATOR)

    # Act
    await nodes.classify_intent(
        {"messages": [old_human, old_tool_call, old_tool_message, old_final_answer, current_human]}
    )

    # Assert
    sent = model.captured_structured_inputs[-1].to_messages()
    assert sent[1:] == [old_human, old_final_answer, current_human]


async def test_extract_information_sends_the_model_only_the_filtered_context():
    # Arrange
    old_human, old_tool_call, old_tool_message, old_final_answer = _old_request()
    current_human = HumanMessage(content="new question")
    model = ScriptedChatModel(chat_responses=iter([]), structured_responses=iter([ExpenseClaim()]))
    nodes = AgentNodes(model, model, [], CALCULATOR)

    # Act
    await nodes.extract_information(
        {"messages": [old_human, old_tool_call, old_tool_message, old_final_answer, current_human]}
    )

    # Assert
    sent = model.captured_structured_inputs[-1].to_messages()
    assert sent[1:] == [old_human, old_final_answer, current_human]


async def test_agent_step_sends_the_model_only_the_filtered_context():
    # Arrange
    old_human, old_tool_call, old_tool_message, old_final_answer = _old_request()
    current_human = HumanMessage(content="new question")
    model = ScriptedChatModel(chat_responses=iter([AIMessage(content="")]))
    nodes = AgentNodes(model, model, [], CALCULATOR)

    # Act
    await nodes.agent_step(
        {"messages": [old_human, old_tool_call, old_tool_message, old_final_answer, current_human]}
    )

    # Assert
    sent = model.captured_chat_messages[-1]
    assert old_tool_call not in sent
    assert old_tool_message not in sent
    assert old_final_answer in sent
    assert current_human in sent


async def test_generate_response_sends_the_model_only_the_filtered_context():
    # Arrange
    old_human, old_tool_call, old_tool_message, old_final_answer = _old_request()
    current_human = HumanMessage(content="new question")
    current_tool_call = AIMessage(
        content="", tool_calls=[{"name": "search_policies", "args": {}, "id": "2"}]
    )
    current_tool_message = ToolMessage(
        content="found again", tool_call_id="2", name="search_policies"
    )
    current_draft = AIMessage(content="A draft answer the agent step wrote itself.")
    model = ScriptedChatModel(chat_responses=iter([AIMessage(content="Final answer.")]))
    nodes = AgentNodes(model, model, [], CALCULATOR)

    # Act
    await nodes.generate_response(
        {
            "messages": [
                old_human,
                old_tool_call,
                old_tool_message,
                old_final_answer,
                current_human,
                current_tool_call,
                current_tool_message,
                current_draft,
            ]
        }
    )

    # Assert
    sent = model.captured_chat_messages[-1]
    assert old_tool_call not in sent
    assert old_tool_message not in sent
    assert old_final_answer in sent
    assert current_human in sent
    assert current_tool_call in sent
    assert current_tool_message in sent
    assert current_draft not in sent
