from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agent.message_history import MessageHistory


def test_messages_starts_at_latest_human_message():
    # Arrange
    messages = [
        HumanMessage(content="first"),
        AIMessage(content="answer 1"),
        HumanMessage(content="second"),
        AIMessage(content="answer 2"),
    ]

    # Act
    latest = MessageHistory(messages).messages()

    # Assert
    assert latest == messages[2:]


def test_agent_step_count_counts_only_tool_calling_ai_messages_in_the_current_request():
    # Arrange
    messages = [
        HumanMessage(content="q"),
        AIMessage(content="", tool_calls=[{"name": "search_policies", "args": {}, "id": "1"}]),
        ToolMessage(content="ok", tool_call_id="1", name="search_policies"),
        AIMessage(content="final answer"),
    ]

    assert MessageHistory(messages).agent_step_count() == 1


def test_tool_error_count_counts_error_status_tool_messages():
    # Arrange
    messages = [
        HumanMessage(content="q"),
        ToolMessage(content="bad args", tool_call_id="1", name="calculate", status="error"),
        ToolMessage(content="ok", tool_call_id="2", name="calculate", status="success"),
    ]

    assert MessageHistory(messages).tool_error_count("calculate") == 1


def test_find_duplicate_call_reuses_the_matching_successful_tool_message():
    # Arrange
    messages = [
        HumanMessage(content="q"),
        AIMessage(
            content="",
            tool_calls=[{"name": "search_policies", "args": {"question": "x"}, "id": "1"}],
        ),
        ToolMessage(content="found", tool_call_id="1", name="search_policies"),
        AIMessage(
            content="",
            tool_calls=[{"name": "search_policies", "args": {"question": "x"}, "id": "2"}],
        ),
    ]

    # Act
    duplicate = MessageHistory(messages).find_duplicate_call("search_policies", {"question": "x"})

    # Assert
    assert duplicate is not None
    assert duplicate.tool_call_id == "1"


def test_find_duplicate_call_returns_none_when_no_match():
    # Arrange
    messages = [HumanMessage(content="q")]

    assert (
        MessageHistory(messages).find_duplicate_call("search_policies", {"question": "x"}) is None
    )


def test_model_context_condenses_a_single_previous_request_to_human_and_final_answer():
    # Arrange
    old_human = HumanMessage(content="old question")
    old_tool_call = AIMessage(
        content="", tool_calls=[{"name": "search_policies", "args": {}, "id": "1"}]
    )
    old_tool_message = ToolMessage(content="found", tool_call_id="1", name="search_policies")
    old_final_answer = AIMessage(content="old final answer")
    current_human = HumanMessage(content="new question")
    current_tool_call = AIMessage(
        content="", tool_calls=[{"name": "search_policies", "args": {}, "id": "2"}]
    )
    current_tool_message = ToolMessage(
        content="found again", tool_call_id="2", name="search_policies"
    )
    messages = [
        old_human,
        old_tool_call,
        old_tool_message,
        old_final_answer,
        current_human,
        current_tool_call,
        current_tool_message,
    ]

    # Act
    context = MessageHistory(messages).model_context()

    # Assert
    assert context == [
        old_human,
        old_final_answer,
        current_human,
        current_tool_call,
        current_tool_message,
    ]


def test_model_context_condenses_multiple_previous_requests():
    # Arrange
    first_human = HumanMessage(content="first question")
    first_tool_call = AIMessage(
        content="", tool_calls=[{"name": "search_policies", "args": {}, "id": "1"}]
    )
    first_tool_message = ToolMessage(content="found", tool_call_id="1", name="search_policies")
    first_final_answer = AIMessage(content="first final answer")
    second_human = HumanMessage(content="second question")
    second_final_answer = AIMessage(content="second final answer")
    current_human = HumanMessage(content="third question")
    messages = [
        first_human,
        first_tool_call,
        first_tool_message,
        first_final_answer,
        second_human,
        second_final_answer,
        current_human,
    ]

    # Act
    context = MessageHistory(messages).model_context()

    # Assert
    assert context == [
        first_human,
        first_final_answer,
        second_human,
        second_final_answer,
        current_human,
    ]


def test_model_context_keeps_every_message_of_the_current_request():
    # Arrange
    old_human = HumanMessage(content="old question")
    old_final_answer = AIMessage(content="old final answer")
    current_human = HumanMessage(content="new question")
    current_tool_call = AIMessage(
        content="", tool_calls=[{"name": "calculate", "args": {}, "id": "1"}]
    )
    current_tool_message = ToolMessage(content="42", tool_call_id="1", name="calculate")
    current_second_call = AIMessage(
        content="", tool_calls=[{"name": "check_rules", "args": {}, "id": "2"}]
    )
    current_second_tool_message = ToolMessage(content="ok", tool_call_id="2", name="check_rules")
    messages = [
        old_human,
        old_final_answer,
        current_human,
        current_tool_call,
        current_tool_message,
        current_second_call,
        current_second_tool_message,
    ]

    # Act
    context = MessageHistory(messages).model_context()

    # Assert
    assert context[-5:] == [
        current_human,
        current_tool_call,
        current_tool_message,
        current_second_call,
        current_second_tool_message,
    ]


def test_model_context_returns_the_full_history_when_there_is_no_human_message():
    # Arrange
    messages = [
        AIMessage(content="", tool_calls=[{"name": "calculate", "args": {}, "id": "1"}]),
        ToolMessage(content="42", tool_call_id="1", name="calculate"),
    ]

    # Act
    context = MessageHistory(messages).model_context()

    # Assert
    assert context == messages
