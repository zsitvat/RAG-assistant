from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agent.current_request import CurrentRequest


def test_messages_starts_at_latest_human_message():
    messages = [
        HumanMessage(content="first"),
        AIMessage(content="answer 1"),
        HumanMessage(content="second"),
        AIMessage(content="answer 2"),
    ]

    latest = CurrentRequest(messages).messages()

    assert latest == messages[2:]


def test_agent_step_count_counts_only_tool_calling_ai_messages_in_the_current_request():
    messages = [
        HumanMessage(content="q"),
        AIMessage(content="", tool_calls=[{"name": "search_policies", "args": {}, "id": "1"}]),
        ToolMessage(content="ok", tool_call_id="1", name="search_policies"),
        AIMessage(content="final answer"),
    ]

    assert CurrentRequest(messages).agent_step_count() == 1


def test_tool_error_count_counts_error_status_tool_messages():
    messages = [
        HumanMessage(content="q"),
        ToolMessage(content="bad args", tool_call_id="1", name="calculate", status="error"),
        ToolMessage(content="ok", tool_call_id="2", name="calculate", status="success"),
    ]

    assert CurrentRequest(messages).tool_error_count("calculate") == 1


def test_find_duplicate_call_reuses_the_matching_successful_tool_message():
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

    duplicate = CurrentRequest(messages).find_duplicate_call("search_policies", {"question": "x"})

    assert duplicate is not None
    assert duplicate.tool_call_id == "1"


def test_find_duplicate_call_returns_none_when_no_match():
    messages = [HumanMessage(content="q")]

    assert (
        CurrentRequest(messages).find_duplicate_call("search_policies", {"question": "x"}) is None
    )


def test_model_context_condenses_a_single_previous_request_to_human_and_final_answer():
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

    context = CurrentRequest(messages).model_context()

    assert context == [
        old_human,
        old_final_answer,
        current_human,
        current_tool_call,
        current_tool_message,
    ]


def test_model_context_condenses_multiple_previous_requests():
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

    context = CurrentRequest(messages).model_context()

    assert context == [
        first_human,
        first_final_answer,
        second_human,
        second_final_answer,
        current_human,
    ]


def test_model_context_keeps_every_message_of_the_current_request():
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

    context = CurrentRequest(messages).model_context()

    assert context[-5:] == [
        current_human,
        current_tool_call,
        current_tool_message,
        current_second_call,
        current_second_tool_message,
    ]


def test_model_context_returns_the_full_history_when_there_is_no_human_message():
    messages = [
        AIMessage(content="", tool_calls=[{"name": "calculate", "args": {}, "id": "1"}]),
        ToolMessage(content="42", tool_call_id="1", name="calculate"),
    ]

    context = CurrentRequest(messages).model_context()

    assert context == messages
