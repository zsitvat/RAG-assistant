from app.api.schemas import ChatSource, StreamEvent, parse_sse_lines


def test_parse_sse_lines_yields_no_pairs_for_a_data_line_without_a_preceding_event():
    # Arrange
    lines = ['data: {"data": "orphan"}']

    # Act / Assert
    assert list(parse_sse_lines(lines)) == []


def test_parse_sse_lines_yields_one_pair_for_a_single_event():
    # Arrange
    lines = ["event: step", 'data: {"data": "Intent classified"}']

    # Act / Assert
    assert list(parse_sse_lines(lines)) == [("step", "Intent classified")]


def test_parse_sse_lines_carries_the_event_name_across_multiple_data_lines():
    # Arrange
    lines = ["event: token", 'data: {"data": "a"}', 'data: {"data": "b"}']

    # Act / Assert
    assert list(parse_sse_lines(lines)) == [("token", "a"), ("token", "b")]


def test_parse_sse_lines_switches_event_name_on_a_new_event_line():
    # Arrange
    lines = [
        "event: step",
        'data: {"data": "Intent classified"}',
        "event: token",
        'data: {"data": "hi"}',
    ]

    # Act / Assert
    assert list(parse_sse_lines(lines)) == [
        ("step", "Intent classified"),
        ("token", "hi"),
    ]


def test_parse_sse_lines_is_the_inverse_of_stream_event_to_sse():
    # Arrange
    event = StreamEvent(event="step", data="Intent classified")
    lines = event.to_sse().splitlines()

    # Act / Assert
    assert list(parse_sse_lines(lines)) == [("step", "Intent classified")]


def test_parse_sse_lines_decodes_a_structured_source_payload():
    # Arrange
    source = ChatSource(source_id="S1", doc_id="01", title="General Policy", section="Meals")
    event = StreamEvent(event="source", data=source)
    lines = event.to_sse().splitlines()

    # Act
    [(event_name, data)] = list(parse_sse_lines(lines))

    # Assert
    assert event_name == "source"
    assert data == source.model_dump()
