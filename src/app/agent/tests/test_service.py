from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agent.service import AgentService
from app.rag.model import Citation, RagResult


class _FakeGraph:
    def __init__(self, messages: list) -> None:
        self._messages = messages

    async def ainvoke(self, state, config=None):
        return {"messages": [*state["messages"], *self._messages]}


def _rag_result(*citations: Citation) -> RagResult:
    return RagResult(citations=list(citations))


async def test_respond_projects_answer_sources_and_steps():
    # Arrange
    tool_message = ToolMessage(
        content="[S1] Doc",
        artifact=_rag_result(
            Citation(marker="S1", doc_id="01", doc_title="Doc 01", section="4. Meals")
        ),
        tool_call_id="1",
        name="search_policies",
    )
    graph = _FakeGraph([tool_message, AIMessage(content="The limit is 15,000 HUF [S1].")])
    service = AgentService(graph)

    # Act
    response = await service.ainvoke_graph("t1", "What is the meal limit?")

    # Assert
    assert response.thread_id == "t1"
    assert response.answer == "The limit is 15,000 HUF [S1]."
    assert len(response.sources) == 1
    assert response.sources[0].source_id == "S1"
    assert response.sources[0].doc_id == "01"
    assert response.sources[0].title == "Doc 01"
    assert response.sources[0].section == "4. Meals"
    assert response.steps == [
        "Intent classified",
        "Details extracted",
        "Policies searched",
        "Answer generated",
    ]
    assert response.response_time_ms >= 0


async def test_respond_deduplicates_sources_across_multiple_search_calls():
    # Arrange
    citation = Citation(marker="S1", doc_id="01", doc_title="Doc 01", section="4. Meals")
    graph = _FakeGraph(
        [
            ToolMessage(
                content="a",
                artifact=_rag_result(citation),
                tool_call_id="1",
                name="search_policies",
            ),
            ToolMessage(
                content="b",
                artifact=_rag_result(citation),
                tool_call_id="2",
                name="search_policies",
            ),
            AIMessage(content="Final."),
        ]
    )
    service = AgentService(graph)

    # Act
    response = await service.ainvoke_graph("t1", "question")

    # Assert
    assert len(response.sources) == 1


async def test_respond_only_considers_messages_from_the_current_request():
    # Arrange
    old_human = HumanMessage(content="previous request question")
    old_tool = ToolMessage(
        content="old",
        artifact=_rag_result(Citation(marker="S1", doc_id="99", doc_title="Old", section=None)),
        tool_call_id="0",
        name="search_policies",
    )
    graph = _StatefulFakeGraph(
        prior_messages=[old_human, old_tool],
        new_messages=[AIMessage(content="Answer with no tools for this request.")],
    )
    service = AgentService(graph)

    # Act
    response = await service.ainvoke_graph("t1", "new question")

    # Assert
    assert response.sources == []
    assert response.steps == ["Intent classified", "Details extracted", "Answer generated"]


class _StatefulFakeGraph:
    def __init__(self, prior_messages: list, new_messages: list) -> None:
        self._prior_messages = prior_messages
        self._new_messages = new_messages

    async def ainvoke(self, state, config=None):
        return {"messages": [*self._prior_messages, *state["messages"], *self._new_messages]}
