from app.rag.graph import build_rag_graph
from app.rag.tool import NO_EVIDENCE_CONTENT, build_search_policies_tool


class _FakeRetriever:
    def __init__(self, docs: list) -> None:
        self._docs = docs

    async def asearch(self, query: str, category: str | None) -> list:
        return self._docs


async def _call(tool, question: str, category: str | None = None):
    return await tool.ainvoke(
        {
            "name": tool.name,
            "args": {"question": question, "category": category},
            "id": "call-1",
            "type": "tool_call",
        }
    )


async def test_tool_returns_context_as_content_and_rag_result_as_artifact():
    # Arrange
    from langchain_core.documents import Document

    doc = Document(
        page_content="Capped at 15000.",
        metadata={
            "doc_id": "01",
            "doc_title": "Doc 01",
            "section": "4. Business meals",
            "section_id": None,
            "categories": ["meal"],
            "rule_ids": ["R-MEAL-01"],
            "source_path": "01.docx",
            "similarity": 0.9,
        },
    )
    graph = build_rag_graph(_FakeRetriever([doc]))
    tool = build_search_policies_tool(graph)

    # Act
    message = await _call(tool, "meal limit?", "meal")

    # Assert
    assert message.content == "[S1] Doc 01 › 4. Business meals\nCapped at 15000."
    assert message.artifact.results[0].rule_ids == ["R-MEAL-01"]
    assert message.artifact.citations[0].marker == "S1"


async def test_tool_reports_no_evidence_explicitly_when_nothing_is_found():
    # Arrange
    graph = build_rag_graph(_FakeRetriever([]))
    tool = build_search_policies_tool(graph)

    # Act
    message = await _call(tool, "unrelated question", None)

    # Assert
    assert message.content == NO_EVIDENCE_CONTENT
    assert message.artifact.results == []
    assert message.artifact.confidence == 0.0
