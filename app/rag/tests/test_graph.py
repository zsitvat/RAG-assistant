from langchain_core.documents import Document

from app.rag.graph import build_rag_graph
from app.rag.index_schema import CONTEXT_TOKEN_BUDGET


class _RecordingRetriever:
    def __init__(self, *doc_lists: list[Document]) -> None:
        self._doc_lists = list(doc_lists)
        self.calls: list[str | None] = []

    def search(self, query: str, category: str | None) -> list[Document]:
        self.calls.append(category)
        return self._doc_lists.pop(0) if self._doc_lists else []


def _doc(doc_id: str, section: str, similarity: float, content: str = "content", **overrides):
    metadata = {
        "doc_id": doc_id,
        "doc_title": f"Doc {doc_id}",
        "section": section,
        "section_id": None,
        "categories": ["meal"],
        "rule_ids": [],
        "source_path": f"{doc_id}.docx",
        "similarity": similarity,
    }
    metadata.update(overrides)
    return Document(page_content=content, metadata=metadata)


def test_filtered_query_requests_the_active_category():
    retriever = _RecordingRetriever([_doc("01", "4. Business meals", 0.9)])
    graph = build_rag_graph(retriever)

    graph.invoke({"question": "meal limit?", "category": "meal"})

    assert retriever.calls == ["meal"]


def test_unfiltered_query_when_no_category_given():
    retriever = _RecordingRetriever([_doc("01", "4. Business meals", 0.9)])
    graph = build_rag_graph(retriever)

    graph.invoke({"question": "meal limit?", "category": None})

    assert retriever.calls == [None]


def test_empty_filtered_result_retries_once_without_category():
    retriever = _RecordingRetriever([], [_doc("07", "2. Business meals", 0.85)])
    graph = build_rag_graph(retriever)

    result = graph.invoke({"question": "meal limit?", "category": "meal"})["result"]

    assert retriever.calls == ["meal", None]
    assert result.category is None
    assert len(result.results) == 1


def test_empty_result_stays_empty_when_fallback_also_empty():
    retriever = _RecordingRetriever([], [])
    graph = build_rag_graph(retriever)

    result = graph.invoke({"question": "meal limit?", "category": "meal"})["result"]

    assert result.results == []
    assert result.context == ""
    assert result.citations == []
    assert result.confidence == 0.0


def test_results_are_ranked_by_similarity_descending():
    retriever = _RecordingRetriever(
        [_doc("01", "A", 0.7), _doc("02", "B", 0.95), _doc("03", "C", 0.8)]
    )
    graph = build_rag_graph(retriever)

    rag_result = graph.invoke({"question": "q", "category": None})["result"]

    assert [item.similarity for item in rag_result.results] == [0.95, 0.8, 0.7]
    assert rag_result.confidence == 0.95


def test_citations_are_deduplicated_by_document_and_section():
    retriever = _RecordingRetriever(
        [
            _doc("01", "4. Business meals", 0.9, content="first half"),
            _doc("01", "4. Business meals", 0.85, content="second half"),
            _doc("02", "1. Other", 0.8, content="unrelated"),
        ]
    )
    graph = build_rag_graph(retriever)

    result = graph.invoke({"question": "q", "category": None})["result"]

    assert len(result.citations) == 2
    assert [c.marker for c in result.citations] == ["S1", "S2"]
    assert "first half" in result.context
    assert "second half" not in result.context


def test_context_uses_numbered_markers_with_title_and_section():
    retriever = _RecordingRetriever(
        [_doc("01", "4. Business meals", 0.9, content="Capped at 15000.")]
    )
    graph = build_rag_graph(retriever)

    result = graph.invoke({"question": "q", "category": None})["result"]

    assert result.context == "[S1] Doc 01 › 4. Business meals\nCapped at 15000."
    assert result.citations[0].marker == "S1"
    assert result.citations[0].doc_id == "01"
    assert result.citations[0].section == "4. Business meals"


def test_context_stays_within_the_token_budget():
    long_content = "x" * (CONTEXT_TOKEN_BUDGET * 4)
    retriever = _RecordingRetriever(
        [
            _doc("01", "A", 0.95, content=long_content),
            _doc("02", "B", 0.9, content="short overflow content"),
        ]
    )
    graph = build_rag_graph(retriever)

    result = graph.invoke({"question": "q", "category": None})["result"]

    assert len(result.citations) == 1
    assert result.citations[0].doc_id == "01"
    assert "overflow" not in result.context
