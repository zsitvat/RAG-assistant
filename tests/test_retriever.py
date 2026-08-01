from unittest.mock import MagicMock

from langchain_core.documents import Document
from langchain_redis import RedisVectorStore

from app.rag.retriever import Retriever


def _vector_store_returning(*doc_and_distance):
    vector_store = MagicMock(spec=RedisVectorStore)
    vector_store.similarity_search_with_score.return_value = list(doc_and_distance)
    return vector_store


def test_filter_expression_includes_category_and_general():
    assert Retriever._filter_expression("meal") == "@categories:{meal|general}"


def test_filter_expression_is_none_without_a_category():
    assert Retriever._filter_expression(None) is None


def test_search_converts_distance_to_similarity():
    doc = Document(page_content="x", metadata={})
    vector_store = _vector_store_returning((doc, 0.1))
    retriever = Retriever(vector_store)

    results = retriever.search("question", None)

    assert results[0].metadata["similarity"] == 0.9


def test_search_passes_k_and_filter_to_the_vector_store():
    vector_store = _vector_store_returning()
    retriever = Retriever(vector_store, k=3)

    retriever.search("question", "mileage")

    vector_store.similarity_search_with_score.assert_called_once_with(
        "question", k=3, filter="@categories:{mileage|general}"
    )
