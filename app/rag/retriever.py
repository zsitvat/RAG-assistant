from langchain_core.documents import Document
from langchain_redis import RedisVectorStore

from app.rag.index_schema import TOP_K
from app.rules.model import Category


class Retriever:
    """Wraps the policy vector store; the category filter is chosen per search call."""

    def __init__(self, vector_store: RedisVectorStore, k: int = TOP_K) -> None:
        """Stores the vector store and the number of results to retrieve per search."""
        self._vector_store = vector_store
        self._k = k

    def search(self, query: str, category: Category | None) -> list[Document]:
        """Searches the vector store for the query, optionally filtered by category."""
        results = self._vector_store.similarity_search_with_score(
            query, k=self._k, filter=self._filter_expression(category)
        )
        return [self._with_similarity(doc, distance) for doc, distance in results]

    @staticmethod
    def _filter_expression(category: Category | None) -> str | None:
        """Builds the Redis category filter for a retrieval query."""
        if category is None:
            return None
        return f"@categories:{{{category}|general}}"

    @staticmethod
    def _with_similarity(doc: Document, distance: float) -> Document:
        """Attaches a normalized similarity score to a retrieved document."""
        doc.metadata["similarity"] = 1.0 - distance
        return doc
