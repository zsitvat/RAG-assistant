from typing import Any

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_redis import RedisVectorStore

from app.rag.index_schema import TOP_K
from app.rules.model import Category


class Retriever(BaseRetriever):
    """LangChain retriever over the policy vector store; category filter chosen per search call."""

    vector_store: RedisVectorStore
    k: int = TOP_K
    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, vector_store: RedisVectorStore, k: int = TOP_K, **kwargs: Any) -> None:
        """Stores the vector store and the number of results to retrieve per search."""
        super().__init__(vector_store=vector_store, k=k, **kwargs)

    async def asearch(self, query: str, category: Category | None) -> list[Document]:
        """Searches the vector store for the query, optionally filtered by category."""
        return await self.ainvoke(query, category=category)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        category: Category | None = None,
    ) -> list[Document]:
        """Runs the dense similarity search through the retriever interface and attaches scores."""
        results = self.vector_store.similarity_search_with_score(
            query, k=self.k, filter=self._filter_expression(category)
        )
        return [self._with_similarity(doc, distance) for doc, distance in results]

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
        category: Category | None = None,
    ) -> list[Document]:
        """Runs the dense similarity search natively async, attaching scores to each hit."""
        results = await self.vector_store.asimilarity_search_with_score(
            query, k=self.k, filter=self._filter_expression(category)
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
