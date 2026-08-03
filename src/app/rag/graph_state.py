from typing import TypedDict

from app.rag.model import RagResult
from app.rules.model import Category


class RagState(TypedDict, total=False):
    """Holds the question, category filter and result threaded through the RAG subgraph."""

    question: str
    category: Category | None
    result: RagResult
