from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.rag.index_schema import CONTEXT_TOKEN_BUDGET
from app.rag.model import Citation, RagResult, RetrievedResult
from app.rag.retriever import Retriever
from app.rag.state import RagState

CHARS_PER_TOKEN = 4


class RagNodes:
    """The subgraph's two nodes, bound to an injected retriever."""

    def __init__(self, retriever: Retriever) -> None:
        """Stores the policy retriever used to search documents."""
        self._retriever = retriever

    def retrieve_documents(self, state: RagState) -> RagState:
        """Searches for policy documents matching the question, retrying without the category."""
        category = state.get("category")
        docs = self._retriever.search(state["question"], category)
        if not docs and category is not None:
            docs = self._retriever.search(state["question"], None)
            category = None
        results = sorted(
            (self._to_result(doc) for doc in docs),
            key=lambda result: result.similarity,
            reverse=True,
        )
        return {"result": RagResult(results=results, category=category)}

    def build_context(self, state: RagState) -> RagState:
        """Assembles the retrieved results into a citation-annotated context string."""
        result = state["result"]
        context, citations = self._build_context(result.results)
        return {"result": result.model_copy(update={"context": context, "citations": citations})}

    @staticmethod
    def _to_result(doc: Document) -> RetrievedResult:
        """Converts a retrieved document into the typed RAG result item."""
        metadata = doc.metadata
        return RetrievedResult(
            doc_id=metadata["doc_id"],
            doc_title=metadata["doc_title"],
            section_id=metadata.get("section_id"),
            section=metadata.get("section"),
            categories=list(metadata.get("categories", [])),
            rule_ids=list(metadata.get("rule_ids", [])),
            source_path=metadata["source_path"],
            content=doc.page_content,
            similarity=metadata["similarity"],
        )

    @staticmethod
    def _build_context(results: list[RetrievedResult]) -> tuple[str, list[Citation]]:
        """Builds budgeted citation context from ranked retrieval results."""
        blocks: list[str] = []
        citations: list[Citation] = []
        seen_sections: set[tuple[str, str | None]] = set()
        budget_chars = CONTEXT_TOKEN_BUDGET * CHARS_PER_TOKEN
        used_chars = 0

        for result in results:
            section_key = (result.doc_id, result.section)
            if section_key in seen_sections:
                continue

            marker = f"S{len(citations) + 1}"
            label = f"{result.doc_title} › {result.section}" if result.section else result.doc_title
            block = f"[{marker}] {label}\n{result.content}"
            if blocks and used_chars + len(block) > budget_chars:
                break

            seen_sections.add(section_key)
            blocks.append(block)
            citations.append(
                Citation(
                    marker=marker,
                    doc_id=result.doc_id,
                    doc_title=result.doc_title,
                    section=result.section,
                )
            )
            used_chars += len(block)

        return "\n\n".join(blocks), citations


def build_rag_graph(retriever: Retriever) -> CompiledStateGraph:
    """Compiles the standalone retrieve_documents -> build_context RAG subgraph."""

    nodes = RagNodes(retriever)
    graph = StateGraph(RagState)

    graph.add_node("retrieve_documents", nodes.retrieve_documents)
    graph.add_node("build_context", nodes.build_context)

    graph.add_edge(START, "retrieve_documents")
    graph.add_edge("retrieve_documents", "build_context")
    graph.add_edge("build_context", END)

    return graph.compile()
