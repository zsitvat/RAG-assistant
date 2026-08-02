from langchain_core.tools import BaseTool, tool
from langgraph.graph.state import CompiledStateGraph

from app.rag.model import RagResult
from app.rules.model import Category

NO_EVIDENCE_CONTENT = "No relevant policy information found."


def build_search_policies_tool(rag_graph: CompiledStateGraph) -> BaseTool:
    """Builds the tool that runs the RAG subgraph to search the corpus."""

    @tool(response_format="content_and_artifact")
    async def search_policies(
        question: str, category: Category | None = None
    ) -> tuple[str, RagResult]:
        """Searches the corpus and returns grounded, cited evidence."""
        state = await rag_graph.ainvoke({"question": question, "category": category})
        result: RagResult = state["result"]
        return result.context or NO_EVIDENCE_CONTENT, result

    return search_policies
