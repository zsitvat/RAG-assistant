from collections.abc import Iterator
from datetime import date
from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from pydantic import ConfigDict, Field

from app.agent.calculator import ReimbursementCalculator
from app.agent.deadline_check import DeadlineChecker
from app.agent.rule_checker import RuleChecker
from app.agent.tools import build_tools
from app.rag.graph import build_rag_graph
from app.rules.model import RuleCatalogue


class FixedRetriever:
    """Returns a fixed document list for every policy search."""

    def __init__(self, documents: list[Document]) -> None:
        self._documents = documents

    async def asearch(self, query: str, category: str | None) -> list[Document]:
        """Returns the configured documents regardless of query or category."""

        return self._documents


def policy_document(
    doc_id: str,
    section: str,
    content: str,
    rule_ids: list[str],
    categories: list[str],
) -> Document:
    """Builds a retrieved policy document for journey tests."""

    return Document(
        page_content=content,
        metadata={
            "doc_id": doc_id,
            "doc_title": f"Doc {doc_id}",
            "section": section,
            "section_id": None,
            "categories": categories,
            "rule_ids": rule_ids,
            "source_path": f"{doc_id}.docx",
            "similarity": 0.9,
        },
    )


def build_agent_tools(document: Document, catalogue: RuleCatalogue):
    """Builds deterministic agent tools around a fixed policy document."""

    rag_graph = build_rag_graph(FixedRetriever([document]))
    calculator = ReimbursementCalculator(catalogue)
    rule_checker = RuleChecker(catalogue, DeadlineChecker(catalogue.submission.deadline_days))
    return build_tools(rag_graph, calculator, rule_checker, lambda: date(2026, 8, 1))


def tool_message(result: dict, name: str):
    """Returns the named tool message from an invoked graph result."""

    return next(message for message in result["messages"] if getattr(message, "name", None) == name)


def tool_calls(result: dict) -> list[str]:
    """Returns tool names in call order from an invoked graph result."""

    return [
        call["name"]
        for message in result["messages"]
        if hasattr(message, "tool_calls")
        for call in message.tool_calls
    ]


class _StructuredOutputStub(Runnable):
    """Records each invocation's input before returning the next scripted structured response."""

    def __init__(self, responses: Iterator[Any], captured_inputs: list[Any]) -> None:
        """Stores the shared response iterator and the list to record inputs into."""
        self._responses = responses
        self._captured_inputs = captured_inputs

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        """Records the input and returns the next scripted response."""
        self._captured_inputs.append(input)
        return next(self._responses)


class ScriptedChatModel(BaseChatModel):
    """chat_responses feeds plain chat/tool-calling turns; structured_responses feeds
    with_structured_output() calls, independently and in order."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    chat_responses: Iterator[BaseMessage]
    structured_responses: Iterator[Any] = iter(())
    captured_chat_messages: list[list[BaseMessage]] = Field(default_factory=list)
    captured_structured_inputs: list[Any] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.captured_chat_messages.append(messages)
        message = next(self.chat_responses)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedChatModel":
        return self

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Runnable:
        return _StructuredOutputStub(self.structured_responses, self.captured_structured_inputs)
