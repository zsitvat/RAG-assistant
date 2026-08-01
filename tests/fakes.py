from collections.abc import Iterator
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from pydantic import ConfigDict, Field


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
