import logging
from typing import NamedTuple

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class StructuredResult[SchemaT](NamedTuple):
    """Pairs a structured-output value with whether it is the caller-supplied fallback."""

    value: SchemaT
    degraded: bool


class StructuredOutputRunner[SchemaT: BaseModel]:
    """Any failure to parse or validate the model's output degrades to a caller-supplied default."""

    def __init__(
        self, chat_model: BaseChatModel, prompt: ChatPromptTemplate, schema: type[SchemaT]
    ) -> None:
        """Stores the chat model, prompt and schema used for structured output."""
        self._chat_model = chat_model
        self._prompt = prompt
        self._schema = schema

    def run(self, messages: list[BaseMessage], fallback: SchemaT) -> StructuredResult[SchemaT]:
        """Runs the prompt against the chat model, retrying once before falling back."""
        runnable = self._build_runnable()
        if runnable is None:
            return StructuredResult(fallback, degraded=True)

        try:
            return StructuredResult(runnable.invoke({"messages": messages}), degraded=False)
        except Exception as exc:
            logger.warning(
                f"structured output for {self._schema.__name__} failed; retrying once: "
                f"{type(exc).__name__}: {exc}"
            )

        repair_messages = [
            *messages,
            HumanMessage(content="Your previous output was invalid or unparsable. Try again."),
        ]
        try:
            return StructuredResult(runnable.invoke({"messages": repair_messages}), degraded=False)
        except Exception as exc:
            logger.warning(
                f"structured output for {self._schema.__name__} degraded to fallback after "
                f"repair retry: {type(exc).__name__}: {exc}"
            )
            return StructuredResult(fallback, degraded=True)

    def _build_runnable(self):
        """Builds the structured-output runnable when supported by the model."""
        try:
            return self._prompt | self._chat_model.with_structured_output(self._schema)
        except Exception as exc:
            logger.warning(
                f"structured output unsupported for {self._schema.__name__} on this chat model; "
                f"using fallback: {type(exc).__name__}: {exc}"
            )
            return None
