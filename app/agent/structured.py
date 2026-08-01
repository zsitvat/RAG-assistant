import logging
from typing import TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

logger = logging.getLogger(__name__)

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class StructuredOutputRunner:
    """Any failure to parse or validate the model's output degrades to a caller-supplied default."""

    def __init__(
        self, chat_model: BaseChatModel, prompt: ChatPromptTemplate, schema: type[SchemaT]
    ) -> None:
        """Stores the chat model, prompt and schema used for structured output."""
        self._chat_model = chat_model
        self._prompt = prompt
        self._schema = schema

    def run(self, messages: list[BaseMessage], fallback: SchemaT) -> SchemaT:
        """Runs the prompt against the chat model, retrying once before falling back."""
        runnable = self._build_runnable()
        if runnable is None:
            return fallback

        try:
            return runnable.invoke({"messages": messages})
        except Exception:
            logger.warning("structured output for %s failed; retrying once", self._schema.__name__)

        repair_messages = [
            *messages,
            HumanMessage(content="Your previous output was invalid or unparsable. Try again."),
        ]
        try:
            return runnable.invoke({"messages": repair_messages})
        except Exception:
            logger.warning(
                "structured output for %s degraded to fallback after repair retry",
                self._schema.__name__,
            )
            return fallback

    def _build_runnable(self):
        try:
            return self._prompt | self._chat_model.with_structured_output(self._schema)
        except Exception:
            logger.warning(
                "structured output unsupported for %s on this chat model; using fallback",
                self._schema.__name__,
            )
            return None
