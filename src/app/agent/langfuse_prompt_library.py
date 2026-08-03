import logging

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel

from app.agent.prompts import (
    AGENT_STEP_PROMPT,
    CLASSIFY_INTENT_PROMPT,
    EXTRACT_INFORMATION_PROMPT,
    GENERATE_RESPONSE_PROMPT,
)
from app.integrations.langfuse import PRODUCTION_LABEL, Observability

logger = logging.getLogger(__name__)

EMBEDDED_PROMPTS = {
    "classify_intent": CLASSIFY_INTENT_PROMPT,
    "extract_information": EXTRACT_INFORMATION_PROMPT,
    "agent_step": AGENT_STEP_PROMPT,
    "generate_response": GENERATE_RESPONSE_PROMPT,
}


class ResolvedPrompt(BaseModel):
    """A chat prompt template together with the identity of the version that produced it."""

    name: str
    template: ChatPromptTemplate
    source: str
    version: int | None = None


class PromptLibrary:
    """Resolves each prompt from the production-labelled Langfuse version or the embedded one."""

    def __init__(self, observability: Observability) -> None:
        """Stores the observability adapter used to reach Langfuse, and caches resolutions."""
        self._observability = observability
        self._cache: dict[str, ResolvedPrompt] = {}

    def get(self, name: str) -> ResolvedPrompt:
        """Returns the prompt for a name, resolving it at most once."""
        if name not in self._cache:
            self._cache[name] = self._resolve(name)
        return self._cache[name]

    def _resolve(self, name: str) -> ResolvedPrompt:
        """Resolves the named prompt from Langfuse, or its embedded fallback."""
        remote = self._resolve_remote(name)
        if remote is not None:
            return remote
        return ResolvedPrompt(name=name, template=EMBEDDED_PROMPTS[name], source="embedded")

    def _resolve_remote(self, name: str) -> ResolvedPrompt | None:
        """Returns the production-labelled prompt from Langfuse when available."""
        client = self._observability.client
        if client is None:
            return None
        try:
            prompt = client.get_prompt(name, label=PRODUCTION_LABEL, type="chat")
            template = self._to_template(prompt.get_langchain_prompt())
        except Exception:
            logger.warning(f"langfuse prompt {name} unavailable; using embedded")
            return None
        return ResolvedPrompt(
            name=name, template=template, source="langfuse", version=prompt.version
        )

    @staticmethod
    def _to_template(messages: list) -> ChatPromptTemplate:
        """Builds a chat template from Langfuse messages, appending the conversation placeholder."""
        return ChatPromptTemplate.from_messages([*messages, MessagesPlaceholder("messages")])
