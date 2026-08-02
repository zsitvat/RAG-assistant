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

MESSAGES_VARIABLE = "messages"
NO_FABRICATION_PHRASES = ("never invent", "never fabricate", "never guess")
CITATION_MARKER_PHRASE = "[s1]"
SCHEMA_PHRASE = "schema"


class PromptSpec(BaseModel):
    """Declares what a prompt must contain for either source to be accepted."""

    name: str
    requires_no_fabrication: bool = True
    requires_citation_markers: bool = False
    requires_schema: bool = False


PROMPT_SPECS = {
    "classify_intent": PromptSpec(name="classify_intent"),
    "extract_information": PromptSpec(name="extract_information", requires_schema=True),
    "agent_step": PromptSpec(name="agent_step"),
    "generate_response": PromptSpec(name="generate_response", requires_citation_markers=True),
}
EMBEDDED_PROMPTS = {
    "classify_intent": CLASSIFY_INTENT_PROMPT,
    "extract_information": EXTRACT_INFORMATION_PROMPT,
    "agent_step": AGENT_STEP_PROMPT,
    "generate_response": GENERATE_RESPONSE_PROMPT,
}


class PromptValidationError(ValueError):
    """Raised when a prompt does not satisfy its declared requirements."""


class ResolvedPrompt(BaseModel):
    """A validated chat prompt together with the identity of the version that produced it."""

    model_config = {"arbitrary_types_allowed": True}

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
        """Returns the validated prompt for a name, resolving it at most once."""
        if name not in self._cache:
            self._cache[name] = self._resolve(name)
        return self._cache[name]

    def _resolve(self, name: str) -> ResolvedPrompt:
        """Resolves a remote prompt or returns its validated embedded fallback."""
        spec = PROMPT_SPECS[name]
        remote = self._resolve_remote(spec)
        if remote is not None:
            return remote
        template = EMBEDDED_PROMPTS[name]
        self.validate(spec, template)
        return ResolvedPrompt(name=name, template=template, source="embedded")

    def _resolve_remote(self, spec: PromptSpec) -> ResolvedPrompt | None:
        """Returns a validated production prompt from Langfuse when available."""
        client = self._observability.client
        if client is None:
            return None
        try:
            prompt = client.get_prompt(spec.name, label=PRODUCTION_LABEL, type="chat")
            template = self._to_template(prompt.get_langchain_prompt())
            self.validate(spec, template)
        except Exception:
            logger.warning(f"langfuse prompt {spec.name} unavailable or invalid; using embedded")
            return None
        return ResolvedPrompt(
            name=spec.name, template=template, source="langfuse", version=prompt.version
        )

    @staticmethod
    def _to_template(messages: list) -> ChatPromptTemplate:
        """Builds a chat template from Langfuse messages, appending the conversation placeholder."""
        return ChatPromptTemplate.from_messages([*messages, MessagesPlaceholder(MESSAGES_VARIABLE)])

    @staticmethod
    def validate(spec: PromptSpec, template: ChatPromptTemplate) -> None:
        """Rejects a prompt missing the conversation variable or a required guardrail phrase."""
        if MESSAGES_VARIABLE not in template.input_variables:
            raise PromptValidationError(
                f"prompt {spec.name!r} must accept a {MESSAGES_VARIABLE!r} placeholder"
            )
        text = " ".join(
            message.prompt.template
            for message in template.messages
            if hasattr(message, "prompt") and hasattr(message.prompt, "template")
        ).lower()
        if spec.requires_no_fabrication and not any(p in text for p in NO_FABRICATION_PHRASES):
            raise PromptValidationError(
                f"prompt {spec.name!r} must forbid inventing policy numbers or citations"
            )
        if spec.requires_citation_markers and CITATION_MARKER_PHRASE not in text:
            raise PromptValidationError(
                f"prompt {spec.name!r} must describe the citation marker format"
            )
        if spec.requires_schema and SCHEMA_PHRASE not in text:
            raise PromptValidationError(
                f"prompt {spec.name!r} must reference the structured-output schema"
            )
