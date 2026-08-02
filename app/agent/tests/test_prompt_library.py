from unittest.mock import MagicMock

import pytest
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.agent.prompt_library import (
    EMBEDDED_PROMPTS,
    PROMPT_SPECS,
    PromptLibrary,
    PromptSpec,
    PromptValidationError,
)
from app.integrations.langfuse import Observability


def _disabled_library() -> PromptLibrary:
    return PromptLibrary(Observability(None))


@pytest.mark.parametrize("name", list(PROMPT_SPECS))
def test_every_embedded_prompt_is_a_valid_development_fallback(name):
    resolved = _disabled_library().get(name)

    assert resolved.source == "embedded"
    assert "messages" in resolved.template.input_variables


def test_resolution_is_cached_after_the_first_call():
    library = _disabled_library()

    first = library.get("classify_intent")
    second = library.get("classify_intent")

    assert first is second


def test_validate_rejects_a_prompt_missing_the_messages_placeholder():
    spec = PromptSpec(name="x", requires_no_fabrication=False)
    template = ChatPromptTemplate.from_messages([("system", "no placeholder here")])

    with pytest.raises(PromptValidationError, match="messages"):
        PromptLibrary.validate(spec, template)


def test_validate_rejects_a_prompt_missing_the_no_fabrication_guardrail():
    spec = PromptSpec(name="x")
    template = ChatPromptTemplate.from_messages(
        [("system", "just answer nicely"), MessagesPlaceholder("messages")]
    )

    with pytest.raises(PromptValidationError, match="invent"):
        PromptLibrary.validate(spec, template)


def test_validate_rejects_a_generate_response_prompt_missing_citation_markers():
    spec = PromptSpec(name="x", requires_citation_markers=True)
    template = ChatPromptTemplate.from_messages(
        [("system", "never invent a fact"), MessagesPlaceholder("messages")]
    )

    with pytest.raises(PromptValidationError, match="citation"):
        PromptLibrary.validate(spec, template)


def test_validate_rejects_an_extraction_prompt_missing_the_schema_reference():
    spec = PromptSpec(name="x", requires_schema=True)
    template = ChatPromptTemplate.from_messages(
        [("system", "never guess a value"), MessagesPlaceholder("messages")]
    )

    with pytest.raises(PromptValidationError, match="schema"):
        PromptLibrary.validate(spec, template)


def test_validate_accepts_a_prompt_satisfying_every_requirement():
    spec = PromptSpec(name="x", requires_citation_markers=True, requires_schema=True)
    template = ChatPromptTemplate.from_messages(
        [
            ("system", "never fabricate a fact; cite [S1]; follow the schema"),
            MessagesPlaceholder("messages"),
        ]
    )

    PromptLibrary.validate(spec, template)  # must not raise


def test_remote_prompt_is_used_when_valid():
    client = MagicMock()
    client.get_prompt.return_value = MagicMock(
        get_langchain_prompt=lambda: [("system", "never invent a fact")], version=3
    )
    library = PromptLibrary(Observability(client))

    resolved = library.get("classify_intent")

    assert resolved.source == "langfuse"
    assert resolved.version == 3
    client.get_prompt.assert_called_once()


def test_falls_back_to_embedded_when_the_remote_prompt_is_invalid():
    client = MagicMock()
    client.get_prompt.return_value = MagicMock(
        get_langchain_prompt=lambda: [("system", "missing the guardrail phrase")], version=1
    )
    library = PromptLibrary(Observability(client))

    resolved = library.get("classify_intent")

    assert resolved.source == "embedded"
    assert resolved.template is EMBEDDED_PROMPTS["classify_intent"]


def test_falls_back_to_embedded_when_langfuse_raises():
    client = MagicMock()
    client.get_prompt.side_effect = RuntimeError("network down")
    library = PromptLibrary(Observability(client))

    resolved = library.get("classify_intent")

    assert resolved.source == "embedded"


def test_disabled_observability_never_attempts_a_remote_call():
    library = _disabled_library()

    library.get("classify_intent")

    assert library._observability.client is None
