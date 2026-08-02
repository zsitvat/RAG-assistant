from unittest.mock import MagicMock

from app.agent.prompt_library import EMBEDDED_PROMPTS, PromptLibrary
from app.integrations.langfuse import Observability


def _disabled_library() -> PromptLibrary:
    return PromptLibrary(Observability(None))


def test_every_embedded_prompt_is_a_valid_development_fallback():
    for name in EMBEDDED_PROMPTS:
        resolved = _disabled_library().get(name)

        assert resolved.source == "embedded"
        assert "messages" in resolved.template.input_variables


def test_resolution_is_cached_after_the_first_call():
    library = _disabled_library()

    first = library.get("classify_intent")
    second = library.get("classify_intent")

    assert first is second


def test_remote_prompt_is_requested_with_the_production_label():
    client = MagicMock()
    client.get_prompt.return_value = MagicMock(
        get_langchain_prompt=lambda: [("system", "anything")], version=3
    )
    library = PromptLibrary(Observability(client))

    resolved = library.get("classify_intent")

    assert resolved.source == "langfuse"
    assert resolved.version == 3
    client.get_prompt.assert_called_once_with("classify_intent", label="production", type="chat")


def test_falls_back_to_embedded_when_langfuse_raises():
    client = MagicMock()
    client.get_prompt.side_effect = RuntimeError("network down")
    library = PromptLibrary(Observability(client))

    resolved = library.get("classify_intent")

    assert resolved.source == "embedded"
    assert resolved.template is EMBEDDED_PROMPTS["classify_intent"]


def test_disabled_observability_never_attempts_a_remote_call():
    library = _disabled_library()

    library.get("classify_intent")

    assert library._observability.client is None
