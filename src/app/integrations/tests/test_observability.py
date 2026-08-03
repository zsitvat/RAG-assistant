from unittest.mock import MagicMock, patch

from app.integrations.langfuse import Observability
from app.settings import Settings


def _settings(**overrides) -> Settings:
    # Explicit None so this test-only Settings instance never picks up real
    # Langfuse credentials from a developer's local .env file.
    defaults = {"langfuse_public_key": None, "langfuse_secret_key": None}
    return Settings(llm_backend="dummy", **{**defaults, **overrides})


def test_disabled_by_configuration():
    # Act
    observability = Observability.build(_settings(langfuse_enabled=False))

    # Assert
    assert observability.enabled is False
    assert observability.client is None


def test_disabled_when_credentials_are_missing():
    # Act
    observability = Observability.build(_settings(langfuse_enabled=True))

    # Assert
    assert observability.enabled is False


def test_disabled_when_client_construction_fails():
    # Arrange
    settings = _settings(langfuse_enabled=True, langfuse_public_key="pk", langfuse_secret_key="sk")

    # Act
    with patch("app.integrations.langfuse.Langfuse", side_effect=RuntimeError("boom")):
        observability = Observability.build(settings)

    # Assert
    assert observability.enabled is False


def test_enabled_when_credentials_and_client_are_valid():
    # Arrange
    settings = _settings(langfuse_enabled=True, langfuse_public_key="pk", langfuse_secret_key="sk")

    # Act
    with patch("app.integrations.langfuse.Langfuse", return_value=MagicMock()):
        observability = Observability.build(settings)

    # Assert
    assert observability.enabled is True
    assert observability.client is not None


def test_traced_turn_yields_an_empty_config_and_a_no_op_updater_when_disabled():
    # Arrange
    observability = Observability(None)

    # Act
    with observability.traced_turn("thread-1", "hi") as (config, update_trace):
        update_trace(decision="eligible")  # must not raise

    # Assert
    assert config == {}


def test_traced_turn_attaches_a_callback_and_session_id_when_enabled():
    # Arrange
    observability = Observability(MagicMock())

    # Act
    with observability.traced_turn("thread-1", "hi") as (config, _update_trace):
        pass

    # Assert
    assert "callbacks" in config
    assert config["metadata"]["langfuse_session_id"] == "thread-1"


def test_traced_turn_sets_the_message_as_the_span_input():
    # Arrange
    client = MagicMock()
    observability = Observability(client)

    # Act
    with observability.traced_turn("thread-1", "hi") as (_config, _update_trace):
        pass

    # Assert
    client.start_as_current_observation.assert_called_once_with(name="chat_turn", input="hi")


def test_traced_turn_updates_the_turn_span_with_the_answer_and_outcome_attributes():
    # Arrange
    client = MagicMock()
    span = client.start_as_current_observation.return_value.__enter__.return_value
    observability = Observability(client)

    # Act
    with observability.traced_turn("thread-1", "hi") as (_config, update_trace):
        update_trace(decision="eligible", output="the answer")

    # Assert
    span.update.assert_called_once_with(metadata={"decision": "eligible"}, output="the answer")


def test_traced_turn_swallows_a_span_update_failure():
    # Arrange
    client = MagicMock()
    span = client.start_as_current_observation.return_value.__enter__.return_value
    span.update.side_effect = RuntimeError("boom")
    observability = Observability(client)

    # Act
    with observability.traced_turn("thread-1", "hi") as (_config, update_trace):
        update_trace(decision="eligible")  # must not raise

    # Assert
    span.update.assert_called_once()
