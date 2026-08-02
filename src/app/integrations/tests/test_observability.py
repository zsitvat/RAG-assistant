from unittest.mock import MagicMock, patch

from app.integrations.langfuse import Observability
from app.settings import Settings


def _settings(**overrides) -> Settings:
    # Explicit None so this test-only Settings instance never picks up real
    # Langfuse credentials from a developer's local .env file.
    defaults = {"langfuse_public_key": None, "langfuse_secret_key": None}
    return Settings(llm_backend="dummy", **{**defaults, **overrides})


def test_disabled_by_configuration():
    observability = Observability.build(_settings(langfuse_enabled=False))

    assert observability.enabled is False
    assert observability.client is None


def test_disabled_when_credentials_are_missing():
    observability = Observability.build(_settings(langfuse_enabled=True))

    assert observability.enabled is False


def test_disabled_when_client_construction_fails():
    settings = _settings(langfuse_enabled=True, langfuse_public_key="pk", langfuse_secret_key="sk")
    with patch("app.integrations.langfuse.Langfuse", side_effect=RuntimeError("boom")):
        observability = Observability.build(settings)

    assert observability.enabled is False


def test_enabled_when_credentials_and_client_are_valid():
    settings = _settings(langfuse_enabled=True, langfuse_public_key="pk", langfuse_secret_key="sk")
    with patch("app.integrations.langfuse.Langfuse", return_value=MagicMock()):
        observability = Observability.build(settings)

    assert observability.enabled is True
    assert observability.client is not None


def test_trace_config_is_empty_when_disabled():
    observability = Observability(None)

    assert observability.trace_config("thread-1") == {}


def test_trace_config_attaches_a_callback_and_session_id_when_enabled():
    observability = Observability(MagicMock())

    config = observability.trace_config("thread-1")

    assert "callbacks" in config
    assert config["metadata"]["langfuse_session_id"] == "thread-1"


def test_update_trace_is_a_no_op_when_disabled():
    observability = Observability(None)

    observability.update_trace(decision="eligible")  # must not raise


def test_update_trace_swallows_a_client_failure():
    client = MagicMock()
    client.update_current_trace.side_effect = RuntimeError("boom")
    observability = Observability(client)

    observability.update_trace(decision="eligible")  # must not raise

    client.update_current_trace.assert_called_once()
