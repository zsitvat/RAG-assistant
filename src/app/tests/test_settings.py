import pytest

from app.settings import Settings, SettingsError, get_settings


def test_get_settings_wraps_a_validation_error(monkeypatch):
    # Arrange
    monkeypatch.setenv("LLM_BACKEND", "Ollama")
    get_settings.cache_clear()

    # Act / Assert
    with pytest.raises(SettingsError, match="llm_backend"):
        get_settings()

    get_settings.cache_clear()


def test_get_settings_returns_a_settings_instance_for_valid_configuration(monkeypatch):
    # Arrange
    monkeypatch.setenv("LLM_BACKEND", "dummy")
    get_settings.cache_clear()

    # Act
    settings = get_settings()

    # Assert
    assert isinstance(settings, Settings)
    assert settings.llm_backend == "dummy"
    get_settings.cache_clear()
