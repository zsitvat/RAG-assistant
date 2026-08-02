from unittest.mock import MagicMock, patch

import httpx2

from app.integrations.ollama import check_ollama_ready


def test_ready_when_the_configured_model_is_pulled():
    response = MagicMock(json=lambda: {"models": [{"name": "qwen2.5:7b-instruct-q4_K_M"}]})
    response.raise_for_status.return_value = None
    with patch("app.integrations.ollama.httpx2.get", return_value=response):
        result = check_ollama_ready("http://ollama:11434", "qwen2.5:7b-instruct-q4_K_M")

    assert result.ready is True


def test_ready_when_the_model_matches_without_an_explicit_latest_tag():
    response = MagicMock(json=lambda: {"models": [{"name": "qwen2.5:7b-instruct-q4_K_M:latest"}]})
    response.raise_for_status.return_value = None
    with patch("app.integrations.ollama.httpx2.get", return_value=response):
        result = check_ollama_ready("http://ollama:11434", "qwen2.5:7b-instruct-q4_K_M:latest")

    assert result.ready is True


def test_not_ready_when_the_model_is_not_pulled():
    response = MagicMock(json=lambda: {"models": [{"name": "llama3:8b"}]})
    response.raise_for_status.return_value = None
    with patch("app.integrations.ollama.httpx2.get", return_value=response):
        result = check_ollama_ready("http://ollama:11434", "qwen2.5:7b-instruct-q4_K_M")

    assert result.ready is False
    assert "not pulled" in result.detail


def test_not_ready_when_ollama_is_unreachable():
    with patch("app.integrations.ollama.httpx2.get", side_effect=httpx2.ConnectError("refused")):
        result = check_ollama_ready("http://ollama:11434", "qwen2.5:7b-instruct-q4_K_M")

    assert result.ready is False
    assert "unreachable" in result.detail
