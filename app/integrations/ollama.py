import httpx2

TAGS_TIMEOUT_SECONDS = 5.0


class OllamaCheckResult:
    """Reports whether Ollama is reachable and the configured model is pulled."""

    def __init__(self, ready: bool, detail: str) -> None:
        """Stores the readiness outcome and a human-readable reason."""
        self.ready = ready
        self.detail = detail


def check_ollama_ready(base_url: str, model: str) -> OllamaCheckResult:
    """Checks that Ollama answers and has the configured model available."""
    try:
        response = httpx2.get(f"{base_url}/api/tags", timeout=TAGS_TIMEOUT_SECONDS)
        response.raise_for_status()
    except httpx2.HTTPError as exc:
        return OllamaCheckResult(False, f"Ollama at {base_url} is unreachable: {exc}")

    available = {entry["name"] for entry in response.json().get("models", [])}
    if model in available or _base_tag(model) in {_base_tag(name) for name in available}:
        return OllamaCheckResult(True, f"Ollama is reachable and {model!r} is available.")
    return OllamaCheckResult(False, f"Ollama is reachable but {model!r} is not pulled yet.")


def _base_tag(model: str) -> str:
    """Returns a model name without an explicit ':latest' suffix for comparison."""
    return model.removesuffix(":latest")
