import json
import logging
import logging.handlers
import sys
from datetime import UTC, datetime
from pathlib import Path

LOG_RETENTION_DAYS = 7
DEFAULT_LOG_DIR = Path("logs")
_CAPTURED_LOGGERS = ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi", "streamlit")
_NOISY_THIRD_PARTY_LOGGERS = (
    "httpcore",
    "httpx",
    "httpx2",
    "urllib3",
    "asyncio",
    "sentence_transformers",
    "transformers",
    "redisvl",
    "huggingface_hub",
    "filelock",
)


class _JsonFormatter(logging.Formatter):
    """Formats application log records as structured JSON lines."""

    def __init__(self, service: str) -> None:
        """Stores the service name included in every log record."""
        super().__init__()
        self._service = service

    def format(self, record: logging.LogRecord) -> str:
        """Renders the log record as a single JSON line."""
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "service": self._service,
            "logger": record.name,
            "event": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(service: str, log_level: str, log_dir: Path = DEFAULT_LOG_DIR) -> None:
    """Sets up JSON stdout and rotating file logging for the service."""
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(f"Log directory '{log_dir.resolve()}' is not writable: {exc}") from exc

    formatter = _JsonFormatter(service=service)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)

    file_handler = logging.handlers.TimedRotatingFileHandler(
        log_dir / f"{service}.jsonl", when="midnight", utc=True, backupCount=LOG_RETENTION_DAYS
    )
    file_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(log_level)
    root.addHandler(stdout_handler)
    root.addHandler(file_handler)

    for logger_name in _CAPTURED_LOGGERS:
        captured_logger = logging.getLogger(logger_name)
        captured_logger.handlers.clear()
        captured_logger.propagate = True

    for logger_name in _NOISY_THIRD_PARTY_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
