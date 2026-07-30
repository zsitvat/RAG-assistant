import contextvars
import json
import logging
import logging.handlers
import sys
from datetime import UTC, datetime
from pathlib import Path

LOG_RETENTION_DAYS = 7
DEFAULT_LOG_DIR = Path("logs")
_CAPTURED_LOGGERS = ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi", "streamlit")

request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id", default=None
)
thread_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "thread_id", default=None
)


class _CorrelationFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        record.thread_id = thread_id_var.get()
        return True


class _JsonFormatter(logging.Formatter):
    def __init__(self, service: str) -> None:
        super().__init__()
        self._service = service

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "service": self._service,
            "logger": record.name,
            "event": record.getMessage(),
            "request_id": getattr(record, "request_id", None),
            "thread_id": getattr(record, "thread_id", None),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(service: str, log_level: str, log_dir: Path = DEFAULT_LOG_DIR) -> None:
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(f"Log directory '{log_dir.resolve()}' is not writable: {exc}") from exc

    formatter = _JsonFormatter(service=service)
    correlation_filter = _CorrelationFilter()

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    stdout_handler.addFilter(correlation_filter)

    file_handler = logging.handlers.TimedRotatingFileHandler(
        log_dir / f"{service}.jsonl", when="midnight", utc=True, backupCount=LOG_RETENTION_DAYS
    )
    file_handler.setFormatter(formatter)
    file_handler.addFilter(correlation_filter)

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(log_level)
    root.addHandler(stdout_handler)
    root.addHandler(file_handler)

    for logger_name in _CAPTURED_LOGGERS:
        captured_logger = logging.getLogger(logger_name)
        captured_logger.handlers.clear()
        captured_logger.propagate = True
