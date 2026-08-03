import json
import logging
import logging.handlers
import re
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

LOG_RETENTION_DAYS = 7
DEFAULT_LOG_DIR = Path("logs")
_ARCHIVE_SUFFIX_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_CAPTURED_LOGGERS = ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi", "streamlit")
# Only these get the configured verbosity; every other logger (any third-party library)
# falls back to the root's WARNING default, so new noisy dependencies need no extra listing.
_VERBOSE_LOGGERS = ("app", *_CAPTURED_LOGGERS)


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


def cleanup_expired_archives(
    log_dir: Path, service: str, retention_days: int = LOG_RETENTION_DAYS, today: date | None = None
) -> list[Path]:
    """Deletes this service's rotated log archives outside the calendar-day retention window.

    Archive files are named ``{service}.jsonl.YYYY-MM-DD`` by ``TimedRotatingFileHandler``.
    Age is judged from that date suffix, not file mtime, so retention stays correct
    even if archives were copied or touched after rotation.
    """
    cutoff = (today or datetime.now(UTC).date()) - timedelta(days=retention_days)
    removed = []
    for path in log_dir.glob(f"{service}.jsonl.*"):
        suffix = path.name.removeprefix(f"{service}.jsonl.")
        if not _ARCHIVE_SUFFIX_RE.match(suffix):
            continue
        if date.fromisoformat(suffix) < cutoff:
            path.unlink(missing_ok=True)
            removed.append(path)
    return removed


def configure_logging(service: str, log_level: str, log_dir: Path = DEFAULT_LOG_DIR) -> None:
    """Sets up JSON stdout and rotating file logging for the service.

    Runs startup retention cleanup first, so archives left behind by a stopped
    service are removed before the service starts handling requests again.
    """
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise RuntimeError(f"Log directory '{log_dir.resolve()}' is not writable: {e}") from e

    cleanup_expired_archives(log_dir, service)

    formatter = _JsonFormatter(service=service)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)

    file_handler = logging.handlers.TimedRotatingFileHandler(
        log_dir / f"{service}.jsonl", when="midnight", utc=True, backupCount=LOG_RETENTION_DAYS
    )
    file_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.WARNING)
    root.addHandler(stdout_handler)
    root.addHandler(file_handler)

    for logger_name in _CAPTURED_LOGGERS:
        captured_logger = logging.getLogger(logger_name)
        captured_logger.handlers.clear()
        captured_logger.propagate = True

    for logger_name in _VERBOSE_LOGGERS:
        logging.getLogger(logger_name).setLevel(log_level)
