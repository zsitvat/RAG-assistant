import json
import logging
import logging.handlers
from datetime import date

import pytest

from app.logging.config import cleanup_expired_archives, configure_logging


def test_configure_logging_writes_json_lines(tmp_path, capsys):
    configure_logging(service="test-service", log_level="INFO", log_dir=tmp_path)
    logging.getLogger("app.test").info("hello world")

    record = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert record["service"] == "test-service"
    assert record["event"] == "hello world"


def test_configure_logging_includes_exception_field(tmp_path, capsys):
    configure_logging(service="test-service", log_level="INFO", log_dir=tmp_path)
    logger = logging.getLogger("app.test.exception")

    try:
        raise ValueError("boom")
    except ValueError:
        logger.exception("failed")

    record = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert "boom" in record["exception"]


def test_configure_logging_rejects_path_that_is_not_a_directory(tmp_path):
    blocking_file = tmp_path / "not-a-directory"
    blocking_file.write_text("x")
    log_dir = blocking_file / "logs"

    with pytest.raises(RuntimeError, match="not writable"):
        configure_logging(service="test-service", log_level="INFO", log_dir=log_dir)


def test_configure_logging_leaves_unlisted_loggers_at_the_warning_default(tmp_path):
    configure_logging(service="test-service", log_level="DEBUG", log_dir=tmp_path)

    assert logging.getLogger("httpcore").getEffectiveLevel() == logging.WARNING
    assert logging.getLogger("sentence_transformers").getEffectiveLevel() == logging.WARNING


def test_configure_logging_applies_configured_level_to_the_app_logger(tmp_path):
    configure_logging(service="test-service", log_level="DEBUG", log_dir=tmp_path)

    assert logging.getLogger("app").getEffectiveLevel() == logging.DEBUG
    assert logging.getLogger("app.some.module").getEffectiveLevel() == logging.DEBUG


def test_cleanup_expired_archives_removes_files_outside_retention_window(tmp_path):
    expired = tmp_path / "api.jsonl.2026-07-01"
    expired.write_text("old")

    removed = cleanup_expired_archives(
        tmp_path, service="api", retention_days=7, today=date(2026, 7, 20)
    )

    assert removed == [expired]
    assert not expired.exists()


def test_cleanup_expired_archives_keeps_files_inside_retention_window(tmp_path):
    fresh = tmp_path / "api.jsonl.2026-07-18"
    fresh.write_text("fresh")

    removed = cleanup_expired_archives(
        tmp_path, service="api", retention_days=7, today=date(2026, 7, 20)
    )

    assert removed == []
    assert fresh.exists()


def test_cleanup_expired_archives_ignores_other_services_and_malformed_names(tmp_path):
    other_service = tmp_path / "ui.jsonl.2026-01-01"
    other_service.write_text("other")
    malformed = tmp_path / "api.jsonl.not-a-date"
    malformed.write_text("malformed")
    live_file = tmp_path / "api.jsonl"
    live_file.write_text("live")

    removed = cleanup_expired_archives(
        tmp_path, service="api", retention_days=7, today=date(2026, 7, 20)
    )

    assert removed == []
    assert other_service.exists()
    assert malformed.exists()
    assert live_file.exists()


def test_configure_logging_removes_expired_archives_at_startup(tmp_path):
    expired = tmp_path / "test-service.jsonl.2000-01-01"
    expired.write_text("stale from a stopped service")

    configure_logging(service="test-service", log_level="INFO", log_dir=tmp_path)

    assert not expired.exists()


def test_configure_logging_rotates_daily_at_utc_midnight_with_seven_day_backup(tmp_path):
    configure_logging(service="test-service", log_level="INFO", log_dir=tmp_path)

    file_handler = next(
        h
        for h in logging.getLogger().handlers
        if isinstance(h, logging.handlers.TimedRotatingFileHandler)
    )
    assert file_handler.when == "MIDNIGHT"
    assert file_handler.utc is True
    assert file_handler.backupCount == 7


def test_no_application_log_call_passes_claim_or_prompt_payload_content():
    """Guards against a future logger.*() call leaking claim/prompt/answer text."""
    import ast
    from pathlib import Path

    forbidden_names = {"claim", "prompt", "answer", "context", "content", "template"}
    offenders = []
    for path in sorted(Path("app").rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in {"info", "debug", "warning", "error", "exception"}
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "logger"
            ):
                for arg in node.args:
                    names = {n.id for n in ast.walk(arg) if isinstance(n, ast.Name)}
                    if names & forbidden_names:
                        offenders.append(f"{path}:{node.lineno}")

    assert offenders == []
