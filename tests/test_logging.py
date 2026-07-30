import json
import logging

import pytest

from app.core.logging import configure_logging, request_id_var


def test_configure_logging_writes_json_with_correlation_fields(tmp_path, capsys):
    configure_logging(service="test-service", log_level="INFO", log_dir=tmp_path)
    token = request_id_var.set("req-123")
    try:
        logging.getLogger("app.test").info("hello world")
    finally:
        request_id_var.reset(token)

    record = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert record["service"] == "test-service"
    assert record["event"] == "hello world"
    assert record["request_id"] == "req-123"


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
