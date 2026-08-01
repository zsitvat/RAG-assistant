import json
import logging

import pytest

from app.logging.config import configure_logging


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


def test_configure_logging_quiets_noisy_third_party_loggers(tmp_path):
    configure_logging(service="test-service", log_level="DEBUG", log_dir=tmp_path)

    assert logging.getLogger("httpcore").getEffectiveLevel() == logging.WARNING
    assert logging.getLogger("sentence_transformers").getEffectiveLevel() == logging.WARNING
