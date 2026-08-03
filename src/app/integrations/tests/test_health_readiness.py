from unittest.mock import MagicMock

import redis

from app.integrations.readiness import ReadinessChecker
from app.rag.index_schema import VECTOR_DIMENSION


def _matching_index() -> MagicMock:
    client = MagicMock()
    client.ping.return_value = True
    client.indexed_vector_dimension.return_value = VECTOR_DIMENSION
    return client


def test_check_redis_reports_unavailable_when_ping_fails():
    # Arrange
    client = _matching_index()
    client.ping.side_effect = redis.ConnectionError("refused")

    # Act
    check = ReadinessChecker()._check_redis(client)

    # Assert
    assert check.status == "unavailable"


def test_check_redis_reports_ok_when_ping_succeeds():
    # Act
    check = ReadinessChecker()._check_redis(_matching_index())

    # Assert
    assert check.status == "ok"


def test_check_redis_reports_unavailable_on_a_vector_dimension_mismatch():
    # Arrange
    client = _matching_index()
    client.indexed_vector_dimension.return_value = VECTOR_DIMENSION + 1

    # Act
    check = ReadinessChecker()._check_redis(client)

    # Assert
    assert check.status == "unavailable"
    assert "dimension" in check.detail
