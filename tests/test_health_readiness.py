from unittest.mock import MagicMock

import redis

from app.api.routes.health import _check_redis


def test_check_redis_reports_unavailable_when_client_is_none():
    check = _check_redis(None)
    assert check.status == "unavailable"


def test_check_redis_reports_unavailable_when_ping_fails():
    client = MagicMock()
    client.ping.side_effect = redis.ConnectionError("refused")

    check = _check_redis(client)

    assert check.status == "unavailable"


def test_check_redis_reports_ok_when_ping_succeeds():
    client = MagicMock()
    client.ping.return_value = True

    check = _check_redis(client)

    assert check.status == "ok"
