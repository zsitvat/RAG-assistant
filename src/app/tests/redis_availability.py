import os

import redis as redis_lib

TEST_REDIS_URL = os.environ.get("TEST_REDIS_URL", "redis://127.0.0.1:6379/0")


def redis_available() -> bool:
    """Pings the test Redis instance; used to skip integration tests when it's unreachable."""

    try:
        redis_lib.Redis.from_url(TEST_REDIS_URL).ping()
    except redis_lib.RedisError:
        return False
    return True
