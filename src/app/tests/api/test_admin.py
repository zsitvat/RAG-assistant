import pytest

from app.rag.ingest.pipeline import _INGEST_LOCK
from app.tests.redis_availability import redis_available

pytestmark = pytest.mark.skipif(not redis_available(), reason="Redis 8 not reachable")


async def test_admin_ingest_returns_the_ingest_result(client):
    # Act
    response = await client.post("/admin/ingest")

    # Assert
    assert response.status_code == 200
    body = response.json()
    assert body["action"] in ("built", "rebuilt", "reused")
    assert isinstance(body["chunk_count"], int)


async def test_admin_ingest_returns_409_while_another_run_holds_the_lock(client):
    # Arrange
    _INGEST_LOCK.acquire()

    try:
        # Act
        response = await client.post("/admin/ingest")
    finally:
        _INGEST_LOCK.release()

    # Assert
    assert response.status_code == 409


async def test_admin_stats_returns_the_index_stats(client):
    # Act
    response = await client.get("/admin/stats")

    # Assert
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["total_chunks"], int)
    assert isinstance(body["category_counts"], dict)
