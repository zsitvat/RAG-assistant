import redis
from redis.commands.search.query import Query

from app.rag.model import IndexBuildInfo

BUILD_INFO_KEY = "build_info:corpus"
CHUNK_INDEX_NAME = "idx:chunks"


class RedisIndex:
    """The Redis connection plus build-info/index-stats operations, kept in one place."""

    def __init__(self, redis_url: str) -> None:
        """Connects to Redis at the given URL."""
        self._client = redis.Redis.from_url(redis_url, decode_responses=True)

    @property
    def client(self) -> redis.Redis:
        """Returns the underlying Redis client."""
        return self._client

    def ping(self) -> bool:
        """Checks whether Redis is reachable."""
        return self._client.ping()

    def read_build_info(self) -> IndexBuildInfo | None:
        """Reads the stored corpus build info, if any."""
        raw = self._client.get(BUILD_INFO_KEY)
        if raw is None:
            return None
        return IndexBuildInfo.model_validate_json(raw)

    def write_build_info(self, build_info: IndexBuildInfo) -> None:
        """Persists the corpus build info."""
        self._client.set(BUILD_INFO_KEY, build_info.model_dump_json())

    def get_index_stats(self) -> dict:
        """Returns the total chunk count and per-category chunk counts from the index."""
        try:
            info = self._client.ft(CHUNK_INDEX_NAME).info()
        except redis.ResponseError:
            return {"total_chunks": 0, "category_counts": {}}

        total_chunks = int(info["num_docs"])
        results = self._client.ft(CHUNK_INDEX_NAME).search(
            Query("*").return_fields("categories").paging(0, max(total_chunks, 1))
        )
        category_counts: dict[str, int] = {}
        for doc in results.docs:
            for category in (getattr(doc, "categories", "") or "").split("|"):
                if category:
                    category_counts[category] = category_counts.get(category, 0) + 1

        return {"total_chunks": total_chunks, "category_counts": category_counts}
