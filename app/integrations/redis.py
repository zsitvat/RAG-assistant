"""Redis connection and lifecycle operations the LangChain integration does not own."""

import redis
from redis.commands.search.query import Query

from app.rag.model import CorpusManifest

MANIFEST_KEY = "manifest:corpus"
CHUNK_INDEX_NAME = "idx:chunks"


class RedisIndexLifecycle:
    """Connection plus manifest/index-stats operations the vector-store integration doesn't own."""

    def __init__(self, client: redis.Redis) -> None:
        self._client = client

    @classmethod
    def connect(cls, redis_url: str) -> "RedisIndexLifecycle":
        return cls(redis.Redis.from_url(redis_url, decode_responses=True))

    def ping(self) -> bool:
        return self._client.ping()

    def read_manifest(self) -> CorpusManifest | None:
        raw = self._client.get(MANIFEST_KEY)
        if raw is None:
            return None
        return CorpusManifest.model_validate_json(raw)

    def write_manifest(self, manifest: CorpusManifest) -> None:
        self._client.set(MANIFEST_KEY, manifest.model_dump_json())

    def get_index_stats(self) -> dict:
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
