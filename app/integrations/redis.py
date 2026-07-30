"""Raw Redis client and lifecycle operations the LangChain integration does not own."""

import contextlib

import redis
from redis.commands.search.query import Query

from app.rag.model import CorpusManifest

MANIFEST_KEY = "manifest:corpus"
CHUNK_INDEX_NAME = "idx:chunks"


def build_redis_client(redis_url: str) -> redis.Redis:
    return redis.Redis.from_url(redis_url, decode_responses=True)


def read_manifest(client: redis.Redis) -> CorpusManifest | None:
    raw = client.get(MANIFEST_KEY)
    if raw is None:
        return None
    return CorpusManifest.model_validate_json(raw)


def write_manifest(client: redis.Redis, manifest: CorpusManifest) -> None:
    client.set(MANIFEST_KEY, manifest.model_dump_json())


def drop_chunk_index(client: redis.Redis) -> None:
    with contextlib.suppress(redis.ResponseError):
        client.ft(CHUNK_INDEX_NAME).dropindex(delete_documents=True)


def get_index_stats(client: redis.Redis) -> dict:
    try:
        info = client.ft(CHUNK_INDEX_NAME).info()
    except redis.ResponseError:
        return {"total_chunks": 0, "category_counts": {}}

    total_chunks = int(info["num_docs"])
    results = client.ft(CHUNK_INDEX_NAME).search(
        Query("*").return_fields("categories").paging(0, max(total_chunks, 1))
    )
    category_counts: dict[str, int] = {}
    for doc in results.docs:
        for category in (getattr(doc, "categories", "") or "").split("|"):
            if category:
                category_counts[category] = category_counts.get(category, 0) + 1

    return {"total_chunks": total_chunks, "category_counts": category_counts}
