from typing import Annotated

import redis
from fastapi import APIRouter, Depends, HTTPException
from langchain_redis import RedisVectorStore
from starlette.concurrency import run_in_threadpool

from app.dependencies import get_redis_client, get_rule_catalogue, get_vector_store
from app.integrations.redis import get_index_stats
from app.rag.ingest import run_ingest
from app.rag.model import IndexStats, IngestResult
from app.rag.store import EMBEDDING_DIMENSION, INDEX_NAME
from app.rules.model import RuleCatalogue

router = APIRouter(prefix="/admin", tags=["admin"])

REDIS_UNAVAILABLE_DETAIL = "Redis is unavailable; the policy index cannot be reached."
REDIS_UNAVAILABLE_RESPONSE = {503: {"description": REDIS_UNAVAILABLE_DETAIL}}


@router.post("/ingest", responses=REDIS_UNAVAILABLE_RESPONSE)
async def ingest(
    redis_client: Annotated[redis.Redis | None, Depends(get_redis_client)],
    vector_store: Annotated[RedisVectorStore | None, Depends(get_vector_store)],
    rule_catalogue: Annotated[RuleCatalogue, Depends(get_rule_catalogue)],
) -> IngestResult:
    if redis_client is None or vector_store is None:
        raise HTTPException(status_code=503, detail=REDIS_UNAVAILABLE_DETAIL)
    return await run_in_threadpool(
        run_ingest, redis_client, vector_store, rule_catalogue=rule_catalogue
    )


@router.get("/stats", responses=REDIS_UNAVAILABLE_RESPONSE)
async def stats(
    redis_client: Annotated[redis.Redis | None, Depends(get_redis_client)],
) -> IndexStats:
    if redis_client is None:
        raise HTTPException(status_code=503, detail=REDIS_UNAVAILABLE_DETAIL)

    raw_stats = await run_in_threadpool(get_index_stats, redis_client)
    return IndexStats(
        index_name=INDEX_NAME,
        dimension=EMBEDDING_DIMENSION,
        total_chunks=raw_stats["total_chunks"],
        category_counts=raw_stats["category_counts"],
    )
