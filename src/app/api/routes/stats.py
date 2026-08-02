from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from starlette.concurrency import run_in_threadpool

from app.dependencies import get_redis_index
from app.integrations.redis import RedisIndex
from app.rag.index_schema import INDEX_NAME, VECTOR_DIMENSION
from app.rag.model import IndexStats

router = APIRouter(prefix="/admin", tags=["admin"])

REDIS_UNAVAILABLE_DETAIL = "Redis is unavailable; the policy index cannot be reached."
REDIS_UNAVAILABLE_RESPONSE = {503: {"description": REDIS_UNAVAILABLE_DETAIL}}


@router.get("/stats", responses=REDIS_UNAVAILABLE_RESPONSE)
async def stats(
    redis_index: Annotated[RedisIndex | None, Depends(get_redis_index)],
) -> IndexStats:
    """Returns the current policy index size and per-category chunk counts."""
    if redis_index is None:
        raise HTTPException(status_code=503, detail=REDIS_UNAVAILABLE_DETAIL)

    raw_stats = await run_in_threadpool(redis_index.get_index_stats)
    return IndexStats(
        index_name=INDEX_NAME,
        dimension=VECTOR_DIMENSION,
        total_chunks=raw_stats["total_chunks"],
        category_counts=raw_stats["category_counts"],
    )
