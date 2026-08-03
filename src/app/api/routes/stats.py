from typing import Annotated

from fastapi import APIRouter, Depends
from starlette.concurrency import run_in_threadpool

from app.dependencies import get_redis_index
from app.integrations.redis import RedisIndex
from app.rag.index_schema import INDEX_NAME, VECTOR_DIMENSION
from app.rag.model import IndexStats

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/stats")
async def stats(
    redis_index: Annotated[RedisIndex, Depends(get_redis_index)],
) -> IndexStats:
    """Returns the current policy index size and per-category chunk counts."""
    raw_stats = await run_in_threadpool(redis_index.get_index_stats)
    return IndexStats(
        index_name=INDEX_NAME,
        dimension=VECTOR_DIMENSION,
        total_chunks=raw_stats["total_chunks"],
        category_counts=raw_stats["category_counts"],
    )
