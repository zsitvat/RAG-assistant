from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from langchain_redis import RedisVectorStore
from starlette.concurrency import run_in_threadpool

from app.dependencies import get_redis_index, get_rule_catalogue, get_vector_store
from app.integrations.redis import RedisIndex
from app.rag.index_schema import INDEX_NAME, VECTOR_DIMENSION
from app.rag.ingest import PolicyCorpusIngestor
from app.rag.model import IndexStats, IngestResult
from app.rules.model import RuleCatalogue

router = APIRouter(prefix="/admin", tags=["admin"])

REDIS_UNAVAILABLE_DETAIL = "Redis is unavailable; the policy index cannot be reached."
REDIS_UNAVAILABLE_RESPONSE = {503: {"description": REDIS_UNAVAILABLE_DETAIL}}


@router.post("/ingest", responses=REDIS_UNAVAILABLE_RESPONSE)
async def ingest(
    redis_index: Annotated[RedisIndex | None, Depends(get_redis_index)],
    vector_store: Annotated[RedisVectorStore | None, Depends(get_vector_store)],
    rule_catalogue: Annotated[RuleCatalogue, Depends(get_rule_catalogue)],
) -> IngestResult:
    """Ingests the policy corpus and rule catalogue into the vector store and Redis index."""
    if redis_index is None or vector_store is None:
        raise HTTPException(status_code=503, detail=REDIS_UNAVAILABLE_DETAIL)
    return await run_in_threadpool(
        PolicyCorpusIngestor().run, redis_index, vector_store, rule_catalogue=rule_catalogue
    )


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
