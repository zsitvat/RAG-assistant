from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from langchain_redis import RedisVectorStore
from starlette.concurrency import run_in_threadpool

from app.dependencies import get_redis_index, get_rule_catalogue, get_vector_store
from app.integrations.redis import RedisIndex
from app.rag.ingest.pipeline import CorpusIngestor
from app.rag.model import IngestResult
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
    """Ingests the corpus and rule catalogue into the vector store and Redis index."""
    if redis_index is None or vector_store is None:
        raise HTTPException(status_code=503, detail=REDIS_UNAVAILABLE_DETAIL)
    return await run_in_threadpool(
        CorpusIngestor().run, redis_index, vector_store, rule_catalogue=rule_catalogue
    )
