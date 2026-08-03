from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from langchain_redis import RedisVectorStore
from starlette.concurrency import run_in_threadpool

from app.dependencies import get_redis_index, get_rule_catalogue, get_vector_store
from app.integrations.redis import RedisIndex
from app.rag.ingest.errors import IngestionInProgressError
from app.rag.ingest.pipeline import CorpusIngestor
from app.rag.model import IngestResult
from app.rules.model import RuleCatalogue

router = APIRouter(prefix="/admin", tags=["admin"])

INGEST_IN_PROGRESS_DETAIL = "An ingestion run is already in progress; try again shortly."
INGEST_IN_PROGRESS_RESPONSE = {409: {"description": INGEST_IN_PROGRESS_DETAIL}}


@router.post("/ingest", responses=INGEST_IN_PROGRESS_RESPONSE)
async def ingest(
    redis_index: Annotated[RedisIndex, Depends(get_redis_index)],
    vector_store: Annotated[RedisVectorStore, Depends(get_vector_store)],
    rule_catalogue: Annotated[RuleCatalogue, Depends(get_rule_catalogue)],
) -> IngestResult:
    """Ingests the corpus and rule catalogue into the vector store and Redis index."""
    try:
        return await run_in_threadpool(
            CorpusIngestor().run, redis_index, vector_store, rule_catalogue=rule_catalogue
        )
    except IngestionInProgressError as e:
        raise HTTPException(status_code=409, detail=INGEST_IN_PROGRESS_DETAIL) from e
