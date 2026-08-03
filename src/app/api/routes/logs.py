from typing import Annotated

from fastapi import APIRouter, Query
from starlette.concurrency import run_in_threadpool

from app.api.schemas import LogEntry, LogsResponse
from app.logging.config import DEFAULT_LOG_DIR, read_recent_lines

SERVICE_NAME = "api"
MAX_LINES = 1000

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/logs")
async def logs(lines: Annotated[int, Query(ge=1, le=MAX_LINES)] = 200) -> LogsResponse:
    """Returns the service's most recent structured log lines, oldest first."""
    raw_lines = await run_in_threadpool(read_recent_lines, DEFAULT_LOG_DIR, SERVICE_NAME, lines)
    return LogsResponse(entries=[LogEntry.model_validate_json(line) for line in raw_lines])
