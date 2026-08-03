import redis

from app.api.schemas import ReadinessCheck, ReadyResponse
from app.integrations.ollama import check_ollama_ready
from app.integrations.redis import RedisIndex
from app.rag.index_schema import VECTOR_DIMENSION
from app.settings import Settings


class ReadinessChecker:
    """Aggregates the LLM and Redis readiness checks behind the /ready endpoint."""

    def check(self, settings: Settings, redis_index: RedisIndex) -> ReadyResponse:
        """Returns the combined readiness result for the LLM and Redis dependencies."""
        checks = [self._check_llm(settings), self._check_redis(redis_index)]
        return ReadyResponse(ready=all(check.status == "ok" for check in checks), checks=checks)

    @staticmethod
    def _check_redis(redis_index: RedisIndex) -> ReadinessCheck:
        """Returns the current Redis readiness check, including an index-dimension mismatch."""
        try:
            redis_index.ping()
            indexed_dimension = redis_index.indexed_vector_dimension()
        except redis.RedisError:
            return ReadinessCheck(name="redis", status="unavailable", detail="Redis ping failed.")
        if indexed_dimension is not None and indexed_dimension != VECTOR_DIMENSION:
            return ReadinessCheck(
                name="redis",
                status="unavailable",
                detail=(
                    f"Indexed vector dimension {indexed_dimension} does not match the configured "
                    f"{VECTOR_DIMENSION}; the index must be rebuilt."
                ),
            )
        return ReadinessCheck(name="redis", status="ok", detail="Redis is reachable.")

    @staticmethod
    def _check_llm(settings: Settings) -> ReadinessCheck:
        """Returns the current LLM backend readiness check."""
        if settings.llm_backend == "dummy":
            return ReadinessCheck(
                name="llm", status="ok", detail="Dummy backend requires no external model."
            )
        result = check_ollama_ready(settings.ollama_base_url, settings.llm_model)
        return ReadinessCheck(
            name="llm", status="ok" if result.ready else "unavailable", detail=result.detail
        )
