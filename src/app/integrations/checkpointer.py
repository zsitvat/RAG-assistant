from langgraph.checkpoint.redis import AsyncRedisSaver

CHECKPOINT_TTL_MINUTES = 24 * 60
CHECKPOINT_PREFIX = "checkpoint"
CHECKPOINT_WRITE_PREFIX = "checkpoint_write"


def unwrap_lc_envelope(value: object) -> object:
    """Recursively strips LangChain's `{"lc": 2, "type": "constructor", "kwargs": {...}}`
    checkpoint envelope from a value, so nested wrapped objects surface as plain dicts too.

    The checkpoint serializer isn't allowlisted to revive our own pydantic types, so it
    restores each of them — however deeply nested — as its own envelope rather than a plain
    dict of fields; callers rebuilding a model from checkpointed state should unwrap first.
    """
    if isinstance(value, dict):
        if value.get("lc") == 2 and value.get("type") == "constructor":
            return unwrap_lc_envelope(value.get("kwargs", {}))
        return {key: unwrap_lc_envelope(v) for key, v in value.items()}
    if isinstance(value, list):
        return [unwrap_lc_envelope(v) for v in value]
    return value


async def build_checkpointer(redis_url: str) -> AsyncRedisSaver:
    """Builds the Redis-backed LangGraph checkpointer used by the async graph calls.

    Must be awaited from the same persistent event loop the app serves requests on (FastAPI's
    lifespan), not a one-off asyncio.run(), since AsyncRedisSaver is bound to the loop active
    during asetup().
    """
    checkpointer = AsyncRedisSaver(
        redis_url,
        ttl={"default_ttl": CHECKPOINT_TTL_MINUTES, "refresh_on_read": True},
        checkpoint_prefix=CHECKPOINT_PREFIX,
        checkpoint_write_prefix=CHECKPOINT_WRITE_PREFIX,
    )
    await checkpointer.asetup()
    return checkpointer
