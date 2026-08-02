from langgraph.checkpoint.redis import AsyncRedisSaver

CHECKPOINT_TTL_MINUTES = 24 * 60
CHECKPOINT_PREFIX = "checkpoint"
CHECKPOINT_WRITE_PREFIX = "checkpoint_write"


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
