from langgraph.checkpoint.redis import RedisSaver

CHECKPOINT_TTL_MINUTES = 24 * 60
CHECKPOINT_PREFIX = "checkpoint"
CHECKPOINT_WRITE_PREFIX = "checkpoint_write"


def build_checkpointer(redis_url: str) -> RedisSaver:
    """Builds the Redis-backed LangGraph checkpointer and creates its indexes."""
    checkpointer = RedisSaver(
        redis_url,
        ttl={"default_ttl": CHECKPOINT_TTL_MINUTES, "refresh_on_read": True},
        checkpoint_prefix=CHECKPOINT_PREFIX,
        checkpoint_write_prefix=CHECKPOINT_WRITE_PREFIX,
    )
    checkpointer.setup()
    return checkpointer
