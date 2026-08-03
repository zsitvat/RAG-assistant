import logging

from langfuse import Langfuse

from app.settings import Settings

logger = logging.getLogger(__name__)

PRODUCTION_LABEL = "production"


class Observability:
    """Owns the Langfuse client, turn tracing metadata and prompt resolution access."""

    def __init__(self, client: Langfuse | None) -> None:
        """Stores the Langfuse client, or None when tracing is disabled."""
        self._client = client

    @classmethod
    def build(cls, settings: Settings) -> "Observability":
        """Builds an observability adapter, degrading to disabled on any setup failure."""
        if not settings.langfuse_enabled:
            logger.info("langfuse tracing disabled by configuration")
            return cls(None)
        if not (settings.langfuse_public_key and settings.langfuse_secret_key):
            logger.warning("langfuse enabled but credentials are missing; tracing disabled")
            return cls(None)
        try:
            client = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host,
            )
        except Exception:
            logger.warning("langfuse client could not be created; tracing disabled")
            return cls(None)
        return cls(client)

    @property
    def enabled(self) -> bool:
        """Reports whether turns are traced."""
        return self._client is not None

    @property
    def client(self) -> Langfuse | None:
        """Returns the Langfuse client when tracing is enabled."""
        return self._client

    def trace_config(
        self, thread_id: str, *, tags: tuple[str, ...] = ("chat",), **metadata: object
    ) -> dict:
        """Builds the LangChain config fragment that attaches one trace to a turn."""
        if self._client is None:
            return {}
        from langfuse.langchain import CallbackHandler

        trace_metadata = {"langfuse_session_id": thread_id, "langfuse_tags": list(tags)}
        trace_metadata.update({key: value for key, value in metadata.items() if value is not None})
        return {"callbacks": [CallbackHandler()], "metadata": trace_metadata}

    def update_trace(self, **attributes: object) -> None:
        """Records turn-level outcome attributes on the active trace."""
        if self._client is None:
            return
        try:
            self._client.update_current_trace(metadata=attributes)
        except Exception as e:
            logger.warning(f"langfuse trace update failed: {type(e).__name__}: {e}")
