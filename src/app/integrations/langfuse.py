import logging
from collections.abc import Callable, Generator
from contextlib import contextmanager

from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

from app.settings import Settings

logger = logging.getLogger(__name__)

PRODUCTION_LABEL = "production"
TURN_SPAN_NAME = "chat_turn"


def _noop_update(**_attributes: object) -> None:
    """Discards outcome attributes when tracing is disabled."""


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

    @contextmanager
    def traced_turn(
        self,
        thread_id: str,
        message: str,
        *,
        tags: tuple[str, ...] = ("chat",),
        **metadata: object,
    ) -> Generator[tuple[dict, Callable[..., None]], None, None]:
        """Opens the turn's root span, with `message` as trace input and the reply as output."""
        if self._client is None:
            yield {}, _noop_update
            return

        trace_metadata = {"langfuse_session_id": thread_id, "langfuse_tags": list(tags)}
        trace_metadata.update({key: value for key, value in metadata.items() if value is not None})
        config = {"callbacks": [CallbackHandler()], "metadata": trace_metadata}

        with self._client.start_as_current_observation(name=TURN_SPAN_NAME, input=message) as span:

            def update(*, output: object = None, **attributes: object) -> None:
                try:
                    span.update(metadata=attributes, output=output)
                except Exception as e:
                    logger.warning(f"langfuse trace update failed: {type(e).__name__}: {e}")

            yield config, update
