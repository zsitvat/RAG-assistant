class IngestionError(RuntimeError):
    """Raised when the corpus cannot be safely converted, chunked or cross-checked."""


class IngestionInProgressError(IngestionError):
    """Raised when an ingestion run is rejected because another run already holds the lock."""
