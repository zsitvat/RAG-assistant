class IngestionError(RuntimeError):
    """Raised when the corpus cannot be safely converted, chunked or cross-checked."""
