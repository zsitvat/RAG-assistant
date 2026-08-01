from typing import Literal

from pydantic import BaseModel

from app.rules.model import Category

IngestAction = Literal["built", "rebuilt", "reused"]


class IndexBuildInfo(BaseModel):
    """Describes the corpus content and chunking/embedding config a policy index was built from."""

    corpus_hash: str
    chunk_size: int
    chunk_overlap: int
    short_section_merge_threshold: int
    embedding_model: str
    embedding_revision: str
    dimension: int


class IngestResult(BaseModel):
    """Reports the outcome of a corpus ingestion run."""

    action: IngestAction
    chunk_count: int
    category_counts: dict[str, int]


class IndexStats(BaseModel):
    """Reports the size and category breakdown of the policy index."""

    index_name: str
    dimension: int
    total_chunks: int
    category_counts: dict[str, int]


class RetrievedResult(BaseModel):
    """Holds a single retrieved chunk and its similarity to the query."""

    doc_id: str
    doc_title: str
    section_id: str | None
    section: str | None
    categories: list[str]
    rule_ids: list[str]
    source_path: str
    content: str
    similarity: float


class Citation(BaseModel):
    """Identifies a policy document section cited in the assembled context."""

    marker: str
    doc_id: str
    doc_title: str
    section: str | None


class RagResult(BaseModel):
    """Holds the retrieved results, assembled context and citations for a policy search."""

    results: list[RetrievedResult] = []
    category: Category | None = None
    context: str = ""
    citations: list[Citation] = []

    @classmethod
    def from_artifact(cls, value: "RagResult | dict | None") -> "RagResult":
        """Rebuilds a result from a tool artifact, which a checkpoint restores as a plain dict."""
        if isinstance(value, cls):
            return value
        if not value:
            return cls()
        return cls.model_validate(value.get("kwargs", value) if value.get("lc") else value)

    @property
    def confidence(self) -> float:
        """Returns the top result's similarity score, or 0.0 if there are no results."""
        return self.results[0].similarity if self.results else 0.0
