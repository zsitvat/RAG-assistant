from typing import Literal

from pydantic import BaseModel

IngestAction = Literal["built", "rebuilt", "reused"]


class CorpusManifest(BaseModel):
    corpus_hash: str
    chunk_size: int
    chunk_overlap: int
    short_section_merge_threshold: int
    embedding_model: str
    embedding_revision: str
    dimension: int


class IngestResult(BaseModel):
    action: IngestAction
    chunk_count: int
    category_counts: dict[str, int]


class IndexStats(BaseModel):
    index_name: str
    dimension: int
    total_chunks: int
    category_counts: dict[str, int]
