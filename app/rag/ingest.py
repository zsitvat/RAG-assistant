import logging
from pathlib import Path

import redis
from langchain_core.documents import Document
from langchain_redis import RedisVectorStore

from app.integrations.redis import RedisIndex
from app.rag.build_info import IndexBuildInfoBuilder
from app.rag.chunker import MarkdownChunker
from app.rag.docx_loader import CORPUS_DIR, DocxMarkdownLoader
from app.rag.errors import IngestionError
from app.rag.index_schema import VECTOR_DIMENSION
from app.rag.model import IngestResult
from app.rag.rule_metadata import RuleMetadataResolver
from app.rag.store import (
    EMBEDDING_MODEL_NAME,
    EMBEDDING_MODEL_REVISION,
    build_embeddings,
    build_vector_store,
)
from app.rules.loader import get_rule_catalogue
from app.rules.model import RuleCatalogue
from app.settings import Settings, get_settings

RULES_PATH = Path("config/rules.yaml")
INGEST_BATCH_SIZE = 128

logger = logging.getLogger(__name__)

__all__ = ["IngestionError", "CorpusIngestor", "connect_and_ingest"]


class CorpusIngestor:
    """Loads, chunks, validates and upserts the corpus into Redis."""

    def __init__(
        self,
        corpus_dir: Path = CORPUS_DIR,
        rules_path: Path = RULES_PATH,
        chunker: MarkdownChunker | None = None,
    ) -> None:
        """Stores the corpus paths and chunker used for ingestion."""
        self._corpus_dir = corpus_dir
        self._rules_path = rules_path
        self._chunker = chunker or MarkdownChunker()
        self._build_info_builder = IndexBuildInfoBuilder(corpus_dir, rules_path)

    def load_and_chunk(
        self, rule_catalogue: RuleCatalogue
    ) -> tuple[list[Document], list[Document]]:
        """Returns (source_documents, chunks), validated against the rule catalogue."""
        source_documents = list(DocxMarkdownLoader(self._corpus_dir).lazy_load())
        chunks = [
            chunk
            for source in source_documents
            for chunk in self._chunker.chunk(
                source.metadata["doc_id"],
                source.metadata["doc_title"],
                source.metadata["source_path"],
                source.page_content,
            )
        ]
        resolver = RuleMetadataResolver(rule_catalogue)
        resolver.attach(chunks)
        resolver.validate_anchors_resolve(chunks)
        resolver.validate_categories_reachable(chunks)
        return source_documents, chunks

    def run(
        self,
        redis_index: RedisIndex,
        vector_store: RedisVectorStore,
        rule_catalogue: RuleCatalogue | None = None,
    ) -> IngestResult:
        """Ingests the corpus into Redis, skipping embed/upsert when the build info matches."""
        rule_catalogue = rule_catalogue or get_rule_catalogue()
        _, chunks = self.load_and_chunk(rule_catalogue)

        build_info = self._build_info_builder.build(
            EMBEDDING_MODEL_NAME, EMBEDDING_MODEL_REVISION, VECTOR_DIMENSION
        )
        existing_build_info = redis_index.read_build_info()

        if existing_build_info == build_info:
            action = "reused"
        else:
            action = "rebuilt" if existing_build_info is not None else "built"
            if existing_build_info is not None:
                # Recreate through SearchIndex so later writes remain indexed.
                vector_store.index.create(overwrite=True, drop=True)
            self._upsert_chunks(vector_store, chunks)
            redis_index.write_build_info(build_info)

        return IngestResult(
            action=action, chunk_count=len(chunks), category_counts=self._count_categories(chunks)
        )

    @staticmethod
    def _upsert_chunks(vector_store: RedisVectorStore, chunks: list[Document]) -> None:
        for start in range(0, len(chunks), INGEST_BATCH_SIZE):
            batch = chunks[start : start + INGEST_BATCH_SIZE]
            vector_store.add_texts(
                texts=[chunk.page_content for chunk in batch],
                metadatas=[chunk.metadata for chunk in batch],
                keys=[
                    f"{chunk.metadata['doc_id']}:{chunk.metadata['chunk_index']}" for chunk in batch
                ],
            )

    @staticmethod
    def _count_categories(chunks: list[Document]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for chunk in chunks:
            for category in chunk.metadata["categories"]:
                counts[category] = counts.get(category, 0) + 1
        return counts


def connect_and_ingest(
    settings: Settings, rule_catalogue: RuleCatalogue
) -> tuple[RedisIndex | None, RedisVectorStore | None]:
    """Connects to Redis and ensures the index is ready. Returns (None, None) if unreachable."""
    try:
        redis_index = RedisIndex(settings.redis_url)
        redis_index.ping()
    except redis.RedisError:
        logger.warning("Redis unavailable at startup")
        return None, None

    vector_store = build_vector_store(settings.redis_url, build_embeddings())
    CorpusIngestor().run(redis_index, vector_store, rule_catalogue=rule_catalogue)
    return redis_index, vector_store


if __name__ == "__main__":
    settings = get_settings()
    result = CorpusIngestor().run(
        RedisIndex(settings.redis_url),
        build_vector_store(settings.redis_url, build_embeddings()),
        rule_catalogue=get_rule_catalogue(),
    )
    print(result.model_dump_json(indent=2))
