"""Embeddings and the LangChain Redis vector-store factory."""

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_redis import RedisConfig, RedisVectorStore

from app.rag.index_schema import (
    DISTANCE_METRIC,
    INDEX_NAME,
    INDEXING_ALGORITHM,
    KEY_PREFIX,
    METADATA_SCHEMA,
    VECTOR_DATATYPE,
    VECTOR_DIMENSION,
)

EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"
EMBEDDING_MODEL_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"


class E5Embeddings(HuggingFaceEmbeddings):
    """Adds the `query:`/`passage:` prefixes `intfloat/multilingual-e5-small` expects."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return super().embed_documents([f"passage: {text}" for text in texts])

    def embed_query(self, text: str) -> list[float]:
        return super().embed_query(f"query: {text}")


def build_embeddings() -> Embeddings:
    return E5Embeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"revision": EMBEDDING_MODEL_REVISION},
    )


def build_vector_store(redis_url: str, embeddings: Embeddings) -> RedisVectorStore:
    config = RedisConfig(
        index_name=INDEX_NAME,
        key_prefix=KEY_PREFIX,
        redis_url=redis_url,
        distance_metric=DISTANCE_METRIC,
        indexing_algorithm=INDEXING_ALGORITHM,
        vector_datatype=VECTOR_DATATYPE,
        embedding_dimensions=VECTOR_DIMENSION,
        metadata_schema=METADATA_SCHEMA,
    )
    return RedisVectorStore(embeddings=embeddings, config=config)
