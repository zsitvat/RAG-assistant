"""Embeddings and the LangChain Redis vector-store/retriever factory."""

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_redis import RedisConfig, RedisVectorStore

EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"
EMBEDDING_MODEL_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
EMBEDDING_DIMENSION = 384

INDEX_NAME = "idx:chunks"
KEY_PREFIX = "chunk"
TOP_K = 4

METADATA_SCHEMA = [
    {"name": "doc_id", "type": "tag"},
    {"name": "section_id", "type": "tag"},
    {"name": "categories", "type": "tag"},
    {"name": "rule_ids", "type": "tag"},
    {"name": "section", "type": "text", "attrs": {"no_stem": True}},
]


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
        distance_metric="COSINE",
        indexing_algorithm="HNSW",
        vector_datatype="FLOAT32",
        embedding_dimensions=EMBEDDING_DIMENSION,
        metadata_schema=METADATA_SCHEMA,
    )
    return RedisVectorStore(embeddings=embeddings, config=config)
