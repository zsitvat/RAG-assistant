import pytest
import redis as redis_lib

from app.integrations.redis import RedisIndex
from app.rag.graph import build_rag_graph
from app.rag.index_schema import VECTOR_DIMENSION
from app.rag.ingest.pipeline import CorpusIngestor
from app.rag.retriever import Retriever
from app.rag.store import build_embeddings, build_vector_store
from app.rules.loader import load_rule_catalogue
from app.tests.redis_availability import TEST_REDIS_URL, redis_available

CATEGORY_QUESTIONS = {
    "general": "how many days do I have to submit an expense claim",
    "meal": "how much can I claim for a business meal per person",
    "equipment": "what is the approval process for buying a laptop",
    "travel": "what is the accommodation limit per night for a business trip",
    "commuting": "how much travel pass support do I get for commuting",
    "mileage": "what is the mileage reimbursement rate for my own car",
    "benefits": "what is the annual recreational benefit budget",
}

pytestmark = pytest.mark.skipif(not redis_available(), reason="Redis 8 not reachable")


@pytest.fixture(scope="module")
def redis_client() -> redis_lib.Redis:
    return redis_lib.Redis.from_url(TEST_REDIS_URL, decode_responses=True)


@pytest.fixture(scope="module")
def redis_index() -> RedisIndex:
    return RedisIndex(TEST_REDIS_URL)


@pytest.fixture(scope="module")
def vector_store():
    return build_vector_store(TEST_REDIS_URL, build_embeddings())


@pytest.fixture(autouse=True)
def _clean_index(redis_client, vector_store):
    vector_store.index.create(overwrite=True, drop=True)
    redis_client.delete("build_info:corpus")
    yield
    vector_store.index.create(overwrite=True, drop=True)
    redis_client.delete("build_info:corpus")


def test_full_ingest_idempotent_rerun_and_dimension_mismatch_rebuild(
    redis_client, redis_index, vector_store
):
    # Arrange
    catalogue = load_rule_catalogue()
    ingestor = CorpusIngestor()

    first = ingestor.run(redis_index, vector_store, rule_catalogue=catalogue)
    assert first.action == "built"
    assert first.chunk_count > 0
    assert int(redis_client.ft("idx:chunks").info()["num_docs"]) == first.chunk_count

    second = ingestor.run(redis_index, vector_store, rule_catalogue=catalogue)
    assert second.action == "reused"
    assert second.chunk_count == first.chunk_count

    stale_build_info = redis_index.read_build_info()
    redis_index.write_build_info(stale_build_info.model_copy(update={"dimension": 1}))

    third = ingestor.run(redis_index, vector_store, rule_catalogue=catalogue)
    assert third.action == "rebuilt"
    assert third.chunk_count == first.chunk_count
    assert int(redis_client.ft("idx:chunks").info()["num_docs"]) == first.chunk_count


def test_similarity_search_returns_relevant_grounded_chunks(redis_index, vector_store):
    # Arrange
    ingestor = CorpusIngestor()
    ingestor.run(redis_index, vector_store, rule_catalogue=load_rule_catalogue())

    # Act
    results = vector_store.similarity_search(
        "how much can I claim for a business meal per person", k=3
    )

    # Assert
    assert len(results) > 0
    assert any(result.metadata["doc_id"] == "01" for result in results)


def test_similarity_search_respects_category_tag_filter(redis_index, vector_store):
    # Arrange
    ingestor = CorpusIngestor()
    ingestor.run(redis_index, vector_store, rule_catalogue=load_rule_catalogue())

    # Act
    results = vector_store.similarity_search(
        "how much can I claim for a business meal per person",
        k=5,
        filter="@categories:{meal}",
    )

    # Assert
    assert len(results) > 0
    assert all("meal" in result.metadata["categories"] for result in results)


def test_indexed_vector_dimension_matches_the_configured_dimension(redis_index, vector_store):
    # Arrange
    ingestor = CorpusIngestor()
    ingestor.run(redis_index, vector_store, rule_catalogue=load_rule_catalogue())

    # Assert
    assert redis_index.indexed_vector_dimension() == VECTOR_DIMENSION


def test_get_index_stats_reports_total_chunks_and_category_counts(redis_index, vector_store):
    # Arrange
    ingestor = CorpusIngestor()
    ingestor.run(redis_index, vector_store, rule_catalogue=load_rule_catalogue())

    # Act
    stats = redis_index.get_index_stats()

    # Assert
    assert stats["total_chunks"] > 0
    assert stats["category_counts"]["meal"] > 0
    assert sum(stats["category_counts"].values()) >= stats["total_chunks"]


@pytest.mark.parametrize("category,question", list(CATEGORY_QUESTIONS.items()))
async def test_rag_graph_returns_grounded_evidence_for_each_category(
    redis_index, vector_store, category, question
):
    # Arrange
    CorpusIngestor().run(redis_index, vector_store, rule_catalogue=load_rule_catalogue())
    graph = build_rag_graph(Retriever(vector_store))

    # Act
    result = (await graph.ainvoke({"question": question, "category": category}))["result"]

    # Assert
    assert len(result.results) > 0
    assert result.confidence >= 0.8
    assert result.citations
    assert result.context.startswith("[S1]")


async def test_rag_graph_flags_low_confidence_for_an_irrelevant_question(redis_index, vector_store):
    # Arrange
    CorpusIngestor().run(redis_index, vector_store, rule_catalogue=load_rule_catalogue())
    graph = build_rag_graph(Retriever(vector_store))

    # Act
    result = (
        await graph.ainvoke(
            {"question": "what is the weather like on mars today", "category": None}
        )
    )["result"]

    # Assert
    assert result.confidence < 0.8
