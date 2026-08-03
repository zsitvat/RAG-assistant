from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from app.integrations.redis import RedisIndex
from app.rag.ingest.build_info import IndexBuildInfoBuilder
from app.rag.ingest.errors import IngestionInProgressError
from app.rag.ingest.pipeline import _INGEST_LOCK, CorpusIngestor
from app.rag.model import IndexBuildInfo


def _build_info(**overrides) -> IndexBuildInfo:
    base = {
        "corpus_hash": "abc",
        "chunk_size": 800,
        "chunk_overlap": 120,
        "short_section_merge_threshold": 200,
        "embedding_model": "m",
        "embedding_revision": "r",
        "dimension": 384,
    }
    base.update(overrides)
    return IndexBuildInfo(**base)


def _fake_chunks() -> list[Document]:
    return [
        Document(
            page_content="Meals are capped at 15,000 HUF.",
            metadata={"doc_id": "01", "chunk_index": 0, "categories": ["meal"]},
        )
    ]


def _patch_corpus_and_build_info(monkeypatch) -> None:
    monkeypatch.setattr(
        CorpusIngestor, "load_and_chunk", lambda self, rule_catalogue: ([], _fake_chunks())
    )
    monkeypatch.setattr(IndexBuildInfoBuilder, "build", lambda self, *a, **k: _build_info())


def _fake_redis_index(existing_build_info: IndexBuildInfo | None) -> MagicMock:
    redis_index = MagicMock(spec=RedisIndex)
    redis_index.read_build_info.return_value = existing_build_info
    return redis_index


def test_run_builds_when_no_build_info_exists(monkeypatch):
    # Arrange
    _patch_corpus_and_build_info(monkeypatch)
    redis_index = _fake_redis_index(existing_build_info=None)
    vector_store = MagicMock()

    # Act
    result = CorpusIngestor().run(redis_index, vector_store, rule_catalogue=MagicMock())

    # Assert
    assert result.action == "built"
    vector_store.index.create.assert_not_called()
    vector_store.add_texts.assert_called_once()
    redis_index.write_build_info.assert_called_once()


def test_run_reuses_when_build_info_matches(monkeypatch):
    # Arrange
    _patch_corpus_and_build_info(monkeypatch)
    redis_index = _fake_redis_index(existing_build_info=_build_info())
    vector_store = MagicMock()

    # Act
    result = CorpusIngestor().run(redis_index, vector_store, rule_catalogue=MagicMock())

    # Assert
    assert result.action == "reused"
    vector_store.add_texts.assert_not_called()
    redis_index.write_build_info.assert_not_called()


def test_run_rebuilds_when_build_info_differs(monkeypatch):
    # Arrange
    _patch_corpus_and_build_info(monkeypatch)
    redis_index = _fake_redis_index(existing_build_info=_build_info(corpus_hash="old"))
    vector_store = MagicMock()

    # Act
    result = CorpusIngestor().run(redis_index, vector_store, rule_catalogue=MagicMock())

    # Assert
    assert result.action == "rebuilt"
    vector_store.index.create.assert_called_once_with(overwrite=True, drop=True)
    vector_store.add_texts.assert_called_once()
    redis_index.write_build_info.assert_called_once()


def test_run_rejects_a_concurrent_call_while_another_run_holds_the_lock(monkeypatch):
    # Arrange
    _patch_corpus_and_build_info(monkeypatch)
    redis_index = _fake_redis_index(existing_build_info=None)
    vector_store = MagicMock()

    # Act
    _INGEST_LOCK.acquire()
    try:
        with pytest.raises(IngestionInProgressError):
            CorpusIngestor().run(redis_index, vector_store, rule_catalogue=MagicMock())
    finally:
        _INGEST_LOCK.release()

    # Assert
    vector_store.add_texts.assert_not_called()


def test_run_releases_the_lock_after_completing(monkeypatch):
    # Arrange
    _patch_corpus_and_build_info(monkeypatch)
    redis_index = _fake_redis_index(existing_build_info=None)
    vector_store = MagicMock()

    # Act
    CorpusIngestor().run(redis_index, vector_store, rule_catalogue=MagicMock())

    # Assert
    assert _INGEST_LOCK.acquire(blocking=False)
    _INGEST_LOCK.release()
