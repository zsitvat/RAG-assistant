from unittest.mock import MagicMock

from langchain_core.documents import Document

from app.integrations.redis import RedisIndex
from app.rag.build_info import IndexBuildInfoBuilder
from app.rag.ingest import PolicyCorpusIngestor
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
        PolicyCorpusIngestor, "load_and_chunk", lambda self, rule_catalogue: ([], _fake_chunks())
    )
    monkeypatch.setattr(IndexBuildInfoBuilder, "build", lambda self, *a, **k: _build_info())


def _fake_redis_index(existing_build_info: IndexBuildInfo | None) -> MagicMock:
    redis_index = MagicMock(spec=RedisIndex)
    redis_index.read_build_info.return_value = existing_build_info
    return redis_index


def test_run_builds_when_no_build_info_exists(monkeypatch):
    _patch_corpus_and_build_info(monkeypatch)
    redis_index = _fake_redis_index(existing_build_info=None)
    vector_store = MagicMock()

    result = PolicyCorpusIngestor().run(redis_index, vector_store, rule_catalogue=MagicMock())

    assert result.action == "built"
    vector_store.index.create.assert_not_called()
    vector_store.add_texts.assert_called_once()
    redis_index.write_build_info.assert_called_once()


def test_run_reuses_when_build_info_matches(monkeypatch):
    _patch_corpus_and_build_info(monkeypatch)
    redis_index = _fake_redis_index(existing_build_info=_build_info())
    vector_store = MagicMock()

    result = PolicyCorpusIngestor().run(redis_index, vector_store, rule_catalogue=MagicMock())

    assert result.action == "reused"
    vector_store.add_texts.assert_not_called()
    redis_index.write_build_info.assert_not_called()


def test_run_rebuilds_when_build_info_differs(monkeypatch):
    _patch_corpus_and_build_info(monkeypatch)
    redis_index = _fake_redis_index(existing_build_info=_build_info(corpus_hash="old"))
    vector_store = MagicMock()

    result = PolicyCorpusIngestor().run(redis_index, vector_store, rule_catalogue=MagicMock())

    assert result.action == "rebuilt"
    vector_store.index.create.assert_called_once_with(overwrite=True, drop=True)
    vector_store.add_texts.assert_called_once()
    redis_index.write_build_info.assert_called_once()
