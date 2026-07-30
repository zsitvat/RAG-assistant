from unittest.mock import MagicMock

from langchain_core.documents import Document

from app.integrations.redis import RedisIndexLifecycle
from app.rag.ingest import PolicyCorpusIngestor
from app.rag.manifest import CorpusManifestBuilder
from app.rag.model import CorpusManifest


def _manifest(**overrides) -> CorpusManifest:
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
    return CorpusManifest(**base)


def _fake_chunks() -> list[Document]:
    return [
        Document(
            page_content="Meals are capped at 15,000 HUF.",
            metadata={"doc_id": "01", "chunk_index": 0, "categories": ["meal"]},
        )
    ]


def _patch(monkeypatch, existing_manifest: CorpusManifest | None) -> None:
    monkeypatch.setattr(
        PolicyCorpusIngestor, "load_and_chunk", lambda self, rule_catalogue: ([], _fake_chunks())
    )
    monkeypatch.setattr(CorpusManifestBuilder, "build", lambda self, *a, **k: _manifest())
    monkeypatch.setattr(RedisIndexLifecycle, "read_manifest", lambda self: existing_manifest)


def test_run_builds_when_no_manifest_exists(monkeypatch):
    _patch(monkeypatch, existing_manifest=None)
    write_manifest_mock = MagicMock()
    monkeypatch.setattr(RedisIndexLifecycle, "write_manifest", write_manifest_mock)
    vector_store = MagicMock()

    result = PolicyCorpusIngestor().run(MagicMock(), vector_store, rule_catalogue=MagicMock())

    assert result.action == "built"
    vector_store.index.create.assert_not_called()
    vector_store.add_texts.assert_called_once()
    write_manifest_mock.assert_called_once()


def test_run_reuses_when_manifest_matches(monkeypatch):
    _patch(monkeypatch, existing_manifest=_manifest())
    write_manifest_mock = MagicMock()
    monkeypatch.setattr(RedisIndexLifecycle, "write_manifest", write_manifest_mock)
    vector_store = MagicMock()

    result = PolicyCorpusIngestor().run(MagicMock(), vector_store, rule_catalogue=MagicMock())

    assert result.action == "reused"
    vector_store.add_texts.assert_not_called()
    write_manifest_mock.assert_not_called()


def test_run_rebuilds_when_manifest_differs(monkeypatch):
    _patch(monkeypatch, existing_manifest=_manifest(corpus_hash="old"))
    write_manifest_mock = MagicMock()
    monkeypatch.setattr(RedisIndexLifecycle, "write_manifest", write_manifest_mock)
    vector_store = MagicMock()

    result = PolicyCorpusIngestor().run(MagicMock(), vector_store, rule_catalogue=MagicMock())

    assert result.action == "rebuilt"
    vector_store.index.create.assert_called_once_with(overwrite=True, drop=True)
    vector_store.add_texts.assert_called_once()
    write_manifest_mock.assert_called_once()
