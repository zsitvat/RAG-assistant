from unittest.mock import MagicMock

from langchain_core.documents import Document

import app.rag.ingest as ingest_module
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


def _patch_corpus(monkeypatch) -> None:
    monkeypatch.setattr(
        ingest_module, "load_and_chunk_corpus", lambda *a, **k: ([], _fake_chunks())
    )
    monkeypatch.setattr(ingest_module, "build_manifest", lambda *a, **k: _manifest())


def test_run_ingest_builds_when_no_manifest_exists(monkeypatch):
    _patch_corpus(monkeypatch)
    monkeypatch.setattr(ingest_module, "read_manifest", lambda client: None)
    write_manifest_mock = MagicMock()
    monkeypatch.setattr(ingest_module, "write_manifest", write_manifest_mock)
    drop_index_mock = MagicMock()
    monkeypatch.setattr(ingest_module, "drop_chunk_index", drop_index_mock)
    vector_store = MagicMock()

    result = ingest_module.run_ingest(MagicMock(), vector_store, rule_catalogue=MagicMock())

    assert result.action == "built"
    drop_index_mock.assert_not_called()
    vector_store.add_texts.assert_called_once()
    write_manifest_mock.assert_called_once()


def test_run_ingest_reuses_when_manifest_matches(monkeypatch):
    _patch_corpus(monkeypatch)
    monkeypatch.setattr(ingest_module, "read_manifest", lambda client: _manifest())
    write_manifest_mock = MagicMock()
    monkeypatch.setattr(ingest_module, "write_manifest", write_manifest_mock)
    vector_store = MagicMock()

    result = ingest_module.run_ingest(MagicMock(), vector_store, rule_catalogue=MagicMock())

    assert result.action == "reused"
    vector_store.add_texts.assert_not_called()
    write_manifest_mock.assert_not_called()


def test_run_ingest_rebuilds_when_manifest_differs(monkeypatch):
    _patch_corpus(monkeypatch)
    monkeypatch.setattr(ingest_module, "read_manifest", lambda client: _manifest(corpus_hash="old"))
    write_manifest_mock = MagicMock()
    monkeypatch.setattr(ingest_module, "write_manifest", write_manifest_mock)
    drop_index_mock = MagicMock()
    monkeypatch.setattr(ingest_module, "drop_chunk_index", drop_index_mock)
    vector_store = MagicMock()

    result = ingest_module.run_ingest(MagicMock(), vector_store, rule_catalogue=MagicMock())

    assert result.action == "rebuilt"
    drop_index_mock.assert_called_once()
    vector_store.add_texts.assert_called_once()
    write_manifest_mock.assert_called_once()
