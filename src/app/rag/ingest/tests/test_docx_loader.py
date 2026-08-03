import docx

from app.rag.ingest.docx_loader import DocxMarkdownLoader


def _build_docx(path, title):
    document = docx.Document()
    document.add_paragraph(title, style="Title")
    document.save(str(path))
    return path


def test_lazy_load_yields_documents_in_sorted_filename_order_with_correct_metadata(tmp_path):
    # Arrange
    _build_docx(tmp_path / "02_second.docx", "Second Policy")
    _build_docx(tmp_path / "01_first.docx", "First Policy")

    # Act
    documents = list(DocxMarkdownLoader(tmp_path).lazy_load())

    # Assert
    assert [doc.metadata["doc_id"] for doc in documents] == ["01", "02"]
    assert documents[0].metadata["doc_title"] == "First Policy"
    assert documents[0].metadata["source_path"] == str(tmp_path / "01_first.docx")
    assert documents[0].page_content == "# First Policy"


def test_lazy_load_yields_nothing_for_an_empty_corpus_directory(tmp_path):
    # Act
    documents = list(DocxMarkdownLoader(tmp_path).lazy_load())

    # Assert
    assert documents == []
