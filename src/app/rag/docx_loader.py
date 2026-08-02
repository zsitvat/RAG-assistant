from collections.abc import Iterator
from pathlib import Path

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

from app.rag.docx_converter import DocxToMarkdownConverter

CORPUS_DIR = Path(".docs/sources/en")


class DocxMarkdownLoader(BaseLoader):
    """LangChain loader converting the `.docx` corpus to Markdown."""

    def __init__(
        self, corpus_dir: Path = CORPUS_DIR, converter: DocxToMarkdownConverter | None = None
    ) -> None:
        """Stores the corpus directory and docx-to-Markdown converter."""
        self._corpus_dir = corpus_dir
        self._converter = converter or DocxToMarkdownConverter()

    def lazy_load(self) -> Iterator[Document]:
        """Yields each policy docx in the corpus directory as a Markdown Document."""
        for path in sorted(self._corpus_dir.glob("*.docx")):
            doc_id = path.name[:2]
            title, markdown = self._converter.convert(path)
            yield Document(
                page_content=markdown,
                metadata={"doc_id": doc_id, "doc_title": title, "source_path": str(path)},
            )
