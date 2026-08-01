from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
SHORT_SECTION_MERGE_THRESHOLD = 200
HEADER_LEVELS = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]


class MarkdownChunker:
    """Header-aware, size-guarded Markdown chunker that keeps table blocks atomic."""

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        short_section_threshold: int = SHORT_SECTION_MERGE_THRESHOLD,
    ) -> None:
        """Stores chunking sizes and builds the header and character splitters."""
        self._short_section_threshold = short_section_threshold
        self._header_splitter = MarkdownHeaderTextSplitter(HEADER_LEVELS, strip_headers=True)
        self._char_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def chunk(self, doc_id: str, doc_title: str, source_path: str, markdown: str) -> list[Document]:
        """Splits a Markdown document into header-aware, size-guarded chunks."""
        header_sections = self._header_splitter.split_text(markdown)
        raw_sections = [
            (self._heading_from_metadata(section.metadata), section.page_content)
            for section in header_sections
            if section.page_content.strip()
        ]
        merged_sections = self._merge_short_sections(raw_sections)

        chunks: list[Document] = []
        for heading, section_text in merged_sections:
            segments = self._split_prose_and_tables(section_text)
            for chunk_text in self._guard_split_segments(segments):
                chunks.append(
                    Document(
                        page_content=chunk_text,
                        metadata={
                            "doc_id": doc_id,
                            "doc_title": doc_title,
                            "section": heading,
                            "chunk_index": len(chunks),
                            "source_path": source_path,
                        },
                    )
                )
        return chunks

    @staticmethod
    def _heading_from_metadata(metadata: dict) -> str | None:
        for key in ("Header 3", "Header 2", "Header 1"):
            if key in metadata:
                return metadata[key]
        return None

    def _merge_short_sections(
        self, sections: list[tuple[str | None, str]]
    ) -> list[tuple[str | None, str]]:
        """Merges sections shorter than the threshold into the following sibling.

        A trailing short section has no following sibling to merge into, so it is kept as
        its own chunk under its own heading rather than absorbed into an earlier section.
        """
        merged: list[tuple[str | None, str]] = []
        pending_heading: str | None = None
        pending_text = ""
        for heading, content in sections:
            combined = f"{pending_text}\n\n{content}".strip() if pending_text else content
            if len(combined) < self._short_section_threshold:
                pending_heading, pending_text = heading, combined
                continue
            merged.append((heading, combined))
            pending_heading, pending_text = None, ""
        if pending_text:
            merged.append((pending_heading, pending_text))
        return merged

    @staticmethod
    def _is_table_line(line: str) -> bool:
        return line.strip().startswith("|")

    def _split_prose_and_tables(self, text: str) -> list[tuple[bool, str]]:
        """Splits section text into ordered segments, keeping table blocks atomic."""
        segments: list[tuple[bool, str]] = []
        buffer: list[str] = []
        buffer_is_table = False

        def flush() -> None:
            """Appends the buffered lines as a segment and clears the buffer."""
            segment = "\n".join(buffer).strip()
            if segment:
                segments.append((buffer_is_table, segment))
            buffer.clear()

        for line in text.split("\n"):
            line_is_table = self._is_table_line(line)
            if buffer and line_is_table != buffer_is_table:
                flush()
            buffer_is_table = line_is_table
            buffer.append(line)
        flush()
        return segments

    def _guard_split_segments(self, segments: list[tuple[bool, str]]) -> list[str]:
        chunks: list[str] = []
        for is_table, text in segments:
            chunks.append(text) if is_table else chunks.extend(self._char_splitter.split_text(text))
        return chunks
