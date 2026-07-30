"""Converts one policy `.docx` file to Markdown, preserving headings, lists and tables."""

from pathlib import Path

import docx
from docx.table import Table as DocxTable
from docx.text.paragraph import Paragraph as DocxParagraph


class DocxToMarkdownConverter:
    """Converts one policy `.docx` file to Markdown, preserving headings, lists and tables."""

    _HEADING_STYLE_TO_MARKER = {"Heading 1": "#", "Heading 2": "##", "Heading 3": "###"}
    _LIST_STYLE_TO_MARKER = {"List Bullet": "-", "List Number": "1."}

    def convert(self, path: Path) -> tuple[str, str]:
        """Returns (title, markdown)."""
        document = docx.Document(str(path))
        title = path.stem
        blocks: list[str] = []
        for item in document.iter_inner_content():
            if isinstance(item, DocxParagraph):
                if item.style and item.style.name == "Title" and item.text.strip():
                    title = item.text.strip()
                markdown_line = self._paragraph_to_markdown(item)
                if markdown_line is not None:
                    blocks.append(markdown_line)
            elif isinstance(item, DocxTable):
                table_markdown = self._table_to_markdown(item)
                if table_markdown:
                    blocks.append(table_markdown)
        return title, "\n\n".join(blocks)

    def _paragraph_to_markdown(self, paragraph: DocxParagraph) -> str | None:
        text = paragraph.text.strip()
        if not text:
            return None
        style = paragraph.style.name if paragraph.style else ""
        if style == "Title":
            return f"# {text}"
        heading_marker = self._HEADING_STYLE_TO_MARKER.get(style)
        if heading_marker:
            return f"{heading_marker} {text}"
        list_marker = self._LIST_STYLE_TO_MARKER.get(style)
        if list_marker:
            return f"{list_marker} {text}"
        return text

    @staticmethod
    def _table_to_markdown(table: DocxTable) -> str:
        rows = [[cell.text.strip().replace("\n", " ") for cell in row.cells] for row in table.rows]
        rows = [row for row in rows if any(row)]
        if not rows:
            return ""
        header, *body = rows
        lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
        lines.extend("| " + " | ".join(row) + " |" for row in body)
        return "\n".join(lines)
