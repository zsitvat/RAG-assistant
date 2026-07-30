"""Attaches rule/citation metadata to chunks and validates rules.yaml section anchors."""

from langchain_core.documents import Document

from app.rag.errors import IngestionError
from app.rules.model import RuleCatalogue


class RuleMetadataResolver:
    """Attaches `section_id`/`rule_ids`/`categories` to chunks and validates rules.yaml anchors."""

    def __init__(self, rule_catalogue: RuleCatalogue) -> None:
        self._catalogue = rule_catalogue
        self._rule_ids_by_section = self._index_rule_ids(rule_catalogue)

    @staticmethod
    def _index_rule_ids(rule_catalogue: RuleCatalogue) -> dict[tuple[str, str], list[str]]:
        index: dict[tuple[str, str], list[str]] = {}
        for category_rules in rule_catalogue.categories.values():
            for rule in category_rules.rules:
                if rule.doc_ref is None:
                    continue
                doc_id, section_id = rule.doc_ref.split("#", 1)
                index.setdefault((doc_id, section_id), []).append(rule.id)
        return index

    def attach(self, chunks: list[Document]) -> list[Document]:
        for chunk in chunks:
            doc_id = chunk.metadata["doc_id"]
            document = self._catalogue.documents.get(doc_id)
            if document is None:
                raise IngestionError(
                    f"Unknown document identifier {doc_id!r}; not declared in rules.yaml"
                )

            heading = chunk.metadata.get("section")
            section_id = self._resolve_section_id(doc_id, heading)
            chunk.metadata["section_id"] = section_id
            chunk.metadata["rule_ids"] = (
                self._rule_ids_by_section.get((doc_id, section_id), []) if section_id else []
            )
            chunk.metadata["categories"] = list(document.categories)
        return chunks

    def _resolve_section_id(self, doc_id: str, heading: str | None) -> str | None:
        if heading is None:
            return None
        document = self._catalogue.documents.get(doc_id)
        if document is None:
            return None
        for section_id, section in document.sections.items():
            if heading in section.headings:
                return section_id
        return None

    def validate_anchors_resolve(self, chunks: list[Document]) -> None:
        """Rejects rules.yaml section anchors whose heading never appears in the corpus."""
        headings_by_doc: dict[str, set[str]] = {}
        for chunk in chunks:
            heading = chunk.metadata.get("section")
            if heading:
                headings_by_doc.setdefault(chunk.metadata["doc_id"], set()).add(heading)

        errors = [
            f"rules.yaml section '{doc_id}#{section_id}' heading {heading!r} not in the corpus"
            for doc_id, document in self._catalogue.documents.items()
            for section_id, section in document.sections.items()
            for heading in section.headings
            if heading not in headings_by_doc.get(doc_id, set())
        ]
        if errors:
            raise IngestionError("; ".join(errors))
