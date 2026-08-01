from langchain_core.documents import Document

from app.rag.errors import IngestionError
from app.rules.model import RuleCatalogue


class RuleMetadataResolver:
    """Attaches `section_id`/`rule_ids`/`categories` to chunks and validates rules.yaml anchors."""

    def __init__(self, rule_catalogue: RuleCatalogue) -> None:
        """Stores the rule catalogue and indexes rule ids by document section."""
        self._catalogue = rule_catalogue
        self._rule_ids_by_section, self._categories_by_section = self._index_section_metadata(
            rule_catalogue
        )

    @staticmethod
    def _index_section_metadata(
        rule_catalogue: RuleCatalogue,
    ) -> tuple[dict[tuple[str, str], list[str]], dict[tuple[str, str], set[str]]]:
        rule_ids: dict[tuple[str, str], list[str]] = {}
        categories: dict[tuple[str, str], set[str]] = {}

        def index(rule_id: str, doc_ref: str | None, category: str) -> None:
            if doc_ref is None:
                return
            doc_id, section_id = doc_ref.split("#", 1)
            key = (doc_id, section_id)
            if rule_id not in rule_ids.setdefault(key, []):
                rule_ids[key].append(rule_id)
            categories.setdefault(key, set()).add(category)

        for category, category_rules in rule_catalogue.categories.items():
            for rule in category_rules.rules:
                index(rule.id, rule.doc_ref, category)
            if category_rules.required_documents_rule_id:
                index(
                    category_rules.required_documents_rule_id,
                    category_rules.required_documents_doc_ref,
                    category,
                )

        submission = rule_catalogue.submission
        for category in rule_catalogue.categories:
            index(submission.deadline_rule_id, submission.deadline_doc_ref, category)
            index(submission.approval_rule_id, submission.approval_doc_ref, category)
            index(submission.receipt_rule_id, submission.receipt_doc_ref, category)
        return rule_ids, categories

    def attach(self, chunks: list[Document]) -> list[Document]:
        """Attaches section_id, rule_ids and categories metadata to each chunk."""
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
            section_key = (doc_id, section_id) if section_id else None
            chunk.metadata["rule_ids"] = (
                self._rule_ids_by_section.get(section_key, []) if section_key else []
            )
            section_categories = (
                self._categories_by_section.get(section_key, set()) if section_key else set()
            )
            chunk.metadata["categories"] = sorted(set(document.categories) | section_categories)
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

    def validate_categories_reachable(self, chunks: list[Document]) -> None:
        """Rejects categories or configured rules that cannot be retrieved as indexed evidence."""
        reachable = {category for chunk in chunks for category in chunk.metadata["categories"]}
        errors = [category for category in self._catalogue.categories if category not in reachable]
        indexed_rule_categories = {
            (rule_id, category)
            for chunk in chunks
            for rule_id in chunk.metadata.get("rule_ids", [])
            for category in chunk.metadata["categories"]
        }
        for category, category_rules in self._catalogue.categories.items():
            expected_ids = [rule.id for rule in category_rules.rules if rule.doc_ref]
            if (
                category_rules.required_documents_rule_id
                and category_rules.required_documents_doc_ref
            ):
                expected_ids.append(category_rules.required_documents_rule_id)
            errors.extend(
                f"{category}:{rule_id}"
                for rule_id in expected_ids
                if (rule_id, category) not in indexed_rule_categories
            )
        if errors:
            raise IngestionError(f"unreachable category or rule evidence: {', '.join(errors)}")
