from app.rag.ingest.docx_loader import CORPUS_DIR, DocxMarkdownLoader
from app.rules.loader import load_rule_catalogue

CATALOGUE = load_rule_catalogue()


def _document_markdown(doc_id: str) -> str:
    documents = {doc.metadata["doc_id"]: doc for doc in DocxMarkdownLoader(CORPUS_DIR).lazy_load()}
    return documents[doc_id].page_content


def test_meal_rule_doc_refs_resolve_to_indexed_sections():
    for rule in CATALOGUE.categories["meal"].rules:
        assert rule.doc_ref is not None
        doc_id, section_id = rule.doc_ref.split("#", 1)
        assert section_id in CATALOGUE.documents[doc_id].sections


def test_meal_per_person_limit_appears_verbatim_in_its_referenced_policy_section():
    rule = next(r for r in CATALOGUE.categories["meal"].rules if r.limit_per_person_huf is not None)
    doc_id, _ = rule.doc_ref.split("#", 1)
    markdown = _document_markdown(doc_id)

    assert f"{rule.limit_per_person_huf:,}" in markdown


def test_meal_excluded_items_appear_verbatim_in_its_referenced_policy_section():
    rule = next(r for r in CATALOGUE.categories["meal"].rules if r.excluded_items)
    doc_id, _ = rule.doc_ref.split("#", 1)
    markdown = _document_markdown(doc_id).lower()

    for item in rule.excluded_items:
        assert item.lower() in markdown
