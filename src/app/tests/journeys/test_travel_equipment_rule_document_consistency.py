from app.rag.ingest.docx_loader import CORPUS_DIR, DocxMarkdownLoader
from app.rules.loader import load_rule_catalogue

CATALOGUE = load_rule_catalogue()


def _document_markdown(doc_id: str) -> str:
    documents = {doc.metadata["doc_id"]: doc for doc in DocxMarkdownLoader(CORPUS_DIR).lazy_load()}
    return documents[doc_id].page_content


def _rule(category: str, rule_id: str):
    return next(r for r in CATALOGUE.categories[category].rules if r.id == rule_id)


def test_travel_and_equipment_rule_doc_refs_resolve_to_indexed_sections():
    for category in ("travel", "equipment"):
        for rule in CATALOGUE.categories[category].rules:
            assert rule.doc_ref is not None
            doc_id, section_id = rule.doc_ref.split("#", 1)
            assert section_id in CATALOGUE.documents[doc_id].sections


def test_travel_department_head_threshold_appears_verbatim_in_its_section():
    rule = _rule("travel", "R-TRAVEL-01")
    doc_id, _ = rule.doc_ref.split("#", 1)
    markdown = _document_markdown(doc_id)

    assert f"{rule.department_head_approval_above_huf:,}" in markdown


def test_travel_accommodation_limits_appear_verbatim_in_their_section():
    rule = _rule("travel", "R-TRAVEL-02")
    doc_id, _ = rule.doc_ref.split("#", 1)
    markdown = _document_markdown(doc_id)

    assert f"{rule.accommodation_limit_huf_per_night['domestic']:,}" in markdown
    assert f"{rule.accommodation_limit_huf_per_night['international']:,}" in markdown


def test_travel_meal_per_diem_limits_appear_verbatim_in_their_section():
    rule = _rule("travel", "R-TRAVEL-03")
    doc_id, _ = rule.doc_ref.split("#", 1)
    markdown = _document_markdown(doc_id)

    assert f"{rule.meal_per_diem_huf['domestic']:,}" in markdown
    assert f"{rule.meal_per_diem_huf['international']:,}" in markdown


def test_travel_excluded_expense_types_appear_verbatim_in_their_section():
    rule = _rule("travel", "R-TRAVEL-04")
    doc_id, _ = rule.doc_ref.split("#", 1)
    markdown = _document_markdown(doc_id).lower()

    for item in rule.excluded_items:
        assert item.lower() in markdown


def test_equipment_approval_tiers_appear_verbatim_in_their_section():
    rule = _rule("equipment", "R-EQUIP-01")
    doc_id, _ = rule.doc_ref.split("#", 1)
    markdown = _document_markdown(doc_id)

    non_unlimited_tiers = [tier for tier in rule.approval_tiers if tier.max_huf is not None]
    for tier in non_unlimited_tiers:
        assert f"{tier.max_huf:,}" in markdown


def test_equipment_business_use_requirement_appears_in_its_section():
    rule = _rule("equipment", "R-EQUIP-02")
    doc_id, _ = rule.doc_ref.split("#", 1)
    markdown = _document_markdown(doc_id).lower()

    assert rule.business_use_required is True
    assert "personal use is not reimbursable" in markdown
