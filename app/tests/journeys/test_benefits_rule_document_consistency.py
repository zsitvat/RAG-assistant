from app.rag.docx_loader import CORPUS_DIR, DocxMarkdownLoader
from app.rules.loader import load_rule_catalogue

CATALOGUE = load_rule_catalogue()


def _document_markdown(doc_id: str) -> str:
    documents = {doc.metadata["doc_id"]: doc for doc in DocxMarkdownLoader(CORPUS_DIR).lazy_load()}
    return documents[doc_id].page_content


def test_benefits_rule_doc_refs_resolve_to_indexed_sections():
    for rule in CATALOGUE.categories["benefits"].rules:
        assert rule.doc_ref is not None
        doc_id, section_id = rule.doc_ref.split("#", 1)
        assert section_id in CATALOGUE.documents[doc_id].sections


def test_benefits_annual_allowances_appear_verbatim_in_their_section():
    markdown = _document_markdown("05")

    for rule in CATALOGUE.categories["benefits"].rules:
        if rule.annual_budget_huf is None:
            continue
        assert f"{rule.annual_budget_huf:,}" in markdown


def test_benefits_tenure_requirement_appears_in_the_eligibility_section():
    markdown = _document_markdown("05").lower()

    tenure_rule = next(
        rule
        for rule in CATALOGUE.categories["benefits"].rules
        if rule.eligible_after_months is not None
    )
    assert tenure_rule.eligible_after_months == 6
    assert "six months" in markdown


def test_benefits_no_carry_over_rule_appears_in_the_eligibility_section():
    markdown = _document_markdown("05").lower()

    carry_over_rule = next(
        rule for rule in CATALOGUE.categories["benefits"].rules if rule.carry_over is not None
    )
    assert carry_over_rule.carry_over is False
    assert "not be carried forward" in markdown
