from app.rag.ingest.docx_loader import CORPUS_DIR, DocxMarkdownLoader
from app.rules.loader import load_rule_catalogue

CATALOGUE = load_rule_catalogue()


def _document_markdown(doc_id: str) -> str:
    documents = {doc.metadata["doc_id"]: doc for doc in DocxMarkdownLoader(CORPUS_DIR).lazy_load()}
    return documents[doc_id].page_content


def _rule(category: str, rule_id: str):
    return next(r for r in CATALOGUE.categories[category].rules if r.id == rule_id)


def test_commuting_and_mileage_doc_refs_resolve_to_indexed_sections():
    for category in ("commuting", "mileage"):
        for rule in CATALOGUE.categories[category].rules:
            assert rule.doc_ref is not None
            doc_id, section_id = rule.doc_ref.split("#", 1)
            assert section_id in CATALOGUE.documents[doc_id].sections


def test_personal_vehicle_rate_and_flat_monthly_cap_appear_verbatim():
    # Arrange
    rule = _rule("commuting", "R-COMM-02")
    markdown = _document_markdown(rule.doc_ref.split("#", 1)[0])

    # Assert
    assert f"HUF {rule.rate_huf_per_km:g} per kilometre" in markdown
    assert f"{rule.monthly_cap_huf:,}" in markdown


def test_minimum_commuting_distance_appears_verbatim():
    # Arrange
    rule = _rule("commuting", "R-COMM-01")
    markdown = _document_markdown(rule.doc_ref.split("#", 1)[0])

    # Assert
    assert f"at least {rule.min_one_way_km:g} km" in markdown


def test_public_transport_pass_ratio_and_cap_appear_verbatim():
    # Arrange
    rule = _rule("commuting", "R-COMM-03")
    markdown = _document_markdown(rule.doc_ref.split("#", 1)[0])

    # Assert
    assert f"{rule.pass_reimbursement_ratio:.0%}" in markdown
    assert f"{rule.monthly_cap_huf:,}" in markdown


def test_individual_ticket_ratio_and_daily_cap_appear_verbatim():
    # Arrange
    rule = _rule("commuting", "R-COMM-04")
    markdown = _document_markdown(rule.doc_ref.split("#", 1)[0])

    # Assert
    assert f"{rule.ticket_reimbursement_ratio:.0%}" in markdown
    assert f"{rule.daily_cap_huf:,}" in markdown


def test_mileage_rate_appears_verbatim_and_applies_to_every_powertrain():
    # Arrange
    rule = _rule("mileage", "R-MILE-01")
    markdown = _document_markdown(rule.doc_ref.split("#", 1)[0])

    # Assert
    assert f"HUF {rule.rate_huf_per_km:g} per kilometre" in markdown
    assert "same reimbursement rate of HUF 45/km applies to all powertrain types" in markdown
