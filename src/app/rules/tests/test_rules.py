import pytest

from app.rules.loader import RuleCatalogueError, get_rule_catalogue, load_rule_catalogue
from app.rules.model import RuleCatalogue

EXPECTED_DOCUMENT_IDS = {"00", "01", "02", "03", "04", "05", "06", "07"}


def test_load_rule_catalogue_parses_the_repository_rules_yaml():
    catalogue = load_rule_catalogue()
    assert set(catalogue.documents) == EXPECTED_DOCUMENT_IDS
    assert catalogue.categories["meal"].rules[0].limit_per_person_huf == 15000


def test_missing_file_raises_actionable_error(tmp_path):
    with pytest.raises(RuleCatalogueError, match="not found"):
        load_rule_catalogue(tmp_path / "does-not-exist.yaml")


def test_malformed_yaml_raises_actionable_error(tmp_path):
    path = tmp_path / "rules.yaml"
    path.write_text("version: not-a-number\n")
    with pytest.raises(RuleCatalogueError, match="invalid"):
        load_rule_catalogue(path)


def _minimal_catalogue(**overrides) -> dict:
    base = {
        "version": 1,
        "currency": "HUF",
        "fx_rates_fixed": {"EUR": 400},
        "documents": {"01": {"categories": ["meal"], "sections": {"a": {"headings": ["A"]}}}},
        "submission": {"deadline_days": 30, "approval_tiers": [{"max_huf": None, "approver": "x"}]},
        "categories": {"meal": {"rules": [], "required_documents": []}},
    }
    base.update(overrides)
    return base


def test_document_categories_must_not_be_empty():
    data = _minimal_catalogue(documents={"01": {"categories": [], "sections": {}}})
    with pytest.raises(ValueError, match="must not be empty"):
        RuleCatalogue.model_validate(data)


def test_rule_doc_ref_must_resolve_to_a_declared_section():
    data = _minimal_catalogue(
        categories={
            "meal": {
                "rules": [{"id": "R-1", "doc_ref": "01#missing-section"}],
                "required_documents": [],
            }
        }
    )
    with pytest.raises(ValueError, match="unresolved section"):
        RuleCatalogue.model_validate(data)


def test_rule_doc_ref_must_point_to_a_known_document():
    data = _minimal_catalogue(
        categories={
            "meal": {
                "rules": [{"id": "R-1", "doc_ref": "99#a"}],
                "required_documents": [],
            }
        }
    )
    with pytest.raises(ValueError, match="unknown document"):
        RuleCatalogue.model_validate(data)


def test_duplicate_rule_ids_are_rejected():
    data = _minimal_catalogue(
        categories={
            "meal": {
                "rules": [{"id": "R-1", "doc_ref": "01#a"}, {"id": "R-1", "doc_ref": "01#a"}],
                "required_documents": [],
            }
        }
    )
    with pytest.raises(ValueError, match="duplicate rule id"):
        RuleCatalogue.model_validate(data)


def test_get_rule_catalogue_is_cached():
    assert get_rule_catalogue() is get_rule_catalogue()
