from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import ValidationError

from app.rules.model import RuleCatalogue

DEFAULT_RULES_PATH = Path("config/rules.yaml")


class RuleCatalogueError(RuntimeError):
    """Raised when rules.yaml is missing or fails validation."""


def load_rule_catalogue(path: Path = DEFAULT_RULES_PATH) -> RuleCatalogue:
    """Loads and validates the rule catalogue from the given YAML file."""
    if not path.exists():
        raise RuleCatalogueError(f"Rule catalogue not found at '{path.resolve()}'")

    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    try:
        return RuleCatalogue.model_validate(raw)
    except ValidationError as exc:
        raise RuleCatalogueError(f"Rule catalogue at '{path.resolve()}' is invalid: {exc}") from exc


@lru_cache
def get_rule_catalogue() -> RuleCatalogue:
    """Returns the cached rule catalogue loaded from the default path."""
    return load_rule_catalogue()
