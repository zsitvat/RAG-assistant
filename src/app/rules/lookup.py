from collections.abc import Callable, Iterable

from app.rules.model import RuleDefinition


def first_matching(
    rules: Iterable[RuleDefinition], predicate: Callable[[RuleDefinition], bool]
) -> RuleDefinition | None:
    """Returns the first rule satisfying predicate, or None."""
    return next((rule for rule in rules if predicate(rule)), None)
