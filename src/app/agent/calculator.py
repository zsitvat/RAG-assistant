import math

from app.agent.model import CalculationResult, ExpenseClaim
from app.rules.model import Category, RuleCatalogue, RuleDefinition

COMMUTING_TRANSIT_CARD = "pass"
COMMUTING_TICKET = "ticket"


class CalculationInputError(RuntimeError):
    """Raised when the claim is missing a field the calculator needs for its category."""


def _round_half_up(value: float) -> int:
    """Rounds with floor(value + 0.5)."""

    return math.floor(value + 0.5)


class ReimbursementCalculator:
    """Computes the reimbursable amount for a validated ExpenseClaim (design §7.1)."""

    def __init__(self, rules: RuleCatalogue) -> None:
        """Stores the validated rule catalogue."""

        self._rules = rules

    def calculate(self, claim: ExpenseClaim) -> CalculationResult:
        """Dispatches the claim to its category-specific calculation."""

        if claim.category is None:
            raise CalculationInputError("claim.category is required to calculate a reimbursement")
        handler = {
            "meal": self._calculate_meal,
            "travel": self._calculate_travel,
            "mileage": self._calculate_mileage,
            "commuting": self._calculate_commuting,
            "equipment": self._calculate_equipment,
            "benefits": self._calculate_benefits,
        }.get(claim.category)
        if handler is None:
            raise CalculationInputError(
                f"no calculation is defined for category {claim.category!r}"
            )
        return handler(claim)

    def calculate_both_directions(self, claim: ExpenseClaim) -> dict[bool, CalculationResult]:
        """Calculates the result for both the one-way and round-trip reading of the distance."""

        return {
            one_way: self.calculate(claim.model_copy(update={"distance_is_one_way": one_way}))
            for one_way in (True, False)
        }

    def _rules_for(self, category: Category) -> list[RuleDefinition]:
        """Returns the rules configured for a category."""

        return self._rules.categories[category].rules

    def _meal_limit_rule(self) -> RuleDefinition | None:
        """Returns the configured meal limit rule when present."""

        return next(
            (rule for rule in self._rules_for("meal") if rule.limit_per_person_huf is not None),
            None,
        )

    def _first_rule_with(self, category: Category, field_name: str) -> RuleDefinition:
        """Returns the first rule in the category with a value for field_name, or raises."""

        rule = next(
            (r for r in self._rules_for(category) if getattr(r, field_name) is not None), None
        )
        if rule is None:
            raise CalculationInputError(
                f"no {field_name} configured for category {category!r} in the rule catalogue"
            )
        return rule

    @staticmethod
    def _require(claim: ExpenseClaim, *fields: str) -> None:
        """Rejects a claim when required calculation fields are missing."""

        missing = [field for field in fields if getattr(claim, field) is None]
        if missing:
            raise CalculationInputError(
                f"missing required fields for {claim.category}: {', '.join(missing)}"
            )

    def _calculate_meal(self, claim: ExpenseClaim) -> CalculationResult:
        """Calculates min(amount - excluded, per-person limit * headcount)."""

        self._require(claim, "amount_huf", "headcount")
        non_reimbursable = claim.non_reimbursable_amount or 0.0
        base = max(0.0, claim.amount_huf - non_reimbursable)
        rule = self._meal_limit_rule()
        if rule is None:
            return CalculationResult(
                amount_huf=_round_half_up(base),
                warnings=["no meal limit is configured; reimbursement confidence is reduced"],
            )
        cap = rule.limit_per_person_huf * claim.headcount
        reimbursable = min(base, cap)
        warnings = []
        if non_reimbursable > 0:
            warnings.append(
                f"{_round_half_up(non_reimbursable)} HUF of explicitly excluded items (e.g. "
                "alcohol, tobacco, tips) was deducted before applying the per-person cap"
            )
        return CalculationResult(
            amount_huf=_round_half_up(reimbursable),
            cap_huf=_round_half_up(cap),
            excess_huf=_round_half_up(max(0.0, base - cap)),
            warnings=warnings,
        )

    def _calculate_travel(self, claim: ExpenseClaim) -> CalculationResult:
        """Calculates min(amount, subtype cap), or the full amount without a cap."""

        self._require(claim, "amount_huf", "expense_type")
        cap = self._travel_cap(claim.expense_type)
        reimbursable = min(claim.amount_huf, cap) if cap is not None else claim.amount_huf
        return CalculationResult(
            amount_huf=_round_half_up(reimbursable),
            cap_huf=_round_half_up(cap) if cap is not None else None,
            excess_huf=_round_half_up(max(0.0, claim.amount_huf - cap)) if cap is not None else 0,
            warnings=[] if cap is not None else [f"no policy cap found for {claim.expense_type!r}"],
        )

    def _travel_cap(self, expense_type: str) -> float | None:
        """Returns the configured cap for a travel subtype."""

        tiers = {
            "accommodation_domestic": ("accommodation_limit_huf_per_night", "domestic"),
            "accommodation_international": ("accommodation_limit_huf_per_night", "international"),
            "meal_per_diem_domestic": ("meal_per_diem_huf", "domestic"),
            "meal_per_diem_international": ("meal_per_diem_huf", "international"),
        }
        tier = tiers.get(expense_type)
        if tier is None:
            return None
        field_name, key = tier
        for rule in self._rules_for("travel"):
            table = getattr(rule, field_name)
            if table and key in table:
                return table[key]
        return None

    def _calculate_mileage(self, claim: ExpenseClaim) -> CalculationResult:
        """Calculates distance * (2 when one-way else 1) * rate."""

        self._require(claim, "distance_km", "distance_is_one_way")
        rule = next(r for r in self._rules_for("mileage") if r.rate_huf_per_km is not None)
        km = claim.distance_km * (2 if claim.distance_is_one_way else 1)
        amount = km * rule.rate_huf_per_km
        return CalculationResult(amount_huf=_round_half_up(amount))

    def _calculate_commuting(self, claim: ExpenseClaim) -> CalculationResult:
        """Dispatches to the pass, ticket or personal-vehicle commuting calculation."""

        if claim.expense_type == COMMUTING_TRANSIT_CARD:
            return self._calculate_commuting_pass(claim)
        if claim.expense_type == COMMUTING_TICKET:
            return self._calculate_commuting_ticket(claim)
        return self._calculate_commuting_vehicle(claim)

    def _calculate_commuting_vehicle(self, claim: ExpenseClaim) -> CalculationResult:
        """Calculates min(office-day round-trip distance * rate, flat monthly cap)."""

        self._require(claim, "distance_km", "distance_is_one_way", "commute_days_per_month")
        rule = self._first_rule_with("commuting", "rate_huf_per_km")
        km_per_day = claim.distance_km * (2 if claim.distance_is_one_way else 1)
        monthly_km = km_per_day * claim.commute_days_per_month
        raw_amount = monthly_km * rule.rate_huf_per_km
        cap = rule.monthly_cap_huf
        return CalculationResult(
            amount_huf=_round_half_up(min(raw_amount, cap)),
            cap_huf=cap,
            excess_huf=_round_half_up(max(0.0, raw_amount - cap)),
        )

    def _calculate_commuting_pass(self, claim: ExpenseClaim) -> CalculationResult:
        """Calculates min(pass price * reimbursement ratio, monthly cap)."""

        self._require(claim, "amount_huf")
        rule = self._first_rule_with("commuting", "pass_reimbursement_ratio")
        raw_amount = claim.amount_huf * rule.pass_reimbursement_ratio
        cap = rule.monthly_cap_huf
        return CalculationResult(
            amount_huf=_round_half_up(min(raw_amount, cap)),
            cap_huf=cap,
            excess_huf=_round_half_up(max(0.0, raw_amount - cap)),
        )

    def _calculate_commuting_ticket(self, claim: ExpenseClaim) -> CalculationResult:
        """Calculates min(ticket spend * reimbursement ratio, daily cap * office days)."""

        self._require(claim, "amount_huf", "commute_days_per_month")
        rule = self._first_rule_with("commuting", "ticket_reimbursement_ratio")
        raw_amount = claim.amount_huf * rule.ticket_reimbursement_ratio
        cap = rule.daily_cap_huf * claim.commute_days_per_month
        return CalculationResult(
            amount_huf=_round_half_up(min(raw_amount, cap)),
            cap_huf=cap,
            excess_huf=_round_half_up(max(0.0, raw_amount - cap)),
        )

    def _calculate_equipment(self, claim: ExpenseClaim) -> CalculationResult:
        """Calculates the reimbursable amount as the submitted amount."""

        self._require(claim, "amount_huf")
        return CalculationResult(amount_huf=_round_half_up(claim.amount_huf))

    def _calculate_benefits(self, claim: ExpenseClaim) -> CalculationResult:
        """Calculates min(amount, max(0, budget - used)) * reimbursement ratio."""

        self._require(claim, "amount_huf", "annual_budget_used_huf")
        rule = self._benefit_rule(claim.expense_type)
        remaining = rule.annual_budget_huf - claim.annual_budget_used_huf
        reimbursable = min(claim.amount_huf, max(0.0, remaining)) * rule.reimbursement_ratio
        return CalculationResult(
            amount_huf=_round_half_up(reimbursable),
            cap_huf=_round_half_up(max(0.0, remaining)),
            excess_huf=_round_half_up(max(0.0, claim.amount_huf - remaining)),
        )

    def _benefit_rule(self, benefit_type: str | None) -> RuleDefinition:
        """Returns the rule configured for a benefit subtype."""

        rules = self._rules_for("benefits")
        if benefit_type is not None:
            for rule in rules:
                if rule.benefit_type == benefit_type:
                    return rule
        raise CalculationInputError(
            f"no benefits rule found for benefit_type {benefit_type!r}; expected one of "
            f"{[r.benefit_type for r in rules]}"
        )
