"""Narrow deterministic pair decisions with abstention as the default."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

type DecisionStatus = Literal["AUTO_SAME", "NEEDS_FURTHER_RESOLUTION"]
type DecisionRule = Literal[
    "legal_suffix_format_equivalence",
    "legal_suffix_addition_equivalence",
    "whitespace_only_equivalence",
    "unique_source_truncation_equivalence",
    "no_deterministic_equivalence",
]
type PositiveDecisionRule = Literal[
    "legal_suffix_format_equivalence",
    "legal_suffix_addition_equivalence",
    "whitespace_only_equivalence",
    "unique_source_truncation_equivalence",
]
type NumericRelation = Literal["none", "same", "one_sided", "conflict"]

AUTO_SAME: Final = "AUTO_SAME"
NEEDS_FURTHER_RESOLUTION: Final = "NEEDS_FURTHER_RESOLUTION"
NO_DETERMINISTIC_EQUIVALENCE: Final = "no_deterministic_equivalence"
RULE_PRECEDENCE: Final[tuple[PositiveDecisionRule, ...]] = (
    "legal_suffix_format_equivalence",
    "legal_suffix_addition_equivalence",
    "whitespace_only_equivalence",
    "unique_source_truncation_equivalence",
)
RULE_EVIDENCE: Final[dict[PositiveDecisionRule, str]] = {
    "legal_suffix_format_equivalence": "terminal_legal_suffix_formatting_only",
    "legal_suffix_addition_equivalence": "terminal_legal_suffix_addition_or_removal_only",
    "whitespace_only_equivalence": "whitespace_boundary_difference_only",
    "unique_source_truncation_equivalence": "unique_exact_source_truncation_continuation",
}


@dataclass(frozen=True, slots=True)
class TerminalLegalSuffix:
    """One terminal designator recognized by existing normalization configuration."""

    canonical: tuple[str, ...]
    variant: tuple[str, ...]
    base: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TruncationRelation:
    """An exact source-boundary prefix continuation, before uniqueness is known."""

    truncated_key: str
    longer_key: str


@dataclass(frozen=True, slots=True)
class PairDecision:
    """One selected decision plus all positive deterministic rule evidence."""

    status: DecisionStatus
    rule: DecisionRule
    evidence: str
    supporting_rules: tuple[PositiveDecisionRule, ...]


def _terminal_legal_suffix(
    name: str, aliases: dict[str, tuple[str, ...]]
) -> TerminalLegalSuffix | None:
    """Parse only suffix variants already recognized by normalization configuration."""
    tokens = tuple(name.split())
    for canonical, variants in aliases.items():
        canonical_tokens = tuple(canonical.upper().split())
        for variant in variants:
            variant_tokens = tuple(variant.upper().split())
            if (
                variant_tokens
                and len(tokens) > len(variant_tokens)
                and tokens[-len(variant_tokens) :] == variant_tokens
            ):
                return TerminalLegalSuffix(
                    canonical=canonical_tokens,
                    variant=variant_tokens,
                    base=tokens[: -len(variant_tokens)],
                )
    return None


def legal_suffix_format_equivalent(
    name_a: str,
    name_b: str,
    aliases: dict[str, tuple[str, ...]],
) -> bool:
    """Accept only formatting variants of the same configured terminal suffix."""
    suffix_a = _terminal_legal_suffix(name_a, aliases)
    suffix_b = _terminal_legal_suffix(name_b, aliases)
    return _parsed_suffix_format_equivalent(suffix_a, suffix_b)


def _parsed_suffix_format_equivalent(
    suffix_a: TerminalLegalSuffix | None,
    suffix_b: TerminalLegalSuffix | None,
) -> bool:
    return (
        suffix_a is not None
        and suffix_b is not None
        and suffix_a.canonical == suffix_b.canonical
        and suffix_a.base == suffix_b.base
        and suffix_a.variant != suffix_b.variant
    )


def legal_suffix_addition_equivalent(
    name_a: str,
    name_b: str,
    aliases: dict[str, tuple[str, ...]],
) -> bool:
    """Accept exact base text plus or minus one configured terminal legal suffix."""
    suffix_a = _terminal_legal_suffix(name_a, aliases)
    suffix_b = _terminal_legal_suffix(name_b, aliases)
    return _parsed_suffix_addition_equivalent(name_a, name_b, suffix_a, suffix_b)


def _parsed_suffix_addition_equivalent(
    name_a: str,
    name_b: str,
    suffix_a: TerminalLegalSuffix | None,
    suffix_b: TerminalLegalSuffix | None,
) -> bool:
    if (suffix_a is None) == (suffix_b is None):
        return False
    suffixed = suffix_a if suffix_a is not None else suffix_b
    assert suffixed is not None
    unsuffixed_name = name_b if suffix_a is not None else name_a
    return " ".join(suffixed.base) == unsuffixed_name


def whitespace_only_equivalent(
    name_a: str,
    name_b: str,
    numeric_relation: NumericRelation,
    minimum_compact_length: int,
) -> bool:
    """Accept only identical character sequences separated by different spaces."""
    compact_a = name_a.replace(" ", "")
    compact_b = name_b.replace(" ", "")
    return (
        name_a != name_b
        and len(compact_a) >= minimum_compact_length
        and compact_a == compact_b
        and numeric_relation in {"none", "same"}
    )


def exact_source_truncation_relation(
    *,
    key_a: str,
    key_b: str,
    name_a: str,
    name_b: str,
    numeric_relation: NumericRelation,
    possible_truncation_by_key: dict[str, bool],
    source_truncation_boundaries: frozenset[int],
) -> TruncationRelation | None:
    """Return exact source-boundary prefix evidence without deciding uniqueness."""
    if len(name_a) == len(name_b) or numeric_relation not in {"none", "same"}:
        return None
    if len(name_a) < len(name_b):
        shorter_key, longer_key = key_a, key_b
        shorter_name, longer_name = name_a, name_b
    else:
        shorter_key, longer_key = key_b, key_a
        shorter_name, longer_name = name_b, name_a
    if (
        not possible_truncation_by_key.get(shorter_key, False)
        or len(shorter_name) not in source_truncation_boundaries
        or not longer_name.startswith(shorter_name)
    ):
        return None
    return TruncationRelation(truncated_key=shorter_key, longer_key=longer_key)


def decide_pair(
    *,
    key_a: str,
    key_b: str,
    name_a: str,
    name_b: str,
    numeric_relation: NumericRelation,
    corporate_suffix_aliases: dict[str, tuple[str, ...]],
    minimum_whitespace_compact_length: int,
    truncation_relation: TruncationRelation | None,
    unique_truncation_keys: frozenset[str],
) -> PairDecision:
    """Apply deterministic rules in explicit precedence order; otherwise abstain."""
    suffix_a = _terminal_legal_suffix(name_a, corporate_suffix_aliases)
    suffix_b = _terminal_legal_suffix(name_b, corporate_suffix_aliases)
    evidence: dict[PositiveDecisionRule, bool] = {
        "legal_suffix_format_equivalence": _parsed_suffix_format_equivalent(
            suffix_a, suffix_b
        ),
        "legal_suffix_addition_equivalence": _parsed_suffix_addition_equivalent(
            name_a, name_b, suffix_a, suffix_b
        ),
        "whitespace_only_equivalence": whitespace_only_equivalent(
            name_a,
            name_b,
            numeric_relation,
            minimum_whitespace_compact_length,
        ),
        "unique_source_truncation_equivalence": (
            truncation_relation is not None
            and truncation_relation.truncated_key in unique_truncation_keys
        ),
    }
    supporting_rules = tuple(rule for rule in RULE_PRECEDENCE if evidence[rule])
    if not supporting_rules:
        return PairDecision(
            status=NEEDS_FURTHER_RESOLUTION,
            rule=NO_DETERMINISTIC_EQUIVALENCE,
            evidence=NO_DETERMINISTIC_EQUIVALENCE,
            supporting_rules=(),
        )
    selected_rule = supporting_rules[0]
    return PairDecision(
        status=AUTO_SAME,
        rule=selected_rule,
        evidence=RULE_EVIDENCE[selected_rule],
        supporting_rules=supporting_rules,
    )
