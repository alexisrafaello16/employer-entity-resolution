"""High-precision deterministic pair decision rules and abstention tests."""

from __future__ import annotations

import inspect

import pytest

import credit_risk_er.matching.decision as decision_module
from credit_risk_er.matching.decision import (
    AUTO_SAME,
    NEEDS_FURTHER_RESOLUTION,
    NO_DETERMINISTIC_EQUIVALENCE,
    PairDecision,
    TruncationRelation,
    decide_pair,
    exact_source_truncation_relation,
)

SUFFIX_ALIASES = {
    "SA": ("S A", "SA"),
    "SAS": ("S A S", "SAS"),
    "INC": ("INC",),
    "CORP": ("CORP",),
    "LLC": ("L L C", "LLC"),
    "LTDA": ("LTDA",),
    "CIA": ("CIA",),
}
TRUNCATED = "INTERNATIONAL TECHNOLOGY SYSTE"


def _decision(
    name_a: str,
    name_b: str,
    *,
    numeric_relation: decision_module.NumericRelation = "none",
    truncation_relation: TruncationRelation | None = None,
    unique_truncation_keys: frozenset[str] = frozenset(),
) -> PairDecision:
    key_a, key_b = sorted((name_a, name_b))
    return decide_pair(
        key_a=key_a,
        key_b=key_b,
        name_a=key_a,
        name_b=key_b,
        numeric_relation=numeric_relation,
        corporate_suffix_aliases=SUFFIX_ALIASES,
        minimum_whitespace_compact_length=5,
        truncation_relation=truncation_relation,
        unique_truncation_keys=unique_truncation_keys,
    )


def _truncation_relation(
    shorter: str,
    longer: str,
    *,
    possible: bool = True,
    numeric_relation: decision_module.NumericRelation = "none",
) -> TruncationRelation | None:
    key_a, key_b = sorted((shorter, longer))
    return exact_source_truncation_relation(
        key_a=key_a,
        key_b=key_b,
        name_a=key_a,
        name_b=key_b,
        numeric_relation=numeric_relation,
        possible_truncation_by_key={shorter: possible, longer: False},
        source_truncation_boundaries=frozenset({30}),
    )


def test_terminal_legal_suffix_formatting_is_auto_same_and_has_precedence() -> None:
    result = _decision("EMPRESA S A", "EMPRESA SA")
    assert result.status == AUTO_SAME
    assert result.rule == "legal_suffix_format_equivalence"
    assert result.evidence == "terminal_legal_suffix_formatting_only"
    assert result.supporting_rules == (
        "legal_suffix_format_equivalence",
        "whitespace_only_equivalence",
    )


@pytest.mark.parametrize(
    ("base", "suffixed"),
    (
        ("ELECTROCENTRO", "ELECTROCENTRO S A"),
        ("LINA CORPORATION", "LINA CORPORATION INC"),
        ("COFFEE ROASTERS UNIDOS", "COFFEE ROASTERS UNIDOS SA"),
    ),
)
def test_configured_terminal_legal_suffix_addition_is_auto_same(
    base: str, suffixed: str
) -> None:
    result = _decision(base, suffixed)
    assert result.status == AUTO_SAME
    assert result.rule == "legal_suffix_addition_equivalence"


@pytest.mark.parametrize(
    ("left", "right"),
    (
        ("MOBIL PHONE", "MOBILPHONE"),
        ("COFFEE VENDING", "COFFEEVENDING"),
        ("J M INDUSTRIALES", "JM INDUSTRIALES"),
        ("ALSTOM S A", "ALSTOMSA"),
    ),
)
def test_nontrivial_whitespace_only_equivalence_is_auto_same(left: str, right: str) -> None:
    result = _decision(left, right)
    assert result.status == AUTO_SAME
    assert result.rule == "whitespace_only_equivalence"


def test_whitespace_only_with_same_numeric_tokens_is_auto_same() -> None:
    result = _decision(
        "MOBIL PHONE 507",
        "MOBILPHONE 507",
        numeric_relation="same",
    )
    assert result.status == AUTO_SAME
    assert result.rule == "whitespace_only_equivalence"


def test_short_whitespace_only_string_abstains() -> None:
    assert _decision("A B", "AB").status == NEEDS_FURTHER_RESOLUTION


@pytest.mark.parametrize(
    ("left", "right", "numeric_relation"),
    (
        ("1 2 3 EDU S A", "123EDUSA", "one_sided"),
        ("10 PM CURFEW", "10PM CURFEW", "one_sided"),
        ("A 12 3", "A 1 23", "conflict"),
    ),
)
def test_asymmetric_or_conflicting_numeric_structure_blocks_whitespace_rule(
    left: str,
    right: str,
    numeric_relation: decision_module.NumericRelation,
) -> None:
    result = _decision(left, right, numeric_relation=numeric_relation)
    assert result.status == NEEDS_FURTHER_RESOLUTION
    assert result.rule == NO_DETERMINISTIC_EQUIVALENCE


@pytest.mark.parametrize(
    ("left", "right", "numeric_relation"),
    (
        ("STUDIO 507", "STUDIO 508", "conflict"),
        ("BANCO GENERAL", "BANCO GENERAL 4", "one_sided"),
        ("PACHOS KITCHEN", "PANCHOS KITCHEN", "none"),
        ("SUBWAY", "SUBWAY INTERNATIONAL BV", "none"),
        ("BANESCO BANCO UNIVERSAL", "BANESCO SEGUROS", "none"),
        ("BANCO GENERAL", "BANCO GENERAL INVERSIONES", "none"),
    ),
)
def test_questionable_or_fuzzy_only_pairs_abstain(
    left: str,
    right: str,
    numeric_relation: decision_module.NumericRelation,
) -> None:
    result = _decision(left, right, numeric_relation=numeric_relation)
    assert result.status == NEEDS_FURTHER_RESOLUTION
    assert result.rule == NO_DETERMINISTIC_EQUIVALENCE


def test_suffix_like_interior_tokens_are_not_removed() -> None:
    result = _decision("EMPRESA S A PANAMA", "EMPRESA PANAMA")
    assert result.status == NEEDS_FURTHER_RESOLUTION


def test_unconfigured_organization_words_are_not_removable_suffixes() -> None:
    for suffix in ("GROUP", "BANK", "HOTEL", "SCHOOL", "INTERNATIONAL", "PANAMA"):
        assert _decision("EMPRESA", f"EMPRESA {suffix}").status == NEEDS_FURTHER_RESOLUTION


def test_unique_exact_source_truncation_continuation_is_auto_same() -> None:
    longer = f"{TRUNCATED}MS INC"
    relation = _truncation_relation(TRUNCATED, longer)
    assert relation is not None
    result = _decision(
        TRUNCATED,
        longer,
        truncation_relation=relation,
        unique_truncation_keys=frozenset({TRUNCATED}),
    )
    assert result.status == AUTO_SAME
    assert result.rule == "unique_source_truncation_equivalence"


def test_two_compatible_continuations_both_abstain_without_uniqueness() -> None:
    longer_names = (f"{TRUNCATED}MS INC", f"{TRUNCATED}MAS LLC")
    for longer in longer_names:
        relation = _truncation_relation(TRUNCATED, longer)
        assert relation is not None
        assert (
            _decision(TRUNCATED, longer, truncation_relation=relation).status
            == NEEDS_FURTHER_RESOLUTION
        )


def test_possible_truncation_false_disables_rule() -> None:
    assert _truncation_relation(TRUNCATED, f"{TRUNCATED}MS INC", possible=False) is None


def test_nonprefix_name_has_no_truncation_relation() -> None:
    assert _truncation_relation(TRUNCATED, "X" + TRUNCATED + "MS INC") is None


@pytest.mark.parametrize("numeric_relation", ("one_sided", "conflict"))
def test_numeric_incompatibility_blocks_truncation(
    numeric_relation: decision_module.NumericRelation,
) -> None:
    assert (
        _truncation_relation(
            TRUNCATED,
            f"{TRUNCATED}MS 4",
            numeric_relation=numeric_relation,
        )
        is None
    )


def test_rule_output_is_deterministic() -> None:
    first = _decision("EMPRESA S A", "EMPRESA SA")
    second = _decision("EMPRESA S A", "EMPRESA SA")
    assert first == second


def test_decision_module_has_no_fuzzy_threshold_or_clustering_implementation() -> None:
    source = inspect.getsource(decision_module).casefold()
    forbidden = (
        "char_ratio",
        "token_set_ratio",
        "partial_ratio",
        "rapidfuzz",
        "connected_component",
        "auto_different",
    )
    assert all(term not in source for term in forbidden)


def test_only_two_decision_statuses_are_emitted() -> None:
    outcomes = {
        _decision("EMPRESA", "EMPRESA SA").status,
        _decision("PACHOS", "PANCHOS").status,
    }
    assert outcomes == {AUTO_SAME, NEEDS_FURTHER_RESOLUTION}
