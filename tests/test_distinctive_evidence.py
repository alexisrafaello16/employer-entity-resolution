"""Focused exact-overlap and token-distinctiveness evidence tests."""

from __future__ import annotations

import pytest

from credit_risk_er.matching.distinctive import compute_distinctive_name_evidence
from credit_risk_er.matching.orthographic import build_orthographic_policy

POLICY = build_orthographic_policy(
    corporate_suffix_aliases={
        "SA": ("S A", "SA"),
        "SAS": ("S A S", "SAS"),
        "INC": ("INC",),
        "CORP": ("CORP",),
    },
    minimum_typo_token_length=6,
    minimum_informative_token_length=5,
)


def _evidence(
    name_a: str,
    name_b: str,
    support: dict[str, int],
    *,
    maximum: int = 50,
):
    return compute_distinctive_name_evidence(
        name_a=name_a,
        name_b=name_b,
        token_support=support,
        maximum_token_frequency=maximum,
        policy=POLICY,
    )


def test_one_common_shared_token_is_not_distinctive() -> None:
    evidence = _evidence("OCEAN FARMS", "BALBOA FARMS", {"FARMS": 100})
    assert evidence.shared_exact_tokens == "FARMS"
    assert evidence.shared_exact_token_count == 1
    assert evidence.shared_distinctive_token_count == 0
    assert evidence.has_exact_overlap_without_distinctive_token
    assert evidence.minimum_shared_token_support == 100


def test_one_rare_shared_token_is_distinctive() -> None:
    evidence = _evidence("OCEAN ZEPHYR", "BALBOA ZEPHYR", {"ZEPHYR": 2})
    assert evidence.shared_distinctive_tokens == "ZEPHYR"
    assert evidence.has_shared_distinctive_token
    assert evidence.minimum_distinctive_token_support == 2
    assert evidence.maximum_distinctive_token_support == 2


def test_multiple_distinctive_shared_tokens() -> None:
    evidence = _evidence(
        "NORTH ZEPHYR LOGISTICS",
        "SOUTH ZEPHYR LOGISTICS",
        {"ZEPHYR": 2, "LOGISTICS": 8},
    )
    assert evidence.shared_distinctive_tokens == "LOGISTICS|ZEPHYR"
    assert evidence.shared_distinctive_token_count == 2
    assert evidence.has_multiple_shared_distinctive_tokens
    assert evidence.minimum_distinctive_token_support == 2
    assert evidence.maximum_distinctive_token_support == 8


def test_zero_exact_overlap() -> None:
    evidence = _evidence("OCEAN FARMS", "BALBOA LOGISTICS", {})
    assert evidence.shared_exact_tokens == ""
    assert evidence.shared_exact_token_count == 0
    assert not evidence.has_shared_exact_token
    assert not evidence.has_exact_overlap_without_distinctive_token
    assert evidence.minimum_shared_token_support is None


def test_repeated_tokens_preserve_overlap_multiplicity() -> None:
    evidence = _evidence(
        "ZEPHYR ZEPHYR LOGISTICS",
        "ZEPHYR ZEPHYR FARMS",
        {"ZEPHYR": 2},
    )
    assert evidence.shared_exact_tokens == "ZEPHYR|ZEPHYR"
    assert evidence.shared_exact_token_count == 2
    assert evidence.shared_distinctive_token_count == 2


def test_known_generic_token_is_reported_but_not_distinctive() -> None:
    evidence = _evidence("ALPHA PANAMA", "BETA PANAMA", {"PANAMA": 2})
    assert evidence.shared_generic_token_count == 1
    assert evidence.shared_distinctive_token_count == 0


def test_nonterminal_legal_token_cannot_be_distinctive() -> None:
    evidence = _evidence("ALPHA SA GROUPER", "BETA SA GROUPER", {"SA": 1, "GROUPER": 60})
    assert evidence.shared_exact_tokens == "GROUPER|SA"
    assert evidence.shared_distinctive_token_count == 0
    assert evidence.minimum_shared_token_support == 60


@pytest.mark.parametrize("token", ["507", "ACME507"])
def test_numeric_and_alphanumeric_tokens_are_not_distinctive(token: str) -> None:
    evidence = _evidence(f"ALPHA {token}", f"BETA {token}", {token: 1})
    assert evidence.shared_exact_tokens == token
    assert evidence.shared_distinctive_token_count == 0
    assert evidence.has_exact_overlap_without_distinctive_token


def test_short_ineligible_exact_overlap_sets_nondistinctive_overlap_flag() -> None:
    evidence = _evidence("ALPHA ABC", "BETA ABC", {"ABC": 1})
    assert evidence.shared_exact_tokens == "ABC"
    assert evidence.shared_distinctive_token_count == 0
    assert evidence.has_exact_overlap_without_distinctive_token


def test_exact_and_distinctive_coverage_use_core_token_multiplicity() -> None:
    evidence = _evidence(
        "NORTH ZEPHYR LOGISTICS",
        "ZEPHYR LOGISTICS",
        {"ZEPHYR": 2, "LOGISTICS": 4},
    )
    assert evidence.exact_coverage_a == pytest.approx(2 / 3, abs=0.000001)
    assert evidence.exact_coverage_b == 1.0
    assert evidence.distinctive_coverage_a == pytest.approx(2 / 3, abs=0.000001)
    assert evidence.distinctive_coverage_b == 1.0
    assert evidence.shorter_name_exact_coverage == 1.0
    assert evidence.longer_name_exact_coverage == pytest.approx(2 / 3, abs=0.000001)


def test_one_token_name_has_defined_coverage() -> None:
    evidence = _evidence("ZEPHYR", "ZEPHYR FARMS", {"ZEPHYR": 2})
    assert evidence.exact_coverage_a == 1.0
    assert evidence.shorter_name_distinctive_coverage == 1.0


def test_terminal_legal_suffix_handling_is_reused() -> None:
    evidence = _evidence("ALPHA ZEPHYR SA", "BETA ZEPHYR", {"ZEPHYR": 2})
    assert evidence.shared_exact_tokens == "ZEPHYR"
    assert evidence.exact_coverage_a == 0.5


def test_shared_token_serialization_is_stable_and_sorted() -> None:
    first = _evidence("ZEPHYR LOGISTICS", "LOGISTICS ZEPHYR", {"ZEPHYR": 2, "LOGISTICS": 3})
    second = _evidence("LOGISTICS ZEPHYR", "ZEPHYR LOGISTICS", {"ZEPHYR": 2, "LOGISTICS": 3})
    assert first.shared_exact_tokens == "LOGISTICS|ZEPHYR"
    assert first == second


def test_existing_maximum_frequency_is_inclusive() -> None:
    at_limit = _evidence("ALPHA ZEPHYR", "BETA ZEPHYR", {"ZEPHYR": 50})
    above_limit = _evidence("ALPHA ZEPHYR", "BETA ZEPHYR", {"ZEPHYR": 51})
    assert at_limit.has_shared_distinctive_token
    assert not above_limit.has_shared_distinctive_token


def test_evidence_has_no_identity_score_or_status() -> None:
    evidence = _evidence("ALPHA ZEPHYR", "BETA ZEPHYR", {"ZEPHYR": 2})
    assert not hasattr(evidence, "status")
    assert not hasattr(evidence, "score")
    assert not hasattr(evidence, "same_entity")
