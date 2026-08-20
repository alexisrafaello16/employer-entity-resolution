"""Focused structural-family tests for Residual Relationship Profiler v1."""

from __future__ import annotations

from typing import cast

import pytest

from credit_risk_er.matching.decision import NumericRelation
from credit_risk_er.matching.orthographic import build_orthographic_policy
from credit_risk_er.matching.residual_profile import profile_residual_relationship

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


def _profile(
    name_a: str,
    name_b: str,
    *,
    numeric_relation: str = "none",
    possible_truncation_a: bool = False,
    possible_truncation_b: bool = False,
    boundaries: frozenset[int] = frozenset(),
    prior_evidence: str = "edit_distance_not_one",
):
    key_a, key_b = sorted((name_a, name_b))
    ordered_name_a = key_a
    ordered_name_b = key_b
    ordered_truncation_a = (
        possible_truncation_a if name_a == ordered_name_a else possible_truncation_b
    )
    ordered_truncation_b = (
        possible_truncation_b if name_b == ordered_name_b else possible_truncation_a
    )
    return profile_residual_relationship(
        key_a=key_a,
        key_b=key_b,
        name_a=ordered_name_a,
        name_b=ordered_name_b,
        numeric_relation=cast(NumericRelation, numeric_relation),
        possible_truncation_a=ordered_truncation_a,
        possible_truncation_b=ordered_truncation_b,
        source_truncation_boundaries=boundaries,
        prior_orthographic_evidence=prior_evidence,
        policy=POLICY,
    )


def test_exact_token_reorder() -> None:
    profile = _profile("GLOBAL TECH", "TECH GLOBAL")
    assert profile.primary_family == "EXACT_TOKEN_REORDER"
    assert profile.is_token_reorder


def test_duplicate_tokens_preserve_multiset_semantics() -> None:
    profile = _profile("ALPHA ALPHA BETA", "ALPHA BETA BETA")
    assert not profile.is_token_reorder
    assert profile.shared_exact_token_count == 2


@pytest.mark.parametrize(
    ("name_a", "name_b"),
    [("ACME", "ACME PANAMA"), ("GLOBAL TECH PANAMA", "GLOBAL TECH")],
)
def test_single_token_addition_or_removal(name_a: str, name_b: str) -> None:
    profile = _profile(name_a, name_b)
    assert profile.primary_family == "SINGLE_TOKEN_ADDITION_REMOVAL"
    assert profile.added_removed_token == "PANAMA"
    assert profile.added_removed_token_count == 1
    assert profile.is_ordered_subsequence


def test_multi_token_addition_and_ordered_subsequence() -> None:
    profile = _profile("ABC LOGISTICS", "ABC INTERNATIONAL LOGISTICS PANAMA")
    assert profile.primary_family == "MULTI_TOKEN_ADDITION_REMOVAL"
    assert profile.added_removed_token_count == 2
    assert profile.is_ordered_subsequence


def test_nonaligned_containment_preserves_multiplicity() -> None:
    profile = _profile("ALPHA BETA", "BETA ALPHA GAMMA")
    assert profile.primary_family == "EXACT_TOKEN_CONTAINMENT_NONALIGNED"
    assert profile.is_token_multiset_containment
    assert not profile.is_ordered_subsequence


@pytest.mark.parametrize(
    ("phrase", "initialism"),
    [
        ("INTERNATIONAL BUSINESS MACHINES", "IBM"),
        ("PANAMA PORTS COMPANY", "PPC"),
    ],
)
def test_conservative_initialism_positive(phrase: str, initialism: str) -> None:
    profile = _profile(phrase, initialism)
    assert profile.primary_family == "ACRONYM_INITIALISM_PATTERN"
    assert profile.has_initialism_pattern


def test_arbitrary_short_string_is_not_initialism() -> None:
    profile = _profile("INTERNATIONAL BUSINESS MACHINES", "IX")
    assert profile.primary_family != "ACRONYM_INITIALISM_PATTERN"
    assert not profile.has_initialism_pattern


def test_legal_suffix_does_not_supply_initials() -> None:
    profile = _profile("INTERNATIONAL BUSINESS MACHINES SA", "IBMSA")
    assert profile.primary_family != "ACRONYM_INITIALISM_PATTERN"
    assert not profile.has_initialism_pattern


def test_multiple_aligned_edit_distance_one_differences() -> None:
    profile = _profile("COMPAIA PANAMEA DE AVIACION", "COMPANIA PANAMENA DE AVIACION")
    assert profile.primary_family == "MULTIPLE_SMALL_TOKEN_EDITS"
    assert profile.differing_token_count == 2
    assert profile.maximum_token_edit_distance == 1
    assert profile.total_token_edit_distance == 2


def test_multiple_small_token_edits_include_distance_two_diagnostic() -> None:
    profile = _profile("ABCDEF GHIJKL CAPITAL", "ABXYEF GHIXKL CAPITAL")
    assert profile.primary_family == "MULTIPLE_SMALL_TOKEN_EDITS"
    assert profile.maximum_token_edit_distance == 2
    assert profile.total_token_edit_distance == 3


@pytest.mark.parametrize(
    ("name_a", "name_b", "prior_evidence"),
    [
        ("ABCDEF CAPITAL", "ABXYEF CAPITAL", "edit_distance_not_one"),
        ("ALPHA CAPITAL", "ALPHE CAPITAL", "short_typo_token"),
        (
            "SERVICE CAPITAL",
            "SERVICES CAPITAL",
            "terminal_edit_requires_further_resolution",
        ),
    ],
)
def test_single_token_larger_or_unsupported_edit_is_diagnostic(
    name_a: str,
    name_b: str,
    prior_evidence: str,
) -> None:
    profile = _profile(name_a, name_b, prior_evidence=prior_evidence)
    assert profile.primary_family == "SINGLE_TOKEN_LARGER_EDIT"
    assert f"orthographic_evidence={prior_evidence}" in profile.family_evidence


@pytest.mark.parametrize("numeric_relation", ["one_sided", "conflict"])
def test_numeric_variation_has_first_precedence(numeric_relation: str) -> None:
    profile = _profile(
        "GLOBAL TECH 1",
        "TECH GLOBAL 2",
        numeric_relation=numeric_relation,
    )
    assert profile.primary_family == "NUMERIC_VARIATION"
    assert profile.numeric_relation == numeric_relation


def test_possible_truncation_reuses_source_boundary_prefix_semantics() -> None:
    shorter = "ALPHA BET"
    longer = "ALPHA BETA GROUP"
    profile = _profile(
        shorter,
        longer,
        possible_truncation_a=True,
        boundaries=frozenset({len(shorter)}),
    )
    assert profile.primary_family == "POSSIBLE_TRUNCATION_RELATIONSHIP"
    assert "exact_source_boundary_prefix" in profile.family_evidence


def test_ambiguous_truncation_remains_descriptive_only() -> None:
    shorter = "ALPHA BET"
    profile = _profile(
        shorter,
        "ALPHA BETA GROUP",
        possible_truncation_a=True,
        boundaries=frozenset({len(shorter)}),
    )
    assert profile.primary_family == "POSSIBLE_TRUNCATION_RELATIONSHIP"
    assert not hasattr(profile, "decision_status")


@pytest.mark.parametrize(
    ("name_a", "name_b"),
    [
        ("ALPHA BETA", "ALPHE BETA GAMMA"),
        ("GLOBAL TEHC", "TECH GLOBAL"),
    ],
)
def test_multiple_transformations_are_mixed(name_a: str, name_b: str) -> None:
    assert _profile(name_a, name_b).primary_family == "MIXED_STRUCTURAL_RELATIONSHIP"


def test_other_residual_fallback() -> None:
    profile = _profile("ALPHA BETA", "GAMMA")
    assert profile.primary_family == "OTHER_RESIDUAL"


def test_approved_legal_suffix_handling_is_reused() -> None:
    profile = _profile("GLOBAL TECH SA", "TECH GLOBAL")
    assert profile.primary_family == "EXACT_TOKEN_REORDER"


def test_profile_contains_no_identity_outcome() -> None:
    profile = _profile("GLOBAL TECH", "TECH GLOBAL")
    assert not hasattr(profile, "status")
    assert not hasattr(profile, "same_entity")
    assert "SAME" not in profile.primary_family
