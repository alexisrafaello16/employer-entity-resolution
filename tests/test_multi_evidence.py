"""Focused tests for diagnostic multi-evidence flags and precedence."""

from __future__ import annotations

import inspect
from typing import Any

import pytest

from credit_risk_er.matching.multi_evidence import (
    ACTIVE_ASSESSMENT_FAMILIES,
    ASSESSMENT_FAMILY_PRECEDENCE,
    LOCALIZED_ORTHOGRAPHIC_EVIDENCE,
    assess_multi_evidence,
)


def _assess(**overrides: Any) -> Any:
    values: dict[str, Any] = {
        "primary_family": "MIXED_STRUCTURAL_RELATIONSHIP",
        "numeric_relation": "none",
        "shared_exact_token_count": 0,
        "shared_distinctive_token_count": 0,
        "has_shared_distinctive_token": False,
        "has_multiple_shared_distinctive_tokens": False,
        "shorter_name_exact_coverage": 0.0,
        "shorter_name_distinctive_coverage": 0.0,
        "prior_orthographic_evidence": "multiple_core_token_differences",
    }
    values.update(overrides)
    return assess_multi_evidence(**values)


@pytest.mark.parametrize("numeric_relation", ["one_sided", "conflict"])
def test_numeric_risk_with_distinctive_name_evidence_has_first_precedence(
    numeric_relation: str,
) -> None:
    assessment = _assess(
        primary_family="EXACT_TOKEN_REORDER",
        numeric_relation=numeric_relation,
        shared_exact_token_count=2,
        shared_distinctive_token_count=2,
        has_shared_distinctive_token=True,
        has_multiple_shared_distinctive_tokens=True,
        shorter_name_exact_coverage=1.0,
    )
    assert assessment.assessment_family == "NUMERIC_RISK_WITH_NAME_EVIDENCE"
    assert assessment.has_numeric_risk


def test_numeric_risk_with_exact_overlap_only_remains_explicit() -> None:
    assessment = _assess(numeric_relation="conflict", shared_exact_token_count=1)
    assert assessment.assessment_family == "NUMERIC_RISK_WITH_NAME_EVIDENCE"


def test_numeric_risk_without_name_overlap_uses_weak_fallback() -> None:
    assessment = _assess(numeric_relation="one_sided")
    assert assessment.assessment_family == "WEAK_OR_MIXED_EVIDENCE"
    assert assessment.has_zero_exact_overlap


@pytest.mark.parametrize(
    ("primary_family", "expected_family"),
    [
        ("EXACT_TOKEN_REORDER", "STRUCTURE_WITH_DISTINCTIVE_TOKEN"),
        ("EXACT_TOKEN_CONTAINMENT_NONALIGNED", "STRUCTURE_WITH_DISTINCTIVE_TOKEN"),
        ("SINGLE_TOKEN_ADDITION_REMOVAL", "STRUCTURE_WITH_DISTINCTIVE_TOKEN"),
        ("MULTI_TOKEN_ADDITION_REMOVAL", "STRUCTURE_WITH_DISTINCTIVE_TOKEN"),
        ("POSSIBLE_TRUNCATION_RELATIONSHIP", "STRUCTURE_WITH_DISTINCTIVE_TOKEN"),
    ],
)
def test_structural_exact_families_converge_with_one_distinctive_token(
    primary_family: str,
    expected_family: str,
) -> None:
    assessment = _assess(
        primary_family=primary_family,
        shared_exact_token_count=1,
        shared_distinctive_token_count=1,
        has_shared_distinctive_token=True,
    )
    assert assessment.assessment_family == expected_family
    assert assessment.has_structural_exact_relation


def test_structural_relation_with_multiple_distinctive_tokens_uses_stronger_family() -> None:
    assessment = _assess(
        primary_family="EXACT_TOKEN_REORDER",
        shared_exact_token_count=2,
        shared_distinctive_token_count=2,
        has_shared_distinctive_token=True,
        has_multiple_shared_distinctive_tokens=True,
    )
    assert assessment.assessment_family == "STRUCTURE_WITH_MULTIPLE_DISTINCTIVE_TOKENS"


def test_localized_orthographic_evidence_converges_with_distinctive_context() -> None:
    assessment = _assess(
        primary_family="SINGLE_TOKEN_LARGER_EDIT",
        shared_exact_token_count=1,
        shared_distinctive_token_count=1,
        has_shared_distinctive_token=True,
        prior_orthographic_evidence="edit_distance_not_one",
    )
    assert assessment.assessment_family == "LOCAL_ORTHOGRAPHIC_WITH_DISTINCTIVE_CONTEXT"
    assert assessment.has_orthographic_signal


def test_multiple_small_edits_converge_with_distinctive_context() -> None:
    assessment = _assess(
        primary_family="MULTIPLE_SMALL_TOKEN_EDITS",
        shared_exact_token_count=1,
        shared_distinctive_token_count=1,
        has_shared_distinctive_token=True,
    )
    assert assessment.assessment_family == "MULTI_EDIT_WITH_DISTINCTIVE_CONTEXT"


def test_full_shorter_exact_coverage_is_descriptive_only() -> None:
    assessment = _assess(shared_exact_token_count=1, shorter_name_exact_coverage=1.0)
    assert assessment.assessment_family == "FULL_SHORTER_NAME_OVERLAP"
    assert assessment.has_full_shorter_exact_coverage


def test_distinctive_overlap_without_structural_convergence_is_single_signal() -> None:
    assessment = _assess(
        shared_exact_token_count=1,
        shared_distinctive_token_count=1,
        has_shared_distinctive_token=True,
    )
    assert assessment.assessment_family == "DISTINCTIVE_SINGLE_SIGNAL"


def test_exact_overlap_without_distinctive_evidence_is_separate() -> None:
    assessment = _assess(shared_exact_token_count=1)
    assert assessment.assessment_family == "EXACT_OVERLAP_WITHOUT_DISTINCTIVE_EVIDENCE"


def test_zero_exact_overlap_is_explicit() -> None:
    assessment = _assess()
    assert assessment.assessment_family == "ZERO_EXACT_OVERLAP"


@pytest.mark.parametrize(
    "reason",
    [
        "core_token_count_mismatch",
        "multiple_core_token_differences",
        "numeric_evidence_incompatible",
    ],
)
def test_nonlocalized_orthographic_gates_do_not_create_signal(reason: str) -> None:
    assert not _assess(prior_orthographic_evidence=reason).has_orthographic_signal


@pytest.mark.parametrize("reason", sorted(LOCALIZED_ORTHOGRAPHIC_EVIDENCE))
def test_approved_localized_orthographic_reasons_create_signal(reason: str) -> None:
    assert _assess(prior_orthographic_evidence=reason).has_orthographic_signal


def test_output_contract_contains_no_identity_outcome_or_score() -> None:
    fields = set(_assess().__dataclass_fields__)
    forbidden = {"status", "match", "same_entity", "confidence", "probability", "score"}
    assert fields.isdisjoint(forbidden)
    assert _assess().assessment_family in ACTIVE_ASSESSMENT_FAMILIES


def test_precedence_is_unique_and_deterministic() -> None:
    assert len(ASSESSMENT_FAMILY_PRECEDENCE) == len(set(ASSESSMENT_FAMILY_PRECEDENCE))
    first = _assess(shared_exact_token_count=1)
    second = _assess(shared_exact_token_count=1)
    assert first == second


def test_high_lexical_family_is_reserved_but_inactive_without_threshold() -> None:
    assert "HIGH_LEXICAL_WITHOUT_DISTINCTIVE_OVERLAP" in ASSESSMENT_FAMILY_PRECEDENCE
    assert "HIGH_LEXICAL_WITHOUT_DISTINCTIVE_OVERLAP" not in ACTIVE_ASSESSMENT_FAMILIES
    signature = inspect.signature(assess_multi_evidence)
    assert not {
        "char_ratio",
        "token_sort_ratio",
        "token_set_ratio",
        "partial_ratio",
    }.intersection(signature.parameters)


@pytest.mark.parametrize(
    "overrides",
    [
        {"shared_exact_token_count": -1},
        {
            "shared_exact_token_count": 1,
            "shared_distinctive_token_count": 2,
            "has_shared_distinctive_token": True,
            "has_multiple_shared_distinctive_tokens": True,
        },
        {"shared_distinctive_token_count": 1},
        {"shorter_name_exact_coverage": 1.01},
    ],
)
def test_inconsistent_evidence_fails_closed(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        _assess(**overrides)
