"""Focused tests for conservative resolution-key mention typing."""

from __future__ import annotations

import inspect
from dataclasses import replace

import pytest

import credit_risk_er.employer_eligibility as eligibility_module
from credit_risk_er.employer_eligibility import (
    ADDRESS,
    AMBIGUOUS,
    EMPLOYER_CANDIDATE,
    NON_EMPLOYER_STATUS,
    RULE_PRECEDENCE,
    EligibilityEvidence,
    classify_eligibility,
    has_structural_street_address,
)

ROAD_TYPES = frozenset({"ST", "STREET", "RD", "ROAD"})


def _evidence(name: str = "ACME") -> EligibilityEvidence:
    return EligibilityEvidence(
        resolution_key=name,
        representative_name=name,
        representative_route="employer_resolution_candidate",
        source_rows=1,
        routes=frozenset({"employer_resolution_candidate"}),
        is_blank=False,
        is_numeric_only=False,
        has_address_signal=False,
        address_signal_strength="none",
        has_occupation_signal=False,
        occupation_signal_strength="none",
        has_activity_description_signal=False,
        has_corporate_suffix=False,
        has_organization_like_tokens=False,
        mixed_address_organization_signal=False,
        mixed_occupation_organization_signal=False,
        possible_truncation=False,
        token_count=len(name.split()),
    )


def _classify(evidence: EligibilityEvidence) -> str:
    return classify_eligibility(evidence, road_type_tokens=ROAD_TYPES).status


def test_corporate_suffix_is_organization_compatible() -> None:
    assert _classify(replace(_evidence("ACME SA"), has_corporate_suffix=True)) == (
        EMPLOYER_CANDIDATE
    )


def test_organization_token_without_suffix_is_eligible() -> None:
    evidence = replace(_evidence("BANCO GENERAL"), has_organization_like_tokens=True)
    assert _classify(evidence) == EMPLOYER_CANDIDATE


@pytest.mark.parametrize("name", ["IBM", "CEMEX", "BANCO GENERAL"])
def test_short_clean_organization_names_are_not_rejected(name: str) -> None:
    assert _classify(_evidence(name)) == EMPLOYER_CANDIDATE


def test_strong_existing_address_signal_is_address() -> None:
    evidence = replace(
        _evidence("CALLE 10 EDIFICIO 2"),
        has_address_signal=True,
        address_signal_strength="strong",
        representative_route="address_candidate",
        routes=frozenset({"address_candidate"}),
    )
    assert _classify(evidence) == ADDRESS


def test_general_structural_us_address_rule_captures_known_shape() -> None:
    name = "548 MARKET ST 18590 CA SAN FRANCISCO"
    decision = classify_eligibility(_evidence(name), road_type_tokens=ROAD_TYPES)
    assert decision.status == ADDRESS
    assert decision.rule == "structural_street_zip_region_address"


def test_structural_address_with_organization_evidence_abstains() -> None:
    name = "548 MARKET ST 18590 CA SAN FRANCISCO UNIVERSITY"
    evidence = replace(_evidence(name), has_organization_like_tokens=True)
    assert _classify(evidence) == AMBIGUOUS


@pytest.mark.parametrize(
    "name",
    [
        "MARKET",
        "ST JOSEPHS SCHOOL",
        "548 MARKET ST",
        "MARKET ST 18590 CA SAN FRANCISCO",
        "548 MARKET ST 18590 CALIFORNIA SAN FRANCISCO",
    ],
)
def test_isolated_or_incomplete_address_clues_do_not_trigger_structural_rule(
    name: str,
) -> None:
    assert has_structural_street_address(name, ROAD_TYPES) is False
    assert _classify(_evidence(name)) == EMPLOYER_CANDIDATE


def test_plaza_with_organization_evidence_abstains() -> None:
    evidence = replace(
        _evidence("PLAZA UNIVERSIDAD"),
        has_address_signal=True,
        address_signal_strength="weak",
        has_organization_like_tokens=True,
        mixed_address_organization_signal=True,
    )
    assert _classify(evidence) == AMBIGUOUS


@pytest.mark.parametrize("name", ["JUBILADO", "DESEMPLEADO", "AMA DE CASA", "ESTUDIANTE"])
def test_pure_status_is_non_employer(name: str) -> None:
    evidence = replace(
        _evidence(name),
        representative_route="non_employer_status_candidate",
        routes=frozenset({"non_employer_status_candidate"}),
        has_occupation_signal=True,
        occupation_signal_strength="strong",
    )
    assert _classify(evidence) == NON_EMPLOYER_STATUS


def test_independiente_uses_existing_signal_strength_instead_of_keyword_matching() -> None:
    pure_status = replace(
        _evidence("INDEPENDIENTE"),
        representative_route="non_employer_status_candidate",
        routes=frozenset({"non_employer_status_candidate"}),
        has_occupation_signal=True,
        occupation_signal_strength="strong",
    )
    descriptive_activity = replace(
        _evidence("INDEPENDIENTE REFRIGERACION"),
        representative_route="ambiguous_review_candidate",
        routes=frozenset({"ambiguous_review_candidate"}),
        has_occupation_signal=True,
        occupation_signal_strength="moderate",
        has_activity_description_signal=True,
    )
    name_only = _evidence("INDEPENDIENTE COMERCIAL SA")
    assert _classify(pure_status) == NON_EMPLOYER_STATUS
    assert _classify(descriptive_activity) == AMBIGUOUS
    assert _classify(name_only) == EMPLOYER_CANDIDATE


def test_mixed_address_and_organization_evidence_abstains() -> None:
    evidence = replace(
        _evidence("GRUPO CALLE 10"),
        has_address_signal=True,
        address_signal_strength="moderate",
        has_organization_like_tokens=True,
        mixed_address_organization_signal=True,
    )
    assert _classify(evidence) == AMBIGUOUS


def test_mixed_occupation_and_organization_evidence_abstains() -> None:
    evidence = replace(
        _evidence("SERVICIOS ESTUDIANTE SA"),
        has_occupation_signal=True,
        occupation_signal_strength="moderate",
        has_corporate_suffix=True,
        mixed_occupation_organization_signal=True,
    )
    assert _classify(evidence) == AMBIGUOUS


@pytest.mark.parametrize(
    "evidence",
    [
        replace(_evidence(""), is_blank=True, token_count=0),
        replace(_evidence("12345"), is_numeric_only=True),
    ],
)
def test_blank_and_numeric_only_content_cannot_be_employer_candidates(
    evidence: EligibilityEvidence,
) -> None:
    assert _classify(evidence) == AMBIGUOUS


def test_prior_ambiguous_route_remains_abstained_without_stronger_evidence() -> None:
    evidence = replace(
        _evidence("UNKNOWN VALUE"),
        representative_route="ambiguous_review_candidate",
        routes=frozenset({"ambiguous_review_candidate"}),
    )
    assert _classify(evidence) == AMBIGUOUS


def test_rule_precedence_is_explicit_and_classification_is_deterministic() -> None:
    evidence = replace(_evidence("ACME SA"), has_corporate_suffix=True)
    first = classify_eligibility(evidence, road_type_tokens=ROAD_TYPES)
    second = classify_eligibility(evidence, road_type_tokens=ROAD_TYPES)
    assert first == second
    assert RULE_PRECEDENCE == (
        "invalid_content_abstention",
        "mixed_evidence_abstention",
        "structural_street_zip_region_address",
        "existing_address_signal",
        "existing_non_employer_status",
        "descriptive_activity_abstention",
        "organization_compatible",
        "prior_employer_route",
        "insufficient_evidence_abstention",
    )


def test_module_contains_no_pair_resolution_fuzzy_web_or_manual_review_logic() -> None:
    source = inspect.getsource(eligibility_module)
    forbidden = ("AUTO_SAME", "key_a", "key_b", "rapidfuzz", "requests", "review_label")
    assert all(term not in source for term in forbidden)
