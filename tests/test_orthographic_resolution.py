"""Precision and safety tests for single-token orthographic equivalence v2.1."""

from __future__ import annotations

import inspect
from collections.abc import Collection, Mapping
from typing import cast

import pytest

import credit_risk_er.matching.orthographic as orthographic_module
from credit_risk_er.matching.orthographic import (
    NEEDS_FURTHER_RESOLUTION,
    NOT_ELIGIBLE_FOR_ORTHOGRAPHIC,
    STRONG_ORTHOGRAPHIC_EVIDENCE,
    NumericRelation,
    OrthographicComparison,
    OrthographicDecision,
    build_orthographic_policy,
    decide_orthographic_pair,
    orthographic_core_tokens,
    prepare_orthographic_pair,
)

ALIASES = {
    "SA": ("S A", "SA"),
    "SAS": ("S A S", "SAS"),
    "INC": ("INC",),
    "CORP": ("CORP",),
}
POLICY = build_orthographic_policy(
    corporate_suffix_aliases=ALIASES,
    minimum_typo_token_length=6,
    minimum_informative_token_length=5,
)
MAXIMUM_TOKEN_FREQUENCY = 50


def _decision(
    name_a: str,
    name_b: str,
    *,
    numeric_relation: str = "none",
    employer_a: bool = True,
    employer_b: bool = True,
    support_overrides: Mapping[str, int] | None = None,
    context_variants: Collection[str] | None = None,
) -> OrthographicDecision:
    relation = cast(NumericRelation, numeric_relation)
    prepared = prepare_orthographic_pair(
        name_a=name_a,
        name_b=name_b,
        numeric_relation=relation,
        employer_candidate_a=employer_a,
        employer_candidate_b=employer_b,
        policy=POLICY,
    )
    support = {
        token: 1
        for token in set(
            orthographic_core_tokens(name_a, POLICY) + orthographic_core_tokens(name_b, POLICY)
        )
    }
    if support_overrides is not None:
        support.update(support_overrides)
    variant_index: dict[tuple[str, ...], frozenset[str]] = {}
    if isinstance(prepared, OrthographicComparison):
        variants = context_variants or (
            prepared.differing_token_a,
            prepared.differing_token_b,
        )
        variant_index[prepared.context_signature_tokens] = frozenset(variants)
    return decide_orthographic_pair(
        name_a=name_a,
        name_b=name_b,
        numeric_relation=relation,
        employer_candidate_a=employer_a,
        employer_candidate_b=employer_b,
        policy=POLICY,
        token_support=support,
        context_variants=variant_index,
        maximum_token_frequency=MAXIMUM_TOKEN_FREQUENCY,
    )


@pytest.mark.parametrize(
    ("name_a", "name_b", "operation", "location"),
    [
        ("FARMACIA SHEKAINAH", "FARMACIA SHEKINAH", "deletion", "inside"),
        ("IMPORTADORA TRANSMUNDI", "IMPORTADORA TRASMUNDI", "deletion", "inside"),
        ("PACHOS KITCHEN", "PANCHOS KITCHEN", "insertion", "inside"),
        ("ANCOUVER CAPITAL", "VANCOUVER CAPITAL", "insertion", "beginning"),
        ("ALPHAAA CAPITAL", "ALPHBAA CAPITAL", "substitution", "inside"),
    ],
)
def test_safe_one_edit_is_auto_same(
    name_a: str,
    name_b: str,
    operation: str,
    location: str,
) -> None:
    decision = _decision(name_a, name_b)
    assert decision.status == STRONG_ORTHOGRAPHIC_EVIDENCE
    assert decision.rule == "single_token_edit_equivalence"
    assert decision.evidence == "one_alphabetic_token_edit_distance_1_with_exact_context"
    assert decision.edit_operation == operation
    assert decision.edit_location == location
    assert decision.context_variant_count == 2
    assert decision.token_support_a == 1
    assert decision.token_support_b == 1


@pytest.mark.parametrize(
    ("name_a", "name_b", "operation"),
    [
        ("AMERICA CAPITAL", "AMERICAN CAPITAL", "insertion"),
        ("SERVICE CAPITAL", "SERVICES CAPITAL", "insertion"),
        ("CONSULTIN GROUPER", "CONSULTING GROUPER", "insertion"),
        ("ALPHAAA CAPITAL", "ALPHAA CAPITAL", "deletion"),
        ("ALPHAAA CAPITAL", "ALPHAAB CAPITAL", "substitution"),
    ],
)
def test_terminal_edits_abstain(
    name_a: str,
    name_b: str,
    operation: str,
) -> None:
    decision = _decision(name_a, name_b)
    assert decision.status == NEEDS_FURTHER_RESOLUTION
    assert decision.evidence == "terminal_edit_requires_further_resolution"
    assert decision.edit_operation == operation
    assert decision.edit_location == "end"
    assert decision.context_variant_count == 2


def test_both_established_differing_tokens_abstain() -> None:
    decision = _decision(
        "INTERNACIONAL CAPITAL",
        "INTERNATIONAL CAPITAL",
        support_overrides={"INTERNACIONAL": 51, "INTERNATIONAL": 70},
    )
    assert decision.status == NEEDS_FURTHER_RESOLUTION
    assert decision.evidence == "both_differing_tokens_established"


def test_one_low_and_one_high_differing_support_can_resolve() -> None:
    decision = _decision(
        "FARMACIA SHEKAINAH",
        "FARMACIA SHEKINAH",
        support_overrides={"SHEKAINAH": 1, "SHEKINAH": 80},
    )
    assert decision.status == STRONG_ORTHOGRAPHIC_EVIDENCE
    assert (decision.token_support_a, decision.token_support_b) == (1, 80)


@pytest.mark.parametrize("variant_count", [3, 4])
def test_multi_variant_context_abstains(variant_count: int) -> None:
    variants = ["SHEKAINAH", "SHEKINAH", "SHEKANAH", "SHEKYNAH"]
    decision = _decision(
        "FARMACIA SHEKAINAH",
        "FARMACIA SHEKINAH",
        context_variants=variants[:variant_count],
    )
    assert decision.status == NEEDS_FURTHER_RESOLUTION
    assert decision.evidence == "multi_variant_context"
    assert decision.context_variant_count == variant_count


def test_common_exact_context_is_not_distinctive() -> None:
    decision = _decision(
        "ALPHAAA CAPITAL",
        "ALPHBAA CAPITAL",
        support_overrides={"CAPITAL": 51},
    )
    assert decision.status == NEEDS_FURTHER_RESOLUTION
    assert decision.evidence == "no_distinctive_exact_context"


def test_at_least_one_low_support_exact_context_is_distinctive() -> None:
    decision = _decision(
        "UNCOMMON ALPHAAA CAPITAL",
        "UNCOMMON ALPHBAA CAPITAL",
        support_overrides={"UNCOMMON": 2, "CAPITAL": 100},
    )
    assert decision.status == STRONG_ORTHOGRAPHIC_EVIDENCE


def test_single_token_typo_without_context_abstains() -> None:
    decision = _decision("PACHOS", "PANCHOS")
    assert decision.status == NEEDS_FURTHER_RESOLUTION
    assert decision.evidence == "no_exact_informative_context"


@pytest.mark.parametrize(
    ("name_a", "name_b"),
    [
        ("SUBWAY", "SUBWAY INTERNATIONAL BV"),
        ("BANESCO BANCO UNIVERSAL", "BANESCO SEGUROS"),
    ],
)
def test_different_core_token_counts_abstain(name_a: str, name_b: str) -> None:
    decision = _decision(name_a, name_b)
    assert decision.status == NEEDS_FURTHER_RESOLUTION
    assert decision.evidence == "core_token_count_mismatch"


@pytest.mark.parametrize("numeric_relation", ["one_sided", "conflict"])
def test_incompatible_persisted_numeric_relation_abstains(numeric_relation: str) -> None:
    decision = _decision(
        "ALPHAAA CAPITAL",
        "ALPHBAA CAPITAL",
        numeric_relation=numeric_relation,
    )
    assert decision.status == NEEDS_FURTHER_RESOLUTION
    assert decision.evidence == "numeric_evidence_incompatible"


def test_same_numeric_evidence_can_coexist_with_rule() -> None:
    decision = _decision(
        "HOTEL 507 ALPHAAA",
        "HOTEL 507 ALPHBAA",
        numeric_relation="same",
    )
    assert decision.status == STRONG_ORTHOGRAPHIC_EVIDENCE


def test_two_differing_tokens_abstain() -> None:
    decision = _decision("COMPAIA PANAMEA DE AVIACION", "COMPANIA PANAMENA DE AVIACION")
    assert decision.evidence == "multiple_core_token_differences"


def test_token_reordering_abstains() -> None:
    assert _decision("GLOBAL TECH", "TECH GLOBAL").evidence == ("multiple_core_token_differences")


def test_edit_distance_two_abstains() -> None:
    assert _decision("ABCDEF CAPITAL", "ABXYEF CAPITAL").evidence == ("edit_distance_not_one")


def test_terminal_legal_suffix_is_not_context() -> None:
    assert _decision("VERMON SA", "VERMAN SA").evidence == ("no_exact_informative_context")


def test_generic_token_is_not_sufficient_context() -> None:
    assert _decision("VERMON GRUPO", "VERMAN GRUPO").evidence == ("no_exact_informative_context")


@pytest.mark.parametrize(
    ("name_a", "name_b"),
    [("ABC CAPITAL", "ABD CAPITAL"), ("ALPHA CAPITAL", "ALPHE CAPITAL")],
)
def test_short_differing_tokens_abstain(name_a: str, name_b: str) -> None:
    assert _decision(name_a, name_b).evidence == "short_typo_token"


def test_terminal_recognized_legal_suffix_may_be_removed() -> None:
    decision = _decision("FARMACIA SHEKAINAH SA", "FARMACIA SHEKINAH")
    assert decision.status == STRONG_ORTHOGRAPHIC_EVIDENCE
    assert orthographic_core_tokens("FARMACIA SHEKAINAH S A", POLICY) == (
        "FARMACIA",
        "SHEKAINAH",
    )


def test_nonterminal_suffix_like_text_is_not_removed() -> None:
    assert _decision("SA VERMON CAPITAL", "VERMAN CAPITAL").evidence == (
        "core_token_count_mismatch"
    )


def test_legal_designator_cannot_be_differing_token() -> None:
    assert _decision("SA CAPITAL", "SAS CAPITAL").evidence == (
        "legal_designator_is_differing_token"
    )


def test_generic_token_cannot_be_differing_token() -> None:
    assert _decision("GRUPO CAPITAL", "GRUPA CAPITAL").evidence == (
        "generic_token_is_differing_token"
    )


def test_alphanumeric_differing_token_abstains() -> None:
    assert _decision("ACME1 CAPITAL", "ACME2 CAPITAL").evidence == (
        "differing_token_not_alphabetic"
    )


@pytest.mark.parametrize(
    ("employer_a", "employer_b"),
    [(False, True), (True, False), (False, False)],
)
def test_both_endpoints_must_be_employer_candidates(
    employer_a: bool,
    employer_b: bool,
) -> None:
    decision = _decision(
        "VERMON CAPITAL",
        "VERMAN CAPITAL",
        employer_a=employer_a,
        employer_b=employer_b,
    )
    assert decision.status == NOT_ELIGIBLE_FOR_ORTHOGRAPHIC
    assert decision.evidence == "not_both_employer_candidate"


def test_success_trace_is_complete_and_deterministic() -> None:
    first = _decision("FARMACIA SHEKAINAH", "FARMACIA SHEKINAH")
    second = _decision("FARMACIA SHEKAINAH", "FARMACIA SHEKINAH")
    assert first == second
    assert first.differing_token_a == "SHEKAINAH"
    assert first.differing_token_b == "SHEKINAH"
    assert first.context_signature == "FARMACIA <DIFF>"
    assert first.context_variant_count == 2


def test_positive_status_is_evidence_not_final_identity() -> None:
    decision = _decision("FARMACIA SHEKAINAH", "FARMACIA SHEKINAH")
    assert decision.status == STRONG_ORTHOGRAPHIC_EVIDENCE
    assert decision.status not in {"AUTO_SAME", "SAME_ENTITY"}


def test_no_fuzzy_score_or_similarity_threshold_controls_rule() -> None:
    signature = inspect.signature(decide_orthographic_pair)
    source = inspect.getsource(orthographic_module)
    forbidden = ("char_ratio", "token_set_ratio", "partial_ratio", "similarity_threshold")
    assert all(parameter not in signature.parameters for parameter in forbidden)
    assert all(term not in source for term in forbidden)
