"""Deterministic structural profiling for unresolved employer-compatible pairs."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Final, Literal

from rapidfuzz.distance import Levenshtein

from credit_risk_er.matching.decision import NumericRelation, exact_source_truncation_relation
from credit_risk_er.matching.orthographic import OrthographicPolicy, orthographic_core_tokens

type PrimaryFamily = Literal[
    "NUMERIC_VARIATION",
    "POSSIBLE_TRUNCATION_RELATIONSHIP",
    "EXACT_TOKEN_REORDER",
    "SINGLE_TOKEN_ADDITION_REMOVAL",
    "MULTI_TOKEN_ADDITION_REMOVAL",
    "EXACT_TOKEN_CONTAINMENT_NONALIGNED",
    "ACRONYM_INITIALISM_PATTERN",
    "MULTIPLE_SMALL_TOKEN_EDITS",
    "SINGLE_TOKEN_LARGER_EDIT",
    "MIXED_STRUCTURAL_RELATIONSHIP",
    "OTHER_RESIDUAL",
]

PRIMARY_FAMILY_PRECEDENCE: Final[tuple[PrimaryFamily, ...]] = (
    "NUMERIC_VARIATION",
    "POSSIBLE_TRUNCATION_RELATIONSHIP",
    "EXACT_TOKEN_REORDER",
    "SINGLE_TOKEN_ADDITION_REMOVAL",
    "MULTI_TOKEN_ADDITION_REMOVAL",
    "EXACT_TOKEN_CONTAINMENT_NONALIGNED",
    "ACRONYM_INITIALISM_PATTERN",
    "MULTIPLE_SMALL_TOKEN_EDITS",
    "SINGLE_TOKEN_LARGER_EDIT",
    "MIXED_STRUCTURAL_RELATIONSHIP",
    "OTHER_RESIDUAL",
)

SMALL_TOKEN_EDIT_DISTANCE: Final = 2


@dataclass(frozen=True, slots=True)
class ResidualRelationshipProfile:
    """Compact structural evidence; no field establishes entity identity."""

    primary_family: PrimaryFamily
    family_evidence: str
    core_token_count_a: int
    core_token_count_b: int
    shared_exact_token_count: int
    differing_token_positions: str | None
    differing_token_count: int
    added_removed_token: str | None
    added_removed_token_count: int
    maximum_token_edit_distance: int | None
    total_token_edit_distance: int | None
    is_token_reorder: bool
    is_ordered_subsequence: bool
    is_token_multiset_containment: bool
    has_initialism_pattern: bool
    possible_truncation_a: bool
    possible_truncation_b: bool
    numeric_relation: NumericRelation


def _proper_subsequence_extras(
    shorter: tuple[str, ...],
    longer: tuple[str, ...],
) -> tuple[str, ...] | None:
    """Return unmatched longer tokens when shorter is a proper ordered subsequence."""
    if len(shorter) >= len(longer):
        return None
    shorter_index = 0
    extras: list[str] = []
    for token in longer:
        if shorter_index < len(shorter) and token == shorter[shorter_index]:
            shorter_index += 1
        else:
            extras.append(token)
    return tuple(extras) if shorter_index == len(shorter) else None


def _counter_contains(container: Counter[str], contained: Counter[str]) -> bool:
    return all(container[token] >= count for token, count in contained.items())


def _initialism_pattern(
    short_tokens: tuple[str, ...],
    long_tokens: tuple[str, ...],
    policy: OrthographicPolicy,
) -> bool:
    if len(short_tokens) != 1 or len(long_tokens) < 2:
        return False
    abbreviation = short_tokens[0]
    # Generic lexical tokens remain in the exact initial sequence (for example,
    # COMPANY in PPC); only configured legal syntax is excluded from initials.
    significant = tuple(
        token
        for token in long_tokens
        if token.isalpha() and token not in policy.legal_designator_tokens
    )
    return (
        abbreviation.isalpha()
        and 2 <= len(abbreviation) <= 10
        and len(significant) >= 2
        and len(abbreviation) == len(significant)
        and abbreviation == "".join(token[0] for token in significant)
    )


def profile_residual_relationship(
    *,
    key_a: str,
    key_b: str,
    name_a: str,
    name_b: str,
    numeric_relation: NumericRelation,
    possible_truncation_a: bool,
    possible_truncation_b: bool,
    source_truncation_boundaries: frozenset[int],
    prior_orthographic_evidence: str,
    policy: OrthographicPolicy,
) -> ResidualRelationshipProfile:
    """Assign one precedence-ordered structural family without deciding identity."""
    core_a = orthographic_core_tokens(name_a, policy)
    core_b = orthographic_core_tokens(name_b, policy)
    counter_a = Counter(core_a)
    counter_b = Counter(core_b)
    shared_exact_token_count = sum((counter_a & counter_b).values())
    is_token_reorder = core_a != core_b and counter_a == counter_b

    extras_when_a_shorter = _proper_subsequence_extras(core_a, core_b)
    extras_when_b_shorter = _proper_subsequence_extras(core_b, core_a)
    ordered_extras = (
        extras_when_a_shorter if extras_when_a_shorter is not None else extras_when_b_shorter
    )
    is_ordered_subsequence = ordered_extras is not None
    contains_a = _counter_contains(counter_b, counter_a)
    contains_b = _counter_contains(counter_a, counter_b)
    is_token_multiset_containment = contains_a or contains_b

    aligned_differences = tuple(
        (index, token_a, token_b)
        for index, (token_a, token_b) in enumerate(zip(core_a, core_b, strict=False))
        if token_a != token_b
    )
    edit_distances = tuple(
        Levenshtein.distance(token_a, token_b) for _, token_a, token_b in aligned_differences
    )
    differing_token_positions = (
        ",".join(str(index) for index, _, _ in aligned_differences) if aligned_differences else None
    )
    maximum_token_edit_distance = max(edit_distances) if edit_distances else None
    total_token_edit_distance = sum(edit_distances) if edit_distances else None
    added_removed_token_count = abs(len(core_a) - len(core_b))
    added_removed_token = (
        ordered_extras[0] if ordered_extras is not None and len(ordered_extras) == 1 else None
    )
    has_initialism_pattern = _initialism_pattern(core_a, core_b, policy) or _initialism_pattern(
        core_b, core_a, policy
    )

    truncation = exact_source_truncation_relation(
        key_a=key_a,
        key_b=key_b,
        name_a=name_a,
        name_b=name_b,
        numeric_relation=numeric_relation,
        possible_truncation_by_key={
            key_a: possible_truncation_a,
            key_b: possible_truncation_b,
        },
        source_truncation_boundaries=source_truncation_boundaries,
    )

    if numeric_relation in {"one_sided", "conflict"}:
        family: PrimaryFamily = "NUMERIC_VARIATION"
        family_evidence = f"persisted_numeric_relation={numeric_relation}"
    elif truncation is not None:
        family = "POSSIBLE_TRUNCATION_RELATIONSHIP"
        truncated_side = "a" if truncation.truncated_key == key_a else "b"
        family_evidence = f"exact_source_boundary_prefix;truncated_side={truncated_side}"
    elif is_token_reorder:
        family = "EXACT_TOKEN_REORDER"
        family_evidence = "equal_token_multiset_with_different_sequence"
    elif ordered_extras is not None and len(ordered_extras) == 1:
        family = "SINGLE_TOKEN_ADDITION_REMOVAL"
        family_evidence = f"ordered_subsequence;additional_token={ordered_extras[0]}"
    elif ordered_extras is not None and len(ordered_extras) >= 2:
        family = "MULTI_TOKEN_ADDITION_REMOVAL"
        family_evidence = f"ordered_subsequence;additional_token_count={len(ordered_extras)}"
    elif is_token_multiset_containment and len(core_a) != len(core_b):
        family = "EXACT_TOKEN_CONTAINMENT_NONALIGNED"
        family_evidence = "multiplicity_preserving_containment_without_ordered_subsequence"
    elif has_initialism_pattern:
        family = "ACRONYM_INITIALISM_PATTERN"
        family_evidence = "single_token_equals_nonlegal_token_initials"
    elif (
        len(core_a) == len(core_b)
        and len(aligned_differences) >= 2
        and all(
            token_a.isalpha() and token_b.isalpha() for _, token_a, token_b in aligned_differences
        )
        and all(distance <= SMALL_TOKEN_EDIT_DISTANCE for distance in edit_distances)
    ):
        family = "MULTIPLE_SMALL_TOKEN_EDITS"
        family_evidence = (
            f"aligned_differences={len(aligned_differences)};"
            f"maximum_edit_distance={maximum_token_edit_distance};"
            f"total_edit_distance={total_token_edit_distance}"
        )
    elif len(core_a) == len(core_b) and len(aligned_differences) == 1:
        family = "SINGLE_TOKEN_LARGER_EDIT"
        family_evidence = (
            f"single_aligned_token_difference;edit_distance={maximum_token_edit_distance};"
            f"orthographic_evidence={prior_orthographic_evidence}"
        )
    elif (len(core_a) == len(core_b) and len(aligned_differences) >= 2) or (
        len(core_a) != len(core_b)
        and (
            shared_exact_token_count > 0
            or any(distance <= SMALL_TOKEN_EDIT_DISTANCE for distance in edit_distances)
        )
    ):
        family = "MIXED_STRUCTURAL_RELATIONSHIP"
        family_evidence = "multiple_structural_transformations_required"
    else:
        family = "OTHER_RESIDUAL"
        family_evidence = "no_supported_deterministic_structural_family"

    return ResidualRelationshipProfile(
        primary_family=family,
        family_evidence=family_evidence,
        core_token_count_a=len(core_a),
        core_token_count_b=len(core_b),
        shared_exact_token_count=shared_exact_token_count,
        differing_token_positions=differing_token_positions,
        differing_token_count=len(aligned_differences),
        added_removed_token=added_removed_token,
        added_removed_token_count=added_removed_token_count,
        maximum_token_edit_distance=maximum_token_edit_distance,
        total_token_edit_distance=total_token_edit_distance,
        is_token_reorder=is_token_reorder,
        is_ordered_subsequence=is_ordered_subsequence,
        is_token_multiset_containment=is_token_multiset_containment,
        has_initialism_pattern=has_initialism_pattern,
        possible_truncation_a=possible_truncation_a,
        possible_truncation_b=possible_truncation_b,
        numeric_relation=numeric_relation,
    )
