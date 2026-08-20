"""Exact shared-token distinctiveness evidence without identity decisions."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass

from credit_risk_er.matching.candidates import GENERIC_TOKENS
from credit_risk_er.matching.orthographic import OrthographicPolicy, orthographic_core_tokens

EVIDENCE_PRECISION = 6


@dataclass(frozen=True, slots=True)
class DistinctiveNameEvidence:
    """Compact exact-overlap evidence; no field is an identity outcome or score."""

    shared_exact_tokens: str
    shared_distinctive_tokens: str
    shared_exact_token_count: int
    shared_distinctive_token_count: int
    shared_generic_token_count: int
    minimum_shared_token_support: int | None
    maximum_shared_token_support: int | None
    minimum_distinctive_token_support: int | None
    maximum_distinctive_token_support: int | None
    exact_coverage_a: float
    exact_coverage_b: float
    distinctive_coverage_a: float
    distinctive_coverage_b: float
    shorter_name_exact_coverage: float
    longer_name_exact_coverage: float
    shorter_name_distinctive_coverage: float
    longer_name_distinctive_coverage: float
    has_shared_exact_token: bool
    has_shared_distinctive_token: bool
    has_multiple_shared_distinctive_tokens: bool
    has_exact_overlap_without_distinctive_token: bool


def employer_core_token_membership(
    name: str,
    policy: OrthographicPolicy,
) -> frozenset[str]:
    """Return distinct core tokens so one key contributes at most one support count."""
    return frozenset(orthographic_core_tokens(name, policy))


def _coverage(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, EVIDENCE_PRECISION) if denominator else 0.0


def compute_distinctive_name_evidence(
    *,
    name_a: str,
    name_b: str,
    token_support: Mapping[str, int],
    maximum_token_frequency: int,
    policy: OrthographicPolicy,
) -> DistinctiveNameEvidence:
    """Describe exact token overlap and empirical rarity without deciding identity."""
    core_a = orthographic_core_tokens(name_a, policy)
    core_b = orthographic_core_tokens(name_b, policy)
    shared_counter = Counter(core_a) & Counter(core_b)
    shared_tokens_with_multiplicity = tuple(sorted(shared_counter.elements()))
    shared_exact_token_count = len(shared_tokens_with_multiplicity)

    missing_support = sorted(token for token in shared_counter if token_support.get(token, 0) < 1)
    if missing_support:
        raise ValueError(f"Shared tokens are absent from employer support: {missing_support}")

    distinctive_counter = Counter(
        {
            token: multiplicity
            for token, multiplicity in shared_counter.items()
            if (
                token.isalpha()
                and len(token) >= policy.minimum_informative_token_length
                and token not in policy.legal_designator_tokens
                and token not in GENERIC_TOKENS
                and token_support[token] <= maximum_token_frequency
            )
        }
    )
    distinctive_tokens_with_multiplicity = tuple(sorted(distinctive_counter.elements()))
    shared_distinctive_token_count = len(distinctive_tokens_with_multiplicity)
    shared_generic_token_count = sum(
        multiplicity for token, multiplicity in shared_counter.items() if token in GENERIC_TOKENS
    )

    shared_nonlegal_supports = tuple(
        token_support[token]
        for token in shared_counter
        if token not in policy.legal_designator_tokens
    )
    distinctive_supports = tuple(token_support[token] for token in distinctive_counter)
    exact_coverage_a = _coverage(shared_exact_token_count, len(core_a))
    exact_coverage_b = _coverage(shared_exact_token_count, len(core_b))
    distinctive_coverage_a = _coverage(shared_distinctive_token_count, len(core_a))
    distinctive_coverage_b = _coverage(shared_distinctive_token_count, len(core_b))

    if len(core_a) <= len(core_b):
        shorter_exact = exact_coverage_a
        longer_exact = exact_coverage_b
        shorter_distinctive = distinctive_coverage_a
        longer_distinctive = distinctive_coverage_b
    else:
        shorter_exact = exact_coverage_b
        longer_exact = exact_coverage_a
        shorter_distinctive = distinctive_coverage_b
        longer_distinctive = distinctive_coverage_a

    has_shared_exact_token = shared_exact_token_count > 0
    has_shared_distinctive_token = shared_distinctive_token_count > 0
    return DistinctiveNameEvidence(
        shared_exact_tokens="|".join(shared_tokens_with_multiplicity),
        shared_distinctive_tokens="|".join(distinctive_tokens_with_multiplicity),
        shared_exact_token_count=shared_exact_token_count,
        shared_distinctive_token_count=shared_distinctive_token_count,
        shared_generic_token_count=shared_generic_token_count,
        minimum_shared_token_support=(
            min(shared_nonlegal_supports) if shared_nonlegal_supports else None
        ),
        maximum_shared_token_support=(
            max(shared_nonlegal_supports) if shared_nonlegal_supports else None
        ),
        minimum_distinctive_token_support=(
            min(distinctive_supports) if distinctive_supports else None
        ),
        maximum_distinctive_token_support=(
            max(distinctive_supports) if distinctive_supports else None
        ),
        exact_coverage_a=exact_coverage_a,
        exact_coverage_b=exact_coverage_b,
        distinctive_coverage_a=distinctive_coverage_a,
        distinctive_coverage_b=distinctive_coverage_b,
        shorter_name_exact_coverage=shorter_exact,
        longer_name_exact_coverage=longer_exact,
        shorter_name_distinctive_coverage=shorter_distinctive,
        longer_name_distinctive_coverage=longer_distinctive,
        has_shared_exact_token=has_shared_exact_token,
        has_shared_distinctive_token=has_shared_distinctive_token,
        has_multiple_shared_distinctive_tokens=shared_distinctive_token_count >= 2,
        has_exact_overlap_without_distinctive_token=(
            has_shared_exact_token and not has_shared_distinctive_token
        ),
    )
