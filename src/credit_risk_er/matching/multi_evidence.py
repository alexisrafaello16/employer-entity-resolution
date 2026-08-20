"""Deterministic integration of approved pair evidence without identity decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

from credit_risk_er.matching.decision import NumericRelation
from credit_risk_er.matching.residual_profile import PrimaryFamily

type AssessmentFamily = Literal[
    "NUMERIC_RISK_WITH_NAME_EVIDENCE",
    "STRUCTURE_WITH_MULTIPLE_DISTINCTIVE_TOKENS",
    "STRUCTURE_WITH_DISTINCTIVE_TOKEN",
    "LOCAL_ORTHOGRAPHIC_WITH_DISTINCTIVE_CONTEXT",
    "MULTI_EDIT_WITH_DISTINCTIVE_CONTEXT",
    "FULL_SHORTER_NAME_OVERLAP",
    "DISTINCTIVE_SINGLE_SIGNAL",
    "HIGH_LEXICAL_WITHOUT_DISTINCTIVE_OVERLAP",
    "EXACT_OVERLAP_WITHOUT_DISTINCTIVE_EVIDENCE",
    "ZERO_EXACT_OVERLAP",
    "WEAK_OR_MIXED_EVIDENCE",
]

ASSESSMENT_FAMILY_PRECEDENCE: Final[tuple[AssessmentFamily, ...]] = (
    "NUMERIC_RISK_WITH_NAME_EVIDENCE",
    "STRUCTURE_WITH_MULTIPLE_DISTINCTIVE_TOKENS",
    "STRUCTURE_WITH_DISTINCTIVE_TOKEN",
    "LOCAL_ORTHOGRAPHIC_WITH_DISTINCTIVE_CONTEXT",
    "MULTI_EDIT_WITH_DISTINCTIVE_CONTEXT",
    "FULL_SHORTER_NAME_OVERLAP",
    "DISTINCTIVE_SINGLE_SIGNAL",
    "HIGH_LEXICAL_WITHOUT_DISTINCTIVE_OVERLAP",
    "EXACT_OVERLAP_WITHOUT_DISTINCTIVE_EVIDENCE",
    "ZERO_EXACT_OVERLAP",
    "WEAK_OR_MIXED_EVIDENCE",
)

# The high-lexical family is intentionally absent: no approved lexical threshold exists.
ACTIVE_ASSESSMENT_FAMILIES: Final[frozenset[AssessmentFamily]] = frozenset(
    family
    for family in ASSESSMENT_FAMILY_PRECEDENCE
    if family != "HIGH_LEXICAL_WITHOUT_DISTINCTIVE_OVERLAP"
)

STRUCTURAL_EXACT_FAMILIES: Final[frozenset[PrimaryFamily]] = frozenset(
    {
        "EXACT_TOKEN_REORDER",
        "SINGLE_TOKEN_ADDITION_REMOVAL",
        "MULTI_TOKEN_ADDITION_REMOVAL",
        "EXACT_TOKEN_CONTAINMENT_NONALIGNED",
        "POSSIBLE_TRUNCATION_RELATIONSHIP",
    }
)

# These approved abstention reasons follow a localized one-token comparison path.
LOCALIZED_ORTHOGRAPHIC_EVIDENCE: Final[frozenset[str]] = frozenset(
    {
        "edit_distance_not_one",
        "short_typo_token",
        "terminal_edit_requires_further_resolution",
        "no_exact_informative_context",
        "no_distinctive_exact_context",
        "multi_variant_context",
        "both_differing_tokens_established",
    }
)


@dataclass(frozen=True, slots=True)
class MultiEvidenceAssessment:
    """One diagnostic convergence family plus its independently auditable flags."""

    assessment_family: AssessmentFamily
    assessment_evidence: str
    has_structural_exact_relation: bool
    has_orthographic_signal: bool
    has_distinctive_exact_overlap: bool
    has_multiple_distinctive_exact_overlap: bool
    has_numeric_risk: bool
    has_zero_exact_overlap: bool
    has_full_shorter_exact_coverage: bool
    has_full_shorter_distinctive_coverage: bool


def assess_multi_evidence(
    *,
    primary_family: PrimaryFamily,
    numeric_relation: NumericRelation,
    shared_exact_token_count: int,
    shared_distinctive_token_count: int,
    has_shared_distinctive_token: bool,
    has_multiple_shared_distinctive_tokens: bool,
    shorter_name_exact_coverage: float,
    shorter_name_distinctive_coverage: float,
    prior_orthographic_evidence: str,
) -> MultiEvidenceAssessment:
    """Describe deterministic evidence convergence without scoring or resolving identity."""
    if shared_exact_token_count < 0 or shared_distinctive_token_count < 0:
        raise ValueError("Token-overlap counts cannot be negative")
    if shared_distinctive_token_count > shared_exact_token_count:
        raise ValueError("Distinctive overlap cannot exceed exact overlap")
    if has_shared_distinctive_token != (shared_distinctive_token_count > 0):
        raise ValueError("Distinctive-overlap flag and count disagree")
    if has_multiple_shared_distinctive_tokens != (shared_distinctive_token_count >= 2):
        raise ValueError("Multiple-distinctive-token flag and count disagree")
    for coverage in (shorter_name_exact_coverage, shorter_name_distinctive_coverage):
        if not 0.0 <= coverage <= 1.0:
            raise ValueError("Coverage must be in the closed interval [0, 1]")

    has_structural_exact_relation = primary_family in STRUCTURAL_EXACT_FAMILIES
    has_orthographic_signal = prior_orthographic_evidence in LOCALIZED_ORTHOGRAPHIC_EVIDENCE
    has_distinctive_exact_overlap = has_shared_distinctive_token
    has_multiple_distinctive_exact_overlap = has_multiple_shared_distinctive_tokens
    has_numeric_risk = numeric_relation in {"one_sided", "conflict"}
    has_zero_exact_overlap = shared_exact_token_count == 0
    has_full_shorter_exact_coverage = shorter_name_exact_coverage == 1.0
    has_full_shorter_distinctive_coverage = shorter_name_distinctive_coverage == 1.0

    # Numeric risk has first precedence only when exact name evidence is present.
    if has_numeric_risk and shared_exact_token_count > 0:
        family: AssessmentFamily = "NUMERIC_RISK_WITH_NAME_EVIDENCE"
    elif has_structural_exact_relation and has_multiple_distinctive_exact_overlap:
        family = "STRUCTURE_WITH_MULTIPLE_DISTINCTIVE_TOKENS"
    elif has_structural_exact_relation and has_distinctive_exact_overlap:
        family = "STRUCTURE_WITH_DISTINCTIVE_TOKEN"
    elif has_orthographic_signal and has_distinctive_exact_overlap:
        family = "LOCAL_ORTHOGRAPHIC_WITH_DISTINCTIVE_CONTEXT"
    elif primary_family == "MULTIPLE_SMALL_TOKEN_EDITS" and has_distinctive_exact_overlap:
        family = "MULTI_EDIT_WITH_DISTINCTIVE_CONTEXT"
    elif has_full_shorter_exact_coverage:
        family = "FULL_SHORTER_NAME_OVERLAP"
    elif has_distinctive_exact_overlap:
        family = "DISTINCTIVE_SINGLE_SIGNAL"
    # HIGH_LEXICAL_WITHOUT_DISTINCTIVE_OVERLAP is reserved and deliberately inactive.
    elif has_numeric_risk:
        family = "WEAK_OR_MIXED_EVIDENCE"
    elif shared_exact_token_count > 0:
        family = "EXACT_OVERLAP_WITHOUT_DISTINCTIVE_EVIDENCE"
    elif has_zero_exact_overlap:
        family = "ZERO_EXACT_OVERLAP"
    else:  # Defensive fallback for future compatible evidence extensions.
        family = "WEAK_OR_MIXED_EVIDENCE"

    evidence = (
        f"primary_family={primary_family};"
        f"numeric_relation={numeric_relation};"
        f"shared_exact_tokens={shared_exact_token_count};"
        f"shared_distinctive_tokens={shared_distinctive_token_count};"
        f"shorter_exact_coverage={shorter_name_exact_coverage:.6f};"
        f"orthographic_evidence={prior_orthographic_evidence}"
    )
    return MultiEvidenceAssessment(
        assessment_family=family,
        assessment_evidence=evidence,
        has_structural_exact_relation=has_structural_exact_relation,
        has_orthographic_signal=has_orthographic_signal,
        has_distinctive_exact_overlap=has_distinctive_exact_overlap,
        has_multiple_distinctive_exact_overlap=has_multiple_distinctive_exact_overlap,
        has_numeric_risk=has_numeric_risk,
        has_zero_exact_overlap=has_zero_exact_overlap,
        has_full_shorter_exact_coverage=has_full_shorter_exact_coverage,
        has_full_shorter_distinctive_coverage=has_full_shorter_distinctive_coverage,
    )
