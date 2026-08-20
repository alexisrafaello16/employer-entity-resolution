"""Precision-first orthographic evidence for employer-compatible residual pairs."""

from __future__ import annotations

from collections.abc import Mapping, Set
from dataclasses import dataclass
from typing import Final, Literal

from rapidfuzz.distance import Levenshtein

from credit_risk_er.matching.candidates import GENERIC_TOKENS

type OrthographicStatus = Literal[
    "STRONG_ORTHOGRAPHIC_EVIDENCE",
    "NEEDS_FURTHER_RESOLUTION",
    "NOT_ELIGIBLE_FOR_ORTHOGRAPHIC",
]
type OrthographicRule = Literal[
    "single_token_edit_equivalence",
    "no_orthographic_equivalence",
    "not_eligible_for_orthographic",
]
type EditOperation = Literal["insertion", "deletion", "substitution"]
type EditLocation = Literal["beginning", "inside", "end"]
type NumericRelation = Literal["none", "same", "one_sided", "conflict"]

STRONG_ORTHOGRAPHIC_EVIDENCE: Final = "STRONG_ORTHOGRAPHIC_EVIDENCE"
NEEDS_FURTHER_RESOLUTION: Final = "NEEDS_FURTHER_RESOLUTION"
NOT_ELIGIBLE_FOR_ORTHOGRAPHIC: Final = "NOT_ELIGIBLE_FOR_ORTHOGRAPHIC"
SINGLE_TOKEN_EDIT_EQUIVALENCE: Final = "single_token_edit_equivalence"
NO_ORTHOGRAPHIC_EQUIVALENCE: Final = "no_orthographic_equivalence"
NOT_ELIGIBLE_RULE: Final = "not_eligible_for_orthographic"
SUCCESS_EVIDENCE: Final = "one_alphabetic_token_edit_distance_1_with_exact_context"


@dataclass(frozen=True, slots=True)
class OrthographicDecision:
    """One gated evidence outcome; a positive status is not final employer identity."""

    status: OrthographicStatus
    rule: OrthographicRule
    evidence: str
    differing_token_a: str | None = None
    differing_token_b: str | None = None
    edit_operation: EditOperation | None = None
    edit_location: EditLocation | None = None
    context_signature: str | None = None
    context_variant_count: int | None = None
    token_support_a: int | None = None
    token_support_b: int | None = None


@dataclass(frozen=True, slots=True)
class OrthographicComparison:
    """A pair that passed the v2 gates and needs population-level v2.1 guards."""

    differing_token_a: str
    differing_token_b: str
    edit_operation: EditOperation
    edit_location: EditLocation
    context_signature_tokens: tuple[str, ...]
    exact_context_tokens: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class OrthographicPolicy:
    """Precomputed immutable legal-token and minimum-length policy."""

    legal_suffixes: tuple[tuple[str, ...], ...]
    legal_designator_tokens: frozenset[str]
    minimum_typo_token_length: int
    minimum_informative_token_length: int


def _configured_legal_suffixes(
    aliases: dict[str, tuple[str, ...]],
) -> tuple[tuple[str, ...], ...]:
    variants = {
        tuple(value.upper().split())
        for canonical, configured_variants in aliases.items()
        for value in (canonical, *configured_variants)
        if value.strip()
    }
    return tuple(sorted(variants, key=lambda tokens: (-len(tokens), tokens)))


def build_orthographic_policy(
    *,
    corporate_suffix_aliases: dict[str, tuple[str, ...]],
    minimum_typo_token_length: int,
    minimum_informative_token_length: int,
) -> OrthographicPolicy:
    """Compile existing configuration once for repeated pair decisions."""
    if minimum_typo_token_length < 2 or minimum_informative_token_length < 2:
        raise ValueError("Orthographic token minimums must be at least two")
    legal_suffixes = _configured_legal_suffixes(corporate_suffix_aliases)
    return OrthographicPolicy(
        legal_suffixes=legal_suffixes,
        legal_designator_tokens=frozenset(token for suffix in legal_suffixes for token in suffix),
        minimum_typo_token_length=minimum_typo_token_length,
        minimum_informative_token_length=minimum_informative_token_length,
    )


def orthographic_core_tokens(
    name: str,
    policy: OrthographicPolicy,
) -> tuple[str, ...]:
    """Remove at most one recognized terminal legal designator mechanically."""
    tokens = tuple(name.split())
    for suffix in policy.legal_suffixes:
        if len(tokens) > len(suffix) and tokens[-len(suffix) :] == suffix:
            return tokens[: -len(suffix)]
    return tokens


def _abstain(
    evidence: str,
    *,
    differing_token_a: str | None = None,
    differing_token_b: str | None = None,
    edit_operation: EditOperation | None = None,
    edit_location: EditLocation | None = None,
) -> OrthographicDecision:
    return OrthographicDecision(
        status=NEEDS_FURTHER_RESOLUTION,
        rule=NO_ORTHOGRAPHIC_EQUIVALENCE,
        evidence=evidence,
        differing_token_a=differing_token_a,
        differing_token_b=differing_token_b,
        edit_operation=edit_operation,
        edit_location=edit_location,
    )


def _edit_details(token_a: str, token_b: str) -> tuple[EditOperation, EditLocation]:
    """Classify the sole edit and its first differing character deterministically."""
    if len(token_a) == len(token_b):
        operation: EditOperation = "substitution"
        edit_index = next(
            index
            for index, (character_a, character_b) in enumerate(zip(token_a, token_b, strict=True))
            if character_a != character_b
        )
        longer_length = len(token_a)
    else:
        if len(token_a) < len(token_b):
            operation = "insertion"
            shorter, longer = token_a, token_b
        else:
            operation = "deletion"
            shorter, longer = token_b, token_a
        edit_index = next(
            (
                index
                for index, (short_character, long_character) in enumerate(
                    zip(shorter, longer, strict=False)
                )
                if short_character != long_character
            ),
            len(shorter),
        )
        longer_length = len(longer)

    if edit_index == 0:
        location: EditLocation = "beginning"
    elif edit_index == longer_length - 1:
        location = "end"
    else:
        location = "inside"
    return operation, location


def prepare_orthographic_pair(
    *,
    name_a: str,
    name_b: str,
    numeric_relation: NumericRelation,
    employer_candidate_a: bool,
    employer_candidate_b: bool,
    policy: OrthographicPolicy,
) -> OrthographicDecision | OrthographicComparison:
    """Apply the original v2 pair-local gates and prepare population evidence."""
    if not employer_candidate_a or not employer_candidate_b:
        return OrthographicDecision(
            status=NOT_ELIGIBLE_FOR_ORTHOGRAPHIC,
            rule=NOT_ELIGIBLE_RULE,
            evidence="not_both_employer_candidate",
        )
    if numeric_relation not in {"none", "same"}:
        return _abstain("numeric_evidence_incompatible")

    core_a = orthographic_core_tokens(name_a, policy)
    core_b = orthographic_core_tokens(name_b, policy)
    if len(core_a) != len(core_b):
        return _abstain("core_token_count_mismatch")

    differing_positions = tuple(
        index
        for index, (token_a, token_b) in enumerate(zip(core_a, core_b, strict=True))
        if token_a != token_b
    )
    if not differing_positions:
        return _abstain("no_core_token_difference")
    if len(differing_positions) != 1:
        return _abstain("multiple_core_token_differences")

    differing_index = differing_positions[0]
    token_a = core_a[differing_index]
    token_b = core_b[differing_index]
    if not token_a.isalpha() or not token_b.isalpha():
        return _abstain(
            "differing_token_not_alphabetic",
            differing_token_a=token_a,
            differing_token_b=token_b,
        )

    if token_a in policy.legal_designator_tokens or token_b in policy.legal_designator_tokens:
        return _abstain(
            "legal_designator_is_differing_token",
            differing_token_a=token_a,
            differing_token_b=token_b,
        )
    if token_a in GENERIC_TOKENS or token_b in GENERIC_TOKENS:
        return _abstain(
            "generic_token_is_differing_token",
            differing_token_a=token_a,
            differing_token_b=token_b,
        )
    if min(len(token_a), len(token_b)) < policy.minimum_typo_token_length:
        return _abstain(
            "short_typo_token",
            differing_token_a=token_a,
            differing_token_b=token_b,
        )
    if Levenshtein.distance(token_a, token_b) != 1:
        return _abstain(
            "edit_distance_not_one",
            differing_token_a=token_a,
            differing_token_b=token_b,
        )

    edit_operation, edit_location = _edit_details(token_a, token_b)

    exact_context = tuple(
        token_a_value
        for index, (token_a_value, token_b_value) in enumerate(zip(core_a, core_b, strict=True))
        if (
            index != differing_index
            and token_a_value == token_b_value
            and token_a_value.isalpha()
            and len(token_a_value) >= policy.minimum_informative_token_length
            and token_a_value not in GENERIC_TOKENS
            and token_a_value not in policy.legal_designator_tokens
        )
    )
    if not exact_context:
        return _abstain(
            "no_exact_informative_context",
            differing_token_a=token_a,
            differing_token_b=token_b,
            edit_operation=edit_operation,
            edit_location=edit_location,
        )

    context_signature = tuple(
        "<DIFF>" if index == differing_index else token for index, token in enumerate(core_a)
    )
    return OrthographicComparison(
        differing_token_a=token_a,
        differing_token_b=token_b,
        edit_operation=edit_operation,
        edit_location=edit_location,
        context_signature_tokens=context_signature,
        exact_context_tokens=exact_context,
    )


def finalize_orthographic_comparison(
    comparison: OrthographicComparison,
    *,
    token_support: Mapping[str, int],
    context_variants: Mapping[tuple[str, ...], Set[str]],
    maximum_token_frequency: int,
) -> OrthographicDecision:
    """Apply deterministic v2.1 population guards to a prepared comparison."""
    support_a = token_support.get(comparison.differing_token_a, 0)
    support_b = token_support.get(comparison.differing_token_b, 0)
    if support_a < 1 or support_b < 1:
        raise ValueError("Differing-token support is missing from the employer universe")

    context_signature = " ".join(comparison.context_signature_tokens)
    variants = context_variants.get(comparison.context_signature_tokens)
    context_variant_count = len(variants) if variants is not None else None

    def traced_decision(
        status: OrthographicStatus,
        rule: OrthographicRule,
        evidence: str,
    ) -> OrthographicDecision:
        return OrthographicDecision(
            status=status,
            rule=rule,
            evidence=evidence,
            differing_token_a=comparison.differing_token_a,
            differing_token_b=comparison.differing_token_b,
            edit_operation=comparison.edit_operation,
            edit_location=comparison.edit_location,
            context_signature=context_signature,
            context_variant_count=context_variant_count,
            token_support_a=support_a,
            token_support_b=support_b,
        )

    if variants is None or context_variant_count is None or context_variant_count < 2:
        raise RuntimeError("Prepared context signature is absent from its employer index")

    # Precedence is precision-first: terminal morphology, distinctive context,
    # context ambiguity, then differing-token establishment.
    if comparison.edit_location == "end":
        return traced_decision(
            NEEDS_FURTHER_RESOLUTION,
            NO_ORTHOGRAPHIC_EQUIVALENCE,
            "terminal_edit_requires_further_resolution",
        )
    if not any(
        token_support.get(token, 0) <= maximum_token_frequency
        for token in comparison.exact_context_tokens
    ):
        return traced_decision(
            NEEDS_FURTHER_RESOLUTION,
            NO_ORTHOGRAPHIC_EQUIVALENCE,
            "no_distinctive_exact_context",
        )
    if context_variant_count >= 3:
        return traced_decision(
            NEEDS_FURTHER_RESOLUTION,
            NO_ORTHOGRAPHIC_EQUIVALENCE,
            "multi_variant_context",
        )
    if support_a > maximum_token_frequency and support_b > maximum_token_frequency:
        return traced_decision(
            NEEDS_FURTHER_RESOLUTION,
            NO_ORTHOGRAPHIC_EQUIVALENCE,
            "both_differing_tokens_established",
        )
    return traced_decision(
        STRONG_ORTHOGRAPHIC_EVIDENCE,
        SINGLE_TOKEN_EDIT_EQUIVALENCE,
        SUCCESS_EVIDENCE,
    )


def decide_orthographic_pair(
    *,
    name_a: str,
    name_b: str,
    numeric_relation: NumericRelation,
    employer_candidate_a: bool,
    employer_candidate_b: bool,
    policy: OrthographicPolicy,
    token_support: Mapping[str, int],
    context_variants: Mapping[tuple[str, ...], Set[str]],
    maximum_token_frequency: int,
) -> OrthographicDecision:
    """Apply the v2 pair-local gates followed by the v2.1 population guards."""
    prepared = prepare_orthographic_pair(
        name_a=name_a,
        name_b=name_b,
        numeric_relation=numeric_relation,
        employer_candidate_a=employer_candidate_a,
        employer_candidate_b=employer_candidate_b,
        policy=policy,
    )
    if isinstance(prepared, OrthographicDecision):
        return prepared
    return finalize_orthographic_comparison(
        prepared,
        token_support=token_support,
        context_variants=context_variants,
        maximum_token_frequency=maximum_token_frequency,
    )
