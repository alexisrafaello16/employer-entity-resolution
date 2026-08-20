"""Deterministic resolution-key mention typing, separate from entity resolution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Literal

type EligibilityStatus = Literal[
    "EMPLOYER_CANDIDATE",
    "ADDRESS",
    "NON_EMPLOYER_STATUS",
    "AMBIGUOUS",
]
type EligibilityRule = Literal[
    "invalid_content_abstention",
    "mixed_evidence_abstention",
    "structural_street_zip_region_address",
    "existing_address_signal",
    "existing_non_employer_status",
    "descriptive_activity_abstention",
    "organization_compatible",
    "prior_employer_route",
    "insufficient_evidence_abstention",
]
type SignalStrength = Literal["none", "weak", "moderate", "strong"]

EMPLOYER_CANDIDATE: Final = "EMPLOYER_CANDIDATE"
ADDRESS: Final = "ADDRESS"
NON_EMPLOYER_STATUS: Final = "NON_EMPLOYER_STATUS"
AMBIGUOUS: Final = "AMBIGUOUS"
RULE_PRECEDENCE: Final[tuple[EligibilityRule, ...]] = (
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
_STRENGTH_ORDER: Final[dict[SignalStrength, int]] = {
    "none": 0,
    "weak": 1,
    "moderate": 2,
    "strong": 3,
}


@dataclass(slots=True)
class EligibilityEvidenceAggregate:
    """Mutable aggregation of persisted preprocessing evidence for one strict key."""

    source_rows: int = 0
    routes: set[str] = field(default_factory=set)
    is_blank: bool = False
    is_numeric_only: bool = False
    has_address_signal: bool = False
    address_signal_strength: SignalStrength = "none"
    has_occupation_signal: bool = False
    occupation_signal_strength: SignalStrength = "none"
    has_activity_description_signal: bool = False
    has_corporate_suffix: bool = False
    has_organization_like_tokens: bool = False
    mixed_address_organization_signal: bool = False
    mixed_occupation_organization_signal: bool = False


@dataclass(frozen=True, slots=True)
class EligibilityEvidence:
    """Complete compact evidence used to classify one resolution key."""

    resolution_key: str
    representative_name: str
    representative_route: str
    source_rows: int
    routes: frozenset[str]
    is_blank: bool
    is_numeric_only: bool
    has_address_signal: bool
    address_signal_strength: SignalStrength
    has_occupation_signal: bool
    occupation_signal_strength: SignalStrength
    has_activity_description_signal: bool
    has_corporate_suffix: bool
    has_organization_like_tokens: bool
    mixed_address_organization_signal: bool
    mixed_occupation_organization_signal: bool
    possible_truncation: bool
    token_count: int


@dataclass(frozen=True, slots=True)
class EligibilityDecision:
    """One automatic mention-type classification with compact evidence."""

    status: EligibilityStatus
    rule: EligibilityRule
    evidence: str


def _stronger_strength(current: SignalStrength, observed: SignalStrength) -> SignalStrength:
    return observed if _STRENGTH_ORDER[observed] > _STRENGTH_ORDER[current] else current


def update_evidence_aggregate(
    aggregate: EligibilityEvidenceAggregate,
    row: dict[str, object],
) -> None:
    """Aggregate existing preprocessing fields without deriving new normalization."""
    route = row.get("route")
    address_strength = row.get("address_signal_strength")
    occupation_strength = row.get("occupation_signal_strength")
    if not isinstance(route, str):
        raise ValueError("Preprocessing route must be a string")
    if address_strength not in _STRENGTH_ORDER:
        raise ValueError(f"Invalid address signal strength: {address_strength!r}")
    if occupation_strength not in _STRENGTH_ORDER:
        raise ValueError(f"Invalid occupation signal strength: {occupation_strength!r}")
    aggregate.source_rows += 1
    aggregate.routes.add(route)
    aggregate.is_blank = aggregate.is_blank or bool(row.get("is_blank"))
    aggregate.is_numeric_only = aggregate.is_numeric_only or bool(row.get("is_numeric_only"))
    aggregate.has_address_signal = aggregate.has_address_signal or bool(
        row.get("has_address_signal")
    )
    aggregate.address_signal_strength = _stronger_strength(
        aggregate.address_signal_strength, address_strength
    )
    aggregate.has_occupation_signal = aggregate.has_occupation_signal or bool(
        row.get("has_occupation_signal")
    )
    aggregate.occupation_signal_strength = _stronger_strength(
        aggregate.occupation_signal_strength, occupation_strength
    )
    aggregate.has_activity_description_signal = aggregate.has_activity_description_signal or bool(
        row.get("has_activity_description_signal")
    )
    aggregate.has_corporate_suffix = aggregate.has_corporate_suffix or bool(
        row.get("has_corporate_suffix")
    )
    aggregate.has_organization_like_tokens = aggregate.has_organization_like_tokens or bool(
        row.get("has_organization_like_tokens")
    )
    aggregate.mixed_address_organization_signal = (
        aggregate.mixed_address_organization_signal
        or bool(row.get("mixed_address_organization_signal"))
    )
    aggregate.mixed_occupation_organization_signal = (
        aggregate.mixed_occupation_organization_signal
        or bool(row.get("mixed_occupation_organization_signal"))
    )


def finalize_evidence(
    *,
    resolution_key: str,
    representative_name: str,
    representative_route: str,
    possible_truncation: bool,
    token_count: int,
    aggregate: EligibilityEvidenceAggregate,
) -> EligibilityEvidence:
    """Join one resolution key to its deterministically aggregated source evidence."""
    if aggregate.source_rows == 0:
        raise ValueError(f"No preprocessing evidence found for resolution key: {resolution_key!r}")
    return EligibilityEvidence(
        resolution_key=resolution_key,
        representative_name=representative_name,
        representative_route=representative_route,
        source_rows=aggregate.source_rows,
        routes=frozenset(aggregate.routes),
        is_blank=aggregate.is_blank,
        is_numeric_only=aggregate.is_numeric_only,
        has_address_signal=aggregate.has_address_signal,
        address_signal_strength=aggregate.address_signal_strength,
        has_occupation_signal=aggregate.has_occupation_signal,
        occupation_signal_strength=aggregate.occupation_signal_strength,
        has_activity_description_signal=aggregate.has_activity_description_signal,
        has_corporate_suffix=aggregate.has_corporate_suffix,
        has_organization_like_tokens=aggregate.has_organization_like_tokens,
        mixed_address_organization_signal=aggregate.mixed_address_organization_signal,
        mixed_occupation_organization_signal=aggregate.mixed_occupation_organization_signal,
        possible_truncation=possible_truncation,
        token_count=token_count,
    )


def has_structural_street_address(
    name: str,
    road_type_tokens: frozenset[str],
) -> bool:
    """Require leading number, road type, ZIP, region code, and trailing locality."""
    tokens = tuple(name.split())
    if len(tokens) < 6 or not tokens[0].isdigit():
        return False
    for zip_index, token in enumerate(tokens):
        if len(token) != 5 or not token.isdigit() or zip_index < 3:
            continue
        if not any(road_token in road_type_tokens for road_token in tokens[1:zip_index]):
            continue
        if zip_index + 2 >= len(tokens):
            continue
        region = tokens[zip_index + 1]
        locality = tokens[zip_index + 2 :]
        if len(region) == 2 and region.isalpha() and any(part.isalpha() for part in locality):
            return True
    return False


def classify_eligibility(
    evidence: EligibilityEvidence,
    *,
    road_type_tokens: frozenset[str],
) -> EligibilityDecision:
    """Classify mention type in explicit fail-closed precedence order."""
    if (
        evidence.is_blank
        or evidence.is_numeric_only
        or "blank_candidate" in evidence.routes
    ):
        return EligibilityDecision(
            AMBIGUOUS,
            "invalid_content_abstention",
            "blank_or_numeric_only_content",
        )

    structural_address = has_structural_street_address(
        evidence.representative_name, road_type_tokens
    )
    has_organization_evidence = (
        evidence.has_corporate_suffix or evidence.has_organization_like_tokens
    )
    has_existing_address_evidence = (
        evidence.address_signal_strength in {"moderate", "strong"}
        or "address_candidate" in evidence.routes
    )
    has_existing_non_employer_evidence = (
        evidence.occupation_signal_strength == "strong"
        or "non_employer_status_candidate" in evidence.routes
    )
    mixed_evidence = (
        evidence.mixed_address_organization_signal
        or evidence.mixed_occupation_organization_signal
        or (
            (structural_address or has_existing_address_evidence)
            and (evidence.has_occupation_signal or has_existing_non_employer_evidence)
        )
        or ((structural_address or has_existing_address_evidence) and has_organization_evidence)
        or (
            (evidence.has_occupation_signal or has_existing_non_employer_evidence)
            and has_organization_evidence
        )
    )
    if mixed_evidence:
        return EligibilityDecision(
            AMBIGUOUS,
            "mixed_evidence_abstention",
            "conflicting_address_occupation_or_organization_evidence",
        )

    if structural_address:
        return EligibilityDecision(
            ADDRESS,
            "structural_street_zip_region_address",
            "leading_number_road_type_zip_region_and_locality",
        )

    if has_existing_address_evidence:
        return EligibilityDecision(
            ADDRESS,
            "existing_address_signal",
            "existing_moderate_or_strong_address_evidence",
        )

    if has_existing_non_employer_evidence:
        return EligibilityDecision(
            NON_EMPLOYER_STATUS,
            "existing_non_employer_status",
            "existing_pure_status_or_non_employer_route",
        )

    if evidence.has_occupation_signal or evidence.has_activity_description_signal:
        return EligibilityDecision(
            AMBIGUOUS,
            "descriptive_activity_abstention",
            "descriptive_occupation_or_activity_requires_abstention",
        )

    if evidence.has_address_signal:
        return EligibilityDecision(
            AMBIGUOUS,
            "insufficient_evidence_abstention",
            "weak_or_prior_ambiguous_evidence",
        )

    if has_organization_evidence:
        return EligibilityDecision(
            EMPLOYER_CANDIDATE,
            "organization_compatible",
            "corporate_suffix_or_organization_token_without_exclusion",
        )

    if evidence.representative_route == "employer_resolution_candidate":
        return EligibilityDecision(
            EMPLOYER_CANDIDATE,
            "prior_employer_route",
            "prior_employer_route_without_stronger_exclusion",
        )

    return EligibilityDecision(
        AMBIGUOUS,
        "insufficient_evidence_abstention",
        "insufficient_evidence_for_safe_mention_type",
    )
