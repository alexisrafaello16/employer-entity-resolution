"""Deterministic exact matching against validated employer knowledge."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from credit_risk_er.employer_master import ExactLookupEntry

type ResolutionStatus = Literal["resolved", "unresolved"]
type ResolutionMethod = Literal["exact_canonical", "exact_alias"]


@dataclass(frozen=True, slots=True)
class ExactResolution:
    entity_id: str | None
    canonical_name: str | None
    resolution_status: ResolutionStatus
    resolution_method: ResolutionMethod | None
    resolution_reason: str


def resolve_exact(
    strict_normalized_name: str | None,
    exact_index: dict[str, ExactLookupEntry],
) -> ExactResolution:
    """Resolve only strict-normalized equality to validated canonical or alias knowledge."""
    entry = exact_index.get(strict_normalized_name) if strict_normalized_name else None
    if entry is None:
        return ExactResolution(
            entity_id=None,
            canonical_name=None,
            resolution_status="unresolved",
            resolution_method=None,
            resolution_reason="no_validated_exact_match",
        )
    method: ResolutionMethod = (
        "exact_canonical" if entry.match_source == "canonical" else "exact_alias"
    )
    reason = (
        "exact_match_validated_canonical"
        if entry.match_source == "canonical"
        else "exact_match_validated_alias"
    )
    return ExactResolution(
        entity_id=entry.entity_id,
        canonical_name=entry.canonical_name,
        resolution_status="resolved",
        resolution_method=method,
        resolution_reason=reason,
    )
