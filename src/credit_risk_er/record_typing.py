"""Conservative structural signals and non-terminal preprocessing routes."""

from __future__ import annotations

import re
from collections.abc import Set
from dataclasses import dataclass
from functools import cache
from typing import Literal

from credit_risk_er.config import RecordTypingConfig

type SignalStrength = Literal["none", "weak", "moderate", "strong"]
type RecordRoute = Literal[
    "blank_candidate",
    "address_candidate",
    "non_employer_status_candidate",
    "employer_resolution_candidate",
    "ambiguous_review_candidate",
]


@dataclass(frozen=True, slots=True)
class RecordTyping:
    """Compact structural features retained for later resolution and review."""

    is_blank: bool
    is_numeric_only: bool
    has_address_signal: bool
    address_signal_strength: SignalStrength
    has_occupation_signal: bool
    occupation_signal_strength: SignalStrength
    has_corporate_suffix: bool
    has_organization_like_tokens: bool
    mixed_address_organization_signal: bool
    mixed_occupation_organization_signal: bool
    has_activity_description_signal: bool
    token_count: int
    normalized_length: int
    route: RecordRoute
    route_reason: str


@cache
def _configured_set(configured: tuple[str, ...]) -> frozenset[str]:
    return frozenset(configured)


def _ordered_matches(
    tokens: tuple[str, ...],
    configured: tuple[str, ...],
) -> tuple[str, ...]:
    configured_set = _configured_set(configured)

    return tuple(
        dict.fromkeys(
            token
            for token in tokens
            if token in configured_set
        )
    )


def _contains_phrase(
    tokens: tuple[str, ...],
    phrase_tokens: tuple[str, ...],
) -> bool:
    width = len(phrase_tokens)

    return any(
        tokens[index : index + width] == phrase_tokens
        for index in range(len(tokens) - width + 1)
    )


@cache
def _configured_phrases(
    phrases: tuple[str, ...],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    return tuple(
        (phrase, tuple(phrase.split()))
        for phrase in sorted(
            phrases,
            key=lambda item: (-len(item.split()), item),
        )
    )


def _matched_status(
    tokens: tuple[str, ...],
    phrases: tuple[str, ...],
) -> str | None:
    return next(
        (
            phrase
            for phrase, phrase_tokens in _configured_phrases(phrases)
            if _contains_phrase(tokens, phrase_tokens)
        ),
        None,
    )


# Narrow status variants observed in the real source data.
#
# These patterns deliberately do not implement generic fuzzy matching.
# They only recover specific status-language structures that the exact
# configured token-sequence matcher cannot recognize because of
# concatenation, truncation, numeric suffixes, or narrowly observed typos.
#
# A full-value match may represent a strong non-employer status.
# An embedded match only contributes occupation evidence; organization
# evidence is evaluated separately and can force later abstention.
_OBSERVED_PURE_STATUS_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"DESEMPLEAD(?:O|A|0)[A-Z0-9]*(?:\s+.*)?"
    ),
    re.compile(
        r"AMA DE CASA[A-Z0-9]*(?:\s+.*)?"
    ),
    re.compile(
        r"(?:"
        r"DEPENDIENTE ECON[A-Z0-9]*|"
        r"DEPENDE ECON[A-Z0-9]*|"
        r"DEPEND ECON[A-Z0-9]*|"
        r"DENEDE ECON[A-Z0-9]*"
        r")(?:\s+.*)?"
    ),
    re.compile(
        r"(?:"
        r"NO LABOR[A-Z0-9]*|"
        r"NO TRABAJ[A-Z0-9]*|"
        r"NO ESTA(?:\s+[A-Z0-9]+){0,2}\s+"
        r"(?:TRABAJ|LABOR)[A-Z0-9]*"
        r")(?:\s+.*)?"
    ),
    re.compile(
        r"ACTUAL(?:MENTE|METE)\s+NO\s+"
        r"(?:"
        r"(?:ESTA\s+)?(?:TRABAJ|LABOR)[A-Z0-9]*|"
        r"EJERCE\s+PROFESION"
        r")(?:\s+.*)?"
    ),
    re.compile(
        r"(?:"
        r"(?:EN ESTOS MOMENTOS\s+)?"
        r"(?:AUN\s+)?"
        r"NO\s+(?:TIENE|TENGO)\s+EMPLEO|"
        r"NO\s+TIENE\s+TRABAJO|"
        r"SIN\s+(?:EMPLEO|TRABAJO)"
        r")(?:\s+.*)?"
    ),
    re.compile(
        r"ACTUALMENTE"
        r"(?:\s+ACABA\s+DE\s+QUEDAR)?"
        r"\s+CESANTE"
    ),
    re.compile(
        r"CESANTE(?:\s+ACTUALMENTE)?"
    ),
    re.compile(
        r"CESANTE\s+EL\s+[A-Z0-9]+"
    ),
    re.compile(
        r"CESANTE\s+PERCIBE\s+INGRESO"
        r"(?:\s+DE\s+SU(?:\s+ESPOSO)?)?"
    ),
    re.compile(
        r"ESTA\s+CESANTE(?:\s+.*)?"
    ),
    re.compile(
        r"(?:JUBILAD|JUBLAD)[A-Z0-9]*"
    ),
    re.compile(
        r"PENSIONAD[A-Z0-9]*"
    ),
    re.compile(
        r"ESTUDIANT[A-Z0-9]*"
    ),
)


_OBSERVED_EMBEDDED_STATUS_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"(?<![A-Z0-9])"
        r"(?:JUBILAD|JUBLAD)[A-Z0-9]*"
        r"(?![A-Z0-9])"
    ),
    re.compile(
        r"(?<![A-Z0-9])"
        r"PENSIONAD[A-Z0-9]*"
        r"(?![A-Z0-9])"
    ),
    re.compile(
        r"(?<![A-Z0-9])"
        r"ESTUDIANT[A-Z0-9]*"
        r"(?![A-Z0-9])"
    ),
    re.compile(
        r"(?<![A-Z0-9])"
        r"NO\s+(?:LABOR|TRABAJ)[A-Z0-9]*"
        r"(?![A-Z0-9])"
    ),
    re.compile(
        r"(?<![A-Z0-9])"
        r"NO\s+ESTA(?:\s+[A-Z0-9]+){0,2}\s+"
        r"(?:TRABAJ|LABOR)[A-Z0-9]*"
        r"(?![A-Z0-9])"
    ),
    re.compile(
        r"(?<![A-Z0-9])"
        r"(?:"
        r"DEPENDIENTE ECON|"
        r"DEPENDE ECON|"
        r"DEPEND ECON|"
        r"DENEDE ECON"
        r")"
        r"[A-Z0-9]*"
        r"(?![A-Z0-9])"
    ),
    re.compile(
        r"(?<![A-Z0-9])"
        r"AMA DE CASA[A-Z0-9]*"
        r"(?![A-Z0-9])"
    ),
    re.compile(
        r"[A-Z]+"
        r"(?:JUBILAD|JUBLAD)[A-Z0-9]*"
        r"(?![A-Z0-9])"
    ),
    re.compile(
        r"(?<![A-Z0-9])CESANTE(?![A-Z0-9])"
    ),
)


# Narrow occupation/activity vocabulary observed leaking into the employer
# resolution population.
#
# Unlike explicit employment-status evidence, these matches are always
# moderate. A profession or activity term is not enough to prove that the
# record is a non-employer. Organization evidence can therefore coexist
# with these signals and cause conservative downstream abstention.
_OBSERVED_OCCUPATION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"(?<![A-Z0-9])"
        r"AGRICULTOR(?:A|ES)?"
        r"(?![A-Z0-9])"
    ),
    re.compile(
        r"(?<![A-Z0-9])"
        r"AGRONOM(?:O|A|OS|AS)"
        r"(?![A-Z0-9])"
    ),
    re.compile(
        r"(?<![A-Z0-9])"
        r"COMERCIANTE(?:S)?"
        r"(?![A-Z0-9])"
    ),
    re.compile(
        r"(?<![A-Z0-9])"
        r"ESTILISTA(?:S)?"
        r"(?![A-Z0-9])"
    ),
    re.compile(
        r"(?<![A-Z0-9])"
        r"LOCUTOR(?:A|ES|AS)?"
        r"(?![A-Z0-9])"
    ),
    re.compile(
        r"(?<![A-Z0-9])"
        r"INDEPEN\s+DIENTE"
        r"(?![A-Z0-9])"
    ),
    re.compile(
        r"(?<![A-Z0-9])"
        r"INDEPENDINETE"
        r"(?![A-Z0-9])"
    ),
)


_GENERIC_NEGATED_WORK_ORGANIZATION_TOKENS = frozenset(
    {
        "EMPRESA",
        "EMPRESAS",
    }
)


_NEGATED_WORK_STATUS_PATTERN = re.compile(
    r"(?:"
    r"NO\s+(?:LABOR|TRABAJ)[A-Z0-9]*|"
    r"NO\s+ESTA(?:\s+[A-Z0-9]+){0,2}\s+"
    r"(?:TRABAJ|LABOR)[A-Z0-9]*|"
    r"ACTUAL(?:MENTE|METE)\s+NO\s+"
    r"(?:"
    r"(?:ESTA\s+)?(?:TRABAJ|LABOR)[A-Z0-9]*|"
    r"EJERCE\s+PROFESION"
    r")"
    r")(?:\s+.*)?"
)


def _matched_observed_status_variant(
    value: str,
) -> str | None:
    """Recover narrowly evidenced status variants without generic fuzzy matching."""
    for pattern in _OBSERVED_PURE_STATUS_PATTERNS:
        if pattern.fullmatch(value):
            return value

    for pattern in _OBSERVED_EMBEDDED_STATUS_PATTERNS:
        match = pattern.search(value)

        if match is not None:
            return match.group(0)

    return None


def _matched_observed_occupation_variant(
    value: str,
) -> str | None:
    """Return a narrowly observed occupational descriptor, if present."""
    for pattern in _OBSERVED_OCCUPATION_PATTERNS:
        match = pattern.search(value)

        if match is not None:
            return match.group(0)

    return None


def _generic_organization_only_inside_negated_work_status(
    *,
    value: str,
    pure_status: bool,
    organization_matches: tuple[str, ...],
    has_corporate_suffix: bool,
) -> bool:
    """Ignore generic EMPRESA(S) only inside an explicit no-work statement."""
    return (
        pure_status
        and not has_corporate_suffix
        and bool(organization_matches)
        and set(organization_matches)
        <= _GENERIC_NEGATED_WORK_ORGANIZATION_TOKENS
        and _NEGATED_WORK_STATUS_PATTERN.fullmatch(value) is not None
    )


def _has_location_number(
    tokens: tuple[str, ...],
    address_tokens: Set[str],
) -> bool:
    for index, token in enumerate(tokens):
        if token not in address_tokens:
            continue

        neighbors = (
            tokens[max(0, index - 1) : index]
            + tokens[index + 1 : index + 3]
        )

        if any(
            neighbor.isdigit()
            or (
                neighbor[:-1].isdigit()
                and neighbor[-1:].isalpha()
            )
            for neighbor in neighbors
        ):
            return True

    return False


def _strength(
    score: int,
    config: RecordTypingConfig,
) -> SignalStrength:
    if score <= 0:
        return "none"

    if score >= config.address.strong_score:
        return "strong"

    if score >= config.address.moderate_score:
        return "moderate"

    return "weak"


def type_record(
    strict: str | None,
    relaxed: str | None,
    config: RecordTypingConfig,
) -> RecordTyping:
    """Derive deterministic signals without making an entity-resolution decision."""
    value = (relaxed or "").strip()
    tokens = tuple(value.split())

    normalized_length = len((strict or "").strip())
    is_blank = strict is None or normalized_length == 0

    is_numeric_only = bool(tokens) and all(
        token.isdigit()
        for token in tokens
    )

    explicit = _ordered_matches(
        tokens,
        config.address.explicit_tokens,
    )

    contextual = _ordered_matches(
        tokens,
        config.address.contextual_tokens,
    )

    address_score = (
        2 * int(bool(explicit))
        + int(bool(contextual))
    )

    all_address_tokens = _configured_set(
        config.address.explicit_tokens
        + config.address.contextual_tokens
    )

    address_score += 2 * int(
        _has_location_number(
            tokens,
            all_address_tokens,
        )
    )

    has_intersection = any(
        _contains_phrase(
            tokens,
            marker_tokens,
        )
        for _, marker_tokens in _configured_phrases(
            config.address.intersection_markers
        )
    )

    address_score += 3 * int(has_intersection)

    address_score += int(
        len(
            set(
                (
                    *explicit,
                    *contextual,
                )
            )
        )
        > 1
    )

    address_strength = _strength(
        address_score,
        config,
    )

    has_address_signal = address_strength != "none"

    status = (
        _matched_status(
            tokens,
            config.occupation.status_phrases,
        )
        if value
        else None
    )

    if status is None and value:
        status = _matched_observed_status_variant(value)

    pure_status = (
        status is not None
        and status == value
    )

    occupation_variant = (
        _matched_observed_occupation_variant(value)
        if value
        else None
    )

    has_status_signal = status is not None
    has_occupation_variant = occupation_variant is not None

    has_occupation_signal = (
        has_status_signal
        or has_occupation_variant
    )

    occupation_strength: SignalStrength = "none"

    if has_status_signal:
        occupation_strength = (
            "strong"
            if pure_status
            else "moderate"
        )
    elif has_occupation_variant:
        occupation_strength = "moderate"

    has_activity_description = (
        has_occupation_signal
        and not pure_status
    )

    suffix_tokens = _configured_set(
        config.organization.corporate_suffix_tokens
    )

    has_corporate_suffix = (
        bool(tokens)
        and tokens[-1] in suffix_tokens
    )

    organization_matches = _ordered_matches(
        tokens,
        config.organization.organization_tokens,
    )

    suppress_generic_organization = (
        _generic_organization_only_inside_negated_work_status(
            value=value,
            pure_status=pure_status,
            organization_matches=organization_matches,
            has_corporate_suffix=has_corporate_suffix,
        )
    )

    has_organization = (
        has_corporate_suffix
        or (
            bool(organization_matches)
            and not suppress_generic_organization
        )
    )

    mixed_address_organization = (
        has_address_signal
        and has_organization
    )

    mixed_occupation_organization = (
        has_occupation_signal
        and has_organization
    )

    if is_blank:
        route: RecordRoute = "blank_candidate"
        reason = "blank_preserved_for_later_validation"

    elif is_numeric_only:
        route = "ambiguous_review_candidate"
        reason = "numeric_only_is_ambiguous"

    elif (
        mixed_address_organization
        or mixed_occupation_organization
    ):
        route = "employer_resolution_candidate"
        reason = (
            "organization_signal_preserves_resolution_eligibility"
        )

    elif (
        has_address_signal
        and has_occupation_signal
    ):
        route = "ambiguous_review_candidate"
        reason = (
            "conflicting_address_and_occupation_signals"
        )

    elif pure_status:
        route = "non_employer_status_candidate"
        reason = "complete_value_matches_status_phrase"

    elif address_strength in {
        "moderate",
        "strong",
    }:
        route = "address_candidate"
        reason = (
            "corroborated_address_without_organization_signal"
        )

    elif (
        has_address_signal
        or has_occupation_signal
    ):
        route = "ambiguous_review_candidate"
        reason = (
            "weak_or_descriptive_nonorganization_signal"
        )

    else:
        route = "employer_resolution_candidate"
        reason = (
            "no_signal_justifies_exclusion_from_resolution"
        )

    return RecordTyping(
        is_blank=is_blank,
        is_numeric_only=is_numeric_only,
        has_address_signal=has_address_signal,
        address_signal_strength=address_strength,
        has_occupation_signal=has_occupation_signal,
        occupation_signal_strength=occupation_strength,
        has_corporate_suffix=has_corporate_suffix,
        has_organization_like_tokens=has_organization,
        mixed_address_organization_signal=(
            mixed_address_organization
        ),
        mixed_occupation_organization_signal=(
            mixed_occupation_organization
        ),
        has_activity_description_signal=(
            has_activity_description
        ),
        token_count=len(tokens),
        normalized_length=normalized_length,
        route=route,
        route_reason=reason,
    )