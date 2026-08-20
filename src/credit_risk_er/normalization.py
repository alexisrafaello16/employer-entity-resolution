"""Pure deterministic employer-name normalization."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Final

from credit_risk_er.config import NormalizationConfig

_WHITESPACE_RE: Final = re.compile(r"\s+")
_TRAILING_NUMERIC_RE: Final = re.compile(r"(?:^|\s)(\d+)$")
_PUNCTUATION_TRANSLATION: Final = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u02bc": "'",
        "`": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
    }
)


@dataclass(frozen=True, slots=True)
class NormalizedEmployer:
    """Only representations and candidates needed by downstream matching."""

    strict: str | None
    relaxed: str | None
    has_trailing_numeric_token: bool
    trailing_numeric_candidate: str | None
    possible_truncation: bool


def _punctuation_to_space(value: str) -> str:
    replaced = "".join(
        " " if unicodedata.category(character).startswith("P") else character for character in value
    )
    return " ".join(replaced.split())


def _normalize_terminal_suffix(value: str, aliases: dict[str, tuple[str, ...]]) -> str:
    tokens = value.split()
    for canonical, variants in aliases.items():
        canonical_tokens = canonical.upper().split()
        for variant in variants:
            variant_tokens = variant.upper().split()
            if (
                len(tokens) >= len(variant_tokens)
                and tokens[-len(variant_tokens) :] == variant_tokens
            ):
                return " ".join([*tokens[: -len(variant_tokens)], *canonical_tokens])
    return value


def normalize_employer(value: str | None, config: NormalizationConfig) -> NormalizedEmployer:
    """Build strict and matching representations without changing the source value."""
    if value is None:
        return NormalizedEmployer(None, None, False, None, False)

    strict = value[1:] if value.startswith("'") else value
    strict = unicodedata.normalize("NFKC", strict)
    strict = strict.translate(_PUNCTUATION_TRANSLATION).upper().strip()
    strict = _WHITESPACE_RE.sub(" ", strict)

    relaxed = _punctuation_to_space(strict)
    relaxed = _normalize_terminal_suffix(relaxed, config.corporate_suffix_aliases)

    trailing_match = _TRAILING_NUMERIC_RE.search(strict)
    has_trailing_numeric = trailing_match is not None
    trailing_candidate: str | None = None
    if trailing_match is not None:
        numeric_token = trailing_match.group(1)
        candidate = strict[: trailing_match.start()].rstrip()
        if candidate and len(numeric_token) <= config.trailing_numeric_candidate_max_digits:
            trailing_candidate = candidate

    content_view = value[1:] if value.startswith("'") else value
    return NormalizedEmployer(
        strict=strict,
        relaxed=relaxed,
        has_trailing_numeric_token=has_trailing_numeric,
        trailing_numeric_candidate=trailing_candidate,
        possible_truncation=len(content_view) in config.possible_truncation_content_lengths,
    )
