"""Independent lexical and structural evidence for existing candidate pairs."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import takewhile
from typing import Literal

from rapidfuzz import fuzz

type NumericRelation = Literal["none", "same", "one_sided", "conflict"]

FEATURE_PRECISION = 6


@dataclass(frozen=True, slots=True)
class PairFeatures:
    """Interpretable evidence dimensions; none is an employer-identity decision."""

    blocking_method_count: int
    char_ratio: float
    token_sort_ratio: float
    token_set_ratio: float
    partial_ratio: float
    length_ratio: float
    common_prefix_ratio: float
    token_jaccard: float
    same_first_token: bool
    numeric_relation: NumericRelation


def _rounded(value: float) -> float:
    return round(value, FEATURE_PRECISION)


def _numeric_tokens(value: str) -> tuple[str, ...]:
    """Return standalone numeric tokens in their original sequence."""
    return tuple(token for token in value.split() if token.isdigit())


def numeric_relation(name_a: str, name_b: str) -> NumericRelation:
    """Describe ordered standalone numeric-token evidence without deciding identity."""
    numbers_a = _numeric_tokens(name_a)
    numbers_b = _numeric_tokens(name_b)
    if not numbers_a and not numbers_b:
        return "none"
    if not numbers_a or not numbers_b:
        return "one_sided"
    return "same" if numbers_a == numbers_b else "conflict"


def compute_pair_features(
    name_a: str, name_b: str, blocking_methods: list[str] | tuple[str, ...]
) -> PairFeatures:
    """Compute complementary evidence from already-normalized, nonblank names."""
    if not name_a or not name_b:
        raise ValueError("Feature scoring requires nonblank normalized names")

    tokens_a = name_a.split()
    tokens_b = name_b.split()
    token_set_a = set(tokens_a)
    token_set_b = set(tokens_b)
    shorter_length = min(len(name_a), len(name_b))
    longer_length = max(len(name_a), len(name_b))
    common_prefix_length = sum(
        1 for _ in takewhile(lambda pair: pair[0] == pair[1], zip(name_a, name_b, strict=False))
    )
    token_union = token_set_a | token_set_b

    return PairFeatures(
        blocking_method_count=len(set(blocking_methods)),
        char_ratio=_rounded(fuzz.ratio(name_a, name_b, processor=None)),
        token_sort_ratio=_rounded(fuzz.token_sort_ratio(name_a, name_b, processor=None)),
        token_set_ratio=_rounded(fuzz.token_set_ratio(name_a, name_b, processor=None)),
        partial_ratio=_rounded(fuzz.partial_ratio(name_a, name_b, processor=None)),
        length_ratio=_rounded(shorter_length / longer_length),
        common_prefix_ratio=_rounded(common_prefix_length / shorter_length),
        token_jaccard=_rounded(len(token_set_a & token_set_b) / len(token_union)),
        same_first_token=tokens_a[0] == tokens_b[0],
        numeric_relation=numeric_relation(name_a, name_b),
    )
