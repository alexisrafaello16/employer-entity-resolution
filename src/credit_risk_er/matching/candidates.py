"""Deterministic, bounded candidate generation for unresolved employer names."""

from __future__ import annotations

import itertools
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Final, Literal, cast

from credit_risk_er.config import CandidateGenerationConfig

type BlockingMethod = Literal[
    "relaxed_key",
    "trailing_numeric",
    "truncation_prefix",
    "informative_token",
    "prefix_signature",
    "typo_token",
    "typo_context",
]
type ExclusionReason = Literal[
    "already_resolved",
    "blank",
    "numeric_only",
    "non_employer_status",
    "address_candidate",
    "ineligible_route",
    "missing_strict_key",
]

METHOD_ORDER: Final[tuple[BlockingMethod, ...]] = (
    "relaxed_key",
    "trailing_numeric",
    "truncation_prefix",
    "informative_token",
    "prefix_signature",
    "typo_token",
    "typo_context",
)
GENERIC_TOKENS: Final = frozenset(
    {
        "SA",
        "SAS",
        "INC",
        "CORP",
        "LLC",
        "LTDA",
        "CIA",
        "CO",
        "COMPANY",
        "GRUPO",
        "EMPRESA",
        "EMPRESAS",
        "SERVICIO",
        "SERVICIOS",
        "PANAMA",
        "DE",
        "DEL",
        "LA",
        "EL",
        "LOS",
        "LAS",
        "Y",
        "EN",
        "PARA",
        "THE",
        "OF",
        "AND",
    }
)
_ALPHANUMERIC_RE: Final = re.compile(r"[^A-Z0-9]+")


@dataclass(frozen=True, slots=True)
class ResolutionKey:
    resolution_key: str
    representative_name: str
    relaxed_key: str | None
    source_row_frequency: int
    representative_record_id: str
    representative_source_row_number: int
    representative_route: str
    trailing_numeric_candidate: str | None
    possible_truncation: bool
    token_count: int


@dataclass(slots=True)
class ResolutionKeyAggregate:
    """Mutable aggregation state for one strict key during a bounded dataset scan."""

    representative_name: str
    relaxed_key: str | None
    source_row_frequency: int
    representative_record_id: str
    representative_source_row_number: int
    representative_route: str
    trailing_numeric_candidate: str | None
    possible_truncation: bool
    token_count: int


@dataclass(frozen=True, slots=True)
class CandidatePair:
    key_a: str
    key_b: str
    name_a: str
    name_b: str
    blocking_methods: tuple[BlockingMethod, ...]


@dataclass(frozen=True, slots=True)
class CandidateGeneration:
    pairs: tuple[CandidatePair, ...]
    method_pair_counts: dict[str, int]
    multi_method_pairs: int
    block_summaries: dict[str, dict[str, int | float]]
    broad_blocks_skipped: dict[str, int]
    candidates_per_key: dict[str, int | float]
    zero_candidate_keys_before_typo: int
    zero_candidate_keys_recovered_by_typo: int
    zero_candidate_keys_remaining_after_typo: int
    typo_candidate_pairs_added: int
    typo_signatures_considered: int
    broad_typo_signatures_skipped: int
    multi_method_pairs_after_typo: int
    zero_candidate_keys_before_context: int
    zero_candidate_keys_recovered_by_context: int
    zero_candidate_keys_remaining_after_context: int
    typo_context_pairs_added: int
    context_lookups_performed: int
    broad_context_tokens_skipped: int


@dataclass(frozen=True, slots=True)
class TypoFallbackMetrics:
    zero_keys_before: int
    zero_keys_recovered: int
    zero_keys_remaining: int
    pairs_added: int
    signatures_considered: int
    broad_signatures_skipped: int
    multi_method_pairs: int


@dataclass(frozen=True, slots=True)
class ContextualTypoMetrics:
    zero_keys_before: int
    zero_keys_recovered: int
    zero_keys_remaining: int
    pairs_added: int
    context_lookups: int
    broad_context_tokens_skipped: int


def exclusion_reason(row: dict[str, object]) -> ExclusionReason | None:
    """Return why a row is excluded, or ``None`` when eligible for discovery."""
    if row["resolution_status"] != "unresolved":
        return "already_resolved"
    if bool(row["is_blank"]):
        return "blank"
    if bool(row["is_numeric_only"]):
        return "numeric_only"
    route = str(row["route"])
    if route == "non_employer_status_candidate":
        return "non_employer_status"
    if route == "address_candidate":
        return "address_candidate"
    if route not in {"employer_resolution_candidate", "ambiguous_review_candidate"}:
        return "ineligible_route"
    if not row["nombre_normalizado"]:
        return "missing_strict_key"
    return None


def update_resolution_key(
    aggregates: dict[str, ResolutionKeyAggregate], row: dict[str, object]
) -> None:
    """Aggregate one eligible row; the lowest source row is representative."""
    key = str(row["nombre_normalizado"])
    source_row = cast(int, row["source_row_number"])
    trailing = row["trailing_numeric_candidate"]
    trailing_value = str(trailing) if trailing is not None else None
    existing = aggregates.get(key)
    if existing is None:
        aggregates[key] = ResolutionKeyAggregate(
            representative_name=key,
            relaxed_key=str(row["nombre_matching"]) if row["nombre_matching"] else None,
            source_row_frequency=1,
            representative_record_id=str(row["record_id"]),
            representative_source_row_number=source_row,
            representative_route=str(row["route"]),
            trailing_numeric_candidate=trailing_value,
            possible_truncation=bool(row["possible_truncation"]),
            token_count=cast(int, row["token_count"]),
        )
        return
    existing.source_row_frequency += 1
    existing.possible_truncation = existing.possible_truncation or bool(row["possible_truncation"])
    existing.token_count = max(existing.token_count, cast(int, row["token_count"]))
    if trailing_value is not None:
        existing.trailing_numeric_candidate = min(
            value
            for value in (existing.trailing_numeric_candidate, trailing_value)
            if value is not None
        )
    if source_row < existing.representative_source_row_number:
        existing.representative_source_row_number = source_row
        existing.representative_record_id = str(row["record_id"])
        existing.representative_route = str(row["route"])


def finalize_resolution_keys(
    aggregates: dict[str, ResolutionKeyAggregate],
) -> tuple[ResolutionKey, ...]:
    """Return one deterministic row per strict key, ordered by the key."""
    return tuple(
        ResolutionKey(
            resolution_key=key,
            representative_name=item.representative_name,
            relaxed_key=item.relaxed_key,
            source_row_frequency=item.source_row_frequency,
            representative_record_id=item.representative_record_id,
            representative_source_row_number=item.representative_source_row_number,
            representative_route=item.representative_route,
            trailing_numeric_candidate=item.trailing_numeric_candidate,
            possible_truncation=item.possible_truncation,
            token_count=item.token_count,
        )
        for key, item in sorted(aggregates.items())
    )


def _pair(key_a: str, key_b: str) -> tuple[str, str] | None:
    if key_a == key_b:
        return None
    return (key_a, key_b) if key_a < key_b else (key_b, key_a)


def _percentile(values: list[int], proportion: float) -> int:
    ordered = sorted(values)
    if not ordered:
        return 0
    index = math.ceil(proportion * len(ordered)) - 1
    return ordered[max(0, index)]


def _block_summary(sizes: list[int]) -> dict[str, int | float]:
    if not sizes:
        return {"blocks": 0, "minimum": 0, "median": 0, "p95": 0, "maximum": 0}
    return {
        "blocks": len(sizes),
        "minimum": min(sizes),
        "median": _percentile(sizes, 0.5),
        "p95": _percentile(sizes, 0.95),
        "maximum": max(sizes),
    }


def _add_pair(
    pairs: dict[tuple[str, str], set[BlockingMethod]],
    key_a: str,
    key_b: str,
    method: BlockingMethod,
) -> None:
    ordered_pair = _pair(key_a, key_b)
    if ordered_pair is not None:
        pairs.setdefault(ordered_pair, set()).add(method)


def _expand_complete_blocks(
    blocks: dict[str, set[str]],
    method: BlockingMethod,
    maximum_block_size: int,
    pairs: dict[tuple[str, str], set[BlockingMethod]],
    block_sizes: dict[str, list[int]],
    skipped: Counter[str],
) -> None:
    for block_key in sorted(blocks):
        members = sorted(blocks[block_key])
        if len(members) < 2:
            continue
        block_sizes[method].append(len(members))
        if len(members) > maximum_block_size:
            skipped[method] += 1
            continue
        for key_a, key_b in itertools.combinations(members, 2):
            _add_pair(pairs, key_a, key_b, method)


def typo_signatures(token: str) -> frozenset[str]:
    """Return the token plus every deterministic one-character deletion."""
    return frozenset({token, *(token[:index] + token[index + 1 :] for index in range(len(token)))})


def _typo_tokens(key: str, minimum_length: int) -> set[str]:
    return {
        token
        for token in key.split()
        if len(token) >= minimum_length and token not in GENERIC_TOKENS and not token.isdigit()
    }


def _add_typo_fallback(
    resolution_keys: tuple[ResolutionKey, ...],
    config: CandidateGenerationConfig,
    pairs: dict[tuple[str, str], set[BlockingMethod]],
    block_sizes: dict[str, list[int]],
    skipped: Counter[str],
) -> TypoFallbackMetrics:
    """Query a bounded typo index only for keys missed by all approved blockers."""
    keys_with_candidates = {key for pair in pairs for key in pair}
    zero_keys = {
        item.resolution_key
        for item in resolution_keys
        if item.resolution_key not in keys_with_candidates
    }
    signature_index: dict[str, set[str]] = {}
    broad_signatures: set[str] = set()
    minimum_length = max(
        config.minimum_typo_token_length,
        config.minimum_informative_token_length,
    )
    for item in resolution_keys:
        for token in _typo_tokens(item.resolution_key, minimum_length):
            for signature in typo_signatures(token):
                if signature in broad_signatures:
                    continue
                members = signature_index.setdefault(signature, set())
                members.add(item.resolution_key)
                if len(members) > config.maximum_typo_signature_frequency:
                    del signature_index[signature]
                    broad_signatures.add(signature)

    block_sizes["typo_token"].extend(
        len(members) for members in signature_index.values() if len(members) >= 2
    )
    skipped["typo_token"] = len(broad_signatures)
    pair_count_before = len(pairs)
    for query_key in sorted(zero_keys):
        for token in sorted(_typo_tokens(query_key, minimum_length)):
            for signature in sorted(typo_signatures(token)):
                for candidate_key in sorted(signature_index.get(signature, ())):
                    _add_pair(pairs, query_key, candidate_key, "typo_token")

    keys_after_fallback = {key for pair in pairs for key in pair}
    recovered = len(zero_keys & keys_after_fallback)
    return TypoFallbackMetrics(
        zero_keys_before=len(zero_keys),
        zero_keys_recovered=recovered,
        zero_keys_remaining=len(zero_keys) - recovered,
        pairs_added=len(pairs) - pair_count_before,
        signatures_considered=len(signature_index) + len(broad_signatures),
        broad_signatures_skipped=len(broad_signatures),
        multi_method_pairs=sum(len(methods) > 1 for methods in pairs.values()),
    )


def _has_contextual_typo_relation(
    query_tokens: frozenset[str],
    candidate_tokens: frozenset[str],
    context_token: str,
    signature_cache: dict[str, frozenset[str]],
) -> bool:
    """Require typo evidence on tokens other than the exact context token."""
    for query_token in sorted(query_tokens):
        if query_token == context_token:
            continue
        query_signatures = _cached_typo_signatures(query_token, signature_cache)
        for candidate_token in sorted(candidate_tokens):
            if candidate_token == context_token or query_token == candidate_token:
                continue
            candidate_signatures = _cached_typo_signatures(candidate_token, signature_cache)
            if query_signatures & candidate_signatures:
                return True
    return False


def _cached_typo_signatures(
    token: str, signature_cache: dict[str, frozenset[str]]
) -> frozenset[str]:
    signatures = signature_cache.get(token)
    if signatures is None:
        signatures = typo_signatures(token)
        signature_cache[token] = signatures
    return signatures


def _build_context_signature_index(
    context_token: str,
    candidate_pool: set[str],
    context_tokens_by_key: dict[str, frozenset[str]],
    signature_cache: dict[str, frozenset[str]],
) -> dict[str, set[str]]:
    """Index one bounded context pool, then discard it after its queries."""
    signature_index: dict[str, set[str]] = defaultdict(set)
    for candidate_key in sorted(candidate_pool):
        for candidate_token in sorted(context_tokens_by_key[candidate_key]):
            if candidate_token == context_token:
                continue
            for signature in _cached_typo_signatures(candidate_token, signature_cache):
                signature_index[signature].add(candidate_key)
    return signature_index


def _prefilter_context_candidates(
    query_tokens: frozenset[str],
    context_token: str,
    signature_index: dict[str, set[str]],
    signature_cache: dict[str, frozenset[str]],
) -> set[str]:
    """Return only context members sharing a possible non-context signature."""
    candidates: set[str] = set()
    for query_token in sorted(query_tokens):
        if query_token == context_token:
            continue
        for signature in _cached_typo_signatures(query_token, signature_cache):
            candidates.update(signature_index.get(signature, ()))
    return candidates


def _add_contextual_typo_fallback(
    resolution_keys: tuple[ResolutionKey, ...],
    token_blocks: dict[str, set[str]],
    context_tokens_by_key: dict[str, frozenset[str]],
    config: CandidateGenerationConfig,
    pairs: dict[tuple[str, str], set[BlockingMethod]],
    block_sizes: dict[str, list[int]],
    skipped: Counter[str],
) -> ContextualTypoMetrics:
    """Use bounded exact-token context for residual zero-candidate typo queries."""
    keys_with_candidates = {key for pair in pairs for key in pair}
    residual_keys = {
        item.resolution_key
        for item in resolution_keys
        if item.resolution_key not in keys_with_candidates
    }
    broad_context_tokens: set[str] = set()
    signature_cache: dict[str, frozenset[str]] = {}
    queries_by_context: dict[str, list[str]] = defaultdict(list)
    context_lookups = 0
    pair_count_before = len(pairs)
    for query_key in sorted(residual_keys):
        for context_token in sorted(context_tokens_by_key[query_key]):
            candidate_pool = token_blocks.get(context_token, set())
            if len(candidate_pool) > config.maximum_typo_context_frequency:
                broad_context_tokens.add(context_token)
                continue
            context_lookups += 1
            if len(candidate_pool) >= 2:
                block_sizes["typo_context"].append(len(candidate_pool))
            queries_by_context[context_token].append(query_key)

    for context_token in sorted(queries_by_context):
        candidate_pool = token_blocks.get(context_token, set())
        signature_index = _build_context_signature_index(
            context_token,
            candidate_pool,
            context_tokens_by_key,
            signature_cache,
        )
        for query_key in queries_by_context[context_token]:
            query_tokens = context_tokens_by_key[query_key]
            possible_candidates = _prefilter_context_candidates(
                query_tokens,
                context_token,
                signature_index,
                signature_cache,
            )
            for candidate_key in sorted(possible_candidates):
                if candidate_key == query_key:
                    continue
                if _has_contextual_typo_relation(
                    query_tokens,
                    context_tokens_by_key[candidate_key],
                    context_token,
                    signature_cache,
                ):
                    _add_pair(pairs, query_key, candidate_key, "typo_context")

    skipped["typo_context"] = len(broad_context_tokens)
    keys_after_fallback = {key for pair in pairs for key in pair}
    recovered = len(residual_keys & keys_after_fallback)
    return ContextualTypoMetrics(
        zero_keys_before=len(residual_keys),
        zero_keys_recovered=recovered,
        zero_keys_remaining=len(residual_keys) - recovered,
        pairs_added=len(pairs) - pair_count_before,
        context_lookups=context_lookups,
        broad_context_tokens_skipped=len(broad_context_tokens),
    )


def generate_candidate_pairs(
    resolution_keys: tuple[ResolutionKey, ...], config: CandidateGenerationConfig
) -> CandidateGeneration:
    """Generate bounded, explainable possibilities without making identity decisions."""
    by_key = {item.resolution_key: item for item in resolution_keys}
    pairs: dict[tuple[str, str], set[BlockingMethod]] = {}
    block_sizes: dict[str, list[int]] = defaultdict(list)
    skipped: Counter[str] = Counter()

    relaxed_blocks: dict[str, set[str]] = defaultdict(set)
    prefix_blocks: dict[str, set[str]] = defaultdict(set)
    token_blocks: dict[str, set[str]] = defaultdict(set)
    context_tokens_by_key: dict[str, frozenset[str]] = {}
    truncation_queries: dict[str, set[str]] = defaultdict(set)
    truncation_universe: dict[str, set[str]] = defaultdict(set)

    for item in resolution_keys:
        key = item.resolution_key
        raw_tokens = set(key.split())
        context_tokens_by_key[key] = frozenset(
            token
            for token in raw_tokens
            if len(token) >= config.minimum_informative_token_length
            and token not in GENERIC_TOKENS
            and not token.isdigit()
        )
        if item.relaxed_key:
            relaxed_blocks[item.relaxed_key].add(key)
        signature = _ALPHANUMERIC_RE.sub("", key)[: config.prefix_signature_length]
        if len(signature) == config.prefix_signature_length:
            prefix_blocks[signature].add(key)
        if len(key) >= config.minimum_truncation_prefix_length:
            prefix = key[: config.minimum_truncation_prefix_length]
            truncation_universe[prefix].add(key)
            if item.possible_truncation:
                truncation_queries[prefix].add(key)
        for token in raw_tokens:
            if (
                len(token) >= config.minimum_informative_token_length
                and token not in GENERIC_TOKENS
            ):
                token_blocks[token].add(key)

    _expand_complete_blocks(
        relaxed_blocks,
        "relaxed_key",
        config.maximum_block_size,
        pairs,
        block_sizes,
        skipped,
    )

    trailing_lookups = 0
    for item in resolution_keys:
        target = item.trailing_numeric_candidate
        if target is not None and target in by_key and target != item.resolution_key:
            trailing_lookups += 1
            _add_pair(pairs, item.resolution_key, target, "trailing_numeric")
    if trailing_lookups:
        block_sizes["trailing_numeric"].extend([2] * trailing_lookups)

    for prefix in sorted(truncation_queries):
        members = sorted(truncation_universe[prefix])
        if len(members) < 2:
            continue
        block_sizes["truncation_prefix"].append(len(members))
        if len(members) > config.maximum_block_size:
            skipped["truncation_prefix"] += 1
            continue
        for query in sorted(truncation_queries[prefix]):
            for candidate in members:
                _add_pair(pairs, query, candidate, "truncation_prefix")

    bounded_token_blocks: dict[str, set[str]] = {}
    for token, token_members in token_blocks.items():
        if len(token_members) > config.maximum_token_frequency:
            if len(token_members) >= 2:
                block_sizes["informative_token"].append(len(token_members))
                skipped["informative_token"] += 1
            continue
        bounded_token_blocks[token] = token_members
    _expand_complete_blocks(
        bounded_token_blocks,
        "informative_token",
        config.maximum_block_size,
        pairs,
        block_sizes,
        skipped,
    )
    _expand_complete_blocks(
        prefix_blocks,
        "prefix_signature",
        config.maximum_block_size,
        pairs,
        block_sizes,
        skipped,
    )

    typo_metrics = _add_typo_fallback(
        resolution_keys,
        config,
        pairs,
        block_sizes,
        skipped,
    )
    context_metrics = _add_contextual_typo_fallback(
        resolution_keys,
        token_blocks,
        context_tokens_by_key,
        config,
        pairs,
        block_sizes,
        skipped,
    )

    candidate_pairs = tuple(
        CandidatePair(
            key_a=key_a,
            key_b=key_b,
            name_a=by_key[key_a].representative_name,
            name_b=by_key[key_b].representative_name,
            blocking_methods=tuple(method for method in METHOD_ORDER if method in methods),
        )
        for (key_a, key_b), methods in sorted(pairs.items())
    )
    method_counts: dict[str, int] = {
        method: sum(method in pair.blocking_methods for pair in candidate_pairs)
        for method in METHOD_ORDER
    }
    per_key: Counter[str] = Counter()
    for pair in candidate_pairs:
        per_key[pair.key_a] += 1
        per_key[pair.key_b] += 1
    candidate_counts = [per_key[item.resolution_key] for item in resolution_keys]
    positive_counts = [count for count in candidate_counts if count > 0]
    candidates_summary: dict[str, int | float] = {
        "keys_with_zero_candidates": sum(count == 0 for count in candidate_counts),
        "keys_with_candidates": len(positive_counts),
        "minimum": min(positive_counts, default=0),
        "median": _percentile(positive_counts, 0.5),
        "p95": _percentile(positive_counts, 0.95),
        "maximum": max(positive_counts, default=0),
        "mean": round(sum(candidate_counts) / len(candidate_counts), 6)
        if candidate_counts
        else 0.0,
    }
    return CandidateGeneration(
        pairs=candidate_pairs,
        method_pair_counts=method_counts,
        multi_method_pairs=sum(len(pair.blocking_methods) > 1 for pair in candidate_pairs),
        block_summaries={method: _block_summary(block_sizes[method]) for method in METHOD_ORDER},
        broad_blocks_skipped={method: skipped[method] for method in METHOD_ORDER},
        candidates_per_key=candidates_summary,
        zero_candidate_keys_before_typo=typo_metrics.zero_keys_before,
        zero_candidate_keys_recovered_by_typo=typo_metrics.zero_keys_recovered,
        zero_candidate_keys_remaining_after_typo=typo_metrics.zero_keys_remaining,
        typo_candidate_pairs_added=typo_metrics.pairs_added,
        typo_signatures_considered=typo_metrics.signatures_considered,
        broad_typo_signatures_skipped=typo_metrics.broad_signatures_skipped,
        multi_method_pairs_after_typo=typo_metrics.multi_method_pairs,
        zero_candidate_keys_before_context=context_metrics.zero_keys_before,
        zero_candidate_keys_recovered_by_context=context_metrics.zero_keys_recovered,
        zero_candidate_keys_remaining_after_context=context_metrics.zero_keys_remaining,
        typo_context_pairs_added=context_metrics.pairs_added,
        context_lookups_performed=context_metrics.context_lookups,
        broad_context_tokens_skipped=context_metrics.broad_context_tokens_skipped,
    )
