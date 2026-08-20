"""Semantic oracle for contextual typo retrieval performance refactors."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import pytest

from credit_risk_er.config import CandidateGenerationConfig
from credit_risk_er.matching.candidates import (
    GENERIC_TOKENS,
    BlockingMethod,
    ContextualTypoMetrics,
    ResolutionKey,
    _add_contextual_typo_fallback,
    typo_signatures,
)

type PairKey = tuple[str, str]
type PairMethods = dict[PairKey, set[BlockingMethod]]


def _config(**updates: int | Path) -> CandidateGenerationConfig:
    values: dict[str, int | Path] = {
        "resolution_keys_output": Path("keys.parquet"),
        "candidate_pairs_output": Path("pairs.parquet"),
        "metrics_output": Path("metrics.json"),
        "maximum_block_size": 10,
        "minimum_truncation_prefix_length": 20,
        "minimum_informative_token_length": 5,
        "maximum_token_frequency": 10,
        "prefix_signature_length": 8,
        "minimum_typo_token_length": 6,
        "maximum_typo_signature_frequency": 10,
        "maximum_typo_context_frequency": 20,
    }
    values.update(updates)
    return CandidateGenerationConfig.model_validate(values)


def _keys(names: tuple[str, ...]) -> tuple[ResolutionKey, ...]:
    return tuple(
        ResolutionKey(
            resolution_key=name,
            representative_name=name,
            relaxed_key=name,
            source_row_frequency=1,
            representative_record_id=f"record-{index}",
            representative_source_row_number=index,
            representative_route="employer_resolution_candidate",
            trailing_numeric_candidate=None,
            possible_truncation=False,
            token_count=len(name.split()),
        )
        for index, name in enumerate(names, start=2)
    )


def _context_tokens(key: str, minimum_length: int) -> frozenset[str]:
    return frozenset(
        token
        for token in key.split()
        if len(token) >= minimum_length and token not in GENERIC_TOKENS and not token.isdigit()
    )


def _token_blocks(
    resolution_keys: tuple[ResolutionKey, ...], minimum_length: int
) -> dict[str, set[str]]:
    blocks: dict[str, set[str]] = defaultdict(set)
    for item in resolution_keys:
        for token in set(item.resolution_key.split()):
            if len(token) >= minimum_length and token not in GENERIC_TOKENS:
                blocks[token].add(item.resolution_key)
    return blocks


def _reference_relation(
    query_key: str,
    candidate_key: str,
    context_token: str,
    minimum_length: int,
    signature_cache: dict[str, frozenset[str]],
) -> bool:
    query_tokens = _context_tokens(query_key, minimum_length) - {context_token}
    candidate_tokens = _context_tokens(candidate_key, minimum_length) - {context_token}
    for query_token in sorted(query_tokens):
        query_signatures = signature_cache.setdefault(query_token, typo_signatures(query_token))
        for candidate_token in sorted(candidate_tokens):
            if query_token == candidate_token:
                continue
            candidate_signatures = signature_cache.setdefault(
                candidate_token, typo_signatures(candidate_token)
            )
            if query_signatures & candidate_signatures:
                return True
    return False


def _reference_contextual_fallback(
    resolution_keys: tuple[ResolutionKey, ...],
    token_blocks: dict[str, set[str]],
    config: CandidateGenerationConfig,
    initial_pairs: PairMethods,
) -> tuple[PairMethods, ContextualTypoMetrics, list[int], int]:
    """Reproduce approved v2.1 behavior as an independent test-only oracle."""
    pairs = {pair: set(methods) for pair, methods in initial_pairs.items()}
    keys_with_candidates = {key for pair in pairs for key in pair}
    residual_keys = {
        item.resolution_key
        for item in resolution_keys
        if item.resolution_key not in keys_with_candidates
    }
    broad_context_tokens: set[str] = set()
    signature_cache: dict[str, frozenset[str]] = {}
    context_lookups = 0
    pair_count_before = len(pairs)
    block_sizes: list[int] = []

    for query_key in sorted(residual_keys):
        for context_token in sorted(
            _context_tokens(query_key, config.minimum_informative_token_length)
        ):
            candidate_pool = token_blocks.get(context_token, set())
            if len(candidate_pool) > config.maximum_typo_context_frequency:
                broad_context_tokens.add(context_token)
                continue
            context_lookups += 1
            if len(candidate_pool) >= 2:
                block_sizes.append(len(candidate_pool))
            for candidate_key in sorted(candidate_pool):
                if not _reference_relation(
                    query_key,
                    candidate_key,
                    context_token,
                    config.minimum_informative_token_length,
                    signature_cache,
                ):
                    continue
                pair = (
                    (query_key, candidate_key)
                    if query_key < candidate_key
                    else (candidate_key, query_key)
                )
                if pair[0] != pair[1]:
                    pairs.setdefault(pair, set()).add("typo_context")

    keys_after = {key for pair in pairs for key in pair}
    recovered = len(residual_keys & keys_after)
    metrics = ContextualTypoMetrics(
        zero_keys_before=len(residual_keys),
        zero_keys_recovered=recovered,
        zero_keys_remaining=len(residual_keys) - recovered,
        pairs_added=len(pairs) - pair_count_before,
        context_lookups=context_lookups,
        broad_context_tokens_skipped=len(broad_context_tokens),
    )
    return pairs, metrics, block_sizes, len(broad_context_tokens)


def _production_contextual_fallback(
    resolution_keys: tuple[ResolutionKey, ...],
    token_blocks: dict[str, set[str]],
    config: CandidateGenerationConfig,
    initial_pairs: PairMethods,
) -> tuple[PairMethods, ContextualTypoMetrics, list[int], int]:
    pairs = {pair: set(methods) for pair, methods in initial_pairs.items()}
    block_sizes: dict[str, list[int]] = defaultdict(list)
    skipped: Counter[str] = Counter()
    context_tokens_by_key = {
        item.resolution_key: _context_tokens(
            item.resolution_key, config.minimum_informative_token_length
        )
        for item in resolution_keys
    }
    metrics = _add_contextual_typo_fallback(
        resolution_keys,
        token_blocks,
        context_tokens_by_key,
        config,
        pairs,
        block_sizes,
        skipped,
    )
    return pairs, metrics, block_sizes["typo_context"], skipped["typo_context"]


@pytest.mark.parametrize(
    ("names", "config", "initial_pairs"),
    (
        pytest.param(
            ("AER0NAUTICA CIVIL", "AERONAUTICA CIVIL"),
            _config(),
            {},
            id="aeronautica",
        ),
        pytest.param(
            (
                "ADAMBLEA NAVIONAL DE VENEZUELA",
                "ASAMBLEA NACIONAL DE VENEZUELA",
            ),
            _config(),
            {},
            id="asamblea",
        ),
        pytest.param(
            ("THE OFORD SCHOOL", "THE OXFORD SCHOOL"),
            _config(),
            {},
            id="five-character-token",
        ),
        pytest.param(
            ("CERVECERIA CHIRICAMNA", "CERVECERIA CHIRICANA"),
            _config(),
            {},
            id="cerveceria",
        ),
        pytest.param(
            ("ALPHA SHARED CIVIL", "BRAVO SHARED CIVIL"),
            _config(),
            {},
            id="identical-non-context-token",
        ),
        pytest.param(
            ("AER0NAUTICA CIVIL AVIACION", "AERONAUTICA CIVIL AVIACION"),
            _config(),
            {},
            id="multiple-context-tokens",
        ),
        pytest.param(
            ("AAAAB CIVIL", "AAABA CIVIL"),
            _config(),
            {},
            id="multiple-shared-signatures",
        ),
        pytest.param(
            ("OFORD SCHOOL", "OXFORD SCHOOL", "UNRELATED SCHOOL"),
            _config(maximum_typo_context_frequency=2),
            {},
            id="broad-context-skip",
        ),
        pytest.param(
            ("ALPHA 123456", "ALPXA 123456"),
            _config(),
            {},
            id="numeric-protection",
        ),
        pytest.param(
            ("OFORD EMPRESA", "OXFORD EMPRESA"),
            _config(),
            {},
            id="generic-protection",
        ),
        pytest.param(
            ("AER0NAUTICA AERONAUTICA CIVIL",),
            _config(),
            {},
            id="self-candidate",
        ),
        pytest.param(
            ("ALPHA CIVIL", "BRAVO SCHOOL"),
            _config(),
            {("ALPHA CIVIL", "BRAVO SCHOOL"): {"relaxed_key"}},
            id="no-residual-queries",
        ),
    ),
)
def test_optimized_contextual_fallback_matches_v21_reference(
    names: tuple[str, ...],
    config: CandidateGenerationConfig,
    initial_pairs: PairMethods,
) -> None:
    resolution_keys = _keys(names)
    token_blocks = _token_blocks(resolution_keys, config.minimum_informative_token_length)
    expected = _reference_contextual_fallback(
        resolution_keys, token_blocks, config, initial_pairs
    )
    actual = _production_contextual_fallback(
        resolution_keys, token_blocks, config, initial_pairs
    )
    assert actual == expected
