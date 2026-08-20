"""Context-restricted typo retrieval for keys still isolated after typo_token."""

from __future__ import annotations

from pathlib import Path

from credit_risk_er.config import CandidateGenerationConfig
from credit_risk_er.matching.candidates import (
    CandidateGeneration,
    ResolutionKey,
    generate_candidate_pairs,
)
from credit_risk_er.pipeline import CANDIDATE_PAIRS_SCHEMA


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


def _key(name: str, source_row: int = 2) -> ResolutionKey:
    return ResolutionKey(
        resolution_key=name,
        representative_name=name,
        relaxed_key=name,
        source_row_frequency=1,
        representative_record_id=f"record-{source_row}",
        representative_source_row_number=source_row,
        representative_route="employer_resolution_candidate",
        trailing_numeric_candidate=None,
        possible_truncation=False,
        token_count=len(name.split()),
    )


def _generated(
    names: tuple[str, ...], config: CandidateGenerationConfig
) -> CandidateGeneration:
    return generate_candidate_pairs(
        tuple(_key(name, index + 2) for index, name in enumerate(names)), config
    )


def _methods_for(
    generated: CandidateGeneration, key_a: str, key_b: str
) -> tuple[str, ...]:
    ordered = tuple(sorted((key_a, key_b)))
    for pair in generated.pairs:
        if (pair.key_a, pair.key_b) == ordered:
            return pair.blocking_methods
    return ()


def test_aeronautica_is_recovered_when_standalone_signature_is_broad() -> None:
    names = (
        "AER0NAUTICA CIVIL",
        "AERONAUTICA CIVIL",
        "AER1NAUTICA OTHER",
        "UNRELATED CIVIL",
    )
    generated = _generated(
        names,
        _config(
            maximum_token_frequency=2,
            maximum_typo_signature_frequency=2,
        ),
    )
    assert _methods_for(generated, "AER0NAUTICA CIVIL", "AERONAUTICA CIVIL") == ("typo_context",)


def test_asamblea_is_recovered_with_venezuela_context() -> None:
    names = (
        "ADAMBLEA NAVIONAL DE VENEZUELA",
        "ASAMBLEA NACIONAL DE VENEZUELA",
        "AXAMBLEA OTHER",
        "NAXIONAL OTHER",
        "VENEZUELA OTHER",
    )
    generated = _generated(
        names,
        _config(
            maximum_token_frequency=2,
            maximum_typo_signature_frequency=2,
        ),
    )
    assert _methods_for(
        generated,
        "ADAMBLEA NAVIONAL DE VENEZUELA",
        "ASAMBLEA NACIONAL DE VENEZUELA",
    ) == ("typo_context",)


def test_five_character_oford_typo_is_supported_only_with_context() -> None:
    names = ("THE OFORD SCHOOL", "THE OXFORD SCHOOL", "UNRELATED SCHOOL")
    generated = _generated(
        names,
        _config(maximum_token_frequency=2, maximum_typo_signature_frequency=2),
    )
    assert _methods_for(generated, "THE OFORD SCHOOL", "THE OXFORD SCHOOL") == ("typo_context",)
    assert generated.method_pair_counts["typo_token"] == 0


def test_five_character_typo_without_exact_context_is_not_added() -> None:
    generated = _generated(("OFORD ALPHA", "OXFORD BETA"), _config(maximum_token_frequency=2))
    assert generated.pairs == ()


def test_exact_context_alone_is_insufficient_and_not_reused_as_typo_token() -> None:
    names = ("ALPHA CONTEXT", "BETA CONTEXT", "GAMMA CONTEXT")
    generated = _generated(
        names,
        _config(maximum_token_frequency=2, maximum_typo_signature_frequency=2),
    )
    assert generated.pairs == ()


def test_numeric_only_context_is_ignored() -> None:
    names = ("ALPHA 123456", "ALPXA 123456", "OTHER 123456")
    generated = _generated(
        names,
        _config(maximum_token_frequency=2, maximum_typo_signature_frequency=2),
    )
    assert generated.pairs == ()


def test_numeric_only_typo_tokens_are_ignored() -> None:
    names = ("SCHOOL 123456", "SCHOOL 223456", "UNRELATED SCHOOL")
    generated = _generated(
        names,
        _config(maximum_token_frequency=2, maximum_typo_signature_frequency=2),
    )
    assert generated.pairs == ()


def test_generic_context_is_ignored() -> None:
    generated = _generated(("OFORD EMPRESA", "OXFORD EMPRESA"), _config(maximum_token_frequency=2))
    assert generated.pairs == ()


def test_context_over_frequency_limit_is_skipped_and_counted() -> None:
    names = ("THE OFORD SCHOOL", "THE OXFORD SCHOOL", "UNRELATED SCHOOL")
    generated = _generated(
        names,
        _config(
            maximum_token_frequency=2,
            maximum_typo_signature_frequency=2,
            maximum_typo_context_frequency=2,
        ),
    )
    assert generated.method_pair_counts["typo_context"] == 0
    assert generated.broad_context_tokens_skipped == 1
    assert generated.broad_blocks_skipped["typo_context"] == 1


def test_keys_recovered_by_standalone_typo_are_not_context_queries() -> None:
    generated = _generated(("ODEBRECHT", "ODERBRECHT"), _config())
    assert generated.method_pair_counts["typo_token"] == 1
    assert generated.zero_candidate_keys_before_context == 0
    assert generated.context_lookups_performed == 0
    assert generated.method_pair_counts["typo_context"] == 0


def test_contextual_pair_is_undirected_deduplicated_and_preserves_method() -> None:
    names = ("THE OFORD SCHOOL", "THE OXFORD SCHOOL", "UNRELATED SCHOOL")
    generated = _generated(
        names,
        _config(maximum_token_frequency=2, maximum_typo_signature_frequency=2),
    )
    target_pairs = [
        pair
        for pair in generated.pairs
        if {pair.key_a, pair.key_b} == {"THE OFORD SCHOOL", "THE OXFORD SCHOOL"}
    ]
    assert len(target_pairs) == 1
    assert target_pairs[0].key_a < target_pairs[0].key_b
    assert target_pairs[0].blocking_methods == ("typo_context",)


def test_contextual_fallback_never_self_pairs() -> None:
    generated = _generated(("AER0NAUTICA AERONAUTICA CIVIL",), _config())
    assert generated.pairs == ()


def test_cerveceria_controlled_diagnostic_is_retrieved() -> None:
    names = (
        "CERVECERIA CHIRICAMNA SA 1",
        "CERVECERIA CHIRICANA SA 1",
        "CERVECERIA CHIRICAXNA SA 2",
    )
    generated = _generated(
        names,
        _config(
            maximum_block_size=2,
            maximum_token_frequency=2,
            maximum_typo_signature_frequency=2,
        ),
    )
    assert "typo_context" in _methods_for(
        generated,
        "CERVECERIA CHIRICAMNA SA 1",
        "CERVECERIA CHIRICANA SA 1",
    )


def test_contextual_metrics_reconcile_recovered_residual_keys() -> None:
    names = ("THE OFORD SCHOOL", "THE OXFORD SCHOOL", "UNRELATED SCHOOL")
    generated = _generated(
        names,
        _config(maximum_token_frequency=2, maximum_typo_signature_frequency=2),
    )
    assert generated.zero_candidate_keys_before_context > 0
    assert generated.zero_candidate_keys_recovered_by_context > 0
    assert (
        generated.zero_candidate_keys_remaining_after_context
        == generated.zero_candidate_keys_before_context
        - generated.zero_candidate_keys_recovered_by_context
    )
    assert generated.typo_context_pairs_added == generated.method_pair_counts["typo_context"]
    assert generated.context_lookups_performed > 0


def test_contextual_output_is_deterministic_for_reversed_input() -> None:
    names = ("THE OFORD SCHOOL", "THE OXFORD SCHOOL", "UNRELATED SCHOOL")
    config = _config(maximum_token_frequency=2, maximum_typo_signature_frequency=2)
    first = _generated(names, config)
    second = _generated(tuple(reversed(names)), config)
    assert first.pairs == second.pairs
    assert first.method_pair_counts == second.method_pair_counts


def test_candidate_schema_still_has_no_identity_decision_fields() -> None:
    forbidden = {
        "same_entity",
        "matched",
        "confidence",
        "probability",
        "accepted",
        "canonical_name",
    }
    assert forbidden.isdisjoint(CANDIDATE_PAIRS_SCHEMA.names)
