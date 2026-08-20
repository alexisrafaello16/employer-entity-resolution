"""Typo-tolerant fallback retrieval for zero-candidate resolution keys."""

from __future__ import annotations

from pathlib import Path

from credit_risk_er.config import CandidateGenerationConfig
from credit_risk_er.matching.candidates import (
    ResolutionKey,
    generate_candidate_pairs,
    typo_signatures,
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
        "prefix_signature_length": 6,
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


def _typo_pairs(
    names: tuple[str, ...], config: CandidateGenerationConfig | None = None
) -> set[tuple[str, str]]:
    generated = generate_candidate_pairs(
        tuple(_key(name, index + 2) for index, name in enumerate(names)),
        config or _config(),
    )
    return {
        (pair.key_a, pair.key_b)
        for pair in generated.pairs
        if "typo_token" in pair.blocking_methods
    }


def test_deletion_signatures_include_original_and_each_single_deletion() -> None:
    signatures = typo_signatures("ABCDEF")
    assert "ABCDEF" in signatures
    assert "ACDEF" in signatures
    assert "ABCDE" in signatures
    assert len(signatures) == 7


def test_single_character_insertion_or_deletion_is_retrieved() -> None:
    assert _typo_pairs(("ODEBRECHT", "ODERBRECHT")) == {("ODEBRECHT", "ODERBRECHT")}


def test_single_character_substitution_is_retrieved() -> None:
    assert _typo_pairs(("AER0NAUTICA", "AERONAUTICA")) == {("AER0NAUTICA", "AERONAUTICA")}


def test_missing_character_variant_is_retrieved() -> None:
    assert _typo_pairs(("NUTRICIN", "NUTRICION"), _config(prefix_signature_length=8)) == {
        ("NUTRICIN", "NUTRICION")
    }


def test_keys_with_existing_candidates_do_not_query_typo_fallback() -> None:
    names = (
        "ODEBRECHT",
        "ODEBRECHT BETA",
        "ODERBRECHT",
        "ODERBRECHT ALPHA",
    )
    generated = generate_candidate_pairs(tuple(_key(name) for name in names), _config())
    assert generated.zero_candidate_keys_before_typo == 0
    assert generated.method_pair_counts["typo_token"] == 0
    assert ("ODEBRECHT", "ODERBRECHT") not in {(pair.key_a, pair.key_b) for pair in generated.pairs}


def test_complete_universe_can_supply_candidate_target_for_zero_key() -> None:
    names = ("ODEBRECHT", "ODERBRECHT", "ODERBRECHT ALPHA")
    generated = generate_candidate_pairs(tuple(_key(name) for name in names), _config())
    pair_methods = {(pair.key_a, pair.key_b): pair.blocking_methods for pair in generated.pairs}
    assert "typo_token" in pair_methods[("ODEBRECHT", "ODERBRECHT")]


def test_generic_tokens_do_not_generate_typo_candidates() -> None:
    assert _typo_pairs(("ALPHA EMPRESA", "BETA EMPRESAS")) == set()


def test_tokens_below_minimum_length_are_ignored() -> None:
    assert _typo_pairs(("ABCDE", "XABCDE"), _config(minimum_typo_token_length=7)) == set()


def test_numeric_only_tokens_are_ignored() -> None:
    assert _typo_pairs(("ALPHA 123456", "BETA 123457")) == set()


def test_broad_typo_signatures_are_skipped_and_counted() -> None:
    keys = tuple(_key(name) for name in ("XBCDEFG", "YBCDEFG", "ZBCDEFG"))
    generated = generate_candidate_pairs(keys, _config(maximum_typo_signature_frequency=2))
    assert generated.method_pair_counts["typo_token"] == 0
    assert generated.broad_typo_signatures_skipped == 1
    assert generated.broad_blocks_skipped["typo_token"] == 1


def test_typo_pairs_are_self_free_canonical_and_not_duplicated() -> None:
    generated = generate_candidate_pairs((_key("ODERBRECHT"), _key("ODEBRECHT")), _config())
    assert len(generated.pairs) == 1
    pair = generated.pairs[0]
    assert pair.key_a < pair.key_b
    assert pair.key_a != pair.key_b
    assert pair.blocking_methods == ("typo_token",)


def test_single_key_never_self_pairs_through_its_signatures() -> None:
    assert generate_candidate_pairs((_key("ODERBRECHT"),), _config()).pairs == ()


def test_fallback_metrics_reconcile_recovered_zero_keys() -> None:
    generated = generate_candidate_pairs((_key("ODEBRECHT"), _key("ODERBRECHT")), _config())
    assert generated.zero_candidate_keys_before_typo == 2
    assert generated.zero_candidate_keys_recovered_by_typo == 2
    assert generated.zero_candidate_keys_remaining_after_typo == 0
    assert generated.typo_candidate_pairs_added == 1
    assert generated.method_pair_counts["typo_token"] == 1
    assert generated.typo_signatures_considered > 0


def test_typo_output_is_deterministic_for_reversed_input_order() -> None:
    keys = (_key("ODEBRECHT"), _key("ODERBRECHT"), _key("UNRELATED"))
    first = generate_candidate_pairs(keys, _config())
    second = generate_candidate_pairs(tuple(reversed(keys)), _config())
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
