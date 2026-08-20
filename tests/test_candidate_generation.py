"""Bounded candidate-generation behavior and integration tests."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from credit_risk_er.config import CandidateGenerationConfig
from credit_risk_er.matching.candidates import (
    ResolutionKey,
    ResolutionKeyAggregate,
    exclusion_reason,
    finalize_resolution_keys,
    generate_candidate_pairs,
    update_resolution_key,
)
from credit_risk_er.pipeline import (
    CANDIDATE_PAIRS_SCHEMA,
    RESOLUTION_KEYS_SCHEMA,
    generate_candidates,
    preprocess,
    resolve_employers,
)
from tests.conftest import settings_for


def _config(**updates: int | Path) -> CandidateGenerationConfig:
    payload: dict[str, int | Path] = {
        "resolution_keys_output": Path("keys.parquet"),
        "candidate_pairs_output": Path("pairs.parquet"),
        "metrics_output": Path("metrics.json"),
        "maximum_block_size": 10,
        "minimum_truncation_prefix_length": 10,
        "minimum_informative_token_length": 5,
        "maximum_token_frequency": 10,
        "prefix_signature_length": 6,
        "minimum_typo_token_length": 6,
        "maximum_typo_signature_frequency": 10,
        "maximum_typo_context_frequency": 20,
    }
    payload.update(updates)
    return CandidateGenerationConfig.model_validate(payload)


def _key(
    name: str,
    *,
    relaxed: str | None = None,
    trailing: str | None = None,
    truncated: bool = False,
    source_row: int = 2,
) -> ResolutionKey:
    return ResolutionKey(
        resolution_key=name,
        representative_name=name,
        relaxed_key=relaxed or name,
        source_row_frequency=1,
        representative_record_id=f"record-{source_row}",
        representative_source_row_number=source_row,
        representative_route="employer_resolution_candidate",
        trailing_numeric_candidate=trailing,
        possible_truncation=truncated,
        token_count=len(name.split()),
    )


def _method_pairs(
    keys: tuple[ResolutionKey, ...], method: str, config: CandidateGenerationConfig | None = None
) -> set[tuple[str, str]]:
    generated = generate_candidate_pairs(keys, config or _config())
    return {(pair.key_a, pair.key_b) for pair in generated.pairs if method in pair.blocking_methods}


def test_same_relaxed_key_generates_one_canonical_pair() -> None:
    keys = (
        _key("BANCO GENERAL S.A.", relaxed="BANCO GENERAL SA"),
        _key("BANCO GENERAL S A", relaxed="BANCO GENERAL SA", source_row=3),
    )
    generated = generate_candidate_pairs(keys, _config())
    assert len(generated.pairs) == 1
    pair = generated.pairs[0]
    assert pair.key_a < pair.key_b
    assert "relaxed_key" in pair.blocking_methods


def test_same_strict_key_is_aggregated_once_and_never_self_paired() -> None:
    aggregates: dict[str, ResolutionKeyAggregate] = {}
    base = {
        "nombre_normalizado": "ACME",
        "nombre_matching": "ACME",
        "trailing_numeric_candidate": None,
        "possible_truncation": False,
        "token_count": 1,
        "route": "employer_resolution_candidate",
    }
    update_resolution_key(
        aggregates,
        {
            **base,
            "record_id": "later",
            "source_row_number": 10,
            "possible_truncation": True,
            "token_count": 3,
            "trailing_numeric_candidate": "ACME BASE",
            "route": "ambiguous_review_candidate",
        },
    )
    update_resolution_key(aggregates, {**base, "record_id": "earlier", "source_row_number": 2})
    keys = finalize_resolution_keys(aggregates)
    assert len(keys) == 1
    assert keys[0].source_row_frequency == 2
    assert keys[0].representative_record_id == "earlier"
    assert keys[0].representative_route == "employer_resolution_candidate"
    assert keys[0].possible_truncation is True
    assert keys[0].token_count == 3
    assert keys[0].trailing_numeric_candidate == "ACME BASE"
    assert generate_candidate_pairs(keys, _config()).pairs == ()


def test_trailing_one_digit_candidate_finds_existing_strict_key() -> None:
    keys = (_key("BANCO GENERAL"), _key("BANCO GENERAL 4", trailing="BANCO GENERAL"))
    assert _method_pairs(keys, "trailing_numeric") == {("BANCO GENERAL", "BANCO GENERAL 4")}


def test_studio_507_is_not_reduced_to_studio() -> None:
    keys = (_key("STUDIO"), _key("STUDIO 507", trailing=None))
    assert ("STUDIO", "STUDIO 507") not in _method_pairs(keys, "trailing_numeric")


def test_truncation_prefix_retrieves_longer_candidate() -> None:
    keys = (
        _key("EMPRESA DE TRANSMISION ELECTRIC", truncated=True),
        _key("EMPRESA DE TRANSMISION ELECTRICA SA"),
    )
    assert len(_method_pairs(keys, "truncation_prefix")) == 1


def test_broad_prefix_blocks_are_skipped() -> None:
    keys = tuple(
        _key(name, relaxed=f"RELAXED {index}")
        for index, name in enumerate(("TECNOSERV A", "TECNOSERV B", "TECNOSERV C"))
    )
    generated = generate_candidate_pairs(
        keys, _config(maximum_block_size=2, maximum_token_frequency=2)
    )
    assert generated.method_pair_counts["prefix_signature"] == 0
    assert generated.broad_blocks_skipped["prefix_signature"] == 1


def test_informative_tokens_retrieve_names_but_generic_tokens_do_not() -> None:
    useful = (_key("ALPHA TECNOSERV"), _key("BETA TECNOSERV", source_row=3))
    generic = (_key("ALPHA GRUPO"), _key("BETA GRUPO", source_row=3))
    assert len(_method_pairs(useful, "informative_token")) == 1
    assert _method_pairs(generic, "informative_token") == set()


def test_multiple_methods_are_deduplicated_and_preserved() -> None:
    keys = (
        _key("ACME-SOUTH", relaxed="ACME SOUTH"),
        _key("ACME SOUTH", relaxed="ACME SOUTH", source_row=3),
    )
    generated = generate_candidate_pairs(keys, _config())
    assert len(generated.pairs) == 1
    assert generated.pairs[0].blocking_methods == (
        "relaxed_key",
        "prefix_signature",
    )
    assert generated.multi_method_pairs == 1


@pytest.mark.parametrize(
    ("updates", "expected"),
    [
        ({"is_blank": True, "route": "blank_candidate"}, "blank"),
        ({"is_numeric_only": True}, "numeric_only"),
        ({"route": "non_employer_status_candidate"}, "non_employer_status"),
        ({"route": "address_candidate"}, "address_candidate"),
        ({"resolution_status": "resolved"}, "already_resolved"),
        ({"route": "ambiguous_review_candidate"}, None),
    ],
)
def test_eligibility_policy(updates: dict[str, object], expected: str | None) -> None:
    row: dict[str, object] = {
        "resolution_status": "unresolved",
        "is_blank": False,
        "is_numeric_only": False,
        "route": "employer_resolution_candidate",
        "nombre_normalizado": "ACME",
    }
    row.update(updates)
    assert exclusion_reason(row) == expected


def test_candidate_output_is_deterministic_and_has_no_identity_decisions() -> None:
    keys = (
        _key("ZETA TECNOSERV"),
        _key("ALPHA TECNOSERV", source_row=3),
        _key("BETA TECNOSERV", source_row=4),
    )
    first = generate_candidate_pairs(keys, _config())
    second = generate_candidate_pairs(tuple(reversed(keys)), _config())
    assert first.pairs == second.pairs
    assert list(first.pairs) == sorted(first.pairs, key=lambda pair: (pair.key_a, pair.key_b))
    forbidden = {"same_entity", "matched", "confidence", "probability", "resolved_by_blocking"}
    assert forbidden.isdisjoint(CANDIDATE_PAIRS_SCHEMA.names)


def test_candidate_pipeline_smoke_with_synthetic_resolved_dataset(
    tmp_path: Path,
    workbook_factory: Callable[..., Path],
    reference_factory: Callable[..., tuple[Path, Path]],
) -> None:
    values = [
        "ACME-SOUTH",
        "ACME SOUTH",
        "ACME SOUTH",
        "BANCO GENERAL",
        "BANCO GENERAL 4",
        "STUDIO",
        "STUDIO 507",
        None,
        "123",
        "ESTUDIANTE",
        "CALLE 10",
        "ALPHA TECNOSERV",
        "BETA TECNOSERV",
    ]
    workbook = workbook_factory(tmp_path / "source.xlsx", values)
    settings = settings_for(tmp_path, workbook)
    reference_factory(tmp_path / "data/reference", master_rows=[], alias_rows=[])
    preprocess(tmp_path, settings)
    resolved = resolve_employers(tmp_path, settings)
    input_hash = resolved.output_path.read_bytes()

    result = generate_candidates(tmp_path, settings)
    keys = pq.read_table(result.resolution_keys_path)
    pairs = pq.read_table(result.candidate_pairs_path)
    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))

    assert keys.schema == RESOLUTION_KEYS_SCHEMA
    assert pairs.schema == CANDIDATE_PAIRS_SCHEMA
    assert keys.column("resolution_key").to_pylist() == sorted(
        keys.column("resolution_key").to_pylist()
    )
    acme = (
        keys.filter(pc.equal(keys.column("resolution_key"), "ACME SOUTH"))
        .column("source_row_frequency")[0]
        .as_py()
    )
    assert acme == 2
    assert result.eligible_rows == 9
    assert metrics["excluded_rows"] == {
        "address_candidate": 1,
        "blank": 1,
        "non_employer_status": 1,
        "numeric_only": 1,
    }
    typo_metrics = metrics["typo_fallback"]
    assert (
        typo_metrics["zero_candidate_keys_remaining"]
        == typo_metrics["zero_candidate_keys_before"]
        - typo_metrics["zero_candidate_keys_recovered"]
    )
    assert (
        typo_metrics["pairs_containing_typo_token"]
        == metrics["candidate_pairs_by_method"]["typo_token"]
    )
    context_metrics = metrics["typo_context_fallback"]
    assert (
        context_metrics["zero_candidate_keys_remaining"]
        == context_metrics["zero_candidate_keys_before"]
        - context_metrics["zero_candidate_keys_recovered"]
    )
    assert (
        context_metrics["pairs_containing_typo_context"]
        == metrics["candidate_pairs_by_method"]["typo_context"]
    )
    assert pairs.num_rows > 0
    assert resolved.output_path.read_bytes() == input_hash
