"""Independent fuzzy/structural evidence and batched scoring tests."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from credit_risk_er.cli import main
from credit_risk_er.matching.fuzzy import compute_pair_features, numeric_relation
from credit_risk_er.pipeline import (
    CANDIDATE_FEATURES_SCHEMA,
    CANDIDATE_PAIRS_SCHEMA,
    score_candidate_pairs,
)
from tests.conftest import settings_for


def _write_candidates(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=CANDIDATE_PAIRS_SCHEMA), path)
    return path


def _candidate(
    key_a: str,
    key_b: str,
    methods: list[str] | None = None,
) -> dict[str, object]:
    return {
        "key_a": key_a,
        "key_b": key_b,
        "name_a": key_a,
        "name_b": key_b,
        "blocking_methods": methods or ["prefix_signature"],
    }


def test_identical_strings_have_maximum_character_evidence() -> None:
    evidence = compute_pair_features("AIRLINES", "AIRLINES", ["relaxed_key"])
    assert evidence.char_ratio == 100.0
    assert evidence.token_sort_ratio == 100.0
    assert evidence.token_set_ratio == 100.0
    assert evidence.partial_ratio == 100.0


def test_simple_typo_has_high_but_nonidentical_character_evidence() -> None:
    evidence = compute_pair_features("AIRLINEZ", "AIRLINES", ["prefix_signature"])
    assert 85.0 < evidence.char_ratio < 100.0


def test_token_reordering_is_visible_in_token_sort_evidence() -> None:
    evidence = compute_pair_features("GENERAL BANCO", "BANCO GENERAL", ["informative_token"])
    assert evidence.token_sort_ratio == 100.0
    assert evidence.char_ratio < evidence.token_sort_ratio


def test_additional_token_is_visible_in_token_set_evidence_without_decision() -> None:
    evidence = compute_pair_features(
        "BANCO GENERAL", "BANCO GENERAL INVERSIONES", ["informative_token"]
    )
    assert evidence.token_set_ratio == 100.0
    assert evidence.char_ratio < evidence.token_set_ratio
    assert not hasattr(evidence, "is_match")


def test_truncation_has_partial_and_prefix_evidence() -> None:
    evidence = compute_pair_features(
        "365 CELULAR COMPUTADORAS Y ACC",
        "365 CELULAR COMPUTADORAS Y ACCESORIOS",
        ["truncation_prefix"],
    )
    assert evidence.partial_ratio == 100.0
    assert evidence.common_prefix_ratio == 1.0
    assert evidence.length_ratio < 1.0


def test_length_and_common_prefix_ratios_follow_documented_denominators() -> None:
    evidence = compute_pair_features("ABCD", "ABXYZZZZ", ["prefix_signature"])
    assert evidence.length_ratio == 0.5
    assert evidence.common_prefix_ratio == 0.5


def test_token_jaccard_uses_token_sets() -> None:
    evidence = compute_pair_features("ALPHA BETA", "ALPHA GAMMA", ["informative_token"])
    assert evidence.token_jaccard == pytest.approx(1 / 3, abs=1e-6)


@pytest.mark.parametrize(
    ("name_a", "name_b", "expected"),
    [
        ("BANCO GENERAL", "BANCO GLOBAL", True),
        ("BANCO GENERAL", "GENERAL BANCO", False),
    ],
)
def test_same_first_token(name_a: str, name_b: str, expected: bool) -> None:
    assert compute_pair_features(name_a, name_b, ["informative_token"]).same_first_token is expected


@pytest.mark.parametrize(
    ("name_a", "name_b", "expected"),
    [
        ("BANCO GENERAL", "BANCO GLOBAL", "none"),
        ("STUDIO 507", "STUDIO 507", "same"),
        ("BANCO GENERAL", "BANCO GENERAL 4", "one_sided"),
        ("STUDIO 507", "STUDIO 508", "conflict"),
        ("GRUPO 7 SUCURSAL 2", "GRUPO 2 SUCURSAL 7", "conflict"),
    ],
)
def test_numeric_relation_categories(name_a: str, name_b: str, expected: str) -> None:
    assert numeric_relation(name_a, name_b) == expected


def test_blocking_method_count_is_distinct_and_features_are_deterministic() -> None:
    methods = ["informative_token", "prefix_signature", "informative_token"]
    first = compute_pair_features("TECNOSERV INTENATIONAL", "TECNOSERV INTERNATIONAL", methods)
    second = compute_pair_features("TECNOSERV INTENATIONAL", "TECNOSERV INTERNATIONAL", methods)
    assert first == second
    assert first.blocking_method_count == 2


def test_feature_schema_has_no_identity_decision_or_aggregate_score_fields() -> None:
    forbidden = {
        "same_entity",
        "is_match",
        "matched",
        "accepted",
        "rejected",
        "confidence",
        "probability",
        "resolution_status",
        "fuzzy_resolution",
        "final_score",
        "weighted_score",
        "best_score",
        "overall_score",
        "match_score",
        "ensemble_score",
    }
    assert forbidden.isdisjoint(CANDIDATE_FEATURES_SCHEMA.names)


def test_batched_pipeline_preserves_candidate_rows_and_evidence(
    tmp_path: Path,
    workbook_factory: Callable[..., Path],
) -> None:
    workbook = workbook_factory(tmp_path / "input/source.xlsx", ["unused"])
    settings = settings_for(tmp_path, workbook)
    rows = [
        _candidate("BANCO GENERAL", "BANCO GENERAL 4", ["trailing_numeric"]),
        _candidate(
            "TECNOSERV INTENATIONAL",
            "TECNOSERV INTERNATIONAL",
            ["informative_token", "prefix_signature"],
        ),
    ]
    input_path = _write_candidates(settings.candidate_scoring.input_dataset, rows)
    input_bytes = input_path.read_bytes()

    result = score_candidate_pairs(tmp_path, settings)
    output = pq.read_table(result.output_path)
    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))

    assert output.schema == CANDIDATE_FEATURES_SCHEMA
    assert result.candidate_pairs == result.feature_rows == len(rows)
    assert output.select(CANDIDATE_PAIRS_SCHEMA.names).to_pylist() == rows
    assert output.column("blocking_method_count").to_pylist() == [1, 2]
    assert output.column("numeric_relation").to_pylist() == ["one_sided", "none"]
    assert metrics["reconciliation"]["status"] == "passed"
    assert input_path.read_bytes() == input_bytes


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (_candidate("SAME", "SAME"), "key_a < key_b"),
        (_candidate("ZETA", "ALPHA"), "key_a < key_b"),
        (
            {
                "key_a": "ALPHA",
                "key_b": "BETA",
                "name_a": "",
                "name_b": "BETA",
                "blocking_methods": ["prefix_signature"],
            },
            "nonblank",
        ),
    ],
)
def test_pipeline_rejects_invalid_candidate_contract(
    tmp_path: Path,
    workbook_factory: Callable[..., Path],
    row: dict[str, object],
    message: str,
) -> None:
    workbook = workbook_factory(tmp_path / "input/source.xlsx", ["unused"])
    settings = settings_for(tmp_path, workbook)
    _write_candidates(settings.candidate_scoring.input_dataset, [row])
    with pytest.raises(ValueError, match=message):
        score_candidate_pairs(tmp_path, settings)


def test_score_candidates_cli_smoke(
    tmp_path: Path,
    workbook_factory: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    workbook = workbook_factory(tmp_path / "input/source.xlsx", ["unused"])
    settings = settings_for(tmp_path, workbook)
    input_path = _write_candidates(
        tmp_path / "candidate-input.parquet", [_candidate("AIRLINES", "AIRLINEZ")]
    )
    output_path = tmp_path / "candidate-output.parquet"
    config_path = tmp_path / "config" / "config.yaml"
    monkeypatch.setattr("credit_risk_er.cli.load_settings", lambda _: settings)

    exit_code = main(
        [
            "score-candidates",
            "--config",
            str(config_path),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert pq.ParquetFile(output_path).metadata.num_rows == 1
    assert "Feature rows written: 1" in capsys.readouterr().out
