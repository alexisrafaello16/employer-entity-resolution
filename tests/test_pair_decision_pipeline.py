"""Batched pair-decision output, metrics, reconciliation, and CLI tests."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from credit_risk_er.cli import main
from credit_risk_er.pipeline import (
    CANDIDATE_FEATURES_SCHEMA,
    PAIR_DECISIONS_SCHEMA,
    RESOLUTION_KEYS_SCHEMA,
    decide_candidate_pairs,
)
from tests.conftest import settings_for

UNIQUE_TRUNCATED = "INTERNATIONAL TECHNOLOGY SYSTE"
AMBIGUOUS_TRUNCATED = "INVERSIONES KACH DE CHIRIQUI S"


def _ordered_pair(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left < right else (right, left)


def _feature_row(
    left: str,
    right: str,
    *,
    numeric_relation: str = "none",
    methods: list[str] | None = None,
    char_ratio: float = 99.0,
    token_set_ratio: float = 100.0,
) -> dict[str, object]:
    key_a, key_b = _ordered_pair(left, right)
    blocking_methods = methods or ["informative_token"]
    return {
        "key_a": key_a,
        "key_b": key_b,
        "name_a": key_a,
        "name_b": key_b,
        "blocking_methods": blocking_methods,
        "blocking_method_count": len(set(blocking_methods)),
        "char_ratio": char_ratio,
        "token_sort_ratio": char_ratio,
        "token_set_ratio": token_set_ratio,
        "partial_ratio": 100.0,
        "length_ratio": 0.9,
        "common_prefix_ratio": 0.9,
        "token_jaccard": 0.5,
        "same_first_token": True,
        "numeric_relation": numeric_relation,
    }


def _resolution_row(key: str, *, truncated: bool = False) -> dict[str, object]:
    return {
        "resolution_key": key,
        "representative_name": key,
        "relaxed_key": key,
        "source_row_frequency": 1,
        "representative_record_id": f"record-{key}",
        "representative_source_row_number": 2,
        "representative_route": "employer_resolution_candidate",
        "trailing_numeric_candidate": None,
        "possible_truncation": truncated,
        "token_count": len(key.split()),
    }


def _write_feature_and_key_inputs(
    feature_path: Path,
    keys_path: Path,
    rows: list[dict[str, object]],
    truncated_keys: set[str] | None = None,
) -> None:
    ordered_rows = sorted(rows, key=lambda row: (str(row["key_a"]), str(row["key_b"])))
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(ordered_rows, schema=CANDIDATE_FEATURES_SCHEMA), feature_path)
    all_keys = sorted(
        {
            str(row[column])
            for row in ordered_rows
            for column in ("key_a", "key_b")
        }
    )
    truncated = truncated_keys or set()
    pq.write_table(
        pa.Table.from_pylist(
            [_resolution_row(key, truncated=key in truncated) for key in all_keys],
            schema=RESOLUTION_KEYS_SCHEMA,
        ),
        keys_path,
    )


def test_batched_decision_pipeline_reconciles_every_candidate_pair(
    tmp_path: Path,
    workbook_factory: Callable[..., Path],
) -> None:
    assert len(UNIQUE_TRUNCATED) == len(AMBIGUOUS_TRUNCATED) == 30
    workbook = workbook_factory(tmp_path / "input/source.xlsx", ["unused"])
    settings = settings_for(tmp_path, workbook)
    unique_longer = f"{UNIQUE_TRUNCATED}MS INC"
    ambiguous_longers = (
        f"{AMBIGUOUS_TRUNCATED}A",
        f"{AMBIGUOUS_TRUNCATED}OCIEDAD",
    )
    rows = [
        _feature_row("EMPRESA S A", "EMPRESA SA", methods=["relaxed_key"]),
        _feature_row("ELECTROCENTRO", "ELECTROCENTRO S A"),
        _feature_row("PACHOS KITCHEN", "PANCHOS KITCHEN", char_ratio=100.0),
        _feature_row(UNIQUE_TRUNCATED, unique_longer, methods=["truncation_prefix"]),
        *(
            _feature_row(
                AMBIGUOUS_TRUNCATED,
                longer,
                methods=["truncation_prefix"],
            )
            for longer in ambiguous_longers
        ),
    ]
    _write_feature_and_key_inputs(
        settings.pair_decision.input_dataset,
        settings.pair_decision.resolution_keys_dataset,
        rows,
        {UNIQUE_TRUNCATED, AMBIGUOUS_TRUNCATED},
    )
    input_bytes = settings.pair_decision.input_dataset.read_bytes()
    keys_bytes = settings.pair_decision.resolution_keys_dataset.read_bytes()

    result = decide_candidate_pairs(tmp_path, settings)
    output = pq.read_table(result.output_path)
    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    output_rows = output.to_pylist()

    assert output.schema == PAIR_DECISIONS_SCHEMA
    assert result.candidate_pairs == result.decision_rows == len(rows)
    assert result.auto_same_count == 3
    assert result.needs_further_resolution_count == 3
    assert len({(row["key_a"], row["key_b"]) for row in output_rows}) == len(rows)
    assert output.column("decision_status").to_pylist().count("AUTO_SAME") == 3
    assert "char_ratio" not in output.schema.names
    assert "token_set_ratio" not in output.schema.names
    assert "confidence" not in output.schema.names
    assert metrics["candidate_pairs_read"] == metrics["decision_rows_written"] == len(rows)
    assert metrics["auto_same_by_rule"] == {
        "legal_suffix_addition_equivalence": 1,
        "legal_suffix_format_equivalence": 1,
        "unique_source_truncation_equivalence": 1,
        "whitespace_only_equivalence": 0,
    }
    assert metrics["multiple_rule_evidence_pairs"] == 1
    assert metrics["truncation"] == {
        "ambiguous_multi_continuation_candidates": 2,
        "ambiguous_multi_continuation_keys": 1,
        "ambiguous_multi_continuation_truncations_abstained": 2,
        "exact_truncation_candidates_considered": 3,
        "unique_truncation_auto_resolutions": 1,
    }
    assert metrics["reconciliation"]["status"] == "passed"
    assert settings.pair_decision.input_dataset.read_bytes() == input_bytes
    assert settings.pair_decision.resolution_keys_dataset.read_bytes() == keys_bytes


def test_decision_output_order_is_deterministic_across_repeated_runs(
    tmp_path: Path,
    workbook_factory: Callable[..., Path],
) -> None:
    workbook = workbook_factory(tmp_path / "input/source.xlsx", ["unused"])
    settings = settings_for(tmp_path, workbook)
    rows = [
        _feature_row("MOBIL PHONE", "MOBILPHONE"),
        _feature_row("BANCO GENERAL", "BANCO GENERAL 4", numeric_relation="one_sided"),
    ]
    _write_feature_and_key_inputs(
        settings.pair_decision.input_dataset,
        settings.pair_decision.resolution_keys_dataset,
        rows,
    )

    first = decide_candidate_pairs(tmp_path, settings).output_path.read_bytes()
    second = decide_candidate_pairs(tmp_path, settings).output_path.read_bytes()
    assert first == second


def test_decide_pairs_cli_uses_explicit_paths_and_prints_concise_counts(
    tmp_path: Path,
    workbook_factory: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    workbook = workbook_factory(tmp_path / "input/source.xlsx", ["unused"])
    settings = settings_for(tmp_path, workbook)
    input_path = tmp_path / "explicit-features.parquet"
    keys_path = tmp_path / "explicit-keys.parquet"
    output_path = tmp_path / "explicit-decisions.parquet"
    _write_feature_and_key_inputs(
        input_path,
        keys_path,
        [_feature_row("ELECTROCENTRO", "ELECTROCENTRO S A")],
    )
    monkeypatch.setattr("credit_risk_er.cli.load_settings", lambda _: settings)

    exit_code = main(
        [
            "decide-pairs",
            "--config",
            str(tmp_path / "config/config.yaml"),
            "--input",
            str(input_path),
            "--keys",
            str(keys_path),
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert pq.ParquetFile(output_path).metadata.num_rows == 1
    printed = capsys.readouterr().out
    assert "Decision rows written: 1" in printed
    assert "AUTO_SAME: 1" in printed
    assert "NEEDS_FURTHER_RESOLUTION: 0" in printed
