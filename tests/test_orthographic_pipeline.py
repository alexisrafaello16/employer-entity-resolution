"""Synthetic streaming, gating, reconciliation, metrics, and CLI tests."""

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
    EMPLOYER_ELIGIBILITY_SCHEMA,
    ORTHOGRAPHIC_PAIR_DECISIONS_SCHEMA,
    PAIR_DECISIONS_SCHEMA,
    resolve_orthographic_pairs,
)
from tests.conftest import settings_for


def _ordered_names(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left < right else (right, left)


def _feature_row(
    left: str,
    right: str,
    *,
    numeric_relation: str = "none",
) -> dict[str, object]:
    key_a, key_b = _ordered_names(left, right)
    return {
        "key_a": key_a,
        "key_b": key_b,
        "name_a": key_a,
        "name_b": key_b,
        "blocking_methods": ["typo_context"],
        "blocking_method_count": 1,
        "char_ratio": 100.0,
        "token_sort_ratio": 100.0,
        "token_set_ratio": 100.0,
        "partial_ratio": 100.0,
        "length_ratio": 1.0,
        "common_prefix_ratio": 1.0,
        "token_jaccard": 1.0,
        "same_first_token": True,
        "numeric_relation": numeric_relation,
    }


def _decision_row(
    feature: dict[str, object],
    *,
    status: str = "NEEDS_FURTHER_RESOLUTION",
) -> dict[str, object]:
    return {
        "key_a": feature["key_a"],
        "key_b": feature["key_b"],
        "name_a": feature["name_a"],
        "name_b": feature["name_b"],
        "blocking_methods": feature["blocking_methods"],
        "decision_status": status,
        "decision_rule": (
            "whitespace_only_equivalence"
            if status == "AUTO_SAME"
            else "no_deterministic_equivalence"
        ),
        "decision_evidence": (
            "whitespace_boundary_difference_only"
            if status == "AUTO_SAME"
            else "no_deterministic_equivalence"
        ),
    }


def _eligibility_row(key: str, status: str) -> dict[str, object]:
    return {
        "resolution_key": key,
        "representative_name": key,
        "eligibility_status": status,
        "eligibility_rule": "synthetic_rule",
        "eligibility_evidence": "synthetic_evidence",
    }


def _write_inputs(
    decisions_path: Path,
    eligibility_path: Path,
    features_path: Path,
) -> int:
    feature_rows = [
        _feature_row("FARMACIA SHEKAINAH", "FARMACIA SHEKINAH"),
        _feature_row("PACHOS KITCHEN", "PANCHOS KITCHEN"),
        _feature_row("548 MARKET ST 18590 CA SAN FRA", "ACME CAPITAL"),
        _feature_row("AMBIGUOUS ACTIVITY", "EMPLOYER ACTIVITY"),
        _feature_row("ABCDEF CAPITAL", "ABXYEF CAPITAL"),
        _feature_row(
            "ALPHAAA CAPITAL 1",
            "ALPHAAB CAPITAL",
            numeric_relation="one_sided",
        ),
    ]
    feature_rows.sort(key=lambda row: (str(row["key_a"]), str(row["key_b"])))
    decision_rows = [
        _decision_row(
            row,
            status=(
                "AUTO_SAME"
                if row["key_a"] == min("PACHOS KITCHEN", "PANCHOS KITCHEN")
                else "NEEDS_FURTHER_RESOLUTION"
            ),
        )
        for row in feature_rows
    ]
    statuses = {
        str(row[column]): "EMPLOYER_CANDIDATE"
        for row in feature_rows
        for column in ("key_a", "key_b")
    }
    statuses["548 MARKET ST 18590 CA SAN FRA"] = "ADDRESS"
    statuses["AMBIGUOUS ACTIVITY"] = "AMBIGUOUS"
    statuses["FARMACIA SHEKYNAH"] = "AMBIGUOUS"
    eligibility_rows = [_eligibility_row(key, statuses[key]) for key in sorted(statuses)]

    decisions_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(decision_rows, schema=PAIR_DECISIONS_SCHEMA),
        decisions_path,
    )
    pq.write_table(
        pa.Table.from_pylist(eligibility_rows, schema=EMPLOYER_ELIGIBILITY_SCHEMA),
        eligibility_path,
    )
    pq.write_table(
        pa.Table.from_pylist(feature_rows, schema=CANDIDATE_FEATURES_SCHEMA),
        features_path,
    )
    return len(feature_rows)


def test_orthographic_pipeline_processes_every_prior_residual_once(
    tmp_path: Path,
    workbook_factory: Callable[..., Path],
) -> None:
    workbook = workbook_factory(tmp_path / "input/source.xlsx", ["unused"])
    settings = settings_for(tmp_path, workbook)
    config = settings.orthographic_resolution
    total_pairs = _write_inputs(
        config.pair_decisions_dataset,
        config.employer_eligibility_dataset,
        config.candidate_features_dataset,
    )
    input_bytes = {
        path: path.read_bytes()
        for path in (
            config.pair_decisions_dataset,
            config.employer_eligibility_dataset,
            config.candidate_features_dataset,
        )
    }

    result = resolve_orthographic_pairs(tmp_path, settings)
    first_output = result.output_path.read_bytes()
    output = pq.read_table(result.output_path)
    rows = output.to_pylist()
    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))

    assert output.schema == ORTHOGRAPHIC_PAIR_DECISIONS_SCHEMA
    assert result.residual_pairs == 5
    assert result.strong_orthographic_evidence_count == 1
    assert result.needs_further_resolution_count == 2
    assert result.not_eligible_count == 2
    assert len(rows) == 5
    assert all("PACHOS KITCHEN" not in {row["key_a"], row["key_b"]} for row in rows)
    assert metrics["pair_decisions_read"] == total_pairs
    assert metrics["prior_auto_same_pairs_skipped"] == 1
    assert metrics["residual_pairs_read"] == 5
    assert metrics["eligible_employer_residual_pairs"] == 3
    assert metrics["orthographic_status_counts"] == {
        "STRONG_ORTHOGRAPHIC_EVIDENCE": 1,
        "NEEDS_FURTHER_RESOLUTION": 2,
        "NOT_ELIGIBLE_FOR_ORTHOGRAPHIC": 2,
    }
    assert metrics["successful_numeric_relation_distribution"] == {
        "conflict": 0,
        "none": 1,
        "one_sided": 0,
        "same": 0,
    }
    assert metrics["successful_edit_operation_counts"] == {
        "deletion": 1,
        "insertion": 0,
        "substitution": 0,
    }
    successful_row = next(
        row for row in rows if row["orthographic_status"] == "STRONG_ORTHOGRAPHIC_EVIDENCE"
    )
    assert successful_row["differing_token_a"] == "SHEKAINAH"
    assert successful_row["differing_token_b"] == "SHEKINAH"
    assert successful_row["edit_operation"] == "deletion"
    assert successful_row["context_signature"] == "FARMACIA <DIFF>"
    assert successful_row["context_variant_count"] == 2
    assert successful_row["token_support_a"] == 1
    assert successful_row["token_support_b"] == 1
    assert metrics["v2_1_strong_orthographic_evidence_count"] == 1
    assert metrics["v2_1_safety_abstention_counts"] == {
        "both_differing_tokens_established": 0,
        "multi_variant_context": 0,
        "no_distinctive_exact_context": 0,
        "terminal_edit_requires_further_resolution": 0,
    }
    assert metrics["successful_context_variant_count_distribution"] == {"2": 1}
    assert metrics["successful_token_support_distribution"]["1"] == 2
    assert metrics["edit_location_distribution"]["inside"] == 1
    assert metrics["reconciliation"]["status"] == "passed"
    assert all(path.read_bytes() == content for path, content in input_bytes.items())

    second = resolve_orthographic_pairs(tmp_path, settings)
    assert second.output_path.read_bytes() == first_output
    second_metrics = json.loads(second.metrics_path.read_text(encoding="utf-8"))
    metrics.pop("execution_runtime_seconds")
    second_metrics.pop("execution_runtime_seconds")
    assert second_metrics == metrics


def test_resolve_orthographic_cli_honors_explicit_paths_and_prints_counts(
    tmp_path: Path,
    workbook_factory: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    workbook = workbook_factory(tmp_path / "input/source.xlsx", ["unused"])
    settings = settings_for(tmp_path, workbook)
    decisions_path = tmp_path / "explicit-decisions.parquet"
    eligibility_path = tmp_path / "explicit-eligibility.parquet"
    features_path = tmp_path / "explicit-features.parquet"
    output_path = tmp_path / "explicit-orthographic.parquet"
    _write_inputs(decisions_path, eligibility_path, features_path)
    monkeypatch.setattr("credit_risk_er.cli.load_settings", lambda _: settings)

    exit_code = main(
        [
            "resolve-orthographic",
            "--config",
            str(tmp_path / "config/config.yaml"),
            "--decisions",
            str(decisions_path),
            "--eligibility",
            str(eligibility_path),
            "--features",
            str(features_path),
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert pq.ParquetFile(output_path).metadata.num_rows == 5
    printed = capsys.readouterr().out
    assert "Residual pairs: 5" in printed
    assert "STRONG_ORTHOGRAPHIC_EVIDENCE: 1" in printed
    assert "NEEDS_FURTHER_RESOLUTION: 2" in printed
    assert "NOT_ELIGIBLE_FOR_ORTHOGRAPHIC: 2" in printed
