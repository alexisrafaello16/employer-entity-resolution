"""Synthetic streaming, gating, joining, reconciliation, metrics, and CLI tests."""

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
    RESIDUAL_RELATIONSHIP_PROFILE_SCHEMA,
    RESOLUTION_KEYS_SCHEMA,
    profile_residual_relationships,
)
from tests.conftest import settings_for


def _ordered(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left < right else (right, left)


def _feature_row(
    left: str,
    right: str,
    *,
    numeric_relation: str = "none",
) -> dict[str, object]:
    key_a, key_b = _ordered(left, right)
    return {
        "key_a": key_a,
        "key_b": key_b,
        "name_a": key_a,
        "name_b": key_b,
        "blocking_methods": ["informative_token"],
        "blocking_method_count": 1,
        "char_ratio": 80.0,
        "token_sort_ratio": 80.0,
        "token_set_ratio": 80.0,
        "partial_ratio": 80.0,
        "length_ratio": 0.8,
        "common_prefix_ratio": 0.5,
        "token_jaccard": 0.5,
        "same_first_token": key_a.split()[0] == key_b.split()[0],
        "numeric_relation": numeric_relation,
    }


def _orthographic_row(
    feature: dict[str, object],
    status: str,
    evidence: str,
) -> dict[str, object]:
    return {
        "key_a": feature["key_a"],
        "key_b": feature["key_b"],
        "name_a": feature["name_a"],
        "name_b": feature["name_b"],
        "blocking_methods": feature["blocking_methods"],
        "orthographic_status": status,
        "orthographic_rule": (
            "single_token_edit_equivalence"
            if status == "STRONG_ORTHOGRAPHIC_EVIDENCE"
            else "not_eligible_for_orthographic"
            if status == "NOT_ELIGIBLE_FOR_ORTHOGRAPHIC"
            else "no_orthographic_equivalence"
        ),
        "orthographic_evidence": evidence,
        "differing_token_a": None,
        "differing_token_b": None,
        "edit_operation": None,
        "context_signature": None,
        "context_variant_count": None,
        "token_support_a": None,
        "token_support_b": None,
    }


def _resolution_key_row(key: str) -> dict[str, object]:
    return {
        "resolution_key": key,
        "representative_name": key,
        "relaxed_key": key.replace(" ", ""),
        "source_row_frequency": 1,
        "representative_record_id": f"record-{key}",
        "representative_source_row_number": 2,
        "representative_route": "employer_resolution_candidate",
        "trailing_numeric_candidate": None,
        "possible_truncation": False,
        "token_count": len(key.split()),
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
    orthographic_path: Path,
    features_path: Path,
    keys_path: Path,
    eligibility_path: Path,
) -> tuple[int, int]:
    features = [
        _feature_row("ACME", "ACME PANAMA"),
        _feature_row("ADDRESS MARKET", "ACME CAPITAL"),
        _feature_row("EMPLOYER 1", "EMPLOYER 2", numeric_relation="conflict"),
        _feature_row("FARMACIA SHEKAINAH", "FARMACIA SHEKINAH"),
        _feature_row("GLOBAL TECH", "TECH GLOBAL"),
        _feature_row("PACHOS KITCHEN", "PANCHOS KITCHEN"),
    ]
    features.sort(key=lambda row: (str(row["key_a"]), str(row["key_b"])))
    by_pair = {(str(row["key_a"]), str(row["key_b"])): row for row in features}
    orthographic = [
        _orthographic_row(
            by_pair[_ordered("ACME", "ACME PANAMA")],
            "NEEDS_FURTHER_RESOLUTION",
            "core_token_count_mismatch",
        ),
        _orthographic_row(
            by_pair[_ordered("ADDRESS MARKET", "ACME CAPITAL")],
            "NOT_ELIGIBLE_FOR_ORTHOGRAPHIC",
            "not_both_employer_candidate",
        ),
        _orthographic_row(
            by_pair[_ordered("EMPLOYER 1", "EMPLOYER 2")],
            "NEEDS_FURTHER_RESOLUTION",
            "numeric_evidence_incompatible",
        ),
        _orthographic_row(
            by_pair[_ordered("FARMACIA SHEKAINAH", "FARMACIA SHEKINAH")],
            "STRONG_ORTHOGRAPHIC_EVIDENCE",
            "one_alphabetic_token_edit_distance_1_with_exact_context",
        ),
        _orthographic_row(
            by_pair[_ordered("GLOBAL TECH", "TECH GLOBAL")],
            "NEEDS_FURTHER_RESOLUTION",
            "multiple_core_token_differences",
        ),
    ]
    orthographic.sort(key=lambda row: (str(row["key_a"]), str(row["key_b"])))

    keys = sorted({str(row[column]) for row in features for column in ("key_a", "key_b")})
    eligibility_status = {key: "EMPLOYER_CANDIDATE" for key in keys}
    eligibility_status["ADDRESS MARKET"] = "ADDRESS"

    orthographic_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(orthographic, schema=ORTHOGRAPHIC_PAIR_DECISIONS_SCHEMA),
        orthographic_path,
    )
    pq.write_table(
        pa.Table.from_pylist(features, schema=CANDIDATE_FEATURES_SCHEMA),
        features_path,
    )
    pq.write_table(
        pa.Table.from_pylist(
            [_resolution_key_row(key) for key in keys],
            schema=RESOLUTION_KEYS_SCHEMA,
        ),
        keys_path,
    )
    pq.write_table(
        pa.Table.from_pylist(
            [_eligibility_row(key, eligibility_status[key]) for key in keys],
            schema=EMPLOYER_ELIGIBILITY_SCHEMA,
        ),
        eligibility_path,
    )
    return len(features), len(orthographic)


def test_pipeline_profiles_only_employer_compatible_residuals_deterministically(
    tmp_path: Path,
    workbook_factory: Callable[..., Path],
) -> None:
    workbook = workbook_factory(tmp_path / "input/source.xlsx", ["unused"])
    settings = settings_for(tmp_path, workbook)
    config = settings.residual_profile
    feature_count, orthographic_count = _write_inputs(
        config.orthographic_decisions_dataset,
        config.candidate_features_dataset,
        config.resolution_keys_dataset,
        config.employer_eligibility_dataset,
    )
    input_bytes = {
        path: path.read_bytes()
        for path in (
            config.orthographic_decisions_dataset,
            config.candidate_features_dataset,
            config.resolution_keys_dataset,
            config.employer_eligibility_dataset,
        )
    }

    result = profile_residual_relationships(tmp_path, settings)
    first_output = result.output_path.read_bytes()
    first_metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    table = pq.read_table(result.output_path)
    rows = table.to_pylist()

    assert table.schema == RESIDUAL_RELATIONSHIP_PROFILE_SCHEMA
    assert result.profiled_residual_rows == 3
    assert result.skipped_strong_orthographic_rows == 1
    assert result.skipped_ineligible_rows == 1
    assert [row["primary_family"] for row in rows] == [
        "SINGLE_TOKEN_ADDITION_REMOVAL",
        "NUMERIC_VARIATION",
        "EXACT_TOKEN_REORDER",
    ]
    assert all("FARMACIA SHEKAINAH" not in {row["key_a"], row["key_b"]} for row in rows)
    assert all("ADDRESS MARKET" not in {row["key_a"], row["key_b"]} for row in rows)
    assert all("status" not in row and "same_entity" not in row for row in rows)
    assert first_metrics["pair_rows_read"] == orthographic_count
    assert first_metrics["candidate_feature_rows_read"] == feature_count
    assert first_metrics["profiled_residual_rows"] == 3
    assert first_metrics["skipped_strong_orthographic_rows"] == 1
    assert first_metrics["skipped_ineligible_rows"] == 1
    assert first_metrics["primary_family_distribution"]["NUMERIC_VARIATION"]["count"] == 1
    assert first_metrics["primary_family_by_numeric_relation"]["NUMERIC_VARIATION"] == {
        "conflict": 1,
        "none": 0,
        "one_sided": 0,
        "same": 0,
    }
    assert first_metrics["reconciliation"]["status"] == "passed"
    assert all(path.read_bytes() == content for path, content in input_bytes.items())

    second = profile_residual_relationships(tmp_path, settings)
    second_metrics = json.loads(second.metrics_path.read_text(encoding="utf-8"))
    assert second.output_path.read_bytes() == first_output
    first_metrics.pop("execution_runtime_seconds")
    second_metrics.pop("execution_runtime_seconds")
    assert second_metrics == first_metrics


def test_pipeline_rejects_misaligned_feature_identity(
    tmp_path: Path,
    workbook_factory: Callable[..., Path],
) -> None:
    workbook = workbook_factory(tmp_path / "input/source.xlsx", ["unused"])
    settings = settings_for(tmp_path, workbook)
    config = settings.residual_profile
    _write_inputs(
        config.orthographic_decisions_dataset,
        config.candidate_features_dataset,
        config.resolution_keys_dataset,
        config.employer_eligibility_dataset,
    )
    table = pq.read_table(config.candidate_features_dataset)
    rows = table.to_pylist()
    rows[0]["name_a"] = "MISALIGNED NAME"
    pq.write_table(
        pa.Table.from_pylist(rows, schema=CANDIDATE_FEATURES_SCHEMA),
        config.candidate_features_dataset,
    )

    with pytest.raises(ValueError, match="identities are not aligned"):
        profile_residual_relationships(tmp_path, settings)


def test_profile_residuals_cli_honors_explicit_paths(
    tmp_path: Path,
    workbook_factory: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    workbook = workbook_factory(tmp_path / "input/source.xlsx", ["unused"])
    settings = settings_for(tmp_path, workbook)
    orthographic_path = tmp_path / "explicit-orthographic.parquet"
    features_path = tmp_path / "explicit-features.parquet"
    keys_path = tmp_path / "explicit-keys.parquet"
    eligibility_path = tmp_path / "explicit-eligibility.parquet"
    output_path = tmp_path / "explicit-profile.parquet"
    _write_inputs(orthographic_path, features_path, keys_path, eligibility_path)
    monkeypatch.setattr("credit_risk_er.cli.load_settings", lambda _: settings)

    exit_code = main(
        [
            "profile-residuals",
            "--config",
            str(tmp_path / "config/config.yaml"),
            "--orthographic",
            str(orthographic_path),
            "--features",
            str(features_path),
            "--keys",
            str(keys_path),
            "--eligibility",
            str(eligibility_path),
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert pq.ParquetFile(output_path).metadata.num_rows == 3
    printed = capsys.readouterr().out
    assert "Profiled residual pairs: 3" in printed
    assert "NUMERIC_VARIATION: 1" in printed
    assert "Skipped STRONG_ORTHOGRAPHIC_EVIDENCE: 1" in printed
