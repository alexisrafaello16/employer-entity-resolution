"""Synthetic streaming, alignment, reconciliation, and CLI tests."""

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
    DISTINCTIVE_NAME_EVIDENCE_SCHEMA,
    MULTI_EVIDENCE_ASSESSMENT_SCHEMA,
    ORTHOGRAPHIC_PAIR_DECISIONS_SCHEMA,
    RESIDUAL_RELATIONSHIP_PROFILE_SCHEMA,
    assess_pair_evidence,
)
from tests.conftest import settings_for


def _identity(left: str, right: str) -> dict[str, object]:
    key_a, key_b = sorted((left, right))
    return {"key_a": key_a, "key_b": key_b, "name_a": key_a, "name_b": key_b}


def _profile_row(
    left: str,
    right: str,
    *,
    family: str = "MIXED_STRUCTURAL_RELATIONSHIP",
    exact_count: int = 0,
    numeric_relation: str = "none",
    prior_evidence: str = "multiple_core_token_differences",
) -> dict[str, object]:
    return {
        **_identity(left, right),
        "primary_family": family,
        "family_evidence": "synthetic_profile",
        "core_token_count_a": 2,
        "core_token_count_b": 2,
        "shared_exact_token_count": exact_count,
        "differing_token_positions": "0",
        "differing_token_count": 1,
        "added_removed_token": None,
        "added_removed_token_count": 0,
        "maximum_token_edit_distance": 2,
        "total_token_edit_distance": 2,
        "is_token_reorder": family == "EXACT_TOKEN_REORDER",
        "is_ordered_subsequence": False,
        "is_token_multiset_containment": False,
        "has_initialism_pattern": False,
        "possible_truncation_a": False,
        "possible_truncation_b": False,
        "numeric_relation": numeric_relation,
        "prior_orthographic_evidence": prior_evidence,
    }


def _distinctive_row(
    profile: dict[str, object],
    *,
    distinctive_count: int = 0,
    shorter_exact_coverage: float = 0.0,
) -> dict[str, object]:
    exact_count = int(str(profile["shared_exact_token_count"]))
    has_distinctive = distinctive_count > 0
    return {
        **{column: profile[column] for column in ("key_a", "key_b", "name_a", "name_b")},
        "shared_exact_tokens": "TOKEN" if exact_count else "",
        "shared_distinctive_tokens": "TOKEN" if distinctive_count else "",
        "shared_exact_token_count": exact_count,
        "shared_distinctive_token_count": distinctive_count,
        "shared_generic_token_count": 0,
        "minimum_shared_token_support": 2 if exact_count else None,
        "maximum_shared_token_support": 2 if exact_count else None,
        "minimum_distinctive_token_support": 2 if distinctive_count else None,
        "maximum_distinctive_token_support": 2 if distinctive_count else None,
        "exact_coverage_a": shorter_exact_coverage,
        "exact_coverage_b": shorter_exact_coverage,
        "distinctive_coverage_a": 0.5 if has_distinctive else 0.0,
        "distinctive_coverage_b": 0.5 if has_distinctive else 0.0,
        "shorter_name_exact_coverage": shorter_exact_coverage,
        "longer_name_exact_coverage": shorter_exact_coverage,
        "shorter_name_distinctive_coverage": 0.5 if has_distinctive else 0.0,
        "longer_name_distinctive_coverage": 0.5 if has_distinctive else 0.0,
        "has_shared_exact_token": exact_count > 0,
        "has_shared_distinctive_token": has_distinctive,
        "has_multiple_shared_distinctive_tokens": distinctive_count >= 2,
        "has_exact_overlap_without_distinctive_token": exact_count > 0 and not has_distinctive,
        "primary_family": profile["primary_family"],
        "numeric_relation": profile["numeric_relation"],
    }


def _feature_row(
    identity: dict[str, object], *, numeric_relation: str = "none"
) -> dict[str, object]:
    return {
        **identity,
        "blocking_methods": ["synthetic_block"],
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


def _orthographic_row(
    identity: dict[str, object],
    *,
    status: str = "NEEDS_FURTHER_RESOLUTION",
    evidence: str = "multiple_core_token_differences",
) -> dict[str, object]:
    return {
        **identity,
        "blocking_methods": ["synthetic_block"],
        "orthographic_status": status,
        "orthographic_rule": "no_orthographic_equivalence",
        "orthographic_evidence": evidence,
        "differing_token_a": None,
        "differing_token_b": None,
        "edit_operation": None,
        "context_signature": None,
        "context_variant_count": None,
        "token_support_a": None,
        "token_support_b": None,
    }


def _write_table(path: Path, rows: list[dict[str, object]], schema: pa.Schema) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)


def _write_inputs(root: Path, settings: object) -> tuple[Path, ...]:
    config = settings.multi_evidence_assessment
    structural = _profile_row(
        "ALPHA BETA",
        "BETA ALPHA",
        family="EXACT_TOKEN_REORDER",
        exact_count=2,
    )
    zero = _profile_row("OMEGA ONE", "SIGMA TWO")
    profiles = sorted([structural, zero], key=lambda row: (row["key_a"], row["key_b"]))
    distinctive = [
        _distinctive_row(
            row,
            distinctive_count=2 if row is structural else 0,
            shorter_exact_coverage=1.0 if row is structural else 0.0,
        )
        for row in profiles
    ]
    broad_identities = [
        _identity("A BROAD", "B BROAD"),
        *[
            {column: row[column] for column in ("key_a", "key_b", "name_a", "name_b")}
            for row in profiles
        ],
        _identity("X BROAD", "Y BROAD"),
    ]
    broad_identities.sort(key=lambda row: (row["key_a"], row["key_b"]))
    features = [_feature_row(identity) for identity in broad_identities]
    orthographic = []
    evidence_by_pair = {
        (row["key_a"], row["key_b"]): row["prior_orthographic_evidence"] for row in profiles
    }
    for identity in broad_identities:
        evidence = evidence_by_pair.get(
            (identity["key_a"], identity["key_b"]), "synthetic_non_residual"
        )
        orthographic.append(_orthographic_row(identity, evidence=str(evidence)))

    paths = (
        config.residual_profile_dataset,
        config.distinctive_evidence_dataset,
        config.candidate_features_dataset,
        config.orthographic_decisions_dataset,
    )
    _write_table(paths[0], profiles, RESIDUAL_RELATIONSHIP_PROFILE_SCHEMA)
    _write_table(paths[1], distinctive, DISTINCTIVE_NAME_EVIDENCE_SCHEMA)
    _write_table(paths[2], features, CANDIDATE_FEATURES_SCHEMA)
    _write_table(paths[3], orthographic, ORTHOGRAPHIC_PAIR_DECISIONS_SCHEMA)
    return paths


def test_pipeline_stream_aligns_broader_inputs_and_reconciles_deterministically(
    tmp_path: Path,
    workbook_factory: Callable[..., Path],
) -> None:
    workbook = workbook_factory(tmp_path / "input/source.xlsx", ["unused"])
    settings = settings_for(tmp_path, workbook)
    input_paths = _write_inputs(tmp_path, settings)
    input_bytes = {path: path.read_bytes() for path in input_paths}

    result = assess_pair_evidence(tmp_path, settings)
    first_bytes = result.output_path.read_bytes()
    first_metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    table = pq.read_table(result.output_path)
    rows = table.to_pylist()

    assert table.schema == MULTI_EVIDENCE_ASSESSMENT_SCHEMA
    assert result.output_rows == 2
    assert [row["assessment_family"] for row in rows] == [
        "STRUCTURE_WITH_MULTIPLE_DISTINCTIVE_TOKENS",
        "ZERO_EXACT_OVERLAP",
    ]
    # Maximum raw lexical values do not activate an unapproved lexical family.
    assert all(row["char_ratio"] == 100.0 for row in rows)
    assert all(
        row["assessment_family"] != "HIGH_LEXICAL_WITHOUT_DISTINCTIVE_OVERLAP" for row in rows
    )
    assert first_metrics["input_residual_rows"] == 2
    assert first_metrics["distinctive_rows_aligned"] == 2
    assert first_metrics["candidate_feature_rows_read"] == 4
    assert first_metrics["orthographic_rows_read"] == 4
    assert first_metrics["output_rows"] == 2
    assert first_metrics["reconciliation"]["status"] == "passed"
    assert all(path.read_bytes() == content for path, content in input_bytes.items())

    second = assess_pair_evidence(tmp_path, settings)
    second_metrics = json.loads(second.metrics_path.read_text(encoding="utf-8"))
    assert second.output_path.read_bytes() == first_bytes
    first_metrics.pop("execution_runtime_seconds")
    second_metrics.pop("execution_runtime_seconds")
    assert second_metrics == first_metrics


@pytest.mark.parametrize("defect", ["identity_mismatch", "missing_feature", "wrong_status"])
def test_pipeline_fails_closed_on_alignment_or_population_defects(
    defect: str,
    tmp_path: Path,
    workbook_factory: Callable[..., Path],
) -> None:
    workbook = workbook_factory(tmp_path / "input/source.xlsx", ["unused"])
    settings = settings_for(tmp_path, workbook)
    _write_inputs(tmp_path, settings)
    config = settings.multi_evidence_assessment
    if defect == "identity_mismatch":
        table = pq.read_table(config.distinctive_evidence_dataset)
        rows = table.to_pylist()
        rows[0]["name_a"] = "MISALIGNED NAME"
        _write_table(config.distinctive_evidence_dataset, rows, DISTINCTIVE_NAME_EVIDENCE_SCHEMA)
    elif defect == "missing_feature":
        table = pq.read_table(config.candidate_features_dataset)
        rows = [
            row
            for row in table.to_pylist()
            if (row["key_a"], row["key_b"]) != ("ALPHA BETA", "BETA ALPHA")
        ]
        _write_table(config.candidate_features_dataset, rows, CANDIDATE_FEATURES_SCHEMA)
    else:
        table = pq.read_table(config.orthographic_decisions_dataset)
        rows = table.to_pylist()
        target = next(
            row for row in rows if (row["key_a"], row["key_b"]) == ("ALPHA BETA", "BETA ALPHA")
        )
        target["orthographic_status"] = "STRONG_ORTHOGRAPHIC_EVIDENCE"
        _write_table(
            config.orthographic_decisions_dataset,
            rows,
            ORTHOGRAPHIC_PAIR_DECISIONS_SCHEMA,
        )

    with pytest.raises(ValueError):
        assess_pair_evidence(tmp_path, settings)
    assert not config.output_dataset.exists()


def test_assess_evidence_cli_honors_all_explicit_paths(
    tmp_path: Path,
    workbook_factory: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    workbook = workbook_factory(tmp_path / "input/source.xlsx", ["unused"])
    settings = settings_for(tmp_path, workbook)
    paths = _write_inputs(tmp_path, settings)
    output_path = tmp_path / "explicit-assessment.parquet"
    monkeypatch.setattr("credit_risk_er.cli.load_settings", lambda _: settings)

    exit_code = main(
        [
            "assess-evidence",
            "--config",
            str(tmp_path / "config/config.yaml"),
            "--profile",
            str(paths[0]),
            "--distinctive",
            str(paths[1]),
            "--features",
            str(paths[2]),
            "--orthographic",
            str(paths[3]),
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert pq.ParquetFile(output_path).metadata.num_rows == 2
    assert "Residual pairs assessed: 2" in capsys.readouterr().out
