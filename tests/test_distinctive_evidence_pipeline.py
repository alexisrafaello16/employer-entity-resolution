"""Synthetic support-universe, streaming, reconciliation, determinism, and CLI tests."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from credit_risk_er.cli import main
from credit_risk_er.pipeline import (
    DISTINCTIVE_NAME_EVIDENCE_SCHEMA,
    EMPLOYER_ELIGIBILITY_SCHEMA,
    RESIDUAL_RELATIONSHIP_PROFILE_SCHEMA,
    RESOLUTION_KEYS_SCHEMA,
    compute_distinctive_evidence,
)
from tests.conftest import settings_for


def _profile_row(left: str, right: str, *, family: str) -> dict[str, object]:
    key_a, key_b = sorted((left, right))
    return {
        "key_a": key_a,
        "key_b": key_b,
        "name_a": key_a,
        "name_b": key_b,
        "primary_family": family,
        "family_evidence": "synthetic_structural_evidence",
        "core_token_count_a": len(key_a.split()),
        "core_token_count_b": len(key_b.split()),
        "shared_exact_token_count": 0,
        "differing_token_positions": None,
        "differing_token_count": 0,
        "added_removed_token": None,
        "added_removed_token_count": 0,
        "maximum_token_edit_distance": None,
        "total_token_edit_distance": None,
        "is_token_reorder": False,
        "is_ordered_subsequence": False,
        "is_token_multiset_containment": False,
        "has_initialism_pattern": False,
        "possible_truncation_a": False,
        "possible_truncation_b": False,
        "numeric_relation": "none",
        "prior_orthographic_evidence": "synthetic_orthographic_abstention",
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
    profile_path: Path,
    eligibility_path: Path,
    keys_path: Path,
) -> tuple[int, int]:
    profiles = [
        _profile_row(
            "COMMON ONE",
            "COMMON TWO",
            family="SINGLE_TOKEN_LARGER_EDIT",
        ),
        _profile_row("NONE BETA", "ZERO ALPHA", family="OTHER_RESIDUAL"),
        _profile_row(
            "RARITY BETA",
            "RARITY RARITY ALPHA",
            family="MIXED_STRUCTURAL_RELATIONSHIP",
        ),
    ]
    profiles.sort(key=lambda row: (str(row["key_a"]), str(row["key_b"])))
    profile_keys = {str(row[column]) for row in profiles for column in ("key_a", "key_b")}
    support_keys = {f"COMMON SUPPORT{index:02d}" for index in range(9)}
    all_keys = sorted(profile_keys | support_keys | {"RARITY ADDRESS"})
    status_by_key = {key: "EMPLOYER_CANDIDATE" for key in all_keys}
    status_by_key["RARITY ADDRESS"] = "ADDRESS"

    profile_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(profiles, schema=RESIDUAL_RELATIONSHIP_PROFILE_SCHEMA),
        profile_path,
    )
    pq.write_table(
        pa.Table.from_pylist(
            [_resolution_key_row(key) for key in all_keys],
            schema=RESOLUTION_KEYS_SCHEMA,
        ),
        keys_path,
    )
    pq.write_table(
        pa.Table.from_pylist(
            [_eligibility_row(key, status_by_key[key]) for key in all_keys],
            schema=EMPLOYER_ELIGIBILITY_SCHEMA,
        ),
        eligibility_path,
    )
    return len(profiles), len(all_keys) - 1


def test_pipeline_persists_every_profile_row_with_aligned_support_evidence(
    tmp_path: Path,
    workbook_factory: Callable[..., Path],
) -> None:
    workbook = workbook_factory(tmp_path / "input/source.xlsx", ["unused"])
    settings = settings_for(tmp_path, workbook)
    config = settings.distinctive_evidence
    profile_count, employer_count = _write_inputs(
        config.residual_profile_dataset,
        config.employer_eligibility_dataset,
        config.resolution_keys_dataset,
    )
    input_bytes = {
        path: path.read_bytes()
        for path in (
            config.residual_profile_dataset,
            config.employer_eligibility_dataset,
            config.resolution_keys_dataset,
        )
    }

    result = compute_distinctive_evidence(tmp_path, settings)
    first_output = result.output_path.read_bytes()
    first_metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    table = pq.read_table(result.output_path)
    rows = table.to_pylist()
    by_pair = {(row["key_a"], row["key_b"]): row for row in rows}

    assert table.schema == DISTINCTIVE_NAME_EVIDENCE_SCHEMA
    assert result.output_rows == profile_count == 3
    assert result.pairs_with_exact_overlap == 2
    assert result.pairs_with_distinctive_overlap == 1
    assert result.pairs_with_multiple_distinctive_tokens == 0
    common = by_pair[("COMMON ONE", "COMMON TWO")]
    assert common["shared_exact_tokens"] == "COMMON"
    assert common["shared_distinctive_token_count"] == 0
    assert common["minimum_shared_token_support"] == 11
    assert common["has_exact_overlap_without_distinctive_token"]
    rare = by_pair[("RARITY BETA", "RARITY RARITY ALPHA")]
    assert rare["shared_exact_tokens"] == "RARITY"
    assert rare["minimum_shared_token_support"] == 2
    assert rare["shared_distinctive_token_count"] == 1
    assert rare["primary_family"] == "MIXED_STRUCTURAL_RELATIONSHIP"
    assert rare["numeric_relation"] == "none"
    zero = by_pair[("NONE BETA", "ZERO ALPHA")]
    assert zero["shared_exact_token_count"] == 0
    assert not zero["has_exact_overlap_without_distinctive_token"]
    assert all("status" not in row and "score" not in row for row in rows)
    assert first_metrics["profiled_pairs_read"] == profile_count
    assert first_metrics["output_rows"] == profile_count
    assert first_metrics["employer_keys_used_for_support"] == employer_count
    assert first_metrics["pairs_with_zero_exact_overlap"] == 1
    assert first_metrics["pairs_with_exact_overlap_without_distinctive_token"] == 1
    assert first_metrics["reconciliation"]["status"] == "passed"
    assert all(path.read_bytes() == content for path, content in input_bytes.items())

    second = compute_distinctive_evidence(tmp_path, settings)
    second_metrics = json.loads(second.metrics_path.read_text(encoding="utf-8"))
    assert second.output_path.read_bytes() == first_output
    first_metrics.pop("execution_runtime_seconds")
    second_metrics.pop("execution_runtime_seconds")
    assert second_metrics == first_metrics


def test_compute_distinctive_evidence_cli_honors_explicit_paths(
    tmp_path: Path,
    workbook_factory: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    workbook = workbook_factory(tmp_path / "input/source.xlsx", ["unused"])
    settings = settings_for(tmp_path, workbook)
    profile_path = tmp_path / "explicit-profile.parquet"
    eligibility_path = tmp_path / "explicit-eligibility.parquet"
    keys_path = tmp_path / "explicit-keys.parquet"
    output_path = tmp_path / "explicit-evidence.parquet"
    _write_inputs(profile_path, eligibility_path, keys_path)
    monkeypatch.setattr("credit_risk_er.cli.load_settings", lambda _: settings)

    exit_code = main(
        [
            "compute-distinctive-evidence",
            "--config",
            str(tmp_path / "config/config.yaml"),
            "--profile",
            str(profile_path),
            "--eligibility",
            str(eligibility_path),
            "--keys",
            str(keys_path),
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert pq.ParquetFile(output_path).metadata.num_rows == 3
    printed = capsys.readouterr().out
    assert "Profiled pairs: 3" in printed
    assert "Pairs with exact overlap: 2" in printed
    assert "Pairs with distinctive overlap: 1" in printed
