"""Synthetic persistence, reconciliation, determinism, metrics, and CLI tests."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from credit_risk_er.cli import main
from credit_risk_er.pipeline import (
    EMPLOYER_ELIGIBILITY_SCHEMA,
    PREPROCESSED_SCHEMA,
    RESOLUTION_KEYS_SCHEMA,
    classify_employer_eligibility,
)
from tests.conftest import settings_for


def _preprocessed_row(
    key: str,
    source_row: int,
    *,
    route: str = "employer_resolution_candidate",
    is_numeric_only: bool = False,
    address_strength: str = "none",
    occupation_strength: str = "none",
    activity: bool = False,
    corporate_suffix: bool = False,
    organization: bool = False,
    mixed_address_organization: bool = False,
    mixed_occupation_organization: bool = False,
) -> dict[str, object]:
    return {
        "record_id": f"record-{source_row}",
        "source_row_number": source_row,
        "nombre_original": key,
        "nombre_normalizado": key,
        "nombre_matching": key,
        "has_trailing_numeric_token": False,
        "trailing_numeric_candidate": None,
        "possible_truncation": False,
        "is_blank": False,
        "is_numeric_only": is_numeric_only,
        "has_address_signal": address_strength != "none",
        "address_signal_strength": address_strength,
        "has_occupation_signal": occupation_strength != "none",
        "occupation_signal_strength": occupation_strength,
        "has_corporate_suffix": corporate_suffix,
        "has_organization_like_tokens": organization,
        "mixed_address_organization_signal": mixed_address_organization,
        "mixed_occupation_organization_signal": mixed_occupation_organization,
        "has_activity_description_signal": activity,
        "token_count": len(key.split()),
        "normalized_length": len(key),
        "route": route,
        "route_reason": "synthetic_test_evidence",
    }


def _resolution_key_row(row: dict[str, object]) -> dict[str, object]:
    key = str(row["nombre_normalizado"])
    return {
        "resolution_key": key,
        "representative_name": key,
        "relaxed_key": key,
        "source_row_frequency": 1,
        "representative_record_id": row["record_id"],
        "representative_source_row_number": row["source_row_number"],
        "representative_route": row["route"],
        "trailing_numeric_candidate": None,
        "possible_truncation": False,
        "token_count": len(key.split()),
    }


def _write_inputs(preprocessed_path: Path, keys_path: Path) -> None:
    rows = [
        _preprocessed_row("IBM", 2),
        _preprocessed_row("548 MARKET ST 18590 CA SAN FRANCISCO", 3),
        _preprocessed_row("ACME SA", 4, corporate_suffix=True),
        _preprocessed_row(
            "JUBILADO",
            5,
            route="non_employer_status_candidate",
            occupation_strength="strong",
        ),
        _preprocessed_row(
            "PLAZA UNIVERSIDAD",
            6,
            address_strength="weak",
            organization=True,
            mixed_address_organization=True,
        ),
        _preprocessed_row(
            "12345",
            7,
            route="ambiguous_review_candidate",
            is_numeric_only=True,
        ),
        _preprocessed_row("IBM", 8),
    ]
    preprocessed_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=PREPROCESSED_SCHEMA), preprocessed_path)
    first_by_key = {str(row["nombre_normalizado"]): row for row in rows}
    key_rows = [_resolution_key_row(first_by_key[key]) for key in sorted(first_by_key)]
    pq.write_table(pa.Table.from_pylist(key_rows, schema=RESOLUTION_KEYS_SCHEMA), keys_path)


def test_eligibility_pipeline_writes_one_deterministic_row_per_resolution_key(
    tmp_path: Path,
    workbook_factory: Callable[..., Path],
) -> None:
    workbook = workbook_factory(tmp_path / "input/source.xlsx", ["unused"])
    settings = settings_for(tmp_path, workbook)
    preprocessed_path = settings.employer_eligibility.preprocessed_dataset
    keys_path = settings.employer_eligibility.resolution_keys_dataset
    _write_inputs(preprocessed_path, keys_path)
    preprocessed_bytes = preprocessed_path.read_bytes()
    keys_bytes = keys_path.read_bytes()

    result = classify_employer_eligibility(tmp_path, settings)
    first_output = result.output_path.read_bytes()
    output = pq.read_table(result.output_path)
    rows = {row["resolution_key"]: row for row in output.to_pylist()}
    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))

    assert output.schema == EMPLOYER_ELIGIBILITY_SCHEMA
    assert result.resolution_keys == result.rows_written == 6
    assert len(rows) == 6
    assert rows["ACME SA"]["eligibility_status"] == "EMPLOYER_CANDIDATE"
    assert rows["IBM"]["eligibility_status"] == "EMPLOYER_CANDIDATE"
    assert rows["548 MARKET ST 18590 CA SAN FRANCISCO"]["eligibility_status"] == "ADDRESS"
    assert rows["JUBILADO"]["eligibility_status"] == "NON_EMPLOYER_STATUS"
    assert rows["PLAZA UNIVERSIDAD"]["eligibility_status"] == "AMBIGUOUS"
    assert rows["12345"]["eligibility_status"] == "AMBIGUOUS"
    assert metrics["resolution_keys_read"] == metrics["rows_written"] == 6
    assert metrics["matched_preprocessing_rows"] == 7
    assert metrics["eligibility_status_counts"] == {
        "ADDRESS": 1,
        "AMBIGUOUS": 2,
        "EMPLOYER_CANDIDATE": 2,
        "NON_EMPLOYER_STATUS": 1,
    }
    assert metrics["known_structural_address_check"] == {
        "captured_by_general_structural_address_rule": True,
        "eligibility_rule": "structural_street_zip_region_address",
        "eligibility_status": "ADDRESS",
        "present": True,
        "resolution_key": "548 MARKET ST 18590 CA SAN FRANCISCO",
    }
    assert metrics["reconciliation"]["status"] == "passed"
    assert preprocessed_path.read_bytes() == preprocessed_bytes
    assert keys_path.read_bytes() == keys_bytes

    second_result = classify_employer_eligibility(tmp_path, settings)
    assert second_result.output_path.read_bytes() == first_output


def test_classify_eligibility_cli_honors_explicit_paths_and_prints_counts(
    tmp_path: Path,
    workbook_factory: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    workbook = workbook_factory(tmp_path / "input/source.xlsx", ["unused"])
    settings = settings_for(tmp_path, workbook)
    preprocessed_path = tmp_path / "explicit-preprocessed.parquet"
    keys_path = tmp_path / "explicit-keys.parquet"
    output_path = tmp_path / "explicit-eligibility.parquet"
    _write_inputs(preprocessed_path, keys_path)
    monkeypatch.setattr("credit_risk_er.cli.load_settings", lambda _: settings)

    exit_code = main(
        [
            "classify-eligibility",
            "--config",
            str(tmp_path / "config/config.yaml"),
            "--preprocessed",
            str(preprocessed_path),
            "--keys",
            str(keys_path),
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert pq.ParquetFile(output_path).metadata.num_rows == 6
    printed = capsys.readouterr().out
    assert "Resolution keys: 6" in printed
    assert "EMPLOYER_CANDIDATE: 2" in printed
    assert "ADDRESS: 1" in printed
    assert "NON_EMPLOYER_STATUS: 1" in printed
    assert "AMBIGUOUS: 2" in printed
