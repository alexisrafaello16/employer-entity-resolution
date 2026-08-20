"""End-to-end compact preprocessing tests."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import yaml

from credit_risk_er import cli
from credit_risk_er.pipeline import (
    PREPROCESSED_SCHEMA,
    RESOLVED_SCHEMA,
    preprocess,
    resolve_employers,
)
from tests.conftest import settings_for


def test_preprocess_writes_compact_lossless_dataset_without_sample(
    tmp_path: Path, workbook_factory: Callable[..., Path]
) -> None:
    values = ["  ACME  ", None, "'", "CALLE 10", "ESTUDIANTE", "EMPRESA S A"]
    workbook = workbook_factory(tmp_path / "source.xlsx", values)
    settings = settings_for(tmp_path, workbook)

    result = preprocess(tmp_path, settings)
    table = pq.read_table(result.output_path)
    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert table.schema == PREPROCESSED_SCHEMA
    assert table.column("nombre_original").to_pylist() == values
    assert table.column("source_row_number").to_pylist() == list(range(2, 8))
    assert table.column("route").to_pylist() == [
        "employer_resolution_candidate",
        "blank_candidate",
        "blank_candidate",
        "address_candidate",
        "non_employer_status_candidate",
        "employer_resolution_candidate",
    ]
    assert not (result.output_path.parent / "evaluation_sample.parquet").exists()
    assert metrics["row_counts"] == {
        "null_source_values": 1,
        "source": 6,
        "written": 6,
    }
    assert set(manifest) == {
        "configuration_sha256",
        "execution_timing",
        "output",
        "reconciliation",
        "row_counts",
        "ruleset_versions",
        "processing_fingerprint",
        "source",
        "timestamp_utc",
    }
    assert manifest["reconciliation"] == {
        "original_values_preserved": True,
        "record_ids_deterministic": True,
        "row_count_matches": True,
        "row_order_preserved": True,
    }


def test_preprocessed_parquet_and_processing_fingerprint_are_deterministic(
    tmp_path: Path, workbook_factory: Callable[..., Path]
) -> None:
    workbook = workbook_factory(tmp_path / "source.xlsx", ["ACME", "CALLE 10", None])
    settings = settings_for(tmp_path, workbook)
    first = preprocess(tmp_path, settings)
    first_bytes = first.output_path.read_bytes()
    second = preprocess(tmp_path, settings)
    assert second.output_path.read_bytes() == first_bytes
    assert second.processing_fingerprint == first.processing_fingerprint
    assert len(first.processing_fingerprint) == 64


def test_cli_preprocess_is_standalone(
    tmp_path: Path,
    workbook_factory: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    workbook = workbook_factory(tmp_path / "input/source.xlsx", ["ACME", "CALLE 10"])
    settings = settings_for(tmp_path, workbook)
    config_path = tmp_path / "config/config.yaml"
    config_path.parent.mkdir(parents=True)
    relative_settings = settings.model_copy(
        update={
            "source": settings.source.model_copy(update={"workbook": Path("input/source.xlsx")}),
            "processing": settings.processing.model_copy(
                update={
                    "output_dataset": Path("data/processed/preprocessed_employers.parquet"),
                    "metrics_file": Path("data/processed/preprocessing_metrics.json"),
                    "manifest_file": Path("data/processed/run_manifest.json"),
                }
            ),
        }
    )
    config_path.write_text(
        yaml.safe_dump(relative_settings.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    outside_directory = tmp_path / "outside"
    outside_directory.mkdir()
    monkeypatch.chdir(outside_directory)
    monkeypatch.setattr(cli, "DEFAULT_CONFIG_PATH", config_path)

    assert cli.main(["preprocess"]) == 0
    output = capsys.readouterr().out
    assert "Completed: 2 rows" in output
    assert (tmp_path / "data/processed/preprocessed_employers.parquet").is_file()


def test_cli_explicit_paths_use_shell_relative_semantics(
    tmp_path: Path,
    workbook_factory: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execution_directory = tmp_path / "execution"
    workbook = workbook_factory(execution_directory / "source.xlsx", ["ACME"])
    settings = settings_for(tmp_path, workbook)
    config_path = tmp_path / "config/custom.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        yaml.safe_dump(settings.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
    )
    monkeypatch.chdir(execution_directory)

    status = cli.main(
        [
            "preprocess",
            "--config",
            str(config_path),
            "--input",
            "source.xlsx",
            "--output",
            "custom/preprocessed.parquet",
        ]
    )

    assert status == 0
    assert (execution_directory / "custom/preprocessed.parquet").is_file()


def test_exact_resolution_preserves_preprocessing_and_allows_validated_override(
    tmp_path: Path,
    workbook_factory: Callable[..., Path],
    reference_factory: Callable[..., tuple[Path, Path]],
) -> None:
    values = ["ACME", "ACME ALT", "BANCO GENERAL 4", "STUDIO 507", None, "CALLE 10", "123"]
    workbook = workbook_factory(tmp_path / "source.xlsx", values)
    settings = settings_for(tmp_path, workbook)
    master, aliases = reference_factory(
        tmp_path / "data/reference",
        master_rows=[
            ("EMP-000001", "ACME"),
            ("EMP-000002", "CALLE 10"),
            ("EMP-000003", "NUMERIC EMPLOYER"),
        ],
        alias_rows=[
            ("EMP-000001", "ACME ALT"),
            ("EMP-000003", "123"),
        ],
    )
    preprocessed = preprocess(tmp_path, settings)

    result = resolve_employers(
        tmp_path,
        settings,
        master_override=master,
        aliases_override=aliases,
    )
    input_table = pq.read_table(preprocessed.output_path)
    output_table = pq.read_table(result.output_path)
    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))

    assert output_table.schema == RESOLVED_SCHEMA
    assert output_table.select(PREPROCESSED_SCHEMA.names).equals(input_table)
    assert output_table.column("nombre_original").to_pylist() == values
    assert output_table.column("resolution_status").to_pylist() == [
        "resolved",
        "resolved",
        "unresolved",
        "unresolved",
        "unresolved",
        "resolved",
        "resolved",
    ]
    assert output_table.column("resolution_method").to_pylist() == [
        "exact_canonical",
        "exact_alias",
        None,
        None,
        None,
        "exact_canonical",
        "exact_alias",
    ]
    routes = output_table.column("route").to_pylist()
    assert routes[5] == "address_candidate"
    assert routes[6] == "ambiguous_review_candidate"
    assert output_table.column("canonical_name")[5].as_py() == "CALLE 10"
    assert output_table.column("canonical_name")[6].as_py() == "NUMERIC EMPLOYER"
    assert metrics["resolution_counts"] == {
        "exact_alias": 2,
        "exact_canonical": 2,
        "resolved_exact": 4,
        "unresolved": 3,
    }
    assert metrics["reconciliation"]["preprocessing_route_preserved"] is True


def test_resolve_cli_smoke_with_synthetic_reference_data(
    tmp_path: Path,
    workbook_factory: Callable[..., Path],
    reference_factory: Callable[..., tuple[Path, Path]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    workbook = workbook_factory(tmp_path / "source.xlsx", ["ACME", "UNKNOWN"])
    settings = settings_for(tmp_path, workbook)
    reference_factory(
        tmp_path / "data/reference",
        master_rows=[("EMP-000001", "ACME")],
        alias_rows=[],
    )
    preprocess(tmp_path, settings)
    config_path = tmp_path / "config/config.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        yaml.safe_dump(settings.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
    )

    assert cli.main(["resolve", "--config", str(config_path)]) == 0
    output = capsys.readouterr().out
    assert "Resolved exact: 1" in output
    assert "Unresolved: 1" in output
    assert (tmp_path / "data/processed/resolved_employers.parquet").is_file()
