"""Tests for incremental exact-reference reuse in finalization."""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from credit_risk_er.config import Settings
from credit_risk_er.finalization import (
    FINAL_SCHEMA,
    finalize_employers,
)
from tests.conftest import settings_for

DECISION_SCHEMA = pa.schema(
    [
        pa.field(
            "key_a",
            pa.string(),
            nullable=False,
        ),
        pa.field(
            "key_b",
            pa.string(),
            nullable=False,
        ),
        pa.field(
            "decision_status",
            pa.string(),
            nullable=False,
        ),
        pa.field(
            "decision_rule",
            pa.string(),
            nullable=False,
        ),
    ]
)


def _write_parquet(
    path: Path,
    rows: list[dict[str, object]],
) -> Path:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pq.write_table(
        pa.Table.from_pylist(rows),
        path,
    )

    return path


def _write_decisions(
    path: Path,
    rows: list[dict[str, object]],
) -> Path:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    table = pa.Table.from_pylist(
        rows,
        schema=DECISION_SCHEMA,
    )

    pq.write_table(
        table,
        path,
    )

    return path


def _write_public_enrichment(
    path: Path,
    rows: list[
        tuple[
            str,
            str,
            str,
            str,
            str,
            str,
        ]
    ]
    | None = None,
) -> Path:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.writer(
            stream
        )

        writer.writerow(
            [
                "public_entity_id",
                "resolution_key",
                "canonical_name",
                "sector",
                "source_url",
                "validated_on",
            ]
        )

        for row in rows or []:
            writer.writerow(
                row
            )

    return path


def _preprocessed_row(
    record_id: str,
    source_row_number: int,
    value: str,
) -> dict[str, object]:
    return {
        "record_id": record_id,
        "source_row_number": source_row_number,
        "nombre_original": value,
        "nombre_normalizado": value,
        "route": "employer_resolution_candidate",
        "route_reason": "test_fixture",
    }


def _resolved_row(
    record_id: str,
    source_row_number: int,
    value: str,
    *,
    entity_id: str | None,
    canonical_name: str | None,
    resolution_status: str,
    resolution_method: str | None,
    resolution_reason: str,
) -> dict[str, object]:
    return {
        "record_id": record_id,
        "source_row_number": source_row_number,
        "nombre_original": value,
        "nombre_normalizado": value,
        "route": "employer_resolution_candidate",
        "route_reason": "test_fixture",
        "entity_id": entity_id,
        "canonical_name": canonical_name,
        "resolution_status": resolution_status,
        "resolution_method": resolution_method,
        "resolution_reason": resolution_reason,
    }


def _key_row(
    key: str,
    *,
    frequency: int = 1,
) -> dict[str, object]:
    return {
        "resolution_key": key,
        "representative_name": key,
        "source_row_frequency": frequency,
        "possible_truncation": False,
        "token_count": len(
            key.split()
        ),
        "representative_route": (
            "employer_resolution_candidate"
        ),
    }


def _eligibility_row(
    key: str,
) -> dict[str, object]:
    return {
        "resolution_key": key,
        "eligibility_status": (
            "EMPLOYER_CANDIDATE"
        ),
        "eligibility_rule": (
            "prior_employer_route"
        ),
        "eligibility_evidence": (
            "employer"
        ),
    }


def test_incremental_exact_references_survive_finalization(
    tmp_path: Path,
    workbook_factory: Callable[..., Path],
) -> None:
    values = [
        "ACME SA",
        "ACME S A",
        "METRO DE PANAMA SA",
        "NEW COMPANY",
    ]

    workbook = workbook_factory(
        tmp_path / "input/source.xlsx",
        values,
    )

    settings: Settings = settings_for(
        tmp_path,
        workbook,
    )

    preprocessed = _write_parquet(
        tmp_path
        / "data/processed/preprocessed_employers.parquet",
        [
            _preprocessed_row(
                "r1",
                2,
                "ACME SA",
            ),
            _preprocessed_row(
                "r2",
                3,
                "ACME S A",
            ),
            _preprocessed_row(
                "r3",
                4,
                "METRO DE PANAMA SA",
            ),
            _preprocessed_row(
                "r4",
                5,
                "NEW COMPANY",
            ),
        ],
    )

    resolved = _write_parquet(
        tmp_path
        / "data/processed/resolved_employers.parquet",
        [
            _resolved_row(
                "r1",
                2,
                "ACME SA",
                entity_id="ENT-0123456789ABCDEF",
                canonical_name="ACME SA",
                resolution_status="resolved",
                resolution_method="exact_canonical",
                resolution_reason=(
                    "exact_match_validated_canonical"
                ),
            ),
            _resolved_row(
                "r2",
                3,
                "ACME S A",
                entity_id="ENT-0123456789ABCDEF",
                canonical_name="ACME SA",
                resolution_status="resolved",
                resolution_method="exact_alias",
                resolution_reason=(
                    "exact_match_validated_alias"
                ),
            ),
            _resolved_row(
                "r3",
                4,
                "METRO DE PANAMA SA",
                entity_id="PUB-000001",
                canonical_name=(
                    "Metro de Panamá, S.A."
                ),
                resolution_status="resolved",
                resolution_method="exact_alias",
                resolution_reason=(
                    "exact_match_validated_alias"
                ),
            ),
            _resolved_row(
                "r4",
                5,
                "NEW COMPANY",
                entity_id=None,
                canonical_name=None,
                resolution_status="unresolved",
                resolution_method=None,
                resolution_reason=(
                    "no_validated_exact_match"
                ),
            ),
        ],
    )

    # Exact-resolved keys are deliberately absent.
    # Only the unresolved population continues through
    # candidate generation and eligibility.
    keys = _write_parquet(
        tmp_path
        / "data/processed/resolution_keys.parquet",
        [
            _key_row(
                "NEW COMPANY"
            )
        ],
    )

    eligibility = _write_parquet(
        tmp_path
        / "data/processed/employer_eligibility.parquet",
        [
            _eligibility_row(
                "NEW COMPANY"
            )
        ],
    )

    decisions = _write_decisions(
        tmp_path
        / "data/processed/pair_resolution_decisions.parquet",
        [],
    )

    # This public key is also absent from resolution_keys,
    # because it was already resolved by the persistent
    # reference layer.
    enrichment = _write_public_enrichment(
        tmp_path
        / "data/reference/public_employer_enrichment.csv",
        rows=[
            (
                "PUB-000001",
                "METRO DE PANAMA SA",
                "Metro de Panamá, S.A.",
                "Servicios públicos",
                "https://example.com/metro",
                "2026-08-16",
            )
        ],
    )

    result = finalize_employers(
        tmp_path,
        settings,
        preprocessed_override=preprocessed,
        resolved_override=resolved,
        keys_override=keys,
        decisions_override=decisions,
        eligibility_override=eligibility,
        enrichment_override=enrichment,
    )

    assert result.row_count == 4

    parquet = pq.ParquetFile(
        result.parquet_output_path
    )

    assert (
        parquet.schema_arrow
        == FINAL_SCHEMA
    )

    rows = pq.read_table(
        result.parquet_output_path
    ).to_pylist()

    assert [
        row["record_id"]
        for row in rows
    ] == [
        "r1",
        "r2",
        "r3",
        "r4",
    ]

    # The ENT identity must survive exactly.
    assert (
        rows[0]["entity_id"]
        == rows[1]["entity_id"]
        == "ENT-0123456789ABCDEF"
    )

    assert (
        rows[0]["nombre_propuesto"]
        == rows[1]["nombre_propuesto"]
        == "ACME SA"
    )

    assert (
        rows[0]["resultado_final"]
        == rows[1]["resultado_final"]
        == "EMPRESA_CANONICALIZADA_AUTO"
    )

    assert (
        rows[0][
            "score_confianza_resolucion"
        ]
        == 92
    )

    assert (
        rows[1][
            "score_confianza_resolucion"
        ]
        == 92
    )

    assert (
        rows[0]["metodo_resolucion"]
        == "exact_reference_canonical"
    )

    assert (
        rows[1]["metodo_resolucion"]
        == "exact_reference_alias"
    )

    assert (
        rows[0][
            "canonical_resolution_key"
        ]
        == "ACME SA"
    )

    assert (
        rows[1][
            "canonical_resolution_key"
        ]
        == "ACME SA"
    )

    assert (
        rows[0]["component_key_count"]
        == rows[1]["component_key_count"]
        == 2
    )

    assert (
        rows[0][
            "component_source_row_count"
        ]
        == rows[1][
            "component_source_row_count"
        ]
        == 2
    )

    # Publicly validated identities retain public metadata.
    assert (
        rows[2]["entity_id"]
        == "PUB-000001"
    )

    assert (
        rows[2]["nombre_propuesto"]
        == "Metro de Panamá, S.A."
    )

    assert (
        rows[2]["sector_propuesto"]
        == "Servicios públicos"
    )

    assert (
        rows[2]["resultado_final"]
        == "EMPRESA_VALIDADA_PUBLICAMENTE"
    )

    assert (
        rows[2][
            "score_confianza_resolucion"
        ]
        == 100
    )

    assert (
        rows[2]["metodo_resolucion"]
        == "exact_reference_alias"
    )

    assert (
        rows[2]["public_entity_id"]
        == "PUB-000001"
    )

    assert (
        rows[2]["public_source_url"]
        == "https://example.com/metro"
    )

    assert (
        rows[2][
            "public_validation_date"
        ]
        == "2026-08-16"
    )

    # The unresolved population continues through
    # the existing finalization path unchanged.
    assert (
        rows[3]["entity_id"]
        is not None
    )

    assert (
        rows[3]["nombre_propuesto"]
        == "NEW COMPANY"
    )

    assert (
        rows[3]["resultado_final"]
        == (
            "EMPRESA_NORMALIZADA_"
            "SIN_VALIDACION_PUBLICA"
        )
    )

    assert (
        rows[3][
            "score_confianza_resolucion"
        ]
        == 60
    )

    metrics = json.loads(
        result.metrics_path.read_text(
            encoding="utf-8"
        )
    )

    assert (
        metrics[
            "incremental_reference_reuse"
        ]["exact_reference_rows"]
        == 3
    )

    assert (
        metrics[
            "incremental_reference_reuse"
        ][
            "exact_reference_canonical_rows"
        ]
        == 1
    )

    assert (
        metrics[
            "incremental_reference_reuse"
        ][
            "exact_reference_alias_rows"
        ]
        == 2
    )

    assert (
        metrics[
            "incremental_reference_reuse"
        ][
            "exact_reference_entity_count"
        ]
        == 2
    )


def test_incremental_exact_entity_component_metadata_is_deterministic(
    tmp_path: Path,
    workbook_factory: Callable[..., Path],
) -> None:
    values = [
        "ACME S A",
        "ACME SA",
        "ACME S.A.",
    ]

    workbook = workbook_factory(
        tmp_path / "input/source.xlsx",
        values,
    )

    settings = settings_for(
        tmp_path,
        workbook,
    )

    preprocessed = _write_parquet(
        tmp_path
        / "data/processed/preprocessed_employers.parquet",
        [
            _preprocessed_row(
                "r1",
                2,
                "ACME S A",
            ),
            _preprocessed_row(
                "r2",
                3,
                "ACME SA",
            ),
            _preprocessed_row(
                "r3",
                4,
                "ACME S.A.",
            ),
        ],
    )

    resolved = _write_parquet(
        tmp_path
        / "data/processed/resolved_employers.parquet",
        [
            _resolved_row(
                "r1",
                2,
                "ACME S A",
                entity_id=(
                    "ENT-0123456789ABCDEF"
                ),
                canonical_name="ACME SA",
                resolution_status="resolved",
                resolution_method="exact_alias",
                resolution_reason=(
                    "exact_match_validated_alias"
                ),
            ),
            _resolved_row(
                "r2",
                3,
                "ACME SA",
                entity_id=(
                    "ENT-0123456789ABCDEF"
                ),
                canonical_name="ACME SA",
                resolution_status="resolved",
                resolution_method=(
                    "exact_canonical"
                ),
                resolution_reason=(
                    "exact_match_validated_canonical"
                ),
            ),
            _resolved_row(
                "r3",
                4,
                "ACME S.A.",
                entity_id=(
                    "ENT-0123456789ABCDEF"
                ),
                canonical_name="ACME SA",
                resolution_status="resolved",
                resolution_method="exact_alias",
                resolution_reason=(
                    "exact_match_validated_alias"
                ),
            ),
        ],
    )

    # No unresolved candidate population is required.
    keys = _write_parquet(
        tmp_path
        / "data/processed/resolution_keys.parquet",
        [],
    )

    eligibility = _write_parquet(
        tmp_path
        / "data/processed/employer_eligibility.parquet",
        [],
    )

    decisions = _write_decisions(
        tmp_path
        / "data/processed/pair_resolution_decisions.parquet",
        [],
    )

    enrichment = _write_public_enrichment(
        tmp_path
        / "data/reference/public_employer_enrichment.csv"
    )

    result = finalize_employers(
        tmp_path,
        settings,
        preprocessed_override=preprocessed,
        resolved_override=resolved,
        keys_override=keys,
        decisions_override=decisions,
        eligibility_override=eligibility,
        enrichment_override=enrichment,
    )

    rows = pq.read_table(
        result.parquet_output_path
    ).to_pylist()

    assert len(rows) == 3

    for row in rows:
        assert (
            row["entity_id"]
            == "ENT-0123456789ABCDEF"
        )

        assert (
            row["canonical_resolution_key"]
            == "ACME SA"
        )

        assert (
            row["component_key_count"]
            == 3
        )

        assert (
            row["component_source_row_count"]
            == 3
        )


def test_incremental_rejects_invalid_resolved_reference_contract(
    tmp_path: Path,
    workbook_factory: Callable[..., Path],
) -> None:
    workbook = workbook_factory(
        tmp_path / "input/source.xlsx",
        [
            "BROKEN COMPANY"
        ],
    )

    settings = settings_for(
        tmp_path,
        workbook,
    )

    preprocessed = _write_parquet(
        tmp_path
        / "data/processed/preprocessed_employers.parquet",
        [
            _preprocessed_row(
                "r1",
                2,
                "BROKEN COMPANY",
            )
        ],
    )

    resolved = _write_parquet(
        tmp_path
        / "data/processed/resolved_employers.parquet",
        [
            _resolved_row(
                "r1",
                2,
                "BROKEN COMPANY",
                entity_id=None,
                canonical_name=(
                    "BROKEN COMPANY"
                ),
                resolution_status="resolved",
                resolution_method=(
                    "exact_canonical"
                ),
                resolution_reason=(
                    "exact_match_validated_canonical"
                ),
            )
        ],
    )

    keys = _write_parquet(
        tmp_path
        / "data/processed/resolution_keys.parquet",
        [],
    )

    eligibility = _write_parquet(
        tmp_path
        / "data/processed/employer_eligibility.parquet",
        [],
    )

    decisions = _write_decisions(
        tmp_path
        / "data/processed/pair_resolution_decisions.parquet",
        [],
    )

    enrichment = _write_public_enrichment(
        tmp_path
        / "data/reference/public_employer_enrichment.csv"
    )

    with pytest.raises(
        ValueError,
        match=(
            r"resolved exact reference.*entity_id"
        ),
    ):
        finalize_employers(
            tmp_path,
            settings,
            preprocessed_override=preprocessed,
            resolved_override=resolved,
            keys_override=keys,
            decisions_override=decisions,
            eligibility_override=eligibility,
            enrichment_override=enrichment,
        )