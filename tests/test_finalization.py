"""Product tests for the conservative final employer export."""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from credit_risk_er.config import Settings
from credit_risk_er.finalization import (
    FINAL_SCHEMA,
    finalize_employers,
    infer_sector,
)
from tests.conftest import settings_for

DECISION_SCHEMA = pa.schema(
    [
        pa.field("key_a", pa.string(), nullable=False),
        pa.field("key_b", pa.string(), nullable=False),
        pa.field("decision_status", pa.string(), nullable=False),
        pa.field("decision_rule", pa.string(), nullable=False),
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
    """Write pair decisions with a stable schema even when rows are empty."""
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
    rows: list[tuple[str, str, str, str, str, str]]
    | None = None,
) -> Path:
    """Write a valid public-enrichment catalogue for the test scenario."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.writer(stream)

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
            writer.writerow(row)

    return path


def _key_row(
    key: str,
    *,
    frequency: int = 1,
    possible_truncation: bool = False,
) -> dict[str, object]:
    return {
        "resolution_key": key,
        "representative_name": key,
        "source_row_frequency": frequency,
        "possible_truncation": possible_truncation,
        "token_count": len(key.split()),
        "representative_route": (
            "employer_resolution_candidate"
        ),
    }


def _eligibility_row(
    key: str,
    status: str = "EMPLOYER_CANDIDATE",
) -> dict[str, object]:
    return {
        "resolution_key": key,
        "eligibility_status": status,
        "eligibility_rule": "prior_employer_route",
        "eligibility_evidence": "employer",
    }


def _preprocessed_row(
    record_id: str,
    source_row_number: int,
    value: str,
    *,
    route: str = "employer_resolution_candidate",
) -> dict[str, object]:
    return {
        "record_id": record_id,
        "source_row_number": source_row_number,
        "nombre_original": value,
        "nombre_normalizado": value,
        "route": route,
        "route_reason": "test_fixture",
    }


def test_sector_inference_uses_exact_tokens_and_configured_precedence() -> None:
    rules = {
        "Finanzas": (
            "BANCO",
            "BANK",
        ),
        "Educación": (
            "SCHOOL",
        ),
    }

    assert infer_sector(
        "BANCO SCHOOL",
        rules,
    ) == (
        "Finanzas",
        "keyword_taxonomy",
        "BANCO",
    )

    assert infer_sector(
        "BANCORP",
        rules,
    ) == (
        "No determinado",
        "not_determined",
        None,
    )


def test_finalize_preserves_rows_and_applies_conservative_outcomes(
    tmp_path: Path,
    workbook_factory: Callable[..., Path],
) -> None:
    workbook = workbook_factory(
        tmp_path / "input/source.xlsx",
        [
            "ACME SA",
            "ACME S A",
            "CALLE 1",
            "INDEPENDIENTE",
        ],
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
                "CALLE 1",
                route="address_candidate",
            ),
            _preprocessed_row(
                "r4",
                5,
                "INDEPENDIENTE",
                route="ambiguous_review_candidate",
            ),
        ],
    )

    keys = _write_parquet(
        tmp_path
        / "data/processed/resolution_keys.parquet",
        [
            _key_row("ACME S A"),
            _key_row("ACME SA"),
            {
                "resolution_key": "INDEPENDIENTE",
                "representative_name": "INDEPENDIENTE",
                "source_row_frequency": 1,
                "possible_truncation": False,
                "token_count": 1,
                "representative_route": (
                    "ambiguous_review_candidate"
                ),
            },
        ],
    )

    decisions = _write_decisions(
        tmp_path
        / "data/processed/pair_resolution_decisions.parquet",
        [
            {
                "key_a": "ACME S A",
                "key_b": "ACME SA",
                "decision_status": "AUTO_SAME",
                "decision_rule": (
                    "legal_suffix_format_equivalence"
                ),
            }
        ],
    )

    eligibility = _write_parquet(
        tmp_path
        / "data/processed/employer_eligibility.parquet",
        [
            _eligibility_row("ACME S A"),
            _eligibility_row("ACME SA"),
            {
                "resolution_key": "INDEPENDIENTE",
                "eligibility_status": "AMBIGUOUS",
                "eligibility_rule": (
                    "occupation_signal"
                ),
                "eligibility_evidence": (
                    "occupation"
                ),
            },
        ],
    )

    enrichment = _write_public_enrichment(
        tmp_path
        / "data/reference/public_employer_enrichment.csv",
        rows=[
            (
                "PUB-000001",
                "ACME SA",
                "Acme, S.A.",
                "Manufactura",
                "https://example.com/acme",
                "2026-08-16",
            )
        ],
    )

    result = finalize_employers(
        tmp_path,
        settings,
        preprocessed_override=preprocessed,
        keys_override=keys,
        decisions_override=decisions,
        eligibility_override=eligibility,
        enrichment_override=enrichment,
    )

    assert result.row_count == 4
    assert result.public_enriched_rows == 2

    parquet = pq.ParquetFile(
        result.parquet_output_path
    )

    assert parquet.schema_arrow == FINAL_SCHEMA

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

    assert (
        rows[0]["entity_id"]
        == rows[1]["entity_id"]
        == "PUB-000001"
    )

    assert (
        rows[0]["nombre_propuesto"]
        == rows[1]["nombre_propuesto"]
        == "Acme, S.A."
    )

    assert (
        rows[0]["sector_propuesto"]
        == rows[1]["sector_propuesto"]
        == "Manufactura"
    )

    assert (
        rows[0]["metodo_sector"]
        == rows[1]["metodo_sector"]
        == "public_source"
    )

    assert (
        rows[0]["score_confianza_resolucion"]
        == 100
    )

    assert (
        rows[1]["score_confianza_resolucion"]
        == 100
    )

    assert (
        rows[2]["resultado_final"]
        == "NO_IDENTIFICABLE_DIRECCION"
    )

    assert (
        rows[2]["score_confianza_resolucion"]
        == 100
    )

    assert (
        rows[3]["resultado_final"]
        == "NO_IDENTIFICABLE_FALTA_INFORMACION"
    )

    assert result.csv_output_path.is_file()
    assert result.top_keys_path.is_file()

    metrics = json.loads(
        result.metrics_path.read_text(
            encoding="utf-8"
        )
    )

    assert (
        metrics["reconciliation"][
            "row_count_matches"
        ]
        is True
    )

    assert (
        metrics["pair_decisions"][
            "accepted_employer_compatible_auto_same"
        ]
        == 1
    )


def test_explicit_missing_information_values_are_not_employers(
    tmp_path: Path,
    workbook_factory: Callable[..., Path],
) -> None:
    values = [
        "SIN INFORMACION",
        "NO TIENE",
        "NO APLICA",
        "NOAPLICA",
        "NA",
        "N A",
        "NONE",
        "NINGUNO",
        "NINGUNA",
        "DESCONOCIDO",
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
                f"r{index}",
                index + 2,
                value,
            )
            for index, value in enumerate(values)
        ],
    )

    keys = _write_parquet(
        tmp_path
        / "data/processed/resolution_keys.parquet",
        [
            _key_row(value)
            for value in values
        ],
    )

    eligibility = _write_parquet(
        tmp_path
        / "data/processed/employer_eligibility.parquet",
        [
            _eligibility_row(value)
            for value in values
        ],
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
        keys_override=keys,
        decisions_override=decisions,
        eligibility_override=eligibility,
        enrichment_override=enrichment,
    )

    rows = pq.read_table(
        result.parquet_output_path
    ).to_pylist()

    assert len(rows) == len(values)

    for row in rows:
        assert (
            row["resultado_final"]
            == "NO_IDENTIFICABLE_FALTA_INFORMACION"
        )

        assert (
            row["nombre_propuesto"]
            == "No identificable - Falta informacion"
        )

        assert row["entity_id"] is None
        assert row["canonical_resolution_key"] is None

        assert (
            row["metodo_resolucion"]
            == "explicit_missing_information"
        )

        assert (
            row["score_confianza_resolucion"]
            == 100
        )


def test_transitive_auto_same_does_not_merge_conflicting_legal_forms(
    tmp_path: Path,
    workbook_factory: Callable[..., Path],
) -> None:
    values = [
        "CHIQUITA PANAMA",
        "CHIQUITA PANAMA SA",
        "CHIQUITA PANAMA LLC",
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
                "CHIQUITA PANAMA",
            ),
            _preprocessed_row(
                "r2",
                3,
                "CHIQUITA PANAMA SA",
            ),
            _preprocessed_row(
                "r3",
                4,
                "CHIQUITA PANAMA LLC",
            ),
        ],
    )

    keys = _write_parquet(
        tmp_path
        / "data/processed/resolution_keys.parquet",
        [
            _key_row(
                "CHIQUITA PANAMA"
            ),
            _key_row(
                "CHIQUITA PANAMA SA"
            ),
            _key_row(
                "CHIQUITA PANAMA LLC"
            ),
        ],
    )

    eligibility = _write_parquet(
        tmp_path
        / "data/processed/employer_eligibility.parquet",
        [
            _eligibility_row(
                "CHIQUITA PANAMA"
            ),
            _eligibility_row(
                "CHIQUITA PANAMA SA"
            ),
            _eligibility_row(
                "CHIQUITA PANAMA LLC"
            ),
        ],
    )

    decisions = _write_decisions(
        tmp_path
        / "data/processed/pair_resolution_decisions.parquet",
        [
            {
                "key_a": "CHIQUITA PANAMA",
                "key_b": "CHIQUITA PANAMA SA",
                "decision_status": "AUTO_SAME",
                "decision_rule": (
                    "legal_suffix_addition_equivalence"
                ),
            },
            {
                "key_a": "CHIQUITA PANAMA",
                "key_b": "CHIQUITA PANAMA LLC",
                "decision_status": "AUTO_SAME",
                "decision_rule": (
                    "legal_suffix_addition_equivalence"
                ),
            },
        ],
    )

    enrichment = _write_public_enrichment(
        tmp_path
        / "data/reference/public_employer_enrichment.csv"
    )

    result = finalize_employers(
        tmp_path,
        settings,
        preprocessed_override=preprocessed,
        keys_override=keys,
        decisions_override=decisions,
        eligibility_override=eligibility,
        enrichment_override=enrichment,
    )

    rows = pq.read_table(
        result.parquet_output_path
    ).to_pylist()

    by_name = {
        row["nombre_normalizado"]: row
        for row in rows
    }

    sa_entity = by_name[
        "CHIQUITA PANAMA SA"
    ]["entity_id"]

    llc_entity = by_name[
        "CHIQUITA PANAMA LLC"
    ]["entity_id"]

    assert sa_entity is not None
    assert llc_entity is not None

    assert sa_entity != llc_entity

    metrics = json.loads(
        result.metrics_path.read_text(
            encoding="utf-8"
        )
    )

    assert (
        metrics["pair_decisions"][
            "blocked_legal_form_conflict_auto_same"
        ]
        == 1
    )


def test_same_legal_form_variants_can_still_merge(
    tmp_path: Path,
    workbook_factory: Callable[..., Path],
) -> None:
    values = [
        "ACME",
        "ACME SA",
        "ACME S A",
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
                "ACME",
            ),
            _preprocessed_row(
                "r2",
                3,
                "ACME SA",
            ),
            _preprocessed_row(
                "r3",
                4,
                "ACME S A",
            ),
        ],
    )

    keys = _write_parquet(
        tmp_path
        / "data/processed/resolution_keys.parquet",
        [
            _key_row("ACME"),
            _key_row("ACME SA"),
            _key_row("ACME S A"),
        ],
    )

    eligibility = _write_parquet(
        tmp_path
        / "data/processed/employer_eligibility.parquet",
        [
            _eligibility_row("ACME"),
            _eligibility_row("ACME SA"),
            _eligibility_row("ACME S A"),
        ],
    )

    decisions = _write_decisions(
        tmp_path
        / "data/processed/pair_resolution_decisions.parquet",
        [
            {
                "key_a": "ACME",
                "key_b": "ACME SA",
                "decision_status": "AUTO_SAME",
                "decision_rule": (
                    "legal_suffix_addition_equivalence"
                ),
            },
            {
                "key_a": "ACME",
                "key_b": "ACME S A",
                "decision_status": "AUTO_SAME",
                "decision_rule": (
                    "legal_suffix_format_equivalence"
                ),
            },
        ],
    )

    enrichment = _write_public_enrichment(
        tmp_path
        / "data/reference/public_employer_enrichment.csv"
    )

    result = finalize_employers(
        tmp_path,
        settings,
        preprocessed_override=preprocessed,
        keys_override=keys,
        decisions_override=decisions,
        eligibility_override=eligibility,
        enrichment_override=enrichment,
    )

    rows = pq.read_table(
        result.parquet_output_path
    ).to_pylist()

    entity_ids = {
        row["entity_id"]
        for row in rows
    }

    assert len(entity_ids) == 1