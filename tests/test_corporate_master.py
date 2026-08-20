"""Product tests for the generated corporate employer master."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from credit_risk_er.corporate_master import (
    ALIAS_SCHEMA,
    MASTER_SCHEMA,
    build_corporate_master,
)

RESOLUTION_KEY_SCHEMA = pa.schema(
    [
        pa.field("resolution_key", pa.string(), nullable=False),
        pa.field("representative_name", pa.string(), nullable=False),
        pa.field("source_row_frequency", pa.int64(), nullable=False),
    ]
)


def _write_final_dataset(
    path: Path,
    rows: list[dict[str, object]],
) -> Path:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    table = pa.Table.from_pylist(rows)

    pq.write_table(
        table,
        path,
    )

    return path


def _write_resolution_keys(
    path: Path,
    rows: list[dict[str, object]],
) -> Path:
    """Write resolution keys with a stable schema, including empty fixtures."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    table = pa.Table.from_pylist(
        rows,
        schema=RESOLUTION_KEY_SCHEMA,
    )

    pq.write_table(
        table,
        path,
    )

    return path


def _final_row(
    *,
    record_id: str,
    source_row_number: int,
    original: str,
    resolution_key: str | None,
    entity_id: str | None,
    canonical_name: str,
    sector: str,
    result: str,
    confidence: str,
    confidence_score: int,
    method: str,
    canonical_resolution_key: str | None,
    component_key_count: int,
    component_source_row_count: int,
    public_entity_id: str | None = None,
    public_source_url: str | None = None,
    public_validation_date: str | None = None,
) -> dict[str, object]:
    return {
        "record_id": record_id,
        "source_row_number": source_row_number,
        "nombre_original": original,
        "nombre_normalizado": original,
        "resolution_key": resolution_key,
        "entity_id": entity_id,
        "nombre_propuesto": canonical_name,
        "sector_propuesto": sector,
        "resultado_final": result,
        "confianza_resolucion": confidence,
        "score_confianza_resolucion": confidence_score,
        "metodo_resolucion": method,
        "canonical_resolution_key": canonical_resolution_key,
        "component_key_count": component_key_count,
        "component_source_row_count": component_source_row_count,
        "public_entity_id": public_entity_id,
        "public_source_url": public_source_url,
        "public_validation_date": public_validation_date,
    }


def _key_row(
    key: str,
    *,
    representative_name: str | None = None,
    frequency: int = 1,
) -> dict[str, object]:
    return {
        "resolution_key": key,
        "representative_name": representative_name or key,
        "source_row_frequency": frequency,
    }


def test_build_corporate_master_materializes_entities_and_aliases(
    tmp_path: Path,
) -> None:
    final_path = _write_final_dataset(
        tmp_path / "output/employer_resolution_final.parquet",
        [
            _final_row(
                record_id="r1",
                source_row_number=2,
                original="ACME",
                resolution_key="ACME",
                entity_id="ENT-001",
                canonical_name="ACME SA",
                sector="Manufactura",
                result="EMPRESA_CANONICALIZADA_AUTO",
                confidence="ALTA",
                confidence_score=92,
                method="deterministic_auto_same_component",
                canonical_resolution_key="ACME SA",
                component_key_count=2,
                component_source_row_count=3,
            ),
            _final_row(
                record_id="r2",
                source_row_number=3,
                original="ACME SA",
                resolution_key="ACME SA",
                entity_id="ENT-001",
                canonical_name="ACME SA",
                sector="Manufactura",
                result="EMPRESA_CANONICALIZADA_AUTO",
                confidence="ALTA",
                confidence_score=92,
                method="deterministic_auto_same_component",
                canonical_resolution_key="ACME SA",
                component_key_count=2,
                component_source_row_count=3,
            ),
            _final_row(
                record_id="r3",
                source_row_number=4,
                original="ACME",
                resolution_key="ACME",
                entity_id="ENT-001",
                canonical_name="ACME SA",
                sector="Manufactura",
                result="EMPRESA_CANONICALIZADA_AUTO",
                confidence="ALTA",
                confidence_score=92,
                method="deterministic_auto_same_component",
                canonical_resolution_key="ACME SA",
                component_key_count=2,
                component_source_row_count=3,
            ),
            _final_row(
                record_id="r4",
                source_row_number=5,
                original="BANCO EJEMPLO",
                resolution_key="BANCO EJEMPLO",
                entity_id="PUB-000001",
                canonical_name="Banco Ejemplo, S.A.",
                sector="Servicios financieros",
                result="EMPRESA_VALIDADA_PUBLICAMENTE",
                confidence="ALTA",
                confidence_score=100,
                method="public_source_validation",
                canonical_resolution_key="BANCO EJEMPLO",
                component_key_count=1,
                component_source_row_count=1,
                public_entity_id="PUB-000001",
                public_source_url="https://example.com/bank",
                public_validation_date="2026-08-16",
            ),
            _final_row(
                record_id="r5",
                source_row_number=6,
                original="CALLE 1",
                resolution_key=None,
                entity_id=None,
                canonical_name="No identificable - Direcciones",
                sector="No aplica",
                result="NO_IDENTIFICABLE_DIRECCION",
                confidence="ALTA",
                confidence_score=100,
                method="address_classification",
                canonical_resolution_key=None,
                component_key_count=0,
                component_source_row_count=0,
            ),
        ],
    )

    keys_path = _write_resolution_keys(
        tmp_path / "data/processed/resolution_keys.parquet",
        [
            _key_row(
                "ACME",
                frequency=2,
            ),
            _key_row(
                "ACME SA",
                frequency=1,
            ),
            _key_row(
                "BANCO EJEMPLO",
                frequency=1,
            ),
        ],
    )

    result = build_corporate_master(
        tmp_path,
        final_dataset_override=final_path,
        resolution_keys_override=keys_path,
    )

    assert result.entity_count == 2
    assert result.alias_count == 3
    assert result.source_row_count == 4
    assert result.public_validated_entity_count == 1

    master_table = pq.read_table(
        result.master_parquet_path
    )

    alias_table = pq.read_table(
        result.aliases_parquet_path
    )

    assert master_table.schema == MASTER_SCHEMA
    assert alias_table.schema == ALIAS_SCHEMA

    master_rows = master_table.to_pylist()
    alias_rows = alias_table.to_pylist()

    assert [
        row["entity_id"]
        for row in master_rows
    ] == [
        "ENT-001",
        "PUB-000001",
    ]

    acme = master_rows[0]

    assert acme["canonical_name"] == "ACME SA"
    assert acme["sector"] == "Manufactura"

    assert (
        acme["resolution_method"]
        == "deterministic_auto_same_component"
    )

    assert acme["confidence_level"] == "ALTA"
    assert acme["confidence_score"] == 92
    assert acme["component_key_count"] == 2
    assert acme["source_row_count"] == 3

    assert (
        acme["public_validation_status"]
        == "NOT_PUBLICLY_VALIDATED"
    )

    assert acme["public_entity_id"] is None

    public = master_rows[1]

    assert (
        public["public_validation_status"]
        == "PUBLICLY_VALIDATED"
    )

    assert public["public_entity_id"] == "PUB-000001"
    assert public["confidence_score"] == 100

    assert (
        public["public_source_url"]
        == "https://example.com/bank"
    )

    acme_aliases = [
        row
        for row in alias_rows
        if row["entity_id"] == "ENT-001"
    ]

    assert {
        row["resolution_key"]
        for row in acme_aliases
    } == {
        "ACME",
        "ACME SA",
    }

    assert sum(
        row["source_row_frequency"]
        for row in acme_aliases
    ) == 3

    assert result.master_csv_path.is_file()
    assert result.aliases_csv_path.is_file()
    assert result.metrics_path.is_file()

    metrics = json.loads(
        result.metrics_path.read_text(
            encoding="utf-8"
        )
    )

    assert (
        metrics["reconciliation"][
            "entity_count_matches"
        ]
        is True
    )

    assert (
        metrics["reconciliation"][
            "alias_frequency_matches_source_rows"
        ]
        is True
    )


def test_master_excludes_non_identifiable_rows(
    tmp_path: Path,
) -> None:
    final_path = _write_final_dataset(
        tmp_path / "output/employer_resolution_final.parquet",
        [
            _final_row(
                record_id="r1",
                source_row_number=2,
                original="CALLE 1",
                resolution_key=None,
                entity_id=None,
                canonical_name="No identificable - Direcciones",
                sector="No aplica",
                result="NO_IDENTIFICABLE_DIRECCION",
                confidence="ALTA",
                confidence_score=100,
                method="address_classification",
                canonical_resolution_key=None,
                component_key_count=0,
                component_source_row_count=0,
            ),
            _final_row(
                record_id="r2",
                source_row_number=3,
                original="NO APLICA",
                resolution_key=None,
                entity_id=None,
                canonical_name="No identificable - Falta informacion",
                sector="No aplica",
                result="NO_IDENTIFICABLE_FALTA_INFORMACION",
                confidence="ALTA",
                confidence_score=100,
                method="explicit_missing_information",
                canonical_resolution_key=None,
                component_key_count=0,
                component_source_row_count=0,
            ),
        ],
    )

    keys_path = _write_resolution_keys(
        tmp_path / "data/processed/resolution_keys.parquet",
        [],
    )

    result = build_corporate_master(
        tmp_path,
        final_dataset_override=final_path,
        resolution_keys_override=keys_path,
    )

    assert result.entity_count == 0
    assert result.alias_count == 0
    assert result.source_row_count == 0

    assert (
        pq.read_table(
            result.master_parquet_path
        ).num_rows
        == 0
    )

    assert (
        pq.read_table(
            result.aliases_parquet_path
        ).num_rows
        == 0
    )


def test_one_entity_must_have_one_consistent_canonical_identity(
    tmp_path: Path,
) -> None:
    final_path = _write_final_dataset(
        tmp_path / "output/employer_resolution_final.parquet",
        [
            _final_row(
                record_id="r1",
                source_row_number=2,
                original="ACME",
                resolution_key="ACME",
                entity_id="ENT-001",
                canonical_name="ACME SA",
                sector="Manufactura",
                result="EMPRESA_CANONICALIZADA_AUTO",
                confidence="ALTA",
                confidence_score=92,
                method="deterministic_auto_same_component",
                canonical_resolution_key="ACME SA",
                component_key_count=2,
                component_source_row_count=2,
            ),
            _final_row(
                record_id="r2",
                source_row_number=3,
                original="ACME SA",
                resolution_key="ACME SA",
                entity_id="ENT-001",
                canonical_name="OTRO NOMBRE",
                sector="Manufactura",
                result="EMPRESA_CANONICALIZADA_AUTO",
                confidence="ALTA",
                confidence_score=92,
                method="deterministic_auto_same_component",
                canonical_resolution_key="ACME SA",
                component_key_count=2,
                component_source_row_count=2,
            ),
        ],
    )

    keys_path = _write_resolution_keys(
        tmp_path / "data/processed/resolution_keys.parquet",
        [
            _key_row("ACME"),
            _key_row("ACME SA"),
        ],
    )

    try:
        build_corporate_master(
            tmp_path,
            final_dataset_override=final_path,
            resolution_keys_override=keys_path,
        )
    except ValueError as error:
        assert (
            "inconsistent canonical_name"
            in str(error)
        )
    else:
        raise AssertionError(
            "Expected inconsistent entity metadata to be rejected"
        )


def test_alias_frequency_must_reconcile_with_entity_source_rows(
    tmp_path: Path,
) -> None:
    final_path = _write_final_dataset(
        tmp_path / "output/employer_resolution_final.parquet",
        [
            _final_row(
                record_id="r1",
                source_row_number=2,
                original="ACME",
                resolution_key="ACME",
                entity_id="ENT-001",
                canonical_name="ACME",
                sector="No determinado",
                result=(
                    "EMPRESA_NORMALIZADA_SIN_VALIDACION_PUBLICA"
                ),
                confidence="MEDIA",
                confidence_score=60,
                method="normalized_employer_key",
                canonical_resolution_key="ACME",
                component_key_count=1,
                component_source_row_count=1,
            )
        ],
    )

    keys_path = _write_resolution_keys(
        tmp_path / "data/processed/resolution_keys.parquet",
        [
            _key_row(
                "ACME",
                frequency=2,
            )
        ],
    )

    try:
        build_corporate_master(
            tmp_path,
            final_dataset_override=final_path,
            resolution_keys_override=keys_path,
        )
    except ValueError as error:
        assert (
            "source-row frequency"
            in str(error)
        )
    else:
        raise AssertionError(
            "Expected alias/source-row reconciliation failure"
        )


def test_exact_reference_alias_and_canonical_consolidate_to_reference_reuse(
    tmp_path: Path,
) -> None:
    final_path = _write_final_dataset(
        tmp_path / "output/employer_resolution_final.parquet",
        [
            _final_row(
                record_id="r1",
                source_row_number=2,
                original="ACME SA",
                resolution_key="ACME SA",
                entity_id="ENT-001",
                canonical_name="ACME SA",
                sector="Manufactura",
                result="EMPRESA_CANONICALIZADA_AUTO",
                confidence="ALTA",
                confidence_score=92,
                method="exact_reference_canonical",
                canonical_resolution_key="ACME SA",
                component_key_count=2,
                component_source_row_count=2,
            ),
            _final_row(
                record_id="r2",
                source_row_number=3,
                original="ACME S A",
                resolution_key="ACME S A",
                entity_id="ENT-001",
                canonical_name="ACME SA",
                sector="Manufactura",
                result="EMPRESA_CANONICALIZADA_AUTO",
                confidence="ALTA",
                confidence_score=92,
                method="exact_reference_alias",
                canonical_resolution_key="ACME SA",
                component_key_count=2,
                component_source_row_count=2,
            ),
        ],
    )

    keys_path = _write_resolution_keys(
        tmp_path / "data/processed/resolution_keys.parquet",
        [],
    )

    result = build_corporate_master(
        tmp_path,
        final_dataset_override=final_path,
        resolution_keys_override=keys_path,
    )

    master_rows = pq.read_table(
        result.master_parquet_path
    ).to_pylist()

    alias_rows = pq.read_table(
        result.aliases_parquet_path
    ).to_pylist()

    assert result.entity_count == 1
    assert result.alias_count == 2
    assert result.source_row_count == 2

    assert (
        master_rows[0]["resolution_method"]
        == "reference_exact_reuse"
    )

    assert {
        row["resolution_key"]
        for row in alias_rows
    } == {
        "ACME SA",
        "ACME S A",
    }

    assert sum(
        row["source_row_frequency"]
        for row in alias_rows
    ) == 2


def test_exact_reference_alias_alone_is_entity_level_reuse(
    tmp_path: Path,
) -> None:
    final_path = _write_final_dataset(
        tmp_path / "output/employer_resolution_final.parquet",
        [
            _final_row(
                record_id="r1",
                source_row_number=2,
                original="ACME S A",
                resolution_key="ACME S A",
                entity_id="ENT-001",
                canonical_name="ACME SA",
                sector="Manufactura",
                result="EMPRESA_CANONICALIZADA_AUTO",
                confidence="ALTA",
                confidence_score=92,
                method="exact_reference_alias",
                canonical_resolution_key="ACME S A",
                component_key_count=1,
                component_source_row_count=1,
            ),
        ],
    )

    keys_path = _write_resolution_keys(
        tmp_path / "data/processed/resolution_keys.parquet",
        [],
    )

    result = build_corporate_master(
        tmp_path,
        final_dataset_override=final_path,
        resolution_keys_override=keys_path,
    )

    master_rows = pq.read_table(
        result.master_parquet_path
    ).to_pylist()

    assert (
        master_rows[0]["resolution_method"]
        == "reference_exact_reuse"
    )


def test_exact_reference_canonical_alone_is_entity_level_reuse(
    tmp_path: Path,
) -> None:
    final_path = _write_final_dataset(
        tmp_path / "output/employer_resolution_final.parquet",
        [
            _final_row(
                record_id="r1",
                source_row_number=2,
                original="ACME SA",
                resolution_key="ACME SA",
                entity_id="ENT-001",
                canonical_name="ACME SA",
                sector="Manufactura",
                result="EMPRESA_CANONICALIZADA_AUTO",
                confidence="ALTA",
                confidence_score=92,
                method="exact_reference_canonical",
                canonical_resolution_key="ACME SA",
                component_key_count=1,
                component_source_row_count=1,
            ),
        ],
    )

    keys_path = _write_resolution_keys(
        tmp_path / "data/processed/resolution_keys.parquet",
        [],
    )

    result = build_corporate_master(
        tmp_path,
        final_dataset_override=final_path,
        resolution_keys_override=keys_path,
    )

    master_rows = pq.read_table(
        result.master_parquet_path
    ).to_pylist()

    assert (
        master_rows[0]["resolution_method"]
        == "reference_exact_reuse"
    )


def test_exact_reference_cannot_mix_with_unrelated_resolution_method(
    tmp_path: Path,
) -> None:
    final_path = _write_final_dataset(
        tmp_path / "output/employer_resolution_final.parquet",
        [
            _final_row(
                record_id="r1",
                source_row_number=2,
                original="ACME SA",
                resolution_key="ACME SA",
                entity_id="ENT-001",
                canonical_name="ACME SA",
                sector="Manufactura",
                result="EMPRESA_CANONICALIZADA_AUTO",
                confidence="ALTA",
                confidence_score=92,
                method="exact_reference_alias",
                canonical_resolution_key="ACME SA",
                component_key_count=2,
                component_source_row_count=2,
            ),
            _final_row(
                record_id="r2",
                source_row_number=3,
                original="ACME",
                resolution_key="ACME",
                entity_id="ENT-001",
                canonical_name="ACME SA",
                sector="Manufactura",
                result="EMPRESA_CANONICALIZADA_AUTO",
                confidence="ALTA",
                confidence_score=92,
                method="deterministic_auto_same_component",
                canonical_resolution_key="ACME SA",
                component_key_count=2,
                component_source_row_count=2,
            ),
        ],
    )

    keys_path = _write_resolution_keys(
        tmp_path / "data/processed/resolution_keys.parquet",
        [
            _key_row("ACME"),
        ],
    )

    try:
        build_corporate_master(
            tmp_path,
            final_dataset_override=final_path,
            resolution_keys_override=keys_path,
        )
    except ValueError as error:
        assert (
            "inconsistent resolution_method"
            in str(error)
        )
    else:
        raise AssertionError(
            "Expected incompatible resolution methods "
            "to be rejected"
        )


def test_exact_reference_aliases_can_be_rebuilt_when_missing_from_resolution_keys(
    tmp_path: Path,
) -> None:
    final_path = _write_final_dataset(
        tmp_path / "output/employer_resolution_final.parquet",
        [
            _final_row(
                record_id="r1",
                source_row_number=2,
                original="ACME S A",
                resolution_key="ACME S A",
                entity_id="ENT-001",
                canonical_name="ACME SA",
                sector="Manufactura",
                result="EMPRESA_CANONICALIZADA_AUTO",
                confidence="ALTA",
                confidence_score=92,
                method="exact_reference_alias",
                canonical_resolution_key="ACME SA",
                component_key_count=2,
                component_source_row_count=3,
            ),
            _final_row(
                record_id="r2",
                source_row_number=3,
                original="ACME S A",
                resolution_key="ACME S A",
                entity_id="ENT-001",
                canonical_name="ACME SA",
                sector="Manufactura",
                result="EMPRESA_CANONICALIZADA_AUTO",
                confidence="ALTA",
                confidence_score=92,
                method="exact_reference_alias",
                canonical_resolution_key="ACME SA",
                component_key_count=2,
                component_source_row_count=3,
            ),
            _final_row(
                record_id="r3",
                source_row_number=4,
                original="ACME SA",
                resolution_key="ACME SA",
                entity_id="ENT-001",
                canonical_name="ACME SA",
                sector="Manufactura",
                result="EMPRESA_CANONICALIZADA_AUTO",
                confidence="ALTA",
                confidence_score=92,
                method="exact_reference_canonical",
                canonical_resolution_key="ACME SA",
                component_key_count=2,
                component_source_row_count=3,
            ),
        ],
    )

    keys_path = _write_resolution_keys(
        tmp_path / "data/processed/resolution_keys.parquet",
        [],
    )

    result = build_corporate_master(
        tmp_path,
        final_dataset_override=final_path,
        resolution_keys_override=keys_path,
    )

    alias_rows = pq.read_table(
        result.aliases_parquet_path
    ).to_pylist()

    frequencies = {
        row["resolution_key"]:
        row["source_row_frequency"]
        for row in alias_rows
    }

    assert frequencies == {
        "ACME S A": 2,
        "ACME SA": 1,
    }

    assert sum(frequencies.values()) == 3

def test_reference_reuse_allows_unobserved_historical_canonical_key(
    tmp_path: Path,
) -> None:
    final_path = _write_final_dataset(
        tmp_path / "output/employer_resolution_final.parquet",
        [
            _final_row(
                record_id="r1",
                source_row_number=2,
                original="BANCO EJEMPLO SA",
                resolution_key="BANCO EJEMPLO SA",
                entity_id="PUB-000001",
                canonical_name="Banco Ejemplo, S.A.",
                sector="Servicios financieros",
                result="EMPRESA_VALIDADA_PUBLICAMENTE",
                confidence="ALTA",
                confidence_score=100,
                method="exact_reference_alias",
                canonical_resolution_key="BANCO EJEMPLO",
                component_key_count=1,
                component_source_row_count=1,
                public_entity_id="PUB-000001",
                public_source_url="https://example.com/bank",
                public_validation_date="2026-08-16",
            ),
        ],
    )

    keys_path = _write_resolution_keys(
        tmp_path / "data/processed/resolution_keys.parquet",
        [],
    )

    result = build_corporate_master(
        tmp_path,
        final_dataset_override=final_path,
        resolution_keys_override=keys_path,
    )

    master_rows = pq.read_table(
        result.master_parquet_path
    ).to_pylist()

    alias_rows = pq.read_table(
        result.aliases_parquet_path
    ).to_pylist()

    assert result.entity_count == 1
    assert result.alias_count == 1

    assert (
        master_rows[0]["resolution_method"]
        == "reference_exact_reuse"
    )

    assert (
        master_rows[0]["public_entity_id"]
        == "PUB-000001"
    )

    assert {
        row["resolution_key"]
        for row in alias_rows
    } == {
        "BANCO EJEMPLO SA",
    }