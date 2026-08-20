"""Tests for trusted corporate-reference promotion."""

from __future__ import annotations

import csv
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from credit_risk_er.reference_promotion import (
    promote_references,
)

MASTER_SCHEMA = pa.schema(
    [
        pa.field(
            "entity_id",
            pa.string(),
            nullable=False,
        ),
        pa.field(
            "canonical_name",
            pa.string(),
            nullable=False,
        ),
        pa.field(
            "resolution_status",
            pa.string(),
            nullable=False,
        ),
    ]
)

ALIAS_SCHEMA = pa.schema(
    [
        pa.field(
            "entity_id",
            pa.string(),
            nullable=False,
        ),
        pa.field(
            "resolution_key",
            pa.string(),
            nullable=False,
        ),
        pa.field(
            "representative_name",
            pa.string(),
            nullable=False,
        ),
    ]
)

QUALITY_SCHEMA = pa.schema(
    [
        pa.field(
            "entity_id",
            pa.string(),
            nullable=False,
        ),
        pa.field(
            "quality_status",
            pa.string(),
            nullable=False,
        ),
        pa.field(
            "promotion_eligible",
            pa.bool_(),
            nullable=False,
        ),
    ]
)


def _write_parquet(
    path: Path,
    schema: pa.Schema,
    rows: list[dict[str, object]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    table = pa.Table.from_pylist(
        rows,
        schema=schema,
    )

    pq.write_table(
        table,
        path,
    )


def _read_csv(
    path: Path,
) -> list[dict[str, str]]:
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as stream:
        return list(
            csv.DictReader(stream)
        )


def _prepare_inputs(
    root: Path,
) -> None:
    output = root / "output"

    _write_parquet(
        output / "employer_master_final.parquet",
        MASTER_SCHEMA,
        [
            {
                "entity_id": "PUB-000001",
                "canonical_name": "Metro de Panamá, S.A.",
                "resolution_status": "PUBLIC_VALIDATED",
            },
            {
                "entity_id": "ENT-AAAA",
                "canonical_name": "ACME SA",
                "resolution_status": "AUTO_CANONICALIZED",
            },
            {
                "entity_id": "ENT-BBBB",
                "canonical_name": "CITIBANK PANAMA SA 1",
                "resolution_status": "AUTO_CANONICALIZED",
            },
            {
                "entity_id": "ENT-CCCC",
                "canonical_name": "EMPRESA NORMALIZADA",
                "resolution_status": "NORMALIZED_CANDIDATE",
            },
        ],
    )

    _write_parquet(
        output / "employer_aliases_final.parquet",
        ALIAS_SCHEMA,
        [
            {
                "entity_id": "PUB-000001",
                "resolution_key": "METRO DE PANAMA SA",
                "representative_name": "METRO DE PANAMA SA",
            },
            {
                "entity_id": "ENT-AAAA",
                "resolution_key": "ACME SA",
                "representative_name": "ACME SA",
            },
            {
                "entity_id": "ENT-AAAA",
                "resolution_key": "ACME S A",
                "representative_name": "ACME S A",
            },
            {
                "entity_id": "ENT-BBBB",
                "resolution_key": "CITIBANK PANAMA SA 1",
                "representative_name": "CITIBANK PANAMA SA 1",
            },
            {
                "entity_id": "ENT-CCCC",
                "resolution_key": "EMPRESA NORMALIZADA",
                "representative_name": "EMPRESA NORMALIZADA",
            },
        ],
    )

    _write_parquet(
        output / "canonical_name_quality.parquet",
        QUALITY_SCHEMA,
        [
            {
                "entity_id": "PUB-000001",
                "quality_status": "PUBLIC_VALIDATED",
                "promotion_eligible": True,
            },
            {
                "entity_id": "ENT-AAAA",
                "quality_status": "ACCEPTABLE",
                "promotion_eligible": True,
            },
            {
                "entity_id": "ENT-BBBB",
                "quality_status": "SUSPICIOUS",
                "promotion_eligible": False,
            },
            {
                "entity_id": "ENT-CCCC",
                "quality_status": "NOT_PROMOTABLE",
                "promotion_eligible": False,
            },
        ],
    )


def test_promotes_only_quality_eligible_entities(
    tmp_path: Path,
) -> None:
    _prepare_inputs(
        tmp_path
    )

    result = promote_references(
        tmp_path
    )

    assert result.eligible_entity_count == 2
    assert result.promoted_entity_count == 2
    assert result.final_entity_count == 2

    master_rows = _read_csv(
        result.master_path
    )

    assert master_rows == [
        {
            "entity_id": "ENT-AAAA",
            "canonical_name": "ACME SA",
        },
        {
            "entity_id": "PUB-000001",
            "canonical_name": "Metro de Panamá, S.A.",
        },
    ]


def test_promotes_only_aliases_of_eligible_entities(
    tmp_path: Path,
) -> None:
    _prepare_inputs(
        tmp_path
    )

    result = promote_references(
        tmp_path
    )

    aliases = _read_csv(
        result.aliases_path
    )

    assert {
        row["alias_name"]
        for row in aliases
    } == {
        "ACME SA",
        "ACME S A",
        "METRO DE PANAMA SA",
    }

    assert {
        row["entity_id"]
        for row in aliases
    } == {
        "ENT-AAAA",
        "PUB-000001",
    }


def test_existing_references_are_preserved(
    tmp_path: Path,
) -> None:
    _prepare_inputs(
        tmp_path
    )

    reference_dir = (
        tmp_path
        / "data"
        / "reference"
    )

    reference_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        reference_dir
        / "employer_master.csv"
    ).write_text(
        "entity_id,canonical_name\n"
        "EMP-000001,EXISTING COMPANY\n",
        encoding="utf-8",
    )

    (
        reference_dir
        / "employer_aliases.csv"
    ).write_text(
        "entity_id,alias_name\n"
        "EMP-000001,EXISTING CO\n",
        encoding="utf-8",
    )

    result = promote_references(
        tmp_path
    )

    assert result.existing_entity_count == 1
    assert result.existing_alias_count == 1
    assert result.final_entity_count == 3
    assert result.final_alias_count == 4

    master_rows = _read_csv(
        result.master_path
    )

    assert any(
        row["entity_id"] == "EMP-000001"
        and row["canonical_name"]
        == "EXISTING COMPANY"
        for row in master_rows
    )


def test_repeated_promotion_is_idempotent(
    tmp_path: Path,
) -> None:
    _prepare_inputs(
        tmp_path
    )

    first = promote_references(
        tmp_path
    )

    second = promote_references(
        tmp_path
    )

    assert first.final_entity_count == 2
    assert first.final_alias_count == 3

    assert second.promoted_entity_count == 0
    assert second.promoted_alias_count == 0
    assert second.final_entity_count == 2
    assert second.final_alias_count == 3


def test_rejects_conflicting_existing_entity(
    tmp_path: Path,
) -> None:
    _prepare_inputs(
        tmp_path
    )

    reference_dir = (
        tmp_path
        / "data"
        / "reference"
    )

    reference_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        reference_dir
        / "employer_master.csv"
    ).write_text(
        "entity_id,canonical_name\n"
        "ENT-AAAA,DIFFERENT NAME\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="already exists",
    ):
        promote_references(
            tmp_path
        )


def test_rejects_eligible_entity_without_alias(
    tmp_path: Path,
) -> None:
    _prepare_inputs(
        tmp_path
    )

    aliases_path = (
        tmp_path
        / "output"
        / "employer_aliases_final.parquet"
    )

    _write_parquet(
        aliases_path,
        ALIAS_SCHEMA,
        [
            {
                "entity_id": "PUB-000001",
                "resolution_key": "METRO DE PANAMA SA",
                "representative_name": "METRO DE PANAMA SA",
            }
        ],
    )

    with pytest.raises(
        ValueError,
        match="without reusable aliases",
    ):
        promote_references(
            tmp_path
        )


def test_reference_csvs_are_excel_compatible_utf8(
    tmp_path: Path,
) -> None:
    _prepare_inputs(
        tmp_path
    )

    result = promote_references(
        tmp_path
    )

    assert result.master_path.read_bytes().startswith(
        b"\xef\xbb\xbf"
    )

    assert result.aliases_path.read_bytes().startswith(
        b"\xef\xbb\xbf"
    )