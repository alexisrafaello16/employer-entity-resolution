"""Tests for canonical employer-name quality gating."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from credit_risk_er.canonical_quality import (
    QUALITY_SCHEMA,
    assess_canonical_name,
    assess_corporate_master_quality,
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
        pa.field(
            "confidence_score",
            pa.int32(),
            nullable=False,
        ),
    ]
)


def _write_master(
    path: Path,
    rows: list[dict[str, object]],
) -> Path:
    pq.write_table(
        pa.Table.from_pylist(
            rows,
            schema=MASTER_SCHEMA,
        ),
        path,
    )
    return path


def test_public_name_preserves_official_unicode() -> None:
    decision = assess_canonical_name(
        canonical_name="Metro de Panamá, S.A.",
        resolution_status="PUBLIC_VALIDATED",
    )

    assert (
        decision.quality_status
        == "PUBLIC_VALIDATED"
    )
    assert (
        decision.quality_reason
        == "public_validated_name"
    )
    assert decision.promotion_eligible
    assert not decision.has_invalid_character


def test_legal_suffix_trailing_numeric_is_suspicious() -> None:
    decision = assess_canonical_name(
        canonical_name="CITIBANK PANAMA SA 1",
        resolution_status="AUTO_CANONICALIZED",
    )

    assert (
        decision.quality_status
        == "SUSPICIOUS"
    )
    assert (
        decision.quality_reason
        == "legal_suffix_trailing_numeric"
    )
    assert decision.has_trailing_numeric
    assert (
        decision.legal_suffix_trailing_numeric
    )
    assert not decision.promotion_eligible


def test_other_trailing_numeric_is_suspicious() -> None:
    decision = assess_canonical_name(
        canonical_name="NIKOS CAFE 1",
        resolution_status="AUTO_CANONICALIZED",
    )

    assert (
        decision.quality_status
        == "SUSPICIOUS"
    )
    assert (
        decision.quality_reason
        == "trailing_numeric"
    )
    assert decision.has_trailing_numeric
    assert not (
        decision.legal_suffix_trailing_numeric
    )
    assert not decision.promotion_eligible


def test_normalized_candidate_is_not_promotable() -> None:
    decision = assess_canonical_name(
        canonical_name="EMPRESA DUDOSA",
        resolution_status="NORMALIZED_CANDIDATE",
    )

    assert (
        decision.quality_status
        == "NOT_PROMOTABLE"
    )
    assert (
        decision.quality_reason
        == "normalized_candidate_not_promotable"
    )
    assert not decision.promotion_eligible


def test_clean_auto_canonical_name_is_acceptable() -> None:
    decision = assess_canonical_name(
        canonical_name="MAXICLEANERS SERVICES GROUP",
        resolution_status="AUTO_CANONICALIZED",
    )

    assert (
        decision.quality_status
        == "ACCEPTABLE"
    )
    assert (
        decision.quality_reason
        == "auto_canonical_name_acceptable"
    )
    assert decision.promotion_eligible


def test_replacement_character_blocks_promotion() -> None:
    decision = assess_canonical_name(
        canonical_name="EMPRESA � PANAMA SA",
        resolution_status="AUTO_CANONICALIZED",
    )

    assert (
        decision.quality_reason
        == "invalid_control_or_replacement_character"
    )
    assert decision.has_invalid_character
    assert not decision.promotion_eligible


def test_materialization_reconciles_entity_population(
    tmp_path: Path,
) -> None:
    master = _write_master(
        tmp_path / "master.parquet",
        [
            {
                "entity_id": "PUB-000001",
                "canonical_name": "Metro de Panamá, S.A.",
                "resolution_status": "PUBLIC_VALIDATED",
                "confidence_score": 100,
            },
            {
                "entity_id": "ENT-0123456789ABCDEF",
                "canonical_name": "ACME PANAMA SA",
                "resolution_status": "AUTO_CANONICALIZED",
                "confidence_score": 92,
            },
            {
                "entity_id": "ENT-FEDCBA9876543210",
                "canonical_name": "CITIBANK PANAMA SA 1",
                "resolution_status": "AUTO_CANONICALIZED",
                "confidence_score": 92,
            },
            {
                "entity_id": "ENT-AABBCCDDEEFF0011",
                "canonical_name": "EMPRESA DUDOSA",
                "resolution_status": "NORMALIZED_CANDIDATE",
                "confidence_score": 60,
            },
        ],
    )

    result = assess_corporate_master_quality(
        tmp_path,
        master_override=master,
    )

    assert result.entity_count == 4
    assert result.public_validated_count == 1
    assert result.acceptable_count == 1
    assert result.suspicious_count == 1
    assert result.not_promotable_count == 1

    assert result.promotion_eligible_count == 2
    assert result.promotion_blocked_count == 2

    assert result.trailing_numeric_count == 1
    assert (
        result.legal_suffix_trailing_numeric_count
        == 1
    )
    assert result.other_trailing_numeric_count == 0

    persisted = pq.ParquetFile(
        result.parquet_path
    )

    assert persisted.schema_arrow == QUALITY_SCHEMA
    assert persisted.metadata.num_rows == 4