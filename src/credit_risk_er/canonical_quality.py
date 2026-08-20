"""Assess canonical employer-name quality before persistent promotion."""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, cast

import pyarrow as pa
import pyarrow.parquet as pq

type CanonicalQualityStatus = Literal[
    "PUBLIC_VALIDATED",
    "ACCEPTABLE",
    "SUSPICIOUS",
    "NOT_PROMOTABLE",
]

type CanonicalQualityReason = Literal[
    "public_validated_name",
    "auto_canonical_name_acceptable",
    "legal_suffix_trailing_numeric",
    "trailing_numeric",
    "invalid_control_or_replacement_character",
    "normalized_candidate_not_promotable",
]

PUBLIC_STATUS: Final = "PUBLIC_VALIDATED"
AUTO_STATUS: Final = "AUTO_CANONICALIZED"
NORMALIZED_STATUS: Final = "NORMALIZED_CANDIDATE"

TRAILING_NUMERIC_PATTERN: Final = re.compile(r"\s+\d+$")

LEGAL_SUFFIX_TRAILING_NUMERIC_PATTERN: Final = re.compile(
    r"(?:"
    r"\bSA|"
    r"\bS\s+A|"
    r"\bCA|"
    r"\bC\s+A|"
    r"\bSPA|"
    r"\bINC|"
    r"\bLLC|"
    r"\bLTD|"
    r"\bCORP|"
    r"\bSRL|"
    r"\bS\s+R\s+L"
    r")\s+\d+$",
    re.IGNORECASE,
)

MASTER_REQUIRED_COLUMNS: Final = (
    "entity_id",
    "canonical_name",
    "resolution_status",
    "confidence_score",
)

QUALITY_SCHEMA = pa.schema(
    [
        pa.field("entity_id", pa.string(), nullable=False),
        pa.field("canonical_name", pa.string(), nullable=False),
        pa.field("resolution_status", pa.string(), nullable=False),
        pa.field("confidence_score", pa.int32(), nullable=False),
        pa.field("quality_status", pa.string(), nullable=False),
        pa.field("quality_reason", pa.string(), nullable=False),
        pa.field("has_trailing_numeric", pa.bool_(), nullable=False),
        pa.field("legal_suffix_trailing_numeric", pa.bool_(), nullable=False),
        pa.field("has_invalid_character", pa.bool_(), nullable=False),
        pa.field("promotion_eligible", pa.bool_(), nullable=False),
    ]
)

QUALITY_COLUMNS: Final = tuple(
    field.name for field in QUALITY_SCHEMA
)


@dataclass(frozen=True, slots=True)
class CanonicalQualityDecision:
    """One deterministic quality decision for a canonical employer name."""

    quality_status: CanonicalQualityStatus
    quality_reason: CanonicalQualityReason
    has_trailing_numeric: bool
    legal_suffix_trailing_numeric: bool
    has_invalid_character: bool
    promotion_eligible: bool


@dataclass(frozen=True, slots=True)
class CanonicalQualityResult:
    """Materialized quality audit and reconciliation counts."""

    entity_count: int
    public_validated_count: int
    acceptable_count: int
    suspicious_count: int
    not_promotable_count: int
    promotion_eligible_count: int
    promotion_blocked_count: int
    trailing_numeric_count: int
    legal_suffix_trailing_numeric_count: int
    other_trailing_numeric_count: int
    invalid_character_count: int
    parquet_path: Path
    csv_path: Path
    metrics_path: Path


def _required_string(
    row: dict[str, object],
    field: str,
    *,
    source: str,
) -> str:
    value = row.get(field)

    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{source} contains blank or invalid {field!r}"
        )

    return value.strip()


def _required_int(
    row: dict[str, object],
    field: str,
    *,
    source: str,
) -> int:
    value = row.get(field)

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"{source} contains invalid integer {field!r}"
        )

    return value


def _has_invalid_character(value: str) -> bool:
    """
    Detect actual text corruption/control characters.

    Accents, punctuation and other legitimate Unicode characters are allowed.
    """
    for character in value:
        if character == "\ufffd":
            return True

        category = unicodedata.category(character)

        if category in {"Cc", "Cs"}:
            return True

    return False


def assess_canonical_name(
    *,
    canonical_name: str,
    resolution_status: str,
) -> CanonicalQualityDecision:
    """Apply deterministic canonical-name quality and promotion rules."""

    invalid_character = _has_invalid_character(
        canonical_name
    )

    trailing_numeric = (
        TRAILING_NUMERIC_PATTERN.search(
            canonical_name
        )
        is not None
    )

    legal_suffix_numeric = (
        LEGAL_SUFFIX_TRAILING_NUMERIC_PATTERN.search(
            canonical_name
        )
        is not None
    )

    if invalid_character:
        return CanonicalQualityDecision(
            quality_status="SUSPICIOUS",
            quality_reason=(
                "invalid_control_or_replacement_character"
            ),
            has_trailing_numeric=trailing_numeric,
            legal_suffix_trailing_numeric=legal_suffix_numeric,
            has_invalid_character=True,
            promotion_eligible=False,
        )

    if resolution_status == PUBLIC_STATUS:
        return CanonicalQualityDecision(
            quality_status="PUBLIC_VALIDATED",
            quality_reason="public_validated_name",
            has_trailing_numeric=trailing_numeric,
            legal_suffix_trailing_numeric=legal_suffix_numeric,
            has_invalid_character=False,
            promotion_eligible=True,
        )

    if resolution_status == NORMALIZED_STATUS:
        return CanonicalQualityDecision(
            quality_status="NOT_PROMOTABLE",
            quality_reason=(
                "normalized_candidate_not_promotable"
            ),
            has_trailing_numeric=trailing_numeric,
            legal_suffix_trailing_numeric=legal_suffix_numeric,
            has_invalid_character=False,
            promotion_eligible=False,
        )

    if resolution_status != AUTO_STATUS:
        raise ValueError(
            "Unsupported corporate-master resolution_status: "
            f"{resolution_status!r}"
        )

    if legal_suffix_numeric:
        return CanonicalQualityDecision(
            quality_status="SUSPICIOUS",
            quality_reason=(
                "legal_suffix_trailing_numeric"
            ),
            has_trailing_numeric=True,
            legal_suffix_trailing_numeric=True,
            has_invalid_character=False,
            promotion_eligible=False,
        )

    if trailing_numeric:
        return CanonicalQualityDecision(
            quality_status="SUSPICIOUS",
            quality_reason="trailing_numeric",
            has_trailing_numeric=True,
            legal_suffix_trailing_numeric=False,
            has_invalid_character=False,
            promotion_eligible=False,
        )

    return CanonicalQualityDecision(
        quality_status="ACCEPTABLE",
        quality_reason="auto_canonical_name_acceptable",
        has_trailing_numeric=False,
        legal_suffix_trailing_numeric=False,
        has_invalid_character=False,
        promotion_eligible=True,
    )


def _require_master(
    path: Path,
) -> pq.ParquetFile:
    if not path.is_file():
        raise FileNotFoundError(
            f"Corporate master does not exist: {path}"
        )

    parquet = pq.ParquetFile(path)
    observed = set(
        parquet.schema_arrow.names
    )
    missing = sorted(
        set(MASTER_REQUIRED_COLUMNS) - observed
    )

    if missing:
        raise ValueError(
            f"{path.name} is missing required columns: "
            f"{missing}"
        )

    return parquet


def _write_csv(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(QUALITY_COLUMNS),
        )
        writer.writeheader()
        writer.writerows(rows)


def assess_corporate_master_quality(
    project_root: Path,
    *,
    master_override: Path | None = None,
) -> CanonicalQualityResult:
    """Assess every materialized corporate entity before reference promotion."""

    master_path = (
        master_override
        if master_override is not None
        else project_root
        / "output"
        / "employer_master_final.parquet"
    )

    parquet = _require_master(
        master_path
    )

    rows: list[dict[str, object]] = []

    status_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()

    promotion_eligible_count = 0
    trailing_numeric_count = 0
    legal_suffix_numeric_count = 0
    invalid_character_count = 0

    seen_entities: set[str] = set()

    for batch in parquet.iter_batches(
        columns=list(MASTER_REQUIRED_COLUMNS)
    ):
        for raw_row in batch.to_pylist():
            row = cast(
                dict[str, object],
                raw_row,
            )

            entity_id = _required_string(
                row,
                "entity_id",
                source=master_path.name,
            )
            canonical_name = _required_string(
                row,
                "canonical_name",
                source=master_path.name,
            )
            resolution_status = _required_string(
                row,
                "resolution_status",
                source=master_path.name,
            )
            confidence_score = _required_int(
                row,
                "confidence_score",
                source=master_path.name,
            )

            if entity_id in seen_entities:
                raise ValueError(
                    f"Duplicate entity_id in corporate master: "
                    f"{entity_id!r}"
                )

            seen_entities.add(
                entity_id
            )

            decision = assess_canonical_name(
                canonical_name=canonical_name,
                resolution_status=resolution_status,
            )

            status_counts[
                decision.quality_status
            ] += 1

            reason_counts[
                decision.quality_reason
            ] += 1

            promotion_eligible_count += int(
                decision.promotion_eligible
            )
            trailing_numeric_count += int(
                decision.has_trailing_numeric
            )
            legal_suffix_numeric_count += int(
                decision.legal_suffix_trailing_numeric
            )
            invalid_character_count += int(
                decision.has_invalid_character
            )

            rows.append(
                {
                    "entity_id": entity_id,
                    "canonical_name": canonical_name,
                    "resolution_status": resolution_status,
                    "confidence_score": confidence_score,
                    "quality_status": (
                        decision.quality_status
                    ),
                    "quality_reason": (
                        decision.quality_reason
                    ),
                    "has_trailing_numeric": (
                        decision.has_trailing_numeric
                    ),
                    "legal_suffix_trailing_numeric": (
                        decision.legal_suffix_trailing_numeric
                    ),
                    "has_invalid_character": (
                        decision.has_invalid_character
                    ),
                    "promotion_eligible": (
                        decision.promotion_eligible
                    ),
                }
            )

    if len(rows) != parquet.metadata.num_rows:
        raise RuntimeError(
            "Canonical quality row-count reconciliation failed"
        )

    output_dir = (
        project_root / "output"
    )

    parquet_path = (
        output_dir
        / "canonical_name_quality.parquet"
    )
    csv_path = (
        output_dir
        / "canonical_name_quality.csv"
    )
    metrics_path = (
        output_dir
        / "canonical_name_quality_metrics.json"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    table = pa.Table.from_pylist(
        rows,
        schema=QUALITY_SCHEMA,
    )

    pq.write_table(
        table,
        parquet_path,
    )

    _write_csv(
        csv_path,
        rows,
    )

    persisted = pq.ParquetFile(
        parquet_path
    )

    if persisted.schema_arrow != QUALITY_SCHEMA:
        raise RuntimeError(
            "Canonical quality output schema failed reconciliation"
        )

    if persisted.metadata.num_rows != len(rows):
        raise RuntimeError(
            "Persisted canonical quality row count failed reconciliation"
        )

    promotion_blocked_count = (
        len(rows)
        - promotion_eligible_count
    )

    other_trailing_numeric_count = (
        trailing_numeric_count
        - legal_suffix_numeric_count
    )

    metrics = {
        "entities": {
            "entity_count": len(rows),
            "quality_status_counts": {
                key: status_counts[key]
                for key in sorted(status_counts)
            },
            "quality_reason_counts": {
                key: reason_counts[key]
                for key in sorted(reason_counts)
            },
        },
        "canonical_name_quality": {
            "trailing_numeric_entities": (
                trailing_numeric_count
            ),
            "legal_suffix_trailing_numeric_entities": (
                legal_suffix_numeric_count
            ),
            "other_trailing_numeric_entities": (
                other_trailing_numeric_count
            ),
            "invalid_character_entities": (
                invalid_character_count
            ),
        },
        "promotion": {
            "eligible_entities": (
                promotion_eligible_count
            ),
            "blocked_entities": (
                promotion_blocked_count
            ),
        },
        "reconciliation": {
            "input_entity_count": (
                parquet.metadata.num_rows
            ),
            "output_entity_count": (
                len(rows)
            ),
            "row_count_matches": True,
            "entity_ids_unique": (
                len(seen_entities)
                == len(rows)
            ),
        },
    }

    metrics_path.write_text(
        json.dumps(
            metrics,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return CanonicalQualityResult(
        entity_count=len(rows),
        public_validated_count=status_counts[
            "PUBLIC_VALIDATED"
        ],
        acceptable_count=status_counts[
            "ACCEPTABLE"
        ],
        suspicious_count=status_counts[
            "SUSPICIOUS"
        ],
        not_promotable_count=status_counts[
            "NOT_PROMOTABLE"
        ],
        promotion_eligible_count=(
            promotion_eligible_count
        ),
        promotion_blocked_count=(
            promotion_blocked_count
        ),
        trailing_numeric_count=(
            trailing_numeric_count
        ),
        legal_suffix_trailing_numeric_count=(
            legal_suffix_numeric_count
        ),
        other_trailing_numeric_count=(
            other_trailing_numeric_count
        ),
        invalid_character_count=(
            invalid_character_count
        ),
        parquet_path=parquet_path,
        csv_path=csv_path,
        metrics_path=metrics_path,
    )