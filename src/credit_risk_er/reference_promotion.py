"""Promote trusted employer entities into reusable exact-match references."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

import pyarrow.parquet as pq

MASTER_REQUIRED_COLUMNS: Final = (
    "entity_id",
    "canonical_name",
    "resolution_status",
)

ALIAS_REQUIRED_COLUMNS: Final = (
    "entity_id",
    "resolution_key",
    "representative_name",
)

QUALITY_REQUIRED_COLUMNS: Final = (
    "entity_id",
    "quality_status",
    "promotion_eligible",
)

REFERENCE_MASTER_COLUMNS: Final = (
    "entity_id",
    "canonical_name",
)

REFERENCE_ALIAS_COLUMNS: Final = (
    "entity_id",
    "alias_name",
)

PUBLIC_VALIDATED: Final = "PUBLIC_VALIDATED"
AUTO_CANONICALIZED: Final = "AUTO_CANONICALIZED"

PROMOTABLE_QUALITY_STATUSES: Final = frozenset(
    {
        "PUBLIC_VALIDATED",
        "ACCEPTABLE",
    }
)


@dataclass(frozen=True, slots=True)
class ReferencePromotionResult:
    """Summary of one reference-promotion materialization."""

    eligible_entity_count: int
    promoted_entity_count: int
    promoted_alias_count: int
    existing_entity_count: int
    existing_alias_count: int
    final_entity_count: int
    final_alias_count: int
    master_path: Path
    aliases_path: Path
    metrics_path: Path


def _require_parquet_columns(
    path: Path,
    required: tuple[str, ...],
) -> pq.ParquetFile:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required reference-promotion input does not exist: {path}"
        )

    parquet = pq.ParquetFile(path)
    observed = set(parquet.schema_arrow.names)
    missing = sorted(set(required) - observed)

    if missing:
        raise ValueError(
            f"{path.name} is missing required columns: {missing}"
        )

    return parquet


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


def _required_bool(
    row: dict[str, object],
    field: str,
    *,
    source: str,
) -> bool:
    value = row.get(field)

    if not isinstance(value, bool):
        raise ValueError(
            f"{source} contains invalid boolean {field!r}"
        )

    return value


def _load_quality(
    path: Path,
) -> dict[str, tuple[str, bool]]:
    parquet = _require_parquet_columns(
        path,
        QUALITY_REQUIRED_COLUMNS,
    )

    result: dict[str, tuple[str, bool]] = {}

    for batch in parquet.iter_batches(
        columns=list(QUALITY_REQUIRED_COLUMNS)
    ):
        for raw_row in batch.to_pylist():
            row = cast(
                dict[str, object],
                raw_row,
            )

            entity_id = _required_string(
                row,
                "entity_id",
                source=path.name,
            )

            quality_status = _required_string(
                row,
                "quality_status",
                source=path.name,
            )

            promotion_eligible = _required_bool(
                row,
                "promotion_eligible",
                source=path.name,
            )

            if entity_id in result:
                raise ValueError(
                    "Canonical quality contains duplicate "
                    f"entity_id: {entity_id!r}"
                )

            if (
                promotion_eligible
                and quality_status
                not in PROMOTABLE_QUALITY_STATUSES
            ):
                raise ValueError(
                    f"Entity {entity_id!r} is marked promotion_eligible "
                    f"with unsupported quality status "
                    f"{quality_status!r}"
                )

            result[entity_id] = (
                quality_status,
                promotion_eligible,
            )

    return result


def _load_promotable_master(
    path: Path,
    quality: dict[str, tuple[str, bool]],
) -> dict[str, str]:
    parquet = _require_parquet_columns(
        path,
        MASTER_REQUIRED_COLUMNS,
    )

    master_entities: dict[str, str] = {}
    observed_entity_ids: set[str] = set()

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
                source=path.name,
            )

            if entity_id in observed_entity_ids:
                raise ValueError(
                    "Corporate master contains duplicate "
                    f"entity_id: {entity_id!r}"
                )

            observed_entity_ids.add(
                entity_id
            )

            canonical_name = _required_string(
                row,
                "canonical_name",
                source=path.name,
            )

            resolution_status = _required_string(
                row,
                "resolution_status",
                source=path.name,
            )

            quality_record = quality.get(
                entity_id
            )

            if quality_record is None:
                raise ValueError(
                    f"Corporate master entity {entity_id!r} "
                    "has no canonical quality record"
                )

            (
                quality_status,
                promotion_eligible,
            ) = quality_record

            if not promotion_eligible:
                continue

            if resolution_status == PUBLIC_VALIDATED:
                expected_quality = PUBLIC_VALIDATED
            elif resolution_status == AUTO_CANONICALIZED:
                expected_quality = "ACCEPTABLE"
            else:
                raise ValueError(
                    f"Entity {entity_id!r} is promotion eligible "
                    "but has non-promotable resolution status "
                    f"{resolution_status!r}"
                )

            if quality_status != expected_quality:
                raise ValueError(
                    f"Entity {entity_id!r} has incompatible "
                    "promotion contract: "
                    f"resolution_status={resolution_status!r}, "
                    f"quality_status={quality_status!r}"
                )

            master_entities[
                entity_id
            ] = canonical_name

    missing_master_entities = sorted(
        entity_id
        for (
            entity_id,
            (_, promotion_eligible),
        ) in quality.items()
        if promotion_eligible
        and entity_id not in observed_entity_ids
    )

    if missing_master_entities:
        raise ValueError(
            "Canonical quality contains promotion-eligible "
            "entities missing from corporate master: "
            f"{missing_master_entities[:10]!r}"
        )

    return master_entities


def _build_promoted_alias_mapping(
    path: Path,
    promotable_entities: dict[str, str],
) -> dict[str, str]:
    parquet = _require_parquet_columns(
        path,
        ALIAS_REQUIRED_COLUMNS,
    )

    alias_to_entity: dict[str, str] = {}
    entities_with_aliases: set[str] = set()

    for batch in parquet.iter_batches(
        columns=list(ALIAS_REQUIRED_COLUMNS)
    ):
        for raw_row in batch.to_pylist():
            row = cast(
                dict[str, object],
                raw_row,
            )

            entity_id = _required_string(
                row,
                "entity_id",
                source=path.name,
            )

            if entity_id not in promotable_entities:
                continue

            representative_name = _required_string(
                row,
                "representative_name",
                source=path.name,
            )

            existing_entity = alias_to_entity.get(
                representative_name
            )

            if (
                existing_entity is not None
                and existing_entity != entity_id
            ):
                raise ValueError(
                    f"Promotable alias {representative_name!r} "
                    "maps to multiple entities"
                )

            alias_to_entity[
                representative_name
            ] = entity_id

            entities_with_aliases.add(
                entity_id
            )

    missing_alias_entities = sorted(
        set(promotable_entities)
        - entities_with_aliases
    )

    if missing_alias_entities:
        raise ValueError(
            "Promotion-eligible entities without reusable aliases: "
            f"{missing_alias_entities[:10]!r}"
        )

    return {
        alias_name: alias_to_entity[
            alias_name
        ]
        for alias_name in sorted(
            alias_to_entity
        )
    }


def _load_existing_master(
    path: Path,
) -> dict[str, str]:
    if not path.exists():
        return {}

    result: dict[str, str] = {}

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as stream:
        reader = csv.DictReader(
            stream
        )

        if reader.fieldnames != list(
            REFERENCE_MASTER_COLUMNS
        ):
            raise ValueError(
                f"{path.name} must contain exactly columns "
                f"{list(REFERENCE_MASTER_COLUMNS)!r}; "
                f"observed {reader.fieldnames!r}"
            )

        for row in reader:
            entity_id = row[
                "entity_id"
            ].strip()

            canonical_name = row[
                "canonical_name"
            ].strip()

            if (
                not entity_id
                or not canonical_name
            ):
                raise ValueError(
                    f"{path.name} contains blank "
                    "reference values"
                )

            existing = result.get(
                entity_id
            )

            if (
                existing is not None
                and existing != canonical_name
            ):
                raise ValueError(
                    f"{path.name} contains conflicting "
                    "canonical name for entity "
                    f"{entity_id!r}"
                )

            result[
                entity_id
            ] = canonical_name

    return result


def _load_existing_aliases(
    path: Path,
) -> dict[str, str]:
    if not path.exists():
        return {}

    alias_to_entity: dict[str, str] = {}

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as stream:
        reader = csv.DictReader(
            stream
        )

        if reader.fieldnames != list(
            REFERENCE_ALIAS_COLUMNS
        ):
            raise ValueError(
                f"{path.name} must contain exactly columns "
                f"{list(REFERENCE_ALIAS_COLUMNS)!r}; "
                f"observed {reader.fieldnames!r}"
            )

        for row in reader:
            entity_id = row[
                "entity_id"
            ].strip()

            alias_name = row[
                "alias_name"
            ].strip()

            if (
                not entity_id
                or not alias_name
            ):
                raise ValueError(
                    f"{path.name} contains blank "
                    "reference values"
                )

            existing_entity = (
                alias_to_entity.get(
                    alias_name
                )
            )

            if (
                existing_entity is not None
                and existing_entity
                != entity_id
            ):
                raise ValueError(
                    f"Existing alias {alias_name!r} "
                    "maps to multiple entities"
                )

            alias_to_entity[
                alias_name
            ] = entity_id

    return alias_to_entity


def _merge_master(
    existing: dict[str, str],
    promoted: dict[str, str],
) -> dict[str, str]:
    merged = dict(
        existing
    )

    for (
        entity_id,
        canonical_name,
    ) in promoted.items():
        existing_name = merged.get(
            entity_id
        )

        if (
            existing_name is not None
            and existing_name != canonical_name
        ):
            raise ValueError(
                f"Reference entity {entity_id!r} "
                "already exists with "
                f"canonical_name={existing_name!r}; "
                "attempted promotion uses "
                f"{canonical_name!r}"
            )

        merged[
            entity_id
        ] = canonical_name

    return {
        entity_id: merged[
            entity_id
        ]
        for entity_id in sorted(
            merged
        )
    }


def _merge_aliases(
    existing: dict[str, str],
    promoted: dict[str, str],
) -> dict[str, str]:
    merged = dict(
        existing
    )

    for (
        alias_name,
        entity_id,
    ) in promoted.items():
        existing_entity = merged.get(
            alias_name
        )

        if (
            existing_entity is not None
            and existing_entity != entity_id
        ):
            raise ValueError(
                f"Alias {alias_name!r} already belongs "
                f"to {existing_entity!r}; "
                "attempted promotion belongs to "
                f"{entity_id!r}"
            )

        merged[
            alias_name
        ] = entity_id

    return {
        alias_name: merged[
            alias_name
        ]
        for alias_name in sorted(
            merged
        )
    }


def _validate_alias_entities(
    aliases: dict[str, str],
    master: dict[str, str],
) -> None:
    missing_entities = sorted(
        {
            entity_id
            for entity_id in aliases.values()
            if entity_id not in master
        }
    )

    if missing_entities:
        raise ValueError(
            "Reference aliases contain entity IDs "
            "absent from reference master: "
            f"{missing_entities[:10]!r}"
        )


def _atomic_write_csv(
    path: Path,
    columns: tuple[str, ...],
    rows: list[dict[str, str]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        file_descriptor,
        temporary_name,
    ) = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )

    temporary_path = Path(
        temporary_name
    )

    try:
        with os.fdopen(
            file_descriptor,
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=list(columns),
            )

            writer.writeheader()
            writer.writerows(
                rows
            )

        os.replace(
            temporary_path,
            path,
        )

    except Exception:
        temporary_path.unlink(
            missing_ok=True
        )
        raise


def promote_references(
    project_root: Path,
    *,
    master_override: Path | None = None,
    aliases_override: Path | None = None,
    quality_override: Path | None = None,
    reference_master_override: Path | None = None,
    reference_aliases_override: Path | None = None,
) -> ReferencePromotionResult:
    """Promote only quality-approved entities into exact-match references."""

    master_path = (
        master_override
        if master_override is not None
        else (
            project_root
            / "output"
            / "employer_master_final.parquet"
        )
    )

    aliases_path = (
        aliases_override
        if aliases_override is not None
        else (
            project_root
            / "output"
            / "employer_aliases_final.parquet"
        )
    )

    quality_path = (
        quality_override
        if quality_override is not None
        else (
            project_root
            / "output"
            / "canonical_name_quality.parquet"
        )
    )

    reference_master_path = (
        reference_master_override
        if reference_master_override is not None
        else (
            project_root
            / "data"
            / "reference"
            / "employer_master.csv"
        )
    )

    reference_aliases_path = (
        reference_aliases_override
        if reference_aliases_override is not None
        else (
            project_root
            / "data"
            / "reference"
            / "employer_aliases.csv"
        )
    )

    metrics_path = (
        project_root
        / "output"
        / "reference_promotion_metrics.json"
    )

    quality = _load_quality(
        quality_path
    )

    promotable_master = _load_promotable_master(
        master_path,
        quality,
    )

    promoted_aliases = (
        _build_promoted_alias_mapping(
            aliases_path,
            promotable_master,
        )
    )

    existing_master = _load_existing_master(
        reference_master_path
    )

    existing_aliases = _load_existing_aliases(
        reference_aliases_path
    )

    merged_master = _merge_master(
        existing_master,
        promotable_master,
    )

    merged_aliases = _merge_aliases(
        existing_aliases,
        promoted_aliases,
    )

    _validate_alias_entities(
        merged_aliases,
        merged_master,
    )

    master_rows = [
        {
            "entity_id": entity_id,
            "canonical_name": canonical_name,
        }
        for (
            entity_id,
            canonical_name,
        ) in merged_master.items()
    ]

    alias_rows = [
        {
            "entity_id": entity_id,
            "alias_name": alias_name,
        }
        for (
            alias_name,
            entity_id,
        ) in merged_aliases.items()
    ]

    _atomic_write_csv(
        reference_master_path,
        REFERENCE_MASTER_COLUMNS,
        master_rows,
    )

    _atomic_write_csv(
        reference_aliases_path,
        REFERENCE_ALIAS_COLUMNS,
        alias_rows,
    )

    promoted_new_entities = (
        set(promotable_master)
        - set(existing_master)
    )

    promoted_new_aliases = (
        set(promoted_aliases)
        - set(existing_aliases)
    )

    metrics = {
        "policy": {
            "quality_gate_required": True,
            "promotion_eligible_required": True,
            "promotable_quality_statuses": sorted(
                PROMOTABLE_QUALITY_STATUSES
            ),
            "normalized_candidates_promoted": False,
            "suspicious_entities_promoted": False,
        },
        "input": {
            "quality_entity_count": len(
                quality
            ),
            "eligible_entity_count": len(
                promotable_master
            ),
            "promotable_alias_count": len(
                promoted_aliases
            ),
        },
        "existing_reference": {
            "entity_count": len(
                existing_master
            ),
            "alias_count": len(
                existing_aliases
            ),
        },
        "promotion": {
            "new_entity_count": len(
                promoted_new_entities
            ),
            "new_alias_count": len(
                promoted_new_aliases
            ),
        },
        "final_reference": {
            "entity_count": len(
                merged_master
            ),
            "alias_count": len(
                merged_aliases
            ),
        },
        "reconciliation": {
            "all_promoted_entities_present": (
                set(promotable_master)
                <= set(merged_master)
            ),
            "all_promoted_aliases_present": (
                set(promoted_aliases)
                <= set(merged_aliases)
            ),
            "all_alias_entities_exist_in_master": all(
                entity_id in merged_master
                for entity_id in merged_aliases.values()
            ),
        },
    }

    metrics_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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

    return ReferencePromotionResult(
        eligible_entity_count=len(
            promotable_master
        ),
        promoted_entity_count=len(
            promoted_new_entities
        ),
        promoted_alias_count=len(
            promoted_new_aliases
        ),
        existing_entity_count=len(
            existing_master
        ),
        existing_alias_count=len(
            existing_aliases
        ),
        final_entity_count=len(
            merged_master
        ),
        final_alias_count=len(
            merged_aliases
        ),
        master_path=(
            reference_master_path
        ),
        aliases_path=(
            reference_aliases_path
        ),
        metrics_path=(
            metrics_path
        ),
    )