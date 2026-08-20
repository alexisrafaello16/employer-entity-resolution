"""Materialize the reusable corporate employer master and alias dictionary."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

import pyarrow as pa
import pyarrow.parquet as pq

MASTER_SCHEMA = pa.schema(
    [
        pa.field("entity_id", pa.string(), nullable=False),
        pa.field("canonical_name", pa.string(), nullable=False),
        pa.field("sector", pa.string(), nullable=False),
        pa.field("resolution_status", pa.string(), nullable=False),
        pa.field("resolution_method", pa.string(), nullable=False),
        pa.field("confidence_level", pa.string(), nullable=False),
        pa.field("confidence_score", pa.int32(), nullable=False),
        pa.field("component_key_count", pa.int32(), nullable=False),
        pa.field("source_row_count", pa.int64(), nullable=False),
        pa.field("public_validation_status", pa.string(), nullable=False),
        pa.field("public_entity_id", pa.string(), nullable=True),
        pa.field("public_source_url", pa.string(), nullable=True),
        pa.field("public_validation_date", pa.string(), nullable=True),
    ]
)

ALIAS_SCHEMA = pa.schema(
    [
        pa.field("entity_id", pa.string(), nullable=False),
        pa.field("resolution_key", pa.string(), nullable=False),
        pa.field("representative_name", pa.string(), nullable=False),
        pa.field("alias_method", pa.string(), nullable=False),
        pa.field("alias_confidence_score", pa.int32(), nullable=False),
        pa.field("source_row_frequency", pa.int64(), nullable=False),
    ]
)

MASTER_COLUMNS: Final = tuple(
    field.name for field in MASTER_SCHEMA
)

ALIAS_COLUMNS: Final = tuple(
    field.name for field in ALIAS_SCHEMA
)

PUBLIC_RESULT: Final = "EMPRESA_VALIDADA_PUBLICAMENTE"
AUTO_RESULT: Final = "EMPRESA_CANONICALIZADA_AUTO"
NORMALIZED_RESULT: Final = (
    "EMPRESA_NORMALIZADA_SIN_VALIDACION_PUBLICA"
)

ENTITY_RESULTS: Final = frozenset(
    {
        PUBLIC_RESULT,
        AUTO_RESULT,
        NORMALIZED_RESULT,
    }
)

RESOLUTION_STATUS_BY_RESULT: Final = {
    PUBLIC_RESULT: "PUBLIC_VALIDATED",
    AUTO_RESULT: "AUTO_CANONICALIZED",
    NORMALIZED_RESULT: "NORMALIZED_CANDIDATE",
}

PUBLICLY_VALIDATED: Final = "PUBLICLY_VALIDATED"
NOT_PUBLICLY_VALIDATED: Final = "NOT_PUBLICLY_VALIDATED"

REFERENCE_EXACT_METHODS: Final = frozenset(
    {
        "exact_reference_canonical",
        "exact_reference_alias",
    }
)

REFERENCE_EXACT_REUSE_METHOD: Final = (
    "reference_exact_reuse"
)

FINAL_REQUIRED_COLUMNS: Final = (
    "record_id",
    "resolution_key",
    "entity_id",
    "nombre_propuesto",
    "sector_propuesto",
    "resultado_final",
    "confianza_resolucion",
    "score_confianza_resolucion",
    "metodo_resolucion",
    "canonical_resolution_key",
    "component_key_count",
    "component_source_row_count",
    "public_entity_id",
    "public_source_url",
    "public_validation_date",
)

KEY_REQUIRED_COLUMNS: Final = (
    "resolution_key",
    "representative_name",
    "source_row_frequency",
)


@dataclass(frozen=True, slots=True)
class CorporateMasterResult:
    """Paths and reconciliation counts produced by corporate-master materialization."""

    entity_count: int
    alias_count: int
    source_row_count: int
    public_validated_entity_count: int
    master_parquet_path: Path
    master_csv_path: Path
    aliases_parquet_path: Path
    aliases_csv_path: Path
    metrics_path: Path


@dataclass(frozen=True, slots=True)
class ResolutionKeyMetadata:
    """Reference metadata required to materialize one alias row."""

    representative_name: str
    source_row_frequency: int


@dataclass(frozen=True, slots=True)
class EntityAccumulator:
    """Validated component-level metadata before serializing one master row."""

    entity_id: str
    canonical_name: str
    sector: str
    resolution_status: str
    resolution_method: str
    confidence_level: str
    confidence_score: int
    component_key_count: int
    source_row_count: int
    public_entity_id: str | None
    public_source_url: str | None
    public_validation_date: str | None


def _require_parquet_columns(
    path: Path,
    required: tuple[str, ...],
) -> pq.ParquetFile:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required corporate-master input does not exist: {path}"
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


def _optional_string(
    row: dict[str, object],
    field: str,
) -> str | None:
    value = row.get(field)

    if value is None:
        return None

    if not isinstance(value, str):
        raise ValueError(
            f"Invalid non-string value for optional field {field!r}"
        )

    stripped = value.strip()

    return stripped or None


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


def _load_resolution_keys(
    path: Path,
) -> dict[str, ResolutionKeyMetadata]:
    parquet = _require_parquet_columns(
        path,
        KEY_REQUIRED_COLUMNS,
    )

    result: dict[str, ResolutionKeyMetadata] = {}

    for batch in parquet.iter_batches(
        columns=list(KEY_REQUIRED_COLUMNS)
    ):
        for raw_row in batch.to_pylist():
            row = cast(
                dict[str, object],
                raw_row,
            )

            key = _required_string(
                row,
                "resolution_key",
                source=path.name,
            )

            representative_name = _required_string(
                row,
                "representative_name",
                source=path.name,
            )

            frequency = _required_int(
                row,
                "source_row_frequency",
                source=path.name,
            )

            if frequency < 1:
                raise ValueError(
                    f"{path.name} contains non-positive "
                    "source_row_frequency"
                )

            existing = result.get(key)

            metadata = ResolutionKeyMetadata(
                representative_name,
                frequency,
            )

            if (
                existing is not None
                and existing != metadata
            ):
                raise ValueError(
                    "Duplicate resolution_key with conflicting "
                    f"metadata: {key!r}"
                )

            result[key] = metadata

    return result


def _single_value(
    *,
    entity_id: str,
    field: str,
    values: set[str],
) -> str:
    if len(values) != 1:
        raise ValueError(
            f"Entity {entity_id!r} has inconsistent "
            f"{field}: {sorted(values)!r}"
        )

    return next(iter(values))


def _entity_resolution_method(
    *,
    entity_id: str,
    methods: set[str],
) -> str:
    """
    Consolidate row-level resolution methods into one entity-level method.

    Exact-reference canonical and alias matches are two row-level paths
    into the same persistent entity knowledge. At corporate-master level
    they are represented by one stable reusable resolution method.

    Any mixture between exact-reference reuse and an unrelated resolution
    method remains invalid.
    """
    if methods and methods.issubset(
        REFERENCE_EXACT_METHODS
    ):
        return REFERENCE_EXACT_REUSE_METHOD

    return _single_value(
        entity_id=entity_id,
        field="resolution_method",
        values=methods,
    )


def _single_optional_value(
    *,
    entity_id: str,
    field: str,
    values: set[str | None],
) -> str | None:
    if len(values) != 1:
        rendered = sorted(
            "<NULL>" if value is None else value
            for value in values
        )

        raise ValueError(
            f"Entity {entity_id!r} has inconsistent "
            f"{field}: {rendered!r}"
        )

    return next(iter(values))


def _load_entities_and_keys(
    path: Path,
) -> tuple[
    dict[str, EntityAccumulator],
    dict[str, set[str]],
    dict[tuple[str, str], int],
    int,
]:
    parquet = _require_parquet_columns(
        path,
        FINAL_REQUIRED_COLUMNS,
    )

    rows_by_entity: dict[
        str,
        list[dict[str, object]],
    ] = defaultdict(list)

    source_row_count = 0
    observed_key_frequencies: dict[
        tuple[str, str],
        int,
    ] = defaultdict(int)

    for batch in parquet.iter_batches(
        columns=list(FINAL_REQUIRED_COLUMNS)
    ):
        for raw_row in batch.to_pylist():
            row = cast(
                dict[str, object],
                raw_row,
            )

            result = _required_string(
                row,
                "resultado_final",
                source=path.name,
            )

            entity_id = _optional_string(
                row,
                "entity_id",
            )

            if result not in ENTITY_RESULTS:
                if entity_id is not None:
                    raise ValueError(
                        "Non-identifiable final row unexpectedly "
                        "contains an entity_id"
                    )

                continue

            if entity_id is None:
                raise ValueError(
                    f"Entity-bearing outcome {result!r} "
                    "has no entity_id"
                )

            resolution_key = _optional_string(
                row,
                "resolution_key",
            )

            if resolution_key is None:
                raise ValueError(
                    f"Entity {entity_id!r} contains a row "
                    "without resolution_key"
                )

            rows_by_entity[
                entity_id
            ].append(row)

            observed_key_frequencies[
                (
                    entity_id,
                    resolution_key,
                )
            ] += 1

            source_row_count += 1

    entities: dict[
        str,
        EntityAccumulator,
    ] = {}

    keys_by_entity: dict[
        str,
        set[str],
    ] = {}

    for entity_id in sorted(rows_by_entity):
        rows = rows_by_entity[
            entity_id
        ]

        canonical_names = {
            _required_string(
                row,
                "nombre_propuesto",
                source=path.name,
            )
            for row in rows
        }

        sectors = {
            _required_string(
                row,
                "sector_propuesto",
                source=path.name,
            )
            for row in rows
        }

        results = {
            _required_string(
                row,
                "resultado_final",
                source=path.name,
            )
            for row in rows
        }

        methods = {
            _required_string(
                row,
                "metodo_resolucion",
                source=path.name,
            )
            for row in rows
        }

        confidence_levels = {
            _required_string(
                row,
                "confianza_resolucion",
                source=path.name,
            )
            for row in rows
        }

        confidence_scores = {
            _required_int(
                row,
                "score_confianza_resolucion",
                source=path.name,
            )
            for row in rows
        }

        canonical_keys = {
            _required_string(
                row,
                "canonical_resolution_key",
                source=path.name,
            )
            for row in rows
        }

        component_key_counts = {
            _required_int(
                row,
                "component_key_count",
                source=path.name,
            )
            for row in rows
        }

        component_source_counts = {
            _required_int(
                row,
                "component_source_row_count",
                source=path.name,
            )
            for row in rows
        }

        public_entity_ids = {
            _optional_string(
                row,
                "public_entity_id",
            )
            for row in rows
        }

        public_source_urls = {
            _optional_string(
                row,
                "public_source_url",
            )
            for row in rows
        }

        public_validation_dates = {
            _optional_string(
                row,
                "public_validation_date",
            )
            for row in rows
        }

        resolution_keys = {
            _required_string(
                row,
                "resolution_key",
                source=path.name,
            )
            for row in rows
        }

        canonical_name = _single_value(
            entity_id=entity_id,
            field="canonical_name",
            values=canonical_names,
        )

        sector = _single_value(
            entity_id=entity_id,
            field="sector",
            values=sectors,
        )

        result = _single_value(
            entity_id=entity_id,
            field="resultado_final",
            values=results,
        )

        method = _entity_resolution_method(
            entity_id=entity_id,
            methods=methods,
        )

        confidence_level = _single_value(
            entity_id=entity_id,
            field="confidence_level",
            values=confidence_levels,
        )

        confidence_score = (
            next(iter(confidence_scores))
            if len(confidence_scores) == 1
            else -1
        )

        if confidence_score < 0:
            raise ValueError(
                f"Entity {entity_id!r} has inconsistent "
                "confidence_score: "
                f"{sorted(confidence_scores)!r}"
            )

        canonical_key = _single_value(
            entity_id=entity_id,
            field="canonical_resolution_key",
            values=canonical_keys,
        )

        component_key_count = (
            next(iter(component_key_counts))
            if len(component_key_counts) == 1
            else -1
        )

        if component_key_count < 1:
            raise ValueError(
                f"Entity {entity_id!r} has inconsistent "
                "component_key_count: "
                f"{sorted(component_key_counts)!r}"
            )

        component_source_row_count = (
            next(iter(component_source_counts))
            if len(component_source_counts) == 1
            else -1
        )

        if component_source_row_count < 1:
            raise ValueError(
                f"Entity {entity_id!r} has inconsistent "
                "component_source_row_count: "
                f"{sorted(component_source_counts)!r}"
            )

        if (
            len(rows)
            != component_source_row_count
        ):
            raise ValueError(
                f"Entity {entity_id!r} source-row frequency "
                "does not match component_source_row_count"
            )

        if (
            len(resolution_keys)
            != component_key_count
        ):
            raise ValueError(
                f"Entity {entity_id!r} resolution-key count "
                "does not match component_key_count"
            )

        if (
            method != REFERENCE_EXACT_REUSE_METHOD
            and canonical_key not in resolution_keys
        ):
            raise ValueError(
                f"Entity {entity_id!r} "
                "canonical_resolution_key is absent "
                "from its aliases"
            )

        public_entity_id = _single_optional_value(
            entity_id=entity_id,
            field="public_entity_id",
            values=public_entity_ids,
        )

        public_source_url = _single_optional_value(
            entity_id=entity_id,
            field="public_source_url",
            values=public_source_urls,
        )

        public_validation_date = (
            _single_optional_value(
                entity_id=entity_id,
                field="public_validation_date",
                values=public_validation_dates,
            )
        )

        if result == PUBLIC_RESULT:
            if (
                public_entity_id is None
                or public_source_url is None
                or public_validation_date is None
            ):
                raise ValueError(
                    f"Publicly validated entity {entity_id!r} "
                    "lacks public validation metadata"
                )

        elif any(
            value is not None
            for value in (
                public_entity_id,
                public_source_url,
                public_validation_date,
            )
        ):
            raise ValueError(
                f"Non-public entity {entity_id!r} "
                "unexpectedly contains public validation metadata"
            )

        entities[
            entity_id
        ] = EntityAccumulator(
            entity_id=entity_id,
            canonical_name=canonical_name,
            sector=sector,
            resolution_status=(
                RESOLUTION_STATUS_BY_RESULT[
                    result
                ]
            ),
            resolution_method=method,
            confidence_level=confidence_level,
            confidence_score=confidence_score,
            component_key_count=(
                component_key_count
            ),
            source_row_count=(
                component_source_row_count
            ),
            public_entity_id=(
                public_entity_id
            ),
            public_source_url=(
                public_source_url
            ),
            public_validation_date=(
                public_validation_date
            ),
        )

        keys_by_entity[
            entity_id
        ] = resolution_keys

    return (
        entities,
        keys_by_entity,
        dict(observed_key_frequencies),
        source_row_count,
    )


def _master_rows(
    entities: dict[
        str,
        EntityAccumulator,
    ],
) -> list[dict[str, object]]:
    rows: list[
        dict[str, object]
    ] = []

    for entity_id in sorted(entities):
        entity = entities[
            entity_id
        ]

        rows.append(
            {
                "entity_id": entity.entity_id,
                "canonical_name": (
                    entity.canonical_name
                ),
                "sector": entity.sector,
                "resolution_status": (
                    entity.resolution_status
                ),
                "resolution_method": (
                    entity.resolution_method
                ),
                "confidence_level": (
                    entity.confidence_level
                ),
                "confidence_score": (
                    entity.confidence_score
                ),
                "component_key_count": (
                    entity.component_key_count
                ),
                "source_row_count": (
                    entity.source_row_count
                ),
                "public_validation_status": (
                    PUBLICLY_VALIDATED
                    if entity.public_entity_id
                    is not None
                    else NOT_PUBLICLY_VALIDATED
                ),
                "public_entity_id": (
                    entity.public_entity_id
                ),
                "public_source_url": (
                    entity.public_source_url
                ),
                "public_validation_date": (
                    entity.public_validation_date
                ),
            }
        )

    return rows


def _alias_rows(
    *,
    entities: dict[
        str,
        EntityAccumulator,
    ],
    keys_by_entity: dict[
        str,
        set[str],
    ],
    resolution_keys: dict[
        str,
        ResolutionKeyMetadata,
    ],
    observed_key_frequencies: dict[
        tuple[str, str],
        int,
    ],
) -> list[dict[str, object]]:
    rows: list[
        dict[str, object]
    ] = []

    seen_keys: dict[
        str,
        str,
    ] = {}

    for entity_id in sorted(entities):
        entity = entities[
            entity_id
        ]

        keys = keys_by_entity[
            entity_id
        ]

        for key in sorted(keys):
            metadata = resolution_keys.get(
                key
            )

            observed_frequency = (
                observed_key_frequencies.get(
                    (
                        entity_id,
                        key,
                    )
                )
            )

            if observed_frequency is None:
                raise ValueError(
                    f"Entity {entity_id!r} has no "
                    "observed source-row frequency "
                    f"for resolution key {key!r}"
                )

            if metadata is None:
                metadata = ResolutionKeyMetadata(
                    representative_name=key,
                    source_row_frequency=(
                        observed_frequency
                    ),
                )
            elif (
                metadata.source_row_frequency
                != observed_frequency
            ):
                raise ValueError(
                    f"Entity {entity_id!r} resolution "
                    f"key {key!r} source-row frequency "
                    "does not match the final dataset"
                )

            prior_entity = seen_keys.get(
                key
            )

            if (
                prior_entity is not None
                and prior_entity != entity_id
            ):
                raise ValueError(
                    f"Resolution key {key!r} maps "
                    "to multiple entities: "
                    f"{prior_entity!r} and "
                    f"{entity_id!r}"
                )

            seen_keys[
                key
            ] = entity_id

            rows.append(
                {
                    "entity_id": entity_id,
                    "resolution_key": key,
                    "representative_name": (
                        metadata.representative_name
                    ),
                    "alias_method": (
                        entity.resolution_method
                    ),
                    "alias_confidence_score": (
                        entity.confidence_score
                    ),
                    "source_row_frequency": (
                        metadata.source_row_frequency
                    ),
                }
            )

    return rows


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


def _write_csv(
    path: Path,
    columns: tuple[str, ...],
    rows: list[dict[str, object]],
) -> None:
    """
    Write an Excel-compatible UTF-8 CSV.

    UTF-8 with BOM is used intentionally so Windows/Excel
    correctly detects accents and other legitimate Unicode
    characters without altering canonical employer names.
    """
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(columns),
        )

        writer.writeheader()
        writer.writerows(rows)


def build_corporate_master(
    project_root: Path,
    *,
    final_dataset_override: Path | None = None,
    resolution_keys_override: Path | None = None,
) -> CorporateMasterResult:
    """
    Materialize one corporate master row per entity.

    Also materialize one alias row per observed resolution key.
    """

    final_dataset_path = (
        final_dataset_override
        if final_dataset_override is not None
        else project_root
        / "output"
        / "employer_resolution_final.parquet"
    )

    resolution_keys_path = (
        resolution_keys_override
        if resolution_keys_override is not None
        else project_root
        / "data"
        / "processed"
        / "resolution_keys.parquet"
    )

    output_dir = (
        project_root / "output"
    )

    master_parquet_path = (
        output_dir
        / "employer_master_final.parquet"
    )

    master_csv_path = (
        output_dir
        / "employer_master_final.csv"
    )

    aliases_parquet_path = (
        output_dir
        / "employer_aliases_final.parquet"
    )

    aliases_csv_path = (
        output_dir
        / "employer_aliases_final.csv"
    )

    metrics_path = (
        output_dir
        / "corporate_master_metrics.json"
    )

    (
        entities,
        keys_by_entity,
        observed_key_frequencies,
        source_row_count,
    ) = _load_entities_and_keys(
        final_dataset_path
    )

    resolution_keys = (
        _load_resolution_keys(
            resolution_keys_path
        )
    )

    master_rows = _master_rows(
        entities
    )

    alias_rows = _alias_rows(
        entities=entities,
        keys_by_entity=keys_by_entity,
        resolution_keys=resolution_keys,
        observed_key_frequencies=(
            observed_key_frequencies
        ),
    )

    alias_frequency_by_entity: dict[
        str,
        int,
    ] = defaultdict(int)

    alias_count_by_entity: dict[
        str,
        int,
    ] = defaultdict(int)

    for row in alias_rows:
        entity_id = cast(
            str,
            row["entity_id"],
        )

        alias_frequency_by_entity[
            entity_id
        ] += cast(
            int,
            row[
                "source_row_frequency"
            ],
        )

        alias_count_by_entity[
            entity_id
        ] += 1

    for (
        entity_id,
        entity,
    ) in entities.items():
        if (
            alias_frequency_by_entity[
                entity_id
            ]
            != entity.source_row_count
        ):
            raise ValueError(
                f"Entity {entity_id!r} alias "
                "source-row frequency does not "
                "match master source-row frequency"
            )

        if (
            alias_count_by_entity[
                entity_id
            ]
            != entity.component_key_count
        ):
            raise ValueError(
                f"Entity {entity_id!r} alias "
                "count does not match "
                "component_key_count"
            )

    _write_parquet(
        master_parquet_path,
        MASTER_SCHEMA,
        master_rows,
    )

    _write_csv(
        master_csv_path,
        MASTER_COLUMNS,
        master_rows,
    )

    _write_parquet(
        aliases_parquet_path,
        ALIAS_SCHEMA,
        alias_rows,
    )

    _write_csv(
        aliases_csv_path,
        ALIAS_COLUMNS,
        alias_rows,
    )

    public_validated_entity_count = sum(
        1
        for entity in entities.values()
        if entity.public_entity_id
        is not None
    )

    alias_frequency_total = sum(
        cast(
            int,
            row[
                "source_row_frequency"
            ],
        )
        for row in alias_rows
    )

    metrics = {
        "entities": {
            "entity_count": len(
                master_rows
            ),
            "public_validated_entity_count": (
                public_validated_entity_count
            ),
            "resolution_status_counts": {
                status: sum(
                    1
                    for row in master_rows
                    if row[
                        "resolution_status"
                    ]
                    == status
                )
                for status in sorted(
                    {
                        cast(
                            str,
                            row[
                                "resolution_status"
                            ],
                        )
                        for row in master_rows
                    }
                )
            },
        },
        "aliases": {
            "alias_count": len(
                alias_rows
            ),
            "source_row_frequency_total": (
                alias_frequency_total
            ),
        },
        "reconciliation": {
            "entity_count_matches": (
                len(master_rows)
                == len(entities)
            ),
            "alias_frequency_matches_source_rows": (
                alias_frequency_total
                == source_row_count
            ),
            "all_entity_alias_counts_match_component_key_count": (
                all(
                    alias_count_by_entity[
                        entity_id
                    ]
                    == entity.component_key_count
                    for (
                        entity_id,
                        entity,
                    ) in entities.items()
                )
            ),
        },
        "source": {
            "entity_source_row_count": (
                source_row_count
            ),
        },
    }

    output_dir.mkdir(
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

    return CorporateMasterResult(
        entity_count=len(
            master_rows
        ),
        alias_count=len(
            alias_rows
        ),
        source_row_count=(
            source_row_count
        ),
        public_validated_entity_count=(
            public_validated_entity_count
        ),
        master_parquet_path=(
            master_parquet_path
        ),
        master_csv_path=(
            master_csv_path
        ),
        aliases_parquet_path=(
            aliases_parquet_path
        ),
        aliases_csv_path=(
            aliases_csv_path
        ),
        metrics_path=(
            metrics_path
        ),
    )