"""Optional profiling and deterministic evaluation-sampling utilities.

Neither function is called by the normal ``preprocess`` command.
"""

from __future__ import annotations

import hashlib
import heapq
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_EVALUATION_QUOTAS: Mapping[str, int] = {
    "blank": 10,
    "numeric-only": 30,
    "mixed-occupation-organization": 50,
    "mixed-address-organization": 60,
    "occupation-only": 50,
    "address-candidate": 60,
    "possible-truncation": 30,
    "trailing-digit": 30,
    "corporate-suffix": 30,
    "short-name": 30,
    "unusual-ambiguous": 50,
    "ordinary-employer-like": 50,
}


def profile_dataset(dataset_path: Path) -> dict[str, object]:
    """Run an explicit, reusable deeper scan of the preprocessed dataset."""
    parquet = pq.ParquetFile(dataset_path)
    routes: Counter[str] = Counter()
    normalized_lengths: Counter[int] = Counter()
    nulls = 0
    for batch in parquet.iter_batches(columns=["nombre_original", "normalized_length", "route"]):
        nulls += batch.column("nombre_original").null_count
        routes.update(cast(list[str], batch.column("route").to_pylist()))
        normalized_lengths.update(cast(list[int], batch.column("normalized_length").to_pylist()))
    return {
        "rows": parquet.metadata.num_rows,
        "null_source_values": nulls,
        "routes": dict(sorted(routes.items())),
        "normalized_length_distribution": dict(sorted(normalized_lengths.items())),
    }


def _strata(row: dict[str, Any], short_name_max_length: int) -> tuple[str, ...]:
    strata: list[str] = []
    checks = (
        ("blank", row["is_blank"]),
        ("numeric-only", row["is_numeric_only"]),
        ("mixed-occupation-organization", row["mixed_occupation_organization_signal"]),
        ("mixed-address-organization", row["mixed_address_organization_signal"]),
        (
            "occupation-only",
            row["has_occupation_signal"] and not row["has_organization_like_tokens"],
        ),
        ("address-candidate", row["route"] == "address_candidate"),
        ("possible-truncation", row["possible_truncation"]),
        ("trailing-digit", row["has_trailing_numeric_token"]),
        ("corporate-suffix", row["has_corporate_suffix"]),
        (
            "short-name",
            not row["is_blank"] and row["normalized_length"] <= short_name_max_length,
        ),
        ("unusual-ambiguous", row["route"] == "ambiguous_review_candidate"),
        (
            "ordinary-employer-like",
            row["route"] == "employer_resolution_candidate"
            and not row["has_address_signal"]
            and not row["has_occupation_signal"]
            and not row["possible_truncation"],
        ),
    )
    strata.extend(name for name, selected in checks if selected)
    return tuple(strata)


def create_evaluation_sample(
    dataset_path: Path,
    output_path: Path,
    *,
    quotas: Mapping[str, int] = DEFAULT_EVALUATION_QUOTAS,
    random_seed: int = 20260813,
    short_name_max_length: int = 3,
) -> int:
    """Create a separate deterministic development sample from preprocessed data."""
    parquet = pq.ParquetFile(dataset_path)
    reservoirs: dict[str, list[tuple[int, str, dict[str, Any]]]] = {
        stratum: [] for stratum, quota in quotas.items() if quota > 0
    }
    for batch in parquet.iter_batches(batch_size=10_000):
        for row in batch.to_pylist():
            record_id = cast(str, row["record_id"])
            for stratum in _strata(row, short_name_max_length):
                if stratum not in reservoirs:
                    continue
                score = int.from_bytes(
                    hashlib.sha256(f"{random_seed}|{stratum}|{record_id}".encode()).digest()[:8]
                )
                candidate = (-score, record_id, row)
                reservoir = reservoirs[stratum]
                quota = quotas[stratum]
                if len(reservoir) < quota:
                    heapq.heappush(reservoir, candidate)
                elif candidate > reservoir[0]:
                    heapq.heapreplace(reservoir, candidate)

    selected: dict[str, tuple[dict[str, Any], set[str]]] = {}
    for stratum, reservoir in reservoirs.items():
        for _, record_id, row in reservoir:
            if record_id not in selected:
                selected[record_id] = (row, set())
            selected[record_id][1].add(stratum)

    rows: list[dict[str, Any]] = []
    for row, strata in sorted(selected.values(), key=lambda item: item[0]["source_row_number"]):
        rows.append(
            {
                **row,
                "evaluation_strata": sorted(strata),
                "reviewed_entity": None,
                "reviewed_sector": None,
                "reviewer_notes": None,
            }
        )
    schema = parquet.schema_arrow.append(pa.field("evaluation_strata", pa.list_(pa.string())))
    schema = schema.append(pa.field("reviewed_entity", pa.string()))
    schema = schema.append(pa.field("reviewed_sector", pa.string()))
    schema = schema.append(pa.field("reviewer_notes", pa.string()))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(f"{output_path.suffix}.tmp")
    pq.write_table(
        pa.Table.from_pylist(rows, schema=schema),
        temporary,
        compression="zstd",
        use_dictionary=True,
    )
    temporary.replace(output_path)
    return len(rows)
