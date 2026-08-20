"""Deterministic, unlabeled datasets for human entity-resolution review."""

from __future__ import annotations

import csv
import hashlib
import heapq
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import pyarrow.parquet as pq

from credit_risk_er.config import Settings

type PairStratum = Literal[
    "numeric_conflict",
    "trailing_numeric",
    "high_token_set_low_char",
    "truncation_or_subset",
    "multi_method",
    "truncation_block",
    "informative_token_only",
    "short_or_sparse_names",
    "low_similarity_retrieved",
    "very_high_char",
    "high_char",
    "medium_char",
]
type AuditStratum = Literal[
    "possible_truncation",
    "ambiguous_review",
    "high_frequency",
    "numeric_name",
    "short_name",
    "ordinary_employer",
    "other",
]

# Risk-specific strata precede broad score bands so overlaps retain diagnostic value.
PAIR_STRATUM_PRIORITY: tuple[PairStratum, ...] = (
    "numeric_conflict",
    "trailing_numeric",
    "high_token_set_low_char",
    "truncation_or_subset",
    "multi_method",
    "truncation_block",
    "informative_token_only",
    "short_or_sparse_names",
    "low_similarity_retrieved",
    "very_high_char",
    "high_char",
    "medium_char",
)
AUDIT_STRATUM_PRIORITY: tuple[AuditStratum, ...] = (
    "possible_truncation",
    "ambiguous_review",
    "high_frequency",
    "numeric_name",
    "short_name",
    "ordinary_employer",
    "other",
)

PAIR_INPUT_COLUMNS = (
    "key_a",
    "key_b",
    "name_a",
    "name_b",
    "blocking_methods",
    "blocking_method_count",
    "char_ratio",
    "token_sort_ratio",
    "token_set_ratio",
    "partial_ratio",
    "length_ratio",
    "common_prefix_ratio",
    "token_jaccard",
    "same_first_token",
    "numeric_relation",
)
PAIR_OUTPUT_COLUMNS = (
    "sample_id",
    "stratum",
    "name_a",
    "name_b",
    "key_a",
    "key_b",
    "blocking_methods",
    "blocking_method_count",
    "char_ratio",
    "token_sort_ratio",
    "token_set_ratio",
    "partial_ratio",
    "length_ratio",
    "common_prefix_ratio",
    "token_jaccard",
    "same_first_token",
    "numeric_relation",
    "review_label",
    "review_notes",
)
AUDIT_OUTPUT_COLUMNS = (
    "audit_id",
    "audit_stratum",
    "resolution_key",
    "representative_name",
    "source_row_frequency",
    "representative_route",
    "possible_truncation",
    "token_count",
    "possible_blocking_miss",
    "suspected_variant",
    "review_notes",
)


@dataclass(frozen=True, slots=True)
class PairEvidence:
    key_a: str
    key_b: str
    name_a: str
    name_b: str
    blocking_methods: tuple[str, ...]
    blocking_method_count: int
    char_ratio: float
    token_sort_ratio: float
    token_set_ratio: float
    partial_ratio: float
    length_ratio: float
    common_prefix_ratio: float
    token_jaccard: float
    same_first_token: bool
    numeric_relation: str


@dataclass(frozen=True, slots=True)
class AuditKey:
    resolution_key: str
    representative_name: str
    source_row_frequency: int
    representative_route: str
    possible_truncation: bool
    token_count: int


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    pair_review_path: Path
    blocking_miss_path: Path
    pair_sample_size: int
    pair_stratum_counts: dict[str, int]
    duplicate_pairs: int
    nonblank_review_labels: int
    blocking_miss_sample_size: int
    zero_candidate_population: int


def _is_short_or_sparse(name_a: str, name_b: str) -> bool:
    return min(len(name_a), len(name_b)) <= 8 or min(len(name_a.split()), len(name_b.split())) <= 1


def assign_pair_stratum(pair: PairEvidence) -> PairStratum | None:
    """Assign one diagnostic stratum according to the documented priority."""
    methods = pair.blocking_methods
    checks: dict[PairStratum, bool] = {
        "numeric_conflict": pair.numeric_relation == "conflict",
        "trailing_numeric": (
            "trailing_numeric" in methods and pair.numeric_relation == "one_sided"
        ),
        "high_token_set_low_char": pair.token_set_ratio >= 95 and pair.char_ratio < 80,
        "truncation_or_subset": pair.partial_ratio >= 99 and pair.length_ratio < 0.85,
        "multi_method": pair.blocking_method_count >= 2,
        "truncation_block": "truncation_prefix" in methods,
        "informative_token_only": methods == ("informative_token",),
        "short_or_sparse_names": _is_short_or_sparse(pair.name_a, pair.name_b),
        "low_similarity_retrieved": pair.char_ratio < 60,
        "very_high_char": pair.char_ratio >= 97,
        "high_char": 90 <= pair.char_ratio < 97,
        "medium_char": 80 <= pair.char_ratio < 90,
    }
    return next((stratum for stratum in PAIR_STRATUM_PRIORITY if checks[stratum]), None)


def _target_allocation(total: int, strata: tuple[str, ...]) -> dict[str, int]:
    base, remainder = divmod(total, len(strata))
    return {stratum: base + int(index < remainder) for index, stratum in enumerate(strata)}


def _rank(seed: int, namespace: str, *parts: str) -> int:
    payload = "\x1f".join((str(seed), namespace, *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest(), byteorder="big")


def _pair_from_row(row: dict[str, object]) -> PairEvidence:
    methods = row["blocking_methods"]
    if not isinstance(methods, list) or not methods:
        raise ValueError("blocking_methods must be a nonempty list")
    pair = PairEvidence(
        key_a=cast(str, row["key_a"]),
        key_b=cast(str, row["key_b"]),
        name_a=cast(str, row["name_a"]),
        name_b=cast(str, row["name_b"]),
        blocking_methods=tuple(cast(list[str], methods)),
        blocking_method_count=cast(int, row["blocking_method_count"]),
        char_ratio=cast(float, row["char_ratio"]),
        token_sort_ratio=cast(float, row["token_sort_ratio"]),
        token_set_ratio=cast(float, row["token_set_ratio"]),
        partial_ratio=cast(float, row["partial_ratio"]),
        length_ratio=cast(float, row["length_ratio"]),
        common_prefix_ratio=cast(float, row["common_prefix_ratio"]),
        token_jaccard=cast(float, row["token_jaccard"]),
        same_first_token=cast(bool, row["same_first_token"]),
        numeric_relation=cast(str, row["numeric_relation"]),
    )
    if not all((pair.key_a, pair.key_b, pair.name_a, pair.name_b)) or pair.key_a >= pair.key_b:
        raise ValueError("Evaluation input contains an invalid canonical candidate pair")
    return pair


def sample_review_pairs(
    feature_path: Path, *, sample_size: int, random_seed: int, batch_size: int
) -> list[tuple[PairStratum, PairEvidence]]:
    """Select deterministic hash-ranked reservoirs after exclusive stratum assignment."""
    parquet = pq.ParquetFile(feature_path)
    missing = set(PAIR_INPUT_COLUMNS) - set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"Candidate feature dataset is missing columns: {sorted(missing)}")
    targets = _target_allocation(sample_size, cast(tuple[str, ...], PAIR_STRATUM_PRIORITY))
    heaps: dict[str, list[tuple[int, str, str, PairEvidence]]] = defaultdict(list)
    previous_pair: tuple[str, str] | None = None
    for batch in parquet.iter_batches(batch_size=batch_size, columns=list(PAIR_INPUT_COLUMNS)):
        for row in batch.to_pylist():
            pair = _pair_from_row(row)
            identity = (pair.key_a, pair.key_b)
            if previous_pair is not None and identity <= previous_pair:
                raise ValueError(
                    "Candidate feature pairs must be unique and deterministically ordered"
                )
            previous_pair = identity
            stratum = assign_pair_stratum(pair)
            if stratum is None or targets[stratum] == 0:
                continue
            rank = _rank(random_seed, stratum, pair.key_a, pair.key_b)
            entry = (-rank, pair.key_a, pair.key_b, pair)
            heap = heaps[stratum]
            if len(heap) < targets[stratum]:
                heapq.heappush(heap, entry)
            elif rank < -heap[0][0]:
                heapq.heapreplace(heap, entry)

    selected: list[tuple[PairStratum, PairEvidence]] = []
    for stratum in PAIR_STRATUM_PRIORITY:
        ranked = sorted(
            ((-negative_rank, pair) for negative_rank, _, _, pair in heaps[stratum]),
            key=lambda item: (item[0], item[1].key_a, item[1].key_b),
        )
        selected.extend((stratum, pair) for _, pair in ranked)
    return selected


def derive_zero_candidate_keys(
    resolution_keys_path: Path, candidate_pairs_path: Path, *, batch_size: int
) -> list[AuditKey]:
    """Return resolution keys absent from both sides of every candidate pair."""
    candidate_file = pq.ParquetFile(candidate_pairs_path)
    if not {"key_a", "key_b"}.issubset(candidate_file.schema_arrow.names):
        raise ValueError("Candidate-pair dataset must contain key_a and key_b")
    keys_with_candidates: set[str] = set()
    for batch in candidate_file.iter_batches(batch_size=batch_size, columns=["key_a", "key_b"]):
        keys_with_candidates.update(cast(list[str], batch.column(0).to_pylist()))
        keys_with_candidates.update(cast(list[str], batch.column(1).to_pylist()))

    required = {
        "resolution_key",
        "representative_name",
        "source_row_frequency",
        "representative_route",
        "possible_truncation",
        "token_count",
    }
    resolution_file = pq.ParquetFile(resolution_keys_path)
    missing = required - set(resolution_file.schema_arrow.names)
    if missing:
        raise ValueError(f"Resolution-key dataset is missing columns: {sorted(missing)}")
    zero_keys: list[AuditKey] = []
    columns = sorted(required)
    for batch in resolution_file.iter_batches(batch_size=batch_size, columns=columns):
        for row in batch.to_pylist():
            key = cast(str, row["resolution_key"])
            if key not in keys_with_candidates:
                zero_keys.append(
                    AuditKey(
                        resolution_key=key,
                        representative_name=cast(str, row["representative_name"]),
                        source_row_frequency=cast(int, row["source_row_frequency"]),
                        representative_route=cast(str, row["representative_route"]),
                        possible_truncation=cast(bool, row["possible_truncation"]),
                        token_count=cast(int, row["token_count"]),
                    )
                )
    return zero_keys


def assign_audit_stratum(item: AuditKey) -> AuditStratum:
    if item.possible_truncation:
        return "possible_truncation"
    if item.representative_route == "ambiguous_review_candidate":
        return "ambiguous_review"
    if item.source_row_frequency >= 2:
        return "high_frequency"
    if any(character.isdigit() for character in item.representative_name):
        return "numeric_name"
    if len(item.representative_name) <= 8 or item.token_count <= 1:
        return "short_name"
    if item.representative_route == "employer_resolution_candidate":
        return "ordinary_employer"
    return "other"


def sample_blocking_misses(
    zero_keys: list[AuditKey], *, sample_size: int, random_seed: int
) -> list[tuple[AuditStratum, AuditKey]]:
    """Select diagnostic zero-candidate strata, then deterministically backfill."""
    sampling_strata = AUDIT_STRATUM_PRIORITY[:-1]
    targets = _target_allocation(sample_size, cast(tuple[str, ...], sampling_strata))
    buckets: dict[AuditStratum, list[AuditKey]] = defaultdict(list)
    for item in zero_keys:
        buckets[assign_audit_stratum(item)].append(item)

    selected: list[tuple[AuditStratum, AuditKey]] = []
    selected_keys: set[str] = set()
    for stratum in sampling_strata:
        ranked = sorted(
            buckets[stratum],
            key=lambda item: (
                _rank(random_seed, stratum, item.resolution_key),
                item.resolution_key,
            ),
        )
        for item in ranked[: targets[stratum]]:
            selected.append((stratum, item))
            selected_keys.add(item.resolution_key)

    if len(selected) < sample_size:
        remaining = sorted(
            (item for item in zero_keys if item.resolution_key not in selected_keys),
            key=lambda item: (
                _rank(random_seed, "audit_backfill", item.resolution_key),
                item.resolution_key,
            ),
        )
        selected.extend(
            (assign_audit_stratum(item), item) for item in remaining[: sample_size - len(selected)]
        )
    priority = {stratum: index for index, stratum in enumerate(AUDIT_STRATUM_PRIORITY)}
    return sorted(
        selected,
        key=lambda selected_item: (
            priority[selected_item[0]],
            _rank(random_seed, selected_item[0], selected_item[1].resolution_key),
            selected_item[1].resolution_key,
        ),
    )


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def build_evaluation_sample(
    root: Path, settings: Settings, *, output_directory_override: Path | None = None
) -> EvaluationResult:
    """Build two deterministic, initially unlabeled human-review CSV datasets."""

    def product_path(configured: Path) -> Path:
        return configured if configured.is_absolute() else root / configured

    feature_path = product_path(settings.candidate_scoring.output_dataset)
    resolution_keys_path = product_path(settings.candidate_generation.resolution_keys_output)
    candidate_pairs_path = product_path(settings.candidate_generation.candidate_pairs_output)
    for required_path in (feature_path, resolution_keys_path, candidate_pairs_path):
        if not required_path.is_file():
            raise FileNotFoundError(f"Evaluation input does not exist: {required_path}")
    output_directory = output_directory_override or product_path(
        settings.evaluation.output_directory
    )
    pair_review_path = output_directory / "pair_review_sample.csv"
    blocking_miss_path = output_directory / "blocking_miss_audit.csv"

    selected_pairs = sample_review_pairs(
        feature_path,
        sample_size=settings.evaluation.pair_sample_size,
        random_seed=settings.evaluation.random_seed,
        batch_size=settings.candidate_scoring.batch_size,
    )
    pair_rows: list[dict[str, object]] = []
    for index, (pair_stratum, pair) in enumerate(selected_pairs, start=1):
        pair_rows.append(
            {
                "sample_id": f"PAIR-{index:04d}",
                "stratum": pair_stratum,
                "name_a": pair.name_a,
                "name_b": pair.name_b,
                "key_a": pair.key_a,
                "key_b": pair.key_b,
                "blocking_methods": "|".join(pair.blocking_methods),
                "blocking_method_count": pair.blocking_method_count,
                "char_ratio": pair.char_ratio,
                "token_sort_ratio": pair.token_sort_ratio,
                "token_set_ratio": pair.token_set_ratio,
                "partial_ratio": pair.partial_ratio,
                "length_ratio": pair.length_ratio,
                "common_prefix_ratio": pair.common_prefix_ratio,
                "token_jaccard": pair.token_jaccard,
                "same_first_token": pair.same_first_token,
                "numeric_relation": pair.numeric_relation,
                "review_label": "",
                "review_notes": "",
            }
        )
    _write_csv(pair_review_path, PAIR_OUTPUT_COLUMNS, pair_rows)

    zero_keys = derive_zero_candidate_keys(
        resolution_keys_path,
        candidate_pairs_path,
        batch_size=settings.candidate_scoring.batch_size,
    )
    selected_audit = sample_blocking_misses(
        zero_keys,
        sample_size=settings.evaluation.blocking_miss_sample_size,
        random_seed=settings.evaluation.random_seed,
    )
    audit_rows: list[dict[str, object]] = []
    for index, (audit_stratum, item) in enumerate(selected_audit, start=1):
        audit_rows.append(
            {
                "audit_id": f"AUDIT-{index:04d}",
                "audit_stratum": audit_stratum,
                "resolution_key": item.resolution_key,
                "representative_name": item.representative_name,
                "source_row_frequency": item.source_row_frequency,
                "representative_route": item.representative_route,
                "possible_truncation": item.possible_truncation,
                "token_count": item.token_count,
                "possible_blocking_miss": "",
                "suspected_variant": "",
                "review_notes": "",
            }
        )
    _write_csv(blocking_miss_path, AUDIT_OUTPUT_COLUMNS, audit_rows)

    pair_identities = [(row["key_a"], row["key_b"]) for row in pair_rows]
    duplicate_pairs = len(pair_identities) - len(set(pair_identities))
    nonblank_labels = sum(bool(str(row["review_label"]).strip()) for row in pair_rows)
    return EvaluationResult(
        pair_review_path=pair_review_path,
        blocking_miss_path=blocking_miss_path,
        pair_sample_size=len(pair_rows),
        pair_stratum_counts=dict(sorted(Counter(str(row["stratum"]) for row in pair_rows).items())),
        duplicate_pairs=duplicate_pairs,
        nonblank_review_labels=nonblank_labels,
        blocking_miss_sample_size=len(audit_rows),
        zero_candidate_population=len(zero_keys),
    )
