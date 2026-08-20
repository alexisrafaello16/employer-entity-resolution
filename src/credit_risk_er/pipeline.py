"""Single-pass product preprocessing pipeline."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import Counter
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from credit_risk_er.config import Settings, configuration_fingerprint
from credit_risk_er.employer_eligibility import (
    ADDRESS,
    AMBIGUOUS,
    EMPLOYER_CANDIDATE,
    NON_EMPLOYER_STATUS,
    EligibilityEvidenceAggregate,
    EligibilityStatus,
    classify_eligibility,
    finalize_evidence,
    update_evidence_aggregate,
)
from credit_risk_er.employer_eligibility import (
    RULE_PRECEDENCE as ELIGIBILITY_RULE_PRECEDENCE,
)
from credit_risk_er.employer_master import load_employer_knowledge
from credit_risk_er.ingestion import (
    deterministic_record_id,
    iter_source_batches,
    sha256_file,
    validate_source,
)
from credit_risk_er.matching.candidates import (
    CandidatePair,
    ResolutionKey,
    ResolutionKeyAggregate,
    exclusion_reason,
    finalize_resolution_keys,
    generate_candidate_pairs,
    update_resolution_key,
)
from credit_risk_er.matching.decision import (
    AUTO_SAME,
    NEEDS_FURTHER_RESOLUTION,
    RULE_PRECEDENCE,
    NumericRelation,
    decide_pair,
    exact_source_truncation_relation,
)
from credit_risk_er.matching.distinctive import (
    DistinctiveNameEvidence,
    compute_distinctive_name_evidence,
    employer_core_token_membership,
)
from credit_risk_er.matching.exact import resolve_exact
from credit_risk_er.matching.fuzzy import FEATURE_PRECISION, compute_pair_features
from credit_risk_er.matching.multi_evidence import (
    ACTIVE_ASSESSMENT_FAMILIES,
    ASSESSMENT_FAMILY_PRECEDENCE,
    MultiEvidenceAssessment,
    assess_multi_evidence,
)
from credit_risk_er.matching.orthographic import (
    NO_ORTHOGRAPHIC_EQUIVALENCE,
    NOT_ELIGIBLE_FOR_ORTHOGRAPHIC,
    NOT_ELIGIBLE_RULE,
    SINGLE_TOKEN_EDIT_EQUIVALENCE,
    STRONG_ORTHOGRAPHIC_EVIDENCE,
    EditOperation,
    OrthographicComparison,
    OrthographicDecision,
    OrthographicPolicy,
    OrthographicRule,
    OrthographicStatus,
    build_orthographic_policy,
    finalize_orthographic_comparison,
    orthographic_core_tokens,
    prepare_orthographic_pair,
)
from credit_risk_er.matching.residual_profile import (
    PRIMARY_FAMILY_PRECEDENCE,
    ResidualRelationshipProfile,
    profile_residual_relationship,
)
from credit_risk_er.models import (
    CandidateGenerationResult,
    CandidateScoringResult,
    DistinctiveEvidenceResult,
    EmployerEligibilityResult,
    JsonValue,
    MultiEvidenceAssessmentResult,
    OrthographicResolutionResult,
    PairDecisionResult,
    PreprocessResult,
    ResidualProfileResult,
    ResolutionResult,
)
from credit_risk_er.normalization import NormalizedEmployer, normalize_employer
from credit_risk_er.record_typing import RecordTyping, type_record

LOGGER = logging.getLogger(__name__)

PREPROCESSED_SCHEMA = pa.schema(
    [
        pa.field("record_id", pa.string(), nullable=False),
        pa.field("source_row_number", pa.int64(), nullable=False),
        pa.field("nombre_original", pa.string(), nullable=True),
        pa.field("nombre_normalizado", pa.string(), nullable=True),
        pa.field("nombre_matching", pa.string(), nullable=True),
        pa.field("has_trailing_numeric_token", pa.bool_(), nullable=False),
        pa.field("trailing_numeric_candidate", pa.string(), nullable=True),
        pa.field("possible_truncation", pa.bool_(), nullable=False),
        pa.field("is_blank", pa.bool_(), nullable=False),
        pa.field("is_numeric_only", pa.bool_(), nullable=False),
        pa.field("has_address_signal", pa.bool_(), nullable=False),
        pa.field("address_signal_strength", pa.string(), nullable=False),
        pa.field("has_occupation_signal", pa.bool_(), nullable=False),
        pa.field("occupation_signal_strength", pa.string(), nullable=False),
        pa.field("has_corporate_suffix", pa.bool_(), nullable=False),
        pa.field("has_organization_like_tokens", pa.bool_(), nullable=False),
        pa.field("mixed_address_organization_signal", pa.bool_(), nullable=False),
        pa.field("mixed_occupation_organization_signal", pa.bool_(), nullable=False),
        pa.field("has_activity_description_signal", pa.bool_(), nullable=False),
        pa.field("token_count", pa.int32(), nullable=False),
        pa.field("normalized_length", pa.int32(), nullable=False),
        pa.field("route", pa.string(), nullable=False),
        pa.field("route_reason", pa.string(), nullable=False),
    ]
)

RESOLVED_SCHEMA = pa.schema(
    [
        *PREPROCESSED_SCHEMA,
        pa.field("entity_id", pa.string(), nullable=True),
        pa.field("canonical_name", pa.string(), nullable=True),
        pa.field("resolution_status", pa.string(), nullable=False),
        pa.field("resolution_method", pa.string(), nullable=True),
        pa.field("resolution_reason", pa.string(), nullable=False),
    ]
)

RESOLUTION_KEYS_SCHEMA = pa.schema(
    [
        pa.field("resolution_key", pa.string(), nullable=False),
        pa.field("representative_name", pa.string(), nullable=False),
        pa.field("relaxed_key", pa.string(), nullable=True),
        pa.field("source_row_frequency", pa.int64(), nullable=False),
        pa.field("representative_record_id", pa.string(), nullable=False),
        pa.field("representative_source_row_number", pa.int64(), nullable=False),
        pa.field("representative_route", pa.string(), nullable=False),
        pa.field("trailing_numeric_candidate", pa.string(), nullable=True),
        pa.field("possible_truncation", pa.bool_(), nullable=False),
        pa.field("token_count", pa.int32(), nullable=False),
    ]
)

CANDIDATE_PAIRS_SCHEMA = pa.schema(
    [
        pa.field("key_a", pa.string(), nullable=False),
        pa.field("key_b", pa.string(), nullable=False),
        pa.field("name_a", pa.string(), nullable=False),
        pa.field("name_b", pa.string(), nullable=False),
        pa.field("blocking_methods", pa.list_(pa.string()), nullable=False),
    ]
)

CANDIDATE_FEATURES_SCHEMA = pa.schema(
    [
        *CANDIDATE_PAIRS_SCHEMA,
        pa.field("blocking_method_count", pa.int16(), nullable=False),
        pa.field("char_ratio", pa.float64(), nullable=False),
        pa.field("token_sort_ratio", pa.float64(), nullable=False),
        pa.field("token_set_ratio", pa.float64(), nullable=False),
        pa.field("partial_ratio", pa.float64(), nullable=False),
        pa.field("length_ratio", pa.float64(), nullable=False),
        pa.field("common_prefix_ratio", pa.float64(), nullable=False),
        pa.field("token_jaccard", pa.float64(), nullable=False),
        pa.field("same_first_token", pa.bool_(), nullable=False),
        pa.field("numeric_relation", pa.string(), nullable=False),
    ]
)

PAIR_DECISIONS_SCHEMA = pa.schema(
    [
        *CANDIDATE_PAIRS_SCHEMA,
        pa.field("decision_status", pa.string(), nullable=False),
        pa.field("decision_rule", pa.string(), nullable=False),
        pa.field("decision_evidence", pa.string(), nullable=False),
    ]
)

EMPLOYER_ELIGIBILITY_SCHEMA = pa.schema(
    [
        pa.field("resolution_key", pa.string(), nullable=False),
        pa.field("representative_name", pa.string(), nullable=False),
        pa.field("eligibility_status", pa.string(), nullable=False),
        pa.field("eligibility_rule", pa.string(), nullable=False),
        pa.field("eligibility_evidence", pa.string(), nullable=False),
    ]
)

ORTHOGRAPHIC_PAIR_DECISIONS_SCHEMA = pa.schema(
    [
        *CANDIDATE_PAIRS_SCHEMA,
        pa.field("orthographic_status", pa.string(), nullable=False),
        pa.field("orthographic_rule", pa.string(), nullable=False),
        pa.field("orthographic_evidence", pa.string(), nullable=False),
        pa.field("differing_token_a", pa.string(), nullable=True),
        pa.field("differing_token_b", pa.string(), nullable=True),
        pa.field("edit_operation", pa.string(), nullable=True),
        pa.field("context_signature", pa.string(), nullable=True),
        pa.field("context_variant_count", pa.int32(), nullable=True),
        pa.field("token_support_a", pa.int64(), nullable=True),
        pa.field("token_support_b", pa.int64(), nullable=True),
    ]
)

RESIDUAL_RELATIONSHIP_PROFILE_SCHEMA = pa.schema(
    [
        pa.field("key_a", pa.string(), nullable=False),
        pa.field("key_b", pa.string(), nullable=False),
        pa.field("name_a", pa.string(), nullable=False),
        pa.field("name_b", pa.string(), nullable=False),
        pa.field("primary_family", pa.string(), nullable=False),
        pa.field("family_evidence", pa.string(), nullable=False),
        pa.field("core_token_count_a", pa.int32(), nullable=False),
        pa.field("core_token_count_b", pa.int32(), nullable=False),
        pa.field("shared_exact_token_count", pa.int32(), nullable=False),
        pa.field("differing_token_positions", pa.string(), nullable=True),
        pa.field("differing_token_count", pa.int32(), nullable=False),
        pa.field("added_removed_token", pa.string(), nullable=True),
        pa.field("added_removed_token_count", pa.int32(), nullable=False),
        pa.field("maximum_token_edit_distance", pa.int32(), nullable=True),
        pa.field("total_token_edit_distance", pa.int32(), nullable=True),
        pa.field("is_token_reorder", pa.bool_(), nullable=False),
        pa.field("is_ordered_subsequence", pa.bool_(), nullable=False),
        pa.field("is_token_multiset_containment", pa.bool_(), nullable=False),
        pa.field("has_initialism_pattern", pa.bool_(), nullable=False),
        pa.field("possible_truncation_a", pa.bool_(), nullable=False),
        pa.field("possible_truncation_b", pa.bool_(), nullable=False),
        pa.field("numeric_relation", pa.string(), nullable=False),
        pa.field("prior_orthographic_evidence", pa.string(), nullable=False),
    ]
)

DISTINCTIVE_NAME_EVIDENCE_SCHEMA = pa.schema(
    [
        pa.field("key_a", pa.string(), nullable=False),
        pa.field("key_b", pa.string(), nullable=False),
        pa.field("name_a", pa.string(), nullable=False),
        pa.field("name_b", pa.string(), nullable=False),
        pa.field("shared_exact_tokens", pa.string(), nullable=False),
        pa.field("shared_distinctive_tokens", pa.string(), nullable=False),
        pa.field("shared_exact_token_count", pa.int32(), nullable=False),
        pa.field("shared_distinctive_token_count", pa.int32(), nullable=False),
        pa.field("shared_generic_token_count", pa.int32(), nullable=False),
        pa.field("minimum_shared_token_support", pa.int64(), nullable=True),
        pa.field("maximum_shared_token_support", pa.int64(), nullable=True),
        pa.field("minimum_distinctive_token_support", pa.int64(), nullable=True),
        pa.field("maximum_distinctive_token_support", pa.int64(), nullable=True),
        pa.field("exact_coverage_a", pa.float64(), nullable=False),
        pa.field("exact_coverage_b", pa.float64(), nullable=False),
        pa.field("distinctive_coverage_a", pa.float64(), nullable=False),
        pa.field("distinctive_coverage_b", pa.float64(), nullable=False),
        pa.field("shorter_name_exact_coverage", pa.float64(), nullable=False),
        pa.field("longer_name_exact_coverage", pa.float64(), nullable=False),
        pa.field("shorter_name_distinctive_coverage", pa.float64(), nullable=False),
        pa.field("longer_name_distinctive_coverage", pa.float64(), nullable=False),
        pa.field("has_shared_exact_token", pa.bool_(), nullable=False),
        pa.field("has_shared_distinctive_token", pa.bool_(), nullable=False),
        pa.field("has_multiple_shared_distinctive_tokens", pa.bool_(), nullable=False),
        pa.field(
            "has_exact_overlap_without_distinctive_token",
            pa.bool_(),
            nullable=False,
        ),
        pa.field("primary_family", pa.string(), nullable=False),
        pa.field("numeric_relation", pa.string(), nullable=False),
    ]
)

MULTI_EVIDENCE_ASSESSMENT_SCHEMA = pa.schema(
    [
        pa.field("key_a", pa.string(), nullable=False),
        pa.field("key_b", pa.string(), nullable=False),
        pa.field("name_a", pa.string(), nullable=False),
        pa.field("name_b", pa.string(), nullable=False),
        pa.field("assessment_family", pa.string(), nullable=False),
        pa.field("assessment_evidence", pa.string(), nullable=False),
        pa.field("primary_family", pa.string(), nullable=False),
        pa.field("numeric_relation", pa.string(), nullable=False),
        pa.field("has_structural_exact_relation", pa.bool_(), nullable=False),
        pa.field("has_orthographic_signal", pa.bool_(), nullable=False),
        pa.field("has_distinctive_exact_overlap", pa.bool_(), nullable=False),
        pa.field("has_multiple_distinctive_exact_overlap", pa.bool_(), nullable=False),
        pa.field("has_numeric_risk", pa.bool_(), nullable=False),
        pa.field("has_zero_exact_overlap", pa.bool_(), nullable=False),
        pa.field("has_full_shorter_exact_coverage", pa.bool_(), nullable=False),
        pa.field("has_full_shorter_distinctive_coverage", pa.bool_(), nullable=False),
        pa.field("core_token_count_a", pa.int32(), nullable=False),
        pa.field("core_token_count_b", pa.int32(), nullable=False),
        pa.field("shared_exact_token_count", pa.int32(), nullable=False),
        pa.field("shared_distinctive_token_count", pa.int32(), nullable=False),
        pa.field("minimum_distinctive_token_support", pa.int64(), nullable=True),
        pa.field("maximum_distinctive_token_support", pa.int64(), nullable=True),
        pa.field("exact_coverage_a", pa.float64(), nullable=False),
        pa.field("exact_coverage_b", pa.float64(), nullable=False),
        pa.field("distinctive_coverage_a", pa.float64(), nullable=False),
        pa.field("distinctive_coverage_b", pa.float64(), nullable=False),
        pa.field("shorter_name_exact_coverage", pa.float64(), nullable=False),
        pa.field("longer_name_exact_coverage", pa.float64(), nullable=False),
        pa.field("shorter_name_distinctive_coverage", pa.float64(), nullable=False),
        pa.field("longer_name_distinctive_coverage", pa.float64(), nullable=False),
        pa.field("differing_token_count", pa.int32(), nullable=False),
        pa.field("maximum_token_edit_distance", pa.int32(), nullable=True),
        pa.field("total_token_edit_distance", pa.int32(), nullable=True),
        pa.field("is_token_reorder", pa.bool_(), nullable=False),
        pa.field("is_ordered_subsequence", pa.bool_(), nullable=False),
        pa.field("is_token_multiset_containment", pa.bool_(), nullable=False),
        pa.field("possible_truncation_a", pa.bool_(), nullable=False),
        pa.field("possible_truncation_b", pa.bool_(), nullable=False),
        pa.field("has_shared_distinctive_token", pa.bool_(), nullable=False),
        pa.field("has_multiple_shared_distinctive_tokens", pa.bool_(), nullable=False),
        pa.field(
            "has_exact_overlap_without_distinctive_token",
            pa.bool_(),
            nullable=False,
        ),
        pa.field("char_ratio", pa.float64(), nullable=False),
        pa.field("token_sort_ratio", pa.float64(), nullable=False),
        pa.field("token_set_ratio", pa.float64(), nullable=False),
        pa.field("partial_ratio", pa.float64(), nullable=False),
        pa.field("length_ratio", pa.float64(), nullable=False),
        pa.field("common_prefix_ratio", pa.float64(), nullable=False),
        pa.field("token_jaccard", pa.float64(), nullable=False),
        pa.field("same_first_token", pa.bool_(), nullable=False),
        pa.field("prior_orthographic_evidence", pa.string(), nullable=False),
    ]
)

_CANDIDATE_IDENTITY_COLUMNS = ("key_a", "key_b", "name_a", "name_b", "blocking_methods")
_PAIR_DECISION_INPUT_COLUMNS = (*_CANDIDATE_IDENTITY_COLUMNS, "numeric_relation")
_NUMERIC_FEATURE_COLUMNS = (
    "char_ratio",
    "token_sort_ratio",
    "token_set_ratio",
    "partial_ratio",
    "length_ratio",
    "common_prefix_ratio",
    "token_jaccard",
)
_ELIGIBILITY_STATUSES: tuple[EligibilityStatus, ...] = (
    EMPLOYER_CANDIDATE,
    ADDRESS,
    NON_EMPLOYER_STATUS,
    AMBIGUOUS,
)
_ELIGIBILITY_PREPROCESSING_COLUMNS = (
    "nombre_normalizado",
    "route",
    "is_blank",
    "is_numeric_only",
    "has_address_signal",
    "address_signal_strength",
    "has_occupation_signal",
    "occupation_signal_strength",
    "has_activity_description_signal",
    "has_corporate_suffix",
    "has_organization_like_tokens",
    "mixed_address_organization_signal",
    "mixed_occupation_organization_signal",
)
_ORTHOGRAPHIC_DECISION_INPUT_COLUMNS = (
    *_CANDIDATE_IDENTITY_COLUMNS,
    "decision_status",
)


def _write_json(path: Path, payload: dict[str, JsonValue]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _processing_fingerprint(source_sha256: str, config_sha256: str, settings: Settings) -> str:
    payload = (
        f"preprocess-v1|{source_sha256}|{config_sha256}|"
        f"{settings.normalization.ruleset_version}|{settings.record_typing.ruleset_version}"
    ).encode()
    return hashlib.sha256(payload).hexdigest().upper()


def _row_payload(
    *,
    source_sha256: str,
    sheet_name: str,
    source_row_number: int,
    nombre_original: str | None,
    normalized: NormalizedEmployer,
    typed: RecordTyping,
) -> dict[str, object]:
    return {
        "record_id": deterministic_record_id(source_sha256, sheet_name, source_row_number),
        "source_row_number": source_row_number,
        "nombre_original": nombre_original,
        "nombre_normalizado": normalized.strict,
        "nombre_matching": normalized.relaxed,
        "has_trailing_numeric_token": normalized.has_trailing_numeric_token,
        "trailing_numeric_candidate": normalized.trailing_numeric_candidate,
        "possible_truncation": normalized.possible_truncation,
        "is_blank": typed.is_blank,
        "is_numeric_only": typed.is_numeric_only,
        "has_address_signal": typed.has_address_signal,
        "address_signal_strength": typed.address_signal_strength,
        "has_occupation_signal": typed.has_occupation_signal,
        "occupation_signal_strength": typed.occupation_signal_strength,
        "has_corporate_suffix": typed.has_corporate_suffix,
        "has_organization_like_tokens": typed.has_organization_like_tokens,
        "mixed_address_organization_signal": typed.mixed_address_organization_signal,
        "mixed_occupation_organization_signal": typed.mixed_occupation_organization_signal,
        "has_activity_description_signal": typed.has_activity_description_signal,
        "token_count": typed.token_count,
        "normalized_length": typed.normalized_length,
        "route": typed.route,
        "route_reason": typed.route_reason,
    }


def _value_digest_payload(source_row_number: int, value: str | None) -> bytes:
    row_bytes = source_row_number.to_bytes(8, byteorder="big", signed=False)
    if value is None:
        return row_bytes + b"N"
    encoded = value.encode("utf-8")
    return row_bytes + b"S" + len(encoded).to_bytes(8, byteorder="big") + encoded


def _lineage_digest_payload(
    record_id: str, source_row_number: int, nombre_original: str | None, route: str
) -> bytes:
    fields = (
        record_id.encode("utf-8"),
        _value_digest_payload(source_row_number, nombre_original),
        route.encode("utf-8"),
    )
    return b"".join(len(field).to_bytes(8, byteorder="big") + field for field in fields)


def _reconcile_persisted_output(
    path: Path, *, source_sha256: str, sheet_name: str, expected_rows: int, source_digest: str
) -> None:
    parquet = pq.ParquetFile(path)
    if parquet.metadata.num_rows != expected_rows or parquet.schema_arrow != PREPROCESSED_SCHEMA:
        raise RuntimeError("Persisted Parquet metadata failed reconciliation")

    persisted_digest = hashlib.sha256()
    expected_source_row = 2
    for batch in parquet.iter_batches(
        columns=["record_id", "source_row_number", "nombre_original"]
    ):
        for row in batch.to_pylist():
            source_row_number = row["source_row_number"]
            if source_row_number != expected_source_row:
                raise RuntimeError(
                    f"Persisted row order failed at source row {expected_source_row}"
                )
            expected_id = deterministic_record_id(source_sha256, sheet_name, source_row_number)
            if row["record_id"] != expected_id:
                raise RuntimeError(f"Persisted record ID failed at source row {source_row_number}")
            persisted_digest.update(
                _value_digest_payload(source_row_number, row["nombre_original"])
            )
            expected_source_row += 1
    if persisted_digest.hexdigest() != source_digest:
        raise RuntimeError("Persisted original values failed ordered digest reconciliation")


def preprocess(
    root: Path,
    settings: Settings,
    *,
    source_override: Path | None = None,
    output_override: Path | None = None,
) -> PreprocessResult:
    """Validate, normalize, type, and persist the compact internal dataset."""
    total_started = time.perf_counter()
    source_path = source_override or settings.source.workbook
    output_path = output_override or settings.processing.output_dataset
    if not source_path.is_absolute():
        source_path = root / source_path
    if not output_path.is_absolute():
        output_path = root / output_path
    metrics_path = settings.processing.metrics_file
    manifest_path = settings.processing.manifest_file
    if not metrics_path.is_absolute():
        metrics_path = root / metrics_path
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path

    LOGGER.info("Validating immutable source workbook")
    validation_started = time.perf_counter()
    source = validate_source(source_path, settings.source)
    validation_seconds = time.perf_counter() - validation_started
    config_sha256 = configuration_fingerprint(
        settings, source_path=source_path, output_path=output_path
    )
    processing_fingerprint = _processing_fingerprint(source.sha256, config_sha256, settings)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_suffix(f"{output_path.suffix}.tmp")
    temporary_output.unlink(missing_ok=True)
    writer: pq.ParquetWriter | None = None
    row_count = 0
    null_count = 0
    first_source_row: int | None = None
    last_source_row: int | None = None
    route_counts: Counter[str] = Counter()
    signal_counts: Counter[str] = Counter()
    address_strength_counts: Counter[str] = Counter()
    occupation_strength_counts: Counter[str] = Counter()
    read_seconds = 0.0
    normalization_seconds = 0.0
    typing_seconds = 0.0
    write_seconds = 0.0
    source_values_digest = hashlib.sha256()

    compression = (
        None
        if settings.processing.parquet_compression == "none"
        else settings.processing.parquet_compression
    )
    LOGGER.info("Preprocessing %s source rows in bounded batches", source.data_rows)
    batches = iter_source_batches(source, settings.processing.batch_size)
    try:
        writer = pq.ParquetWriter(
            temporary_output,
            PREPROCESSED_SCHEMA,
            compression=compression,
            use_dictionary=True,
            write_statistics=True,
        )
        while True:
            read_started = time.perf_counter()
            try:
                batch = next(batches)
            except StopIteration:
                read_seconds += time.perf_counter() - read_started
                break
            read_seconds += time.perf_counter() - read_started

            normalization_started = time.perf_counter()
            normalized_batch = [
                normalize_employer(row.nombre_original, settings.normalization) for row in batch
            ]
            normalization_seconds += time.perf_counter() - normalization_started

            typing_started = time.perf_counter()
            typed_batch = [
                type_record(item.strict, item.relaxed, settings.record_typing)
                for item in normalized_batch
            ]
            typing_seconds += time.perf_counter() - typing_started

            rows = [
                _row_payload(
                    source_sha256=source.sha256,
                    sheet_name=source.sheet_name,
                    source_row_number=row.source_row_number,
                    nombre_original=row.nombre_original,
                    normalized=normalized,
                    typed=typed,
                )
                for row, normalized, typed in zip(batch, normalized_batch, typed_batch, strict=True)
            ]
            write_started = time.perf_counter()
            writer.write_table(pa.Table.from_pylist(rows, schema=PREPROCESSED_SCHEMA))
            write_seconds += time.perf_counter() - write_started

            if first_source_row is None:
                first_source_row = batch[0].source_row_number
            last_source_row = batch[-1].source_row_number
            row_count += len(batch)
            null_count += sum(row.nombre_original is None for row in batch)
            for row in batch:
                source_values_digest.update(
                    _value_digest_payload(row.source_row_number, row.nombre_original)
                )
            for normalized, typed in zip(normalized_batch, typed_batch, strict=True):
                route_counts[typed.route] += 1
                address_strength_counts[typed.address_signal_strength] += 1
                occupation_strength_counts[typed.occupation_signal_strength] += 1
                for key, present in (
                    ("blank", typed.is_blank),
                    ("numeric_only", typed.is_numeric_only),
                    ("address", typed.has_address_signal),
                    ("occupation", typed.has_occupation_signal),
                    ("corporate_suffix", typed.has_corporate_suffix),
                    ("organization", typed.has_organization_like_tokens),
                    ("mixed_address_organization", typed.mixed_address_organization_signal),
                    ("mixed_occupation_organization", typed.mixed_occupation_organization_signal),
                    ("activity_description", typed.has_activity_description_signal),
                    ("trailing_numeric_token", normalized.has_trailing_numeric_token),
                    (
                        "trailing_numeric_candidate",
                        normalized.trailing_numeric_candidate is not None,
                    ),
                    ("possible_truncation", normalized.possible_truncation),
                ):
                    signal_counts[key] += int(present)
    except Exception:
        if writer is not None:
            writer.close()
            writer = None
        temporary_output.unlink(missing_ok=True)
        raise
    finally:
        if writer is not None:
            writer.close()

    if row_count != source.data_rows or first_source_row != 2 or last_source_row != source.max_row:
        temporary_output.unlink(missing_ok=True)
        raise RuntimeError(
            "Row reconciliation failed: "
            f"expected {source.data_rows} rows at 2..{source.max_row}, "
            f"observed {row_count} rows at {first_source_row}..{last_source_row}"
        )
    reconciliation_started = time.perf_counter()
    try:
        _reconcile_persisted_output(
            temporary_output,
            source_sha256=source.sha256,
            sheet_name=source.sheet_name,
            expected_rows=row_count,
            source_digest=source_values_digest.hexdigest(),
        )
    except Exception:
        temporary_output.unlink(missing_ok=True)
        raise
    reconciliation_seconds = time.perf_counter() - reconciliation_started
    temporary_output.replace(output_path)

    elapsed_seconds = time.perf_counter() - total_started
    timings: dict[str, JsonValue] = {
        "source_validation_seconds": round(validation_seconds, 6),
        "source_read_seconds": round(read_seconds, 6),
        "normalization_seconds": round(normalization_seconds, 6),
        "record_typing_seconds": round(typing_seconds, 6),
        "parquet_write_seconds": round(write_seconds, 6),
        "reconciliation_seconds": round(reconciliation_seconds, 6),
        "total_seconds": round(elapsed_seconds, 6),
    }
    metrics: dict[str, JsonValue] = {
        "row_counts": {
            "source": source.data_rows,
            "written": row_count,
            "null_source_values": null_count,
        },
        "route_counts": dict(sorted(route_counts.items())),
        "signal_counts": dict(sorted(signal_counts.items())),
        "signal_strength_counts": {
            "address": dict(sorted(address_strength_counts.items())),
            "occupation": dict(sorted(occupation_strength_counts.items())),
        },
        "execution_timing": timings,
    }
    _write_json(metrics_path, metrics)

    output_sha256 = sha256_file(output_path)
    manifest: dict[str, JsonValue] = {
        "processing_fingerprint": processing_fingerprint,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "source": {
            "path": str(source.path),
            "sha256": source.sha256,
            "size_bytes": source.size_bytes,
        },
        "configuration_sha256": config_sha256,
        "ruleset_versions": {
            "normalization": settings.normalization.ruleset_version,
            "record_typing": settings.record_typing.ruleset_version,
        },
        "row_counts": {"source": source.data_rows, "written": row_count},
        "execution_timing": timings,
        "output": {
            "path": str(output_path.resolve()),
            "sha256": output_sha256,
            "size_bytes": output_path.stat().st_size,
        },
        "reconciliation": {
            "row_count_matches": True,
            "row_order_preserved": True,
            "original_values_preserved": True,
            "record_ids_deterministic": True,
        },
    }
    _write_json(manifest_path, manifest)
    LOGGER.info("Wrote %s rows to %s", row_count, output_path)
    return PreprocessResult(
        output_path=output_path,
        metrics_path=metrics_path,
        manifest_path=manifest_path,
        row_count=row_count,
        processing_fingerprint=processing_fingerprint,
        elapsed_seconds=elapsed_seconds,
    )


def _reconcile_resolved_output(
    path: Path, *, expected_rows: int, input_lineage_digest: str
) -> None:
    parquet = pq.ParquetFile(path)
    if parquet.metadata.num_rows != expected_rows or parquet.schema_arrow != RESOLVED_SCHEMA:
        raise RuntimeError("Resolved Parquet metadata failed reconciliation")
    output_digest = hashlib.sha256()
    for batch in parquet.iter_batches(
        columns=["record_id", "source_row_number", "nombre_original", "route"]
    ):
        record_ids = batch.column("record_id").to_pylist()
        source_rows = batch.column("source_row_number").to_pylist()
        original_names = batch.column("nombre_original").to_pylist()
        routes = batch.column("route").to_pylist()
        for record_id, source_row, original_name, route in zip(
            record_ids, source_rows, original_names, routes, strict=True
        ):
            output_digest.update(
                _lineage_digest_payload(record_id, source_row, original_name, route)
            )
    if output_digest.hexdigest() != input_lineage_digest:
        raise RuntimeError("Resolved output failed preprocessing-lineage reconciliation")


def resolve_employers(
    root: Path,
    settings: Settings,
    *,
    input_override: Path | None = None,
    output_override: Path | None = None,
    master_override: Path | None = None,
    aliases_override: Path | None = None,
) -> ResolutionResult:
    """Resolve strict normalized names against validated canonical and alias knowledge."""
    total_started = time.perf_counter()

    def product_path(override: Path | None, configured: Path) -> Path:
        path = override or configured
        return path if path.is_absolute() else root / path

    input_path = product_path(input_override, settings.processing.output_dataset)
    output_path = product_path(output_override, settings.resolution.output_dataset)
    master_path = product_path(master_override, settings.reference_data.employer_master)
    aliases_path = product_path(aliases_override, settings.reference_data.employer_aliases)
    metrics_path = product_path(None, settings.resolution.metrics_output)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Resolved output must not overwrite the preprocessed input dataset")

    reference_started = time.perf_counter()
    knowledge = load_employer_knowledge(master_path, aliases_path, settings.normalization)
    reference_seconds = time.perf_counter() - reference_started

    if not input_path.is_file():
        raise FileNotFoundError(f"Preprocessed dataset does not exist: {input_path}")
    input_parquet = pq.ParquetFile(input_path)
    if input_parquet.schema_arrow != PREPROCESSED_SCHEMA:
        raise ValueError(
            f"Preprocessed schema mismatch for {input_path}; run preprocess with this version"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_suffix(f"{output_path.suffix}.tmp")
    temporary_output.unlink(missing_ok=True)
    writer: pq.ParquetWriter | None = None
    row_count = 0
    resolved_count = 0
    canonical_count = 0
    alias_count = 0
    exact_match_seconds = 0.0
    write_seconds = 0.0
    input_digest = hashlib.sha256()
    compression = (
        None
        if settings.processing.parquet_compression == "none"
        else settings.processing.parquet_compression
    )
    LOGGER.info(
        "Resolving %s rows against %s entities and %s validated aliases",
        input_parquet.metadata.num_rows,
        knowledge.entity_count,
        knowledge.alias_count,
    )
    try:
        writer = pq.ParquetWriter(
            temporary_output,
            RESOLVED_SCHEMA,
            compression=compression,
            use_dictionary=True,
            write_statistics=True,
        )
        for batch in input_parquet.iter_batches(batch_size=settings.processing.batch_size):
            strict_names = batch.column("nombre_normalizado").to_pylist()
            matching_started = time.perf_counter()
            resolutions = [
                resolve_exact(strict_name, knowledge.exact_index) for strict_name in strict_names
            ]
            exact_match_seconds += time.perf_counter() - matching_started

            output_batch = pa.RecordBatch.from_arrays(
                [
                    *batch.columns,
                    pa.array([item.entity_id for item in resolutions], type=pa.string()),
                    pa.array([item.canonical_name for item in resolutions], type=pa.string()),
                    pa.array([item.resolution_status for item in resolutions], type=pa.string()),
                    pa.array([item.resolution_method for item in resolutions], type=pa.string()),
                    pa.array([item.resolution_reason for item in resolutions], type=pa.string()),
                ],
                schema=RESOLVED_SCHEMA,
            )
            write_started = time.perf_counter()
            writer.write_batch(output_batch)
            write_seconds += time.perf_counter() - write_started

            record_ids = batch.column("record_id").to_pylist()
            source_rows = batch.column("source_row_number").to_pylist()
            original_names = batch.column("nombre_original").to_pylist()
            routes = batch.column("route").to_pylist()
            for record_id, source_row, original_name, route in zip(
                record_ids, source_rows, original_names, routes, strict=True
            ):
                input_digest.update(
                    _lineage_digest_payload(record_id, source_row, original_name, route)
                )
            row_count += batch.num_rows
            resolved_count += sum(item.resolution_status == "resolved" for item in resolutions)
            canonical_count += sum(
                item.resolution_method == "exact_canonical" for item in resolutions
            )
            alias_count += sum(item.resolution_method == "exact_alias" for item in resolutions)
    except Exception:
        if writer is not None:
            writer.close()
            writer = None
        temporary_output.unlink(missing_ok=True)
        raise
    finally:
        if writer is not None:
            writer.close()

    if row_count != input_parquet.metadata.num_rows:
        temporary_output.unlink(missing_ok=True)
        raise RuntimeError(
            f"Resolution row reconciliation failed: read {row_count}, "
            f"expected {input_parquet.metadata.num_rows}"
        )
    reconciliation_started = time.perf_counter()
    try:
        _reconcile_resolved_output(
            temporary_output,
            expected_rows=row_count,
            input_lineage_digest=input_digest.hexdigest(),
        )
    except Exception:
        temporary_output.unlink(missing_ok=True)
        raise
    reconciliation_seconds = time.perf_counter() - reconciliation_started
    temporary_output.replace(output_path)

    unresolved_count = row_count - resolved_count
    elapsed_seconds = time.perf_counter() - total_started
    metrics: dict[str, JsonValue] = {
        "rows_processed": row_count,
        "resolution_counts": {
            "resolved_exact": resolved_count,
            "unresolved": unresolved_count,
            "exact_canonical": canonical_count,
            "exact_alias": alias_count,
        },
        "reference_data": {
            "employer_entities": knowledge.entity_count,
            "validated_aliases": knowledge.alias_count,
        },
        "reconciliation": {
            "input_rows": input_parquet.metadata.num_rows,
            "output_rows": row_count,
            "row_count_matches": True,
            "row_order_preserved": True,
            "original_values_preserved": True,
            "preprocessing_route_preserved": True,
        },
        "execution_timing": {
            "reference_load_seconds": round(reference_seconds, 6),
            "exact_matching_seconds": round(exact_match_seconds, 6),
            "parquet_write_seconds": round(write_seconds, 6),
            "reconciliation_seconds": round(reconciliation_seconds, 6),
            "total_seconds": round(elapsed_seconds, 6),
        },
    }
    _write_json(metrics_path, metrics)
    LOGGER.info(
        "Wrote %s resolved and %s unresolved rows to %s",
        resolved_count,
        unresolved_count,
        output_path,
    )
    return ResolutionResult(
        output_path=output_path,
        metrics_path=metrics_path,
        row_count=row_count,
        resolved_count=resolved_count,
        unresolved_count=unresolved_count,
        entity_count=knowledge.entity_count,
        alias_count=knowledge.alias_count,
        elapsed_seconds=elapsed_seconds,
    )


def _write_resolution_keys(
    path: Path,
    resolution_keys: tuple[ResolutionKey, ...],
    *,
    batch_size: int,
    compression: str | None,
) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.unlink(missing_ok=True)
    writer: pq.ParquetWriter | None = None
    try:
        writer = pq.ParquetWriter(
            temporary,
            RESOLUTION_KEYS_SCHEMA,
            compression=compression,
            use_dictionary=True,
            write_statistics=True,
        )
        for start in range(0, len(resolution_keys), batch_size):
            rows = [
                {
                    "resolution_key": item.resolution_key,
                    "representative_name": item.representative_name,
                    "relaxed_key": item.relaxed_key,
                    "source_row_frequency": item.source_row_frequency,
                    "representative_record_id": item.representative_record_id,
                    "representative_source_row_number": item.representative_source_row_number,
                    "representative_route": item.representative_route,
                    "trailing_numeric_candidate": item.trailing_numeric_candidate,
                    "possible_truncation": item.possible_truncation,
                    "token_count": item.token_count,
                }
                for item in resolution_keys[start : start + batch_size]
            ]
            writer.write_table(pa.Table.from_pylist(rows, schema=RESOLUTION_KEYS_SCHEMA))
    except Exception:
        if writer is not None:
            writer.close()
            writer = None
        temporary.unlink(missing_ok=True)
        raise
    finally:
        if writer is not None:
            writer.close()
    temporary.replace(path)


def _write_candidate_pairs(
    path: Path,
    pairs: tuple[CandidatePair, ...],
    *,
    batch_size: int,
    compression: str | None,
) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.unlink(missing_ok=True)
    writer: pq.ParquetWriter | None = None
    try:
        writer = pq.ParquetWriter(
            temporary,
            CANDIDATE_PAIRS_SCHEMA,
            compression=compression,
            use_dictionary=True,
            write_statistics=True,
        )
        for start in range(0, len(pairs), batch_size):
            rows = [
                {
                    "key_a": item.key_a,
                    "key_b": item.key_b,
                    "name_a": item.name_a,
                    "name_b": item.name_b,
                    "blocking_methods": list(item.blocking_methods),
                }
                for item in pairs[start : start + batch_size]
            ]
            writer.write_table(pa.Table.from_pylist(rows, schema=CANDIDATE_PAIRS_SCHEMA))
    except Exception:
        if writer is not None:
            writer.close()
            writer = None
        temporary.unlink(missing_ok=True)
        raise
    finally:
        if writer is not None:
            writer.close()
    temporary.replace(path)


def generate_candidates(
    root: Path,
    settings: Settings,
    *,
    input_override: Path | None = None,
    keys_output_override: Path | None = None,
    pairs_output_override: Path | None = None,
) -> CandidateGenerationResult:
    """Build unique unresolved keys and bounded candidate pairs without identity decisions."""
    total_started = time.perf_counter()

    def product_path(override: Path | None, configured: Path) -> Path:
        path = override or configured
        return path if path.is_absolute() else root / path

    input_path = product_path(input_override, settings.resolution.output_dataset)
    keys_path = product_path(
        keys_output_override, settings.candidate_generation.resolution_keys_output
    )
    pairs_path = product_path(
        pairs_output_override, settings.candidate_generation.candidate_pairs_output
    )
    metrics_path = product_path(None, settings.candidate_generation.metrics_output)
    output_paths = {keys_path.resolve(), pairs_path.resolve()}
    if input_path.resolve() in output_paths or len(output_paths) != 2:
        raise ValueError(
            "Candidate outputs must be distinct and must not overwrite resolution input"
        )
    if not input_path.is_file():
        raise FileNotFoundError(f"Resolved dataset does not exist: {input_path}")

    resolved = pq.ParquetFile(input_path)
    if resolved.schema_arrow != RESOLVED_SCHEMA:
        raise ValueError(
            f"Resolved schema mismatch for {input_path}; run resolve with this version"
        )

    universe_started = time.perf_counter()
    aggregates: dict[str, ResolutionKeyAggregate] = {}
    excluded_counts: Counter[str] = Counter()
    unresolved_rows = 0
    eligible_rows = 0
    columns = [
        "record_id",
        "source_row_number",
        "nombre_normalizado",
        "nombre_matching",
        "trailing_numeric_candidate",
        "possible_truncation",
        "is_blank",
        "is_numeric_only",
        "token_count",
        "route",
        "resolution_status",
    ]
    for batch in resolved.iter_batches(batch_size=settings.processing.batch_size, columns=columns):
        for row in batch.to_pylist():
            unresolved_rows += int(row["resolution_status"] == "unresolved")
            reason = exclusion_reason(row)
            if reason is not None:
                excluded_counts[reason] += 1
                continue
            eligible_rows += 1
            update_resolution_key(aggregates, row)
    resolution_keys = finalize_resolution_keys(aggregates)
    universe_seconds = time.perf_counter() - universe_started

    generation_started = time.perf_counter()
    generated = generate_candidate_pairs(resolution_keys, settings.candidate_generation)
    generation_seconds = time.perf_counter() - generation_started

    keys_path.parent.mkdir(parents=True, exist_ok=True)
    pairs_path.parent.mkdir(parents=True, exist_ok=True)
    compression = (
        None
        if settings.processing.parquet_compression == "none"
        else settings.processing.parquet_compression
    )
    write_started = time.perf_counter()
    _write_resolution_keys(
        keys_path,
        resolution_keys,
        batch_size=settings.processing.batch_size,
        compression=compression,
    )
    _write_candidate_pairs(
        pairs_path,
        generated.pairs,
        batch_size=settings.processing.batch_size,
        compression=compression,
    )
    write_seconds = time.perf_counter() - write_started

    keys_metadata = pq.ParquetFile(keys_path)
    pairs_metadata = pq.ParquetFile(pairs_path)
    if (
        keys_metadata.schema_arrow != RESOLUTION_KEYS_SCHEMA
        or keys_metadata.metadata.num_rows != len(resolution_keys)
    ):
        raise RuntimeError("Resolution-key output failed reconciliation")
    if (
        pairs_metadata.schema_arrow != CANDIDATE_PAIRS_SCHEMA
        or pairs_metadata.metadata.num_rows != len(generated.pairs)
    ):
        raise RuntimeError("Candidate-pair output failed reconciliation")

    key_count = len(resolution_keys)
    pair_count = len(generated.pairs)
    theoretical_pairs = key_count * (key_count - 1) // 2
    reduction = 1.0 - (pair_count / theoretical_pairs) if theoretical_pairs else 1.0
    elapsed_seconds = time.perf_counter() - total_started
    metrics: dict[str, JsonValue] = {
        "input_rows": resolved.metadata.num_rows,
        "unresolved_source_rows_considered": unresolved_rows,
        "eligible_source_rows": eligible_rows,
        "excluded_rows": dict(sorted(excluded_counts.items())),
        "unique_resolution_keys": key_count,
        "candidate_pairs": pair_count,
        "candidate_pairs_by_method": cast(JsonValue, generated.method_pair_counts),
        "multi_method_pairs": generated.multi_method_pairs,
        "typo_fallback": {
            "zero_candidate_keys_before": generated.zero_candidate_keys_before_typo,
            "zero_candidate_keys_recovered": generated.zero_candidate_keys_recovered_by_typo,
            "zero_candidate_keys_remaining": generated.zero_candidate_keys_remaining_after_typo,
            "candidate_pairs_added": generated.typo_candidate_pairs_added,
            "signatures_considered": generated.typo_signatures_considered,
            "broad_signatures_skipped": generated.broad_typo_signatures_skipped,
            "pairs_containing_typo_token": generated.method_pair_counts["typo_token"],
            "multi_method_pairs_after_fallback": generated.multi_method_pairs_after_typo,
        },
        "typo_context_fallback": {
            "zero_candidate_keys_before": generated.zero_candidate_keys_before_context,
            "zero_candidate_keys_recovered": generated.zero_candidate_keys_recovered_by_context,
            "zero_candidate_keys_remaining": generated.zero_candidate_keys_remaining_after_context,
            "candidate_pairs_added": generated.typo_context_pairs_added,
            "context_lookups_performed": generated.context_lookups_performed,
            "broad_context_tokens_skipped": generated.broad_context_tokens_skipped,
            "pairs_containing_typo_context": generated.method_pair_counts["typo_context"],
        },
        "theoretical_unique_key_pairs": theoretical_pairs,
        "pair_reduction_fraction": round(reduction, 12),
        "block_size_summary": cast(JsonValue, generated.block_summaries),
        "broad_blocks_skipped": {
            "by_method": cast(JsonValue, generated.broad_blocks_skipped),
            "total": sum(generated.broad_blocks_skipped.values()),
        },
        "candidates_per_key": cast(JsonValue, generated.candidates_per_key),
        "execution_timing": {
            "universe_construction_seconds": round(universe_seconds, 6),
            "candidate_generation_seconds": round(generation_seconds, 6),
            "parquet_write_seconds": round(write_seconds, 6),
            "total_seconds": round(elapsed_seconds, 6),
        },
        "output_sizes_bytes": {
            "resolution_keys": keys_path.stat().st_size,
            "candidate_pairs": pairs_path.stat().st_size,
        },
    }
    _write_json(metrics_path, metrics)
    LOGGER.info(
        "Generated %s candidate pairs for %s unique eligible keys",
        pair_count,
        key_count,
    )
    return CandidateGenerationResult(
        resolution_keys_path=keys_path,
        candidate_pairs_path=pairs_path,
        metrics_path=metrics_path,
        unresolved_rows=unresolved_rows,
        eligible_rows=eligible_rows,
        unique_keys=key_count,
        candidate_pairs=pair_count,
        elapsed_seconds=elapsed_seconds,
    )


def _update_candidate_identity_digest(
    digest: hashlib._Hash,
    key_a: str,
    key_b: str,
    name_a: str,
    name_b: str,
    blocking_methods: list[str],
) -> None:
    """Add collision-safe candidate identity and retrieval evidence to a digest."""
    for value in (key_a, key_b, name_a, name_b, *blocking_methods):
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big"))
        digest.update(encoded)
    digest.update(len(blocking_methods).to_bytes(4, byteorder="big"))


def _validated_candidate_row(
    row: dict[str, object], previous_pair: tuple[str, str] | None
) -> tuple[str, str, str, str, list[str]]:
    values = [row.get(column) for column in _CANDIDATE_IDENTITY_COLUMNS[:4]]
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError("Candidate keys and names must be nonblank strings")
    key_a, key_b, name_a, name_b = cast(tuple[str, str, str, str], tuple(values))
    if key_a >= key_b:
        raise ValueError(f"Candidate pair must satisfy key_a < key_b: {key_a!r}, {key_b!r}")
    pair = (key_a, key_b)
    if previous_pair is not None and pair <= previous_pair:
        raise ValueError("Candidate pairs must be unique and in deterministic ascending order")
    raw_methods = row.get("blocking_methods")
    if (
        not isinstance(raw_methods, list)
        or not raw_methods
        or any(not isinstance(method, str) or not method.strip() for method in raw_methods)
    ):
        raise ValueError("blocking_methods must be a nonempty list of strings")
    methods = cast(list[str], raw_methods)
    return key_a, key_b, name_a, name_b, methods


def _numeric_feature_summary(path: Path, column: str) -> dict[str, JsonValue]:
    """Compute exact distribution points while loading only one numeric column."""
    values = pq.read_table(path, columns=[column]).column(0)
    labels = ("p10", "p25", "median", "p75", "p90", "p95", "p99")
    if len(values) == 0:
        return {key: None for key in ("minimum", *labels, "maximum")}
    quantiles = pc.quantile(
        values,
        q=[0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99],
        interpolation="linear",
    ).to_pylist()
    summary: dict[str, JsonValue] = {
        "minimum": round(float(pc.min(values).as_py()), FEATURE_PRECISION)
    }
    summary.update(
        {
            label: round(float(value), FEATURE_PRECISION)
            for label, value in zip(labels, quantiles, strict=True)
        }
    )
    summary["maximum"] = round(float(pc.max(values).as_py()), FEATURE_PRECISION)
    return summary


def score_candidate_pairs(
    root: Path,
    settings: Settings,
    *,
    input_override: Path | None = None,
    output_override: Path | None = None,
) -> CandidateScoringResult:
    """Stream independent lexical evidence for existing candidate pairs."""
    total_started = time.perf_counter()

    def product_path(override: Path | None, configured: Path) -> Path:
        path = override or configured
        return path if path.is_absolute() else root / path

    input_path = product_path(input_override, settings.candidate_scoring.input_dataset)
    output_path = product_path(output_override, settings.candidate_scoring.output_dataset)
    metrics_path = product_path(None, settings.candidate_scoring.metrics_output)
    resolved_paths = {path.resolve() for path in (input_path, output_path, metrics_path)}
    if len(resolved_paths) != 3:
        raise ValueError("Scoring input, feature output, and metrics paths must be distinct")
    if not input_path.is_file():
        raise FileNotFoundError(f"Candidate-pair dataset does not exist: {input_path}")

    candidate_file = pq.ParquetFile(input_path)
    missing_columns = set(_CANDIDATE_IDENTITY_COLUMNS) - set(candidate_file.schema_arrow.names)
    if missing_columns:
        raise ValueError(
            f"Candidate-pair dataset is missing required columns: {sorted(missing_columns)}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(f"{output_path.suffix}.tmp")
    temporary.unlink(missing_ok=True)
    compression = (
        None
        if settings.processing.parquet_compression == "none"
        else settings.processing.parquet_compression
    )
    input_digest = hashlib.sha256()
    relation_counts: Counter[str] = Counter()
    blocking_count_distribution: Counter[int] = Counter()
    input_rows = 0
    previous_pair: tuple[str, str] | None = None
    writer: pq.ParquetWriter | None = None
    try:
        writer = pq.ParquetWriter(
            temporary,
            CANDIDATE_FEATURES_SCHEMA,
            compression=compression,
            use_dictionary=True,
            write_statistics=True,
        )
        for batch in candidate_file.iter_batches(
            batch_size=settings.candidate_scoring.batch_size,
            columns=list(_CANDIDATE_IDENTITY_COLUMNS),
        ):
            feature_rows: list[dict[str, object]] = []
            for row in batch.to_pylist():
                key_a, key_b, name_a, name_b, methods = _validated_candidate_row(row, previous_pair)
                previous_pair = (key_a, key_b)
                evidence = compute_pair_features(name_a, name_b, methods)
                _update_candidate_identity_digest(
                    input_digest, key_a, key_b, name_a, name_b, methods
                )
                relation_counts[evidence.numeric_relation] += 1
                blocking_count_distribution[evidence.blocking_method_count] += 1
                input_rows += 1
                feature_rows.append(
                    {
                        "key_a": key_a,
                        "key_b": key_b,
                        "name_a": name_a,
                        "name_b": name_b,
                        "blocking_methods": methods,
                        "blocking_method_count": evidence.blocking_method_count,
                        "char_ratio": evidence.char_ratio,
                        "token_sort_ratio": evidence.token_sort_ratio,
                        "token_set_ratio": evidence.token_set_ratio,
                        "partial_ratio": evidence.partial_ratio,
                        "length_ratio": evidence.length_ratio,
                        "common_prefix_ratio": evidence.common_prefix_ratio,
                        "token_jaccard": evidence.token_jaccard,
                        "same_first_token": evidence.same_first_token,
                        "numeric_relation": evidence.numeric_relation,
                    }
                )
            if feature_rows:
                writer.write_table(
                    pa.Table.from_pylist(feature_rows, schema=CANDIDATE_FEATURES_SCHEMA)
                )
    except Exception:
        if writer is not None:
            writer.close()
            writer = None
        temporary.unlink(missing_ok=True)
        raise
    finally:
        if writer is not None:
            writer.close()
    temporary.replace(output_path)

    output_file = pq.ParquetFile(output_path)
    if output_file.schema_arrow != CANDIDATE_FEATURES_SCHEMA:
        raise RuntimeError("Candidate feature output schema failed reconciliation")
    output_rows = output_file.metadata.num_rows
    output_digest = hashlib.sha256()
    reconciled_rows = 0
    for batch in output_file.iter_batches(
        batch_size=settings.candidate_scoring.batch_size,
        columns=list(_CANDIDATE_IDENTITY_COLUMNS),
    ):
        for row in batch.to_pylist():
            key_a, key_b, name_a, name_b, methods = _validated_candidate_row(row, None)
            _update_candidate_identity_digest(output_digest, key_a, key_b, name_a, name_b, methods)
            reconciled_rows += 1
    identities_preserved = input_digest.digest() == output_digest.digest()
    reconciliation_passed = (
        input_rows == candidate_file.metadata.num_rows == output_rows == reconciled_rows
        and identities_preserved
    )
    if not reconciliation_passed:
        raise RuntimeError("Candidate feature output failed row or identity reconciliation")

    distributions = {
        column: cast(JsonValue, _numeric_feature_summary(output_path, column))
        for column in _NUMERIC_FEATURE_COLUMNS
    }
    elapsed_seconds = time.perf_counter() - total_started
    metrics: dict[str, JsonValue] = {
        "candidate_pairs_read": input_rows,
        "feature_rows_written": output_rows,
        "execution_runtime_seconds": round(elapsed_seconds, 6),
        "output_size_bytes": output_path.stat().st_size,
        "numeric_feature_distributions": distributions,
        "numeric_relation_distribution": dict(sorted(relation_counts.items())),
        "blocking_method_count_distribution": {
            str(count): frequency
            for count, frequency in sorted(blocking_count_distribution.items())
        },
        "reconciliation": {
            "row_counts_equal": True,
            "candidate_identity_and_order_preserved": True,
            "status": "passed",
        },
    }
    _write_json(metrics_path, metrics)
    LOGGER.info("Scored %s candidate pairs without identity decisions", output_rows)
    return CandidateScoringResult(
        output_path=output_path,
        metrics_path=metrics_path,
        candidate_pairs=input_rows,
        feature_rows=output_rows,
        elapsed_seconds=elapsed_seconds,
    )


def _load_possible_truncation_by_key(path: Path, batch_size: int) -> dict[str, bool]:
    """Load only the key metadata required by deterministic truncation decisions."""
    resolution_file = pq.ParquetFile(path)
    required = {"resolution_key", "possible_truncation"}
    missing = required - set(resolution_file.schema_arrow.names)
    if missing:
        raise ValueError(f"Resolution-key dataset is missing required columns: {sorted(missing)}")
    possible_by_key: dict[str, bool] = {}
    for batch in resolution_file.iter_batches(batch_size=batch_size, columns=sorted(required)):
        for row in batch.to_pylist():
            key = row.get("resolution_key")
            possible = row.get("possible_truncation")
            if not isinstance(key, str) or not key.strip() or not isinstance(possible, bool):
                raise ValueError("Resolution-key truncation metadata has invalid values")
            if key in possible_by_key:
                raise ValueError(f"Resolution-key dataset contains a duplicate key: {key!r}")
            possible_by_key[key] = possible
    return possible_by_key


def _validated_numeric_relation(row: dict[str, object]) -> NumericRelation:
    relation = row.get("numeric_relation")
    if relation not in {"none", "same", "one_sided", "conflict"}:
        raise ValueError(f"Invalid numeric_relation value: {relation!r}")
    return cast(NumericRelation, relation)


def decide_candidate_pairs(
    root: Path,
    settings: Settings,
    *,
    input_override: Path | None = None,
    keys_override: Path | None = None,
    output_override: Path | None = None,
) -> PairDecisionResult:
    """Stream high-precision deterministic pair decisions and abstain otherwise."""
    total_started = time.perf_counter()

    def product_path(override: Path | None, configured: Path) -> Path:
        path = override or configured
        return path if path.is_absolute() else root / path

    decision_config = settings.pair_decision
    input_path = product_path(input_override, decision_config.input_dataset)
    keys_path = product_path(keys_override, decision_config.resolution_keys_dataset)
    output_path = product_path(output_override, decision_config.output_dataset)
    metrics_path = product_path(None, decision_config.metrics_output)
    resolved_paths = {path.resolve() for path in (input_path, keys_path, output_path, metrics_path)}
    if len(resolved_paths) != 4:
        raise ValueError("Decision input, key metadata, output, and metrics paths must be distinct")
    if not input_path.is_file():
        raise FileNotFoundError(f"Candidate-feature dataset does not exist: {input_path}")
    if not keys_path.is_file():
        raise FileNotFoundError(f"Resolution-key dataset does not exist: {keys_path}")

    candidate_file = pq.ParquetFile(input_path)
    missing_columns = set(_PAIR_DECISION_INPUT_COLUMNS) - set(candidate_file.schema_arrow.names)
    if missing_columns:
        raise ValueError(
            f"Candidate-feature dataset is missing required columns: {sorted(missing_columns)}"
        )
    possible_truncation_by_key = _load_possible_truncation_by_key(
        keys_path, decision_config.batch_size
    )
    source_boundaries = frozenset(settings.normalization.possible_truncation_content_lengths)

    truncation_candidate_counts: Counter[str] = Counter()
    first_pass_rows = 0
    previous_pair: tuple[str, str] | None = None
    for batch in candidate_file.iter_batches(
        batch_size=decision_config.batch_size,
        columns=list(_PAIR_DECISION_INPUT_COLUMNS),
    ):
        for row in batch.to_pylist():
            key_a, key_b, name_a, name_b, _ = _validated_candidate_row(row, previous_pair)
            previous_pair = (key_a, key_b)
            if key_a not in possible_truncation_by_key or key_b not in possible_truncation_by_key:
                raise ValueError("Candidate-feature key is absent from resolution-key metadata")
            relation = exact_source_truncation_relation(
                key_a=key_a,
                key_b=key_b,
                name_a=name_a,
                name_b=name_b,
                numeric_relation=_validated_numeric_relation(row),
                possible_truncation_by_key=possible_truncation_by_key,
                source_truncation_boundaries=source_boundaries,
            )
            if relation is not None:
                truncation_candidate_counts[relation.truncated_key] += 1
            first_pass_rows += 1
    if first_pass_rows != candidate_file.metadata.num_rows:
        raise RuntimeError("Candidate-feature input row count changed during truncation analysis")

    unique_truncation_keys = frozenset(
        key for key, count in truncation_candidate_counts.items() if count == 1
    )
    ambiguous_truncation_keys = frozenset(
        key for key, count in truncation_candidate_counts.items() if count > 1
    )
    exact_truncation_candidates = sum(truncation_candidate_counts.values())
    ambiguous_truncation_candidates = sum(
        truncation_candidate_counts[key] for key in ambiguous_truncation_keys
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(f"{output_path.suffix}.tmp")
    temporary.unlink(missing_ok=True)
    compression = (
        None
        if settings.processing.parquet_compression == "none"
        else settings.processing.parquet_compression
    )
    input_digest = hashlib.sha256()
    status_counts: Counter[str] = Counter()
    selected_rule_counts: Counter[str] = Counter()
    overlap_combinations: Counter[str] = Counter()
    multiple_rule_pairs = 0
    unique_truncation_auto_resolutions = 0
    ambiguous_truncation_abstentions = 0
    second_pass_rows = 0
    previous_pair = None
    writer: pq.ParquetWriter | None = None
    try:
        writer = pq.ParquetWriter(
            temporary,
            PAIR_DECISIONS_SCHEMA,
            compression=compression,
            use_dictionary=True,
            write_statistics=True,
        )
        for batch in candidate_file.iter_batches(
            batch_size=decision_config.batch_size,
            columns=list(_PAIR_DECISION_INPUT_COLUMNS),
        ):
            decision_rows: list[dict[str, object]] = []
            for row in batch.to_pylist():
                key_a, key_b, name_a, name_b, methods = _validated_candidate_row(row, previous_pair)
                previous_pair = (key_a, key_b)
                numeric_relation = _validated_numeric_relation(row)
                truncation_relation = exact_source_truncation_relation(
                    key_a=key_a,
                    key_b=key_b,
                    name_a=name_a,
                    name_b=name_b,
                    numeric_relation=numeric_relation,
                    possible_truncation_by_key=possible_truncation_by_key,
                    source_truncation_boundaries=source_boundaries,
                )
                decision = decide_pair(
                    key_a=key_a,
                    key_b=key_b,
                    name_a=name_a,
                    name_b=name_b,
                    numeric_relation=numeric_relation,
                    corporate_suffix_aliases=settings.normalization.corporate_suffix_aliases,
                    minimum_whitespace_compact_length=(
                        decision_config.minimum_whitespace_compact_length
                    ),
                    truncation_relation=truncation_relation,
                    unique_truncation_keys=unique_truncation_keys,
                )
                _update_candidate_identity_digest(
                    input_digest, key_a, key_b, name_a, name_b, methods
                )
                status_counts[decision.status] += 1
                selected_rule_counts[decision.rule] += 1
                if len(decision.supporting_rules) > 1:
                    multiple_rule_pairs += 1
                    overlap_combinations["+".join(decision.supporting_rules)] += 1
                if (
                    truncation_relation is not None
                    and truncation_relation.truncated_key in unique_truncation_keys
                ):
                    unique_truncation_auto_resolutions += 1
                if (
                    truncation_relation is not None
                    and truncation_relation.truncated_key in ambiguous_truncation_keys
                    and decision.status == NEEDS_FURTHER_RESOLUTION
                ):
                    ambiguous_truncation_abstentions += 1
                second_pass_rows += 1
                decision_rows.append(
                    {
                        "key_a": key_a,
                        "key_b": key_b,
                        "name_a": name_a,
                        "name_b": name_b,
                        "blocking_methods": methods,
                        "decision_status": decision.status,
                        "decision_rule": decision.rule,
                        "decision_evidence": decision.evidence,
                    }
                )
            if decision_rows:
                writer.write_table(
                    pa.Table.from_pylist(decision_rows, schema=PAIR_DECISIONS_SCHEMA)
                )
    except Exception:
        if writer is not None:
            writer.close()
            writer = None
        temporary.unlink(missing_ok=True)
        raise
    finally:
        if writer is not None:
            writer.close()
    temporary.replace(output_path)

    output_file = pq.ParquetFile(output_path)
    if output_file.schema_arrow != PAIR_DECISIONS_SCHEMA:
        raise RuntimeError("Pair-decision output schema failed reconciliation")
    output_rows = output_file.metadata.num_rows
    output_digest = hashlib.sha256()
    reconciled_rows = 0
    previous_pair = None
    for batch in output_file.iter_batches(
        batch_size=decision_config.batch_size,
        columns=list(_CANDIDATE_IDENTITY_COLUMNS),
    ):
        for row in batch.to_pylist():
            key_a, key_b, name_a, name_b, methods = _validated_candidate_row(row, previous_pair)
            previous_pair = (key_a, key_b)
            _update_candidate_identity_digest(output_digest, key_a, key_b, name_a, name_b, methods)
            reconciled_rows += 1

    identities_preserved = input_digest.digest() == output_digest.digest()
    counts_reconcile = (
        first_pass_rows
        == second_pass_rows
        == output_rows
        == reconciled_rows
        == candidate_file.metadata.num_rows
        == sum(status_counts.values())
    )
    if not counts_reconcile or not identities_preserved:
        raise RuntimeError("Pair-decision output failed row or identity reconciliation")

    auto_same_count = status_counts[AUTO_SAME]
    needs_further_count = status_counts[NEEDS_FURTHER_RESOLUTION]
    elapsed_seconds = time.perf_counter() - total_started
    auto_same_by_rule = {rule: selected_rule_counts[rule] for rule in RULE_PRECEDENCE}
    metrics: dict[str, JsonValue] = {
        "candidate_pairs_read": first_pass_rows,
        "decision_rows_written": output_rows,
        "auto_same_count": auto_same_count,
        "needs_further_resolution_count": needs_further_count,
        "auto_same_by_rule": cast(JsonValue, auto_same_by_rule),
        "percentage_auto_resolved": round(
            (auto_same_count / output_rows * 100.0) if output_rows else 0.0, 6
        ),
        "multiple_rule_evidence_pairs": multiple_rule_pairs,
        "overlap_rule_combinations": cast(JsonValue, dict(sorted(overlap_combinations.items()))),
        "truncation": {
            "exact_truncation_candidates_considered": exact_truncation_candidates,
            "unique_truncation_auto_resolutions": unique_truncation_auto_resolutions,
            "ambiguous_multi_continuation_keys": len(ambiguous_truncation_keys),
            "ambiguous_multi_continuation_candidates": ambiguous_truncation_candidates,
            "ambiguous_multi_continuation_truncations_abstained": (
                ambiguous_truncation_abstentions
            ),
        },
        "execution_runtime_seconds": round(elapsed_seconds, 6),
        "output_size_bytes": output_path.stat().st_size,
        "reconciliation": {
            "candidate_and_decision_row_counts_equal": True,
            "candidate_identity_and_order_preserved": True,
            "status_counts_equal_decision_rows": True,
            "status": "passed",
        },
    }
    _write_json(metrics_path, metrics)
    LOGGER.info(
        "Decided %s candidate pairs: %s AUTO_SAME, %s abstained",
        output_rows,
        auto_same_count,
        needs_further_count,
    )
    return PairDecisionResult(
        output_path=output_path,
        metrics_path=metrics_path,
        candidate_pairs=first_pass_rows,
        decision_rows=output_rows,
        auto_same_count=auto_same_count,
        needs_further_resolution_count=needs_further_count,
        elapsed_seconds=elapsed_seconds,
    )


def _update_eligibility_digest(
    digest: hashlib._Hash,
    *,
    resolution_key: str,
    representative_name: str,
    status: str,
    rule: str,
    evidence: str,
) -> None:
    """Add a collision-safe eligibility row to a reconciliation digest."""
    for value in (resolution_key, representative_name, status, rule, evidence):
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big"))
        digest.update(encoded)


def classify_employer_eligibility(
    root: Path,
    settings: Settings,
    *,
    preprocessed_override: Path | None = None,
    keys_override: Path | None = None,
    output_override: Path | None = None,
) -> EmployerEligibilityResult:
    """Aggregate persisted signals and classify each resolution key's mention type."""
    total_started = time.perf_counter()
    eligibility_config = settings.employer_eligibility

    def product_path(override: Path | None, configured: Path) -> Path:
        path = override or configured
        return path if path.is_absolute() else root / path

    preprocessed_path = product_path(preprocessed_override, eligibility_config.preprocessed_dataset)
    resolution_keys_path = product_path(keys_override, eligibility_config.resolution_keys_dataset)
    output_path = product_path(output_override, eligibility_config.output_dataset)
    metrics_path = product_path(None, eligibility_config.metrics_output)
    paths = (preprocessed_path, resolution_keys_path, output_path, metrics_path)
    if len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("Eligibility inputs, output, and metrics paths must be distinct")
    if not preprocessed_path.is_file():
        raise FileNotFoundError(f"Preprocessed dataset does not exist: {preprocessed_path}")
    if not resolution_keys_path.is_file():
        raise FileNotFoundError(f"Resolution-key dataset does not exist: {resolution_keys_path}")

    preprocessed_file = pq.ParquetFile(preprocessed_path)
    resolution_file = pq.ParquetFile(resolution_keys_path)
    if preprocessed_file.schema_arrow != PREPROCESSED_SCHEMA:
        raise ValueError("Preprocessed dataset does not match the required schema")
    if resolution_file.schema_arrow != RESOLUTION_KEYS_SCHEMA:
        raise ValueError("Resolution-key dataset does not match the required schema")

    key_metadata: list[tuple[str, str, str, bool, int]] = []
    aggregates: dict[str, EligibilityEvidenceAggregate] = {}
    previous_key: str | None = None
    key_columns = (
        "resolution_key",
        "representative_name",
        "representative_route",
        "possible_truncation",
        "token_count",
    )
    for batch in resolution_file.iter_batches(
        batch_size=eligibility_config.batch_size,
        columns=list(key_columns),
    ):
        for row in batch.to_pylist():
            key = row.get("resolution_key")
            name = row.get("representative_name")
            route = row.get("representative_route")
            possible_truncation = row.get("possible_truncation")
            token_count = row.get("token_count")
            if (
                not isinstance(key, str)
                or not key.strip()
                or not isinstance(name, str)
                or not name.strip()
                or not isinstance(route, str)
                or not route.strip()
                or not isinstance(possible_truncation, bool)
                or not isinstance(token_count, int)
                or token_count < 1
            ):
                raise ValueError("Resolution-key dataset contains invalid eligibility metadata")
            if previous_key is not None and key <= previous_key:
                raise ValueError("Resolution keys must be unique and in ascending order")
            previous_key = key
            key_metadata.append((key, name, route, possible_truncation, token_count))
            aggregates[key] = EligibilityEvidenceAggregate()

    matched_preprocessing_rows = 0
    for batch in preprocessed_file.iter_batches(
        batch_size=eligibility_config.batch_size,
        columns=list(_ELIGIBILITY_PREPROCESSING_COLUMNS),
    ):
        for row in batch.to_pylist():
            key = row.get("nombre_normalizado")
            if not isinstance(key, str):
                continue
            aggregate = aggregates.get(key)
            if aggregate is None:
                continue
            update_evidence_aggregate(aggregate, row)
            matched_preprocessing_rows += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(f"{output_path.suffix}.tmp")
    temporary.unlink(missing_ok=True)
    compression = (
        None
        if settings.processing.parquet_compression == "none"
        else settings.processing.parquet_compression
    )
    road_type_tokens = frozenset(
        token.upper() for token in eligibility_config.structural_address_road_type_tokens
    )
    status_counts: Counter[str] = Counter()
    rule_counts: Counter[str] = Counter()
    route_status_counts: Counter[tuple[str, str]] = Counter()
    address_strength_counts: Counter[str] = Counter()
    non_employer_occupation_counts: Counter[str] = Counter()
    non_employer_activity_counts: Counter[str] = Counter()
    expected_digest = hashlib.sha256()
    known_key = "548 MARKET ST 18590 CA SAN FRANCISCO"
    known_check: dict[str, JsonValue] = {
        "resolution_key": known_key,
        "present": False,
        "eligibility_status": None,
        "eligibility_rule": None,
        "captured_by_general_structural_address_rule": False,
    }
    rows_written = 0
    writer: pq.ParquetWriter | None = None
    try:
        writer = pq.ParquetWriter(
            temporary,
            EMPLOYER_ELIGIBILITY_SCHEMA,
            compression=compression,
            use_dictionary=True,
            write_statistics=True,
        )
        output_rows: list[dict[str, object]] = []
        for key, name, route, possible_truncation, token_count in key_metadata:
            evidence = finalize_evidence(
                resolution_key=key,
                representative_name=name,
                representative_route=route,
                possible_truncation=possible_truncation,
                token_count=token_count,
                aggregate=aggregates[key],
            )
            decision = classify_eligibility(evidence, road_type_tokens=road_type_tokens)
            status_counts[decision.status] += 1
            rule_counts[decision.rule] += 1
            route_status_counts[(route, decision.status)] += 1
            if decision.status == ADDRESS:
                address_strength_counts[evidence.address_signal_strength] += 1
            if decision.status == NON_EMPLOYER_STATUS:
                non_employer_occupation_counts[evidence.occupation_signal_strength] += 1
                activity_key = str(evidence.has_activity_description_signal).lower()
                non_employer_activity_counts[activity_key] += 1
            if key == known_key:
                known_check = {
                    "resolution_key": known_key,
                    "present": True,
                    "eligibility_status": decision.status,
                    "eligibility_rule": decision.rule,
                    "captured_by_general_structural_address_rule": (
                        decision.status == ADDRESS
                        and decision.rule == "structural_street_zip_region_address"
                    ),
                }
            output_rows.append(
                {
                    "resolution_key": key,
                    "representative_name": name,
                    "eligibility_status": decision.status,
                    "eligibility_rule": decision.rule,
                    "eligibility_evidence": decision.evidence,
                }
            )
            _update_eligibility_digest(
                expected_digest,
                resolution_key=key,
                representative_name=name,
                status=decision.status,
                rule=decision.rule,
                evidence=decision.evidence,
            )
            rows_written += 1
            if len(output_rows) >= eligibility_config.batch_size:
                writer.write_table(
                    pa.Table.from_pylist(output_rows, schema=EMPLOYER_ELIGIBILITY_SCHEMA)
                )
                output_rows.clear()
        if output_rows:
            writer.write_table(
                pa.Table.from_pylist(output_rows, schema=EMPLOYER_ELIGIBILITY_SCHEMA)
            )
    except Exception:
        if writer is not None:
            writer.close()
            writer = None
        temporary.unlink(missing_ok=True)
        raise
    finally:
        if writer is not None:
            writer.close()
    temporary.replace(output_path)

    output_file = pq.ParquetFile(output_path)
    if output_file.schema_arrow != EMPLOYER_ELIGIBILITY_SCHEMA:
        raise RuntimeError("Employer-eligibility output schema failed reconciliation")
    output_digest = hashlib.sha256()
    reconciled_rows = 0
    previous_key = None
    for batch in output_file.iter_batches(batch_size=eligibility_config.batch_size):
        for row in batch.to_pylist():
            key = row.get("resolution_key")
            name = row.get("representative_name")
            status = row.get("eligibility_status")
            rule = row.get("eligibility_rule")
            evidence = row.get("eligibility_evidence")
            if not all(isinstance(value, str) and value for value in row.values()):
                raise RuntimeError("Employer-eligibility output contains invalid values")
            assert isinstance(key, str)
            assert isinstance(name, str)
            assert isinstance(status, str)
            assert isinstance(rule, str)
            assert isinstance(evidence, str)
            if previous_key is not None and key <= previous_key:
                raise RuntimeError("Employer-eligibility output order is not deterministic")
            previous_key = key
            _update_eligibility_digest(
                output_digest,
                resolution_key=key,
                representative_name=name,
                status=status,
                rule=rule,
                evidence=evidence,
            )
            reconciled_rows += 1

    resolution_keys_read = resolution_file.metadata.num_rows
    counts_reconcile = (
        resolution_keys_read
        == len(key_metadata)
        == rows_written
        == output_file.metadata.num_rows
        == reconciled_rows
        == sum(status_counts.values())
    )
    identities_reconcile = expected_digest.digest() == output_digest.digest()
    if not counts_reconcile or not identities_reconcile:
        raise RuntimeError("Employer-eligibility output failed row or identity reconciliation")

    route_cross_tab = {
        route: {status: route_status_counts[(route, status)] for status in _ELIGIBILITY_STATUSES}
        for route in sorted({route for route, _status in route_status_counts})
    }
    elapsed_seconds = time.perf_counter() - total_started
    metrics: dict[str, JsonValue] = {
        "ruleset_version": eligibility_config.ruleset_version,
        "resolution_keys_read": resolution_keys_read,
        "matched_preprocessing_rows": matched_preprocessing_rows,
        "rows_written": rows_written,
        "eligibility_status_counts": {
            status: status_counts[status] for status in _ELIGIBILITY_STATUSES
        },
        "eligibility_rule_counts": {
            rule: rule_counts[rule] for rule in ELIGIBILITY_RULE_PRECEDENCE
        },
        "prior_route_by_eligibility_status": cast(JsonValue, route_cross_tab),
        "address_classifications_by_prior_address_signal_strength": {
            strength: address_strength_counts[strength]
            for strength in ("none", "weak", "moderate", "strong")
        },
        "non_employer_classifications": {
            "by_occupation_signal_strength": {
                strength: non_employer_occupation_counts[strength]
                for strength in ("none", "weak", "moderate", "strong")
            },
            "by_activity_description_signal": {
                value: non_employer_activity_counts[value] for value in ("false", "true")
            },
        },
        "ambiguous_count": status_counts[AMBIGUOUS],
        "known_structural_address_check": known_check,
        "execution_runtime_seconds": round(elapsed_seconds, 6),
        "output_size_bytes": output_path.stat().st_size,
        "reconciliation": {
            "one_row_per_resolution_key": True,
            "resolution_key_order_preserved": True,
            "persisted_rows_match_classification_digest": True,
            "status_counts_equal_rows_written": True,
            "status": "passed",
        },
    }
    _write_json(metrics_path, metrics)
    LOGGER.info(
        "Classified %s resolution keys: %s employer, %s address, %s non-employer, %s ambiguous",
        rows_written,
        status_counts[EMPLOYER_CANDIDATE],
        status_counts[ADDRESS],
        status_counts[NON_EMPLOYER_STATUS],
        status_counts[AMBIGUOUS],
    )
    return EmployerEligibilityResult(
        output_path=output_path,
        metrics_path=metrics_path,
        resolution_keys=resolution_keys_read,
        rows_written=rows_written,
        employer_candidate_count=status_counts[EMPLOYER_CANDIDATE],
        address_count=status_counts[ADDRESS],
        non_employer_status_count=status_counts[NON_EMPLOYER_STATUS],
        ambiguous_count=status_counts[AMBIGUOUS],
        elapsed_seconds=elapsed_seconds,
    )


def _iter_parquet_rows(
    parquet: pq.ParquetFile,
    *,
    batch_size: int,
    columns: tuple[str, ...],
) -> Iterator[dict[str, object]]:
    """Yield selected Parquet rows while bounding conversion memory by batch size."""
    for batch in parquet.iter_batches(batch_size=batch_size, columns=list(columns)):
        for row in batch.to_pylist():
            yield cast(dict[str, object], row)


def _load_employer_candidate_lookup(
    path: Path,
    *,
    batch_size: int,
    policy: OrthographicPolicy,
) -> tuple[dict[str, bool], dict[str, int]]:
    """Load eligibility and distinct-key token support for employer candidates."""
    eligibility_file = pq.ParquetFile(path)
    if eligibility_file.schema_arrow != EMPLOYER_ELIGIBILITY_SCHEMA:
        raise ValueError("Employer-eligibility dataset does not match the required schema")
    lookup: dict[str, bool] = {}
    token_support: Counter[str] = Counter()
    previous_key: str | None = None
    for row in _iter_parquet_rows(
        eligibility_file,
        batch_size=batch_size,
        columns=("resolution_key", "representative_name", "eligibility_status"),
    ):
        key = row.get("resolution_key")
        representative_name = row.get("representative_name")
        status = row.get("eligibility_status")
        if (
            not isinstance(key, str)
            or not key.strip()
            or not isinstance(representative_name, str)
            or not representative_name.strip()
            or status not in _ELIGIBILITY_STATUSES
        ):
            raise ValueError("Employer-eligibility dataset contains invalid gate values")
        if previous_key is not None and key <= previous_key:
            raise ValueError("Employer-eligibility keys must be unique and in ascending order")
        previous_key = key
        is_employer = status == EMPLOYER_CANDIDATE
        lookup[key] = is_employer
        if is_employer:
            token_support.update(set(orthographic_core_tokens(representative_name, policy)))
    if len(lookup) != eligibility_file.metadata.num_rows:
        raise RuntimeError("Employer-eligibility lookup failed row reconciliation")
    return lookup, dict(token_support)


def _build_context_variant_index(
    path: Path,
    *,
    target_signatures: set[tuple[str, ...]],
    policy: OrthographicPolicy,
    batch_size: int,
) -> dict[tuple[str, ...], frozenset[str]]:
    """Index variants only for pair-derived signatures in the employer universe."""
    if not target_signatures:
        return {}
    eligibility_file = pq.ParquetFile(path)
    variants: dict[tuple[str, ...], set[str]] = {
        signature: set() for signature in target_signatures
    }
    for row in _iter_parquet_rows(
        eligibility_file,
        batch_size=batch_size,
        columns=("representative_name", "eligibility_status"),
    ):
        if row.get("eligibility_status") != EMPLOYER_CANDIDATE:
            continue
        representative_name = row.get("representative_name")
        if not isinstance(representative_name, str) or not representative_name.strip():
            raise ValueError("Employer candidate has an invalid representative name")
        core_tokens = orthographic_core_tokens(representative_name, policy)
        for index, token in enumerate(core_tokens):
            signature = tuple(
                "<DIFF>" if position == index else value
                for position, value in enumerate(core_tokens)
            )
            if signature in variants:
                variants[signature].add(token)
    return {signature: frozenset(values) for signature, values in variants.items()}


def _iter_aligned_orthographic_inputs(
    decision_file: pq.ParquetFile,
    feature_file: pq.ParquetFile,
    *,
    batch_size: int,
) -> Iterator[tuple[str, str, str, str, list[str], str, NumericRelation]]:
    """Validate and yield the two pair datasets in their shared deterministic order."""
    decision_rows = _iter_parquet_rows(
        decision_file,
        batch_size=batch_size,
        columns=_ORTHOGRAPHIC_DECISION_INPUT_COLUMNS,
    )
    feature_rows = _iter_parquet_rows(
        feature_file,
        batch_size=batch_size,
        columns=_PAIR_DECISION_INPUT_COLUMNS,
    )
    previous_decision_pair: tuple[str, str] | None = None
    previous_feature_pair: tuple[str, str] | None = None
    for decision_row, feature_row in zip(decision_rows, feature_rows, strict=True):
        decision_identity = _validated_candidate_row(decision_row, previous_decision_pair)
        feature_identity = _validated_candidate_row(feature_row, previous_feature_pair)
        key_a, key_b, name_a, name_b, blocking_methods = decision_identity
        previous_decision_pair = (key_a, key_b)
        previous_feature_pair = (feature_identity[0], feature_identity[1])
        if decision_identity != feature_identity:
            raise ValueError("Pair-decision and feature identities or order differ")
        prior_status = decision_row.get("decision_status")
        if prior_status not in {AUTO_SAME, NEEDS_FURTHER_RESOLUTION}:
            raise ValueError(f"Invalid prior pair-decision status: {prior_status!r}")
        yield (
            key_a,
            key_b,
            name_a,
            name_b,
            blocking_methods,
            prior_status,
            _validated_numeric_relation(feature_row),
        )


def _update_orthographic_digest(
    digest: hashlib._Hash,
    *,
    key_a: str,
    key_b: str,
    name_a: str,
    name_b: str,
    blocking_methods: list[str],
    decision: OrthographicDecision,
) -> None:
    _update_candidate_identity_digest(digest, key_a, key_b, name_a, name_b, blocking_methods)
    persisted_values: tuple[str | int | None, ...] = (
        decision.status,
        decision.rule,
        decision.evidence,
        decision.differing_token_a,
        decision.differing_token_b,
        decision.edit_operation,
        decision.context_signature,
        decision.context_variant_count,
        decision.token_support_a,
        decision.token_support_b,
    )
    for value in persisted_values:
        if value is None:
            digest.update(b"N")
        elif isinstance(value, int):
            digest.update(b"I")
            digest.update(value.to_bytes(8, byteorder="big", signed=True))
        else:
            digest.update(b"S")
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, byteorder="big"))
            digest.update(encoded)


def _token_support_band(value: int) -> str:
    if value == 1:
        return "1"
    if value <= 5:
        return "2-5"
    if value <= 10:
        return "6-10"
    if value <= 25:
        return "11-25"
    if value <= 50:
        return "26-50"
    if value <= 100:
        return "51-100"
    return "101+"


def resolve_orthographic_pairs(
    root: Path,
    settings: Settings,
    *,
    decisions_override: Path | None = None,
    eligibility_override: Path | None = None,
    features_override: Path | None = None,
    output_override: Path | None = None,
) -> OrthographicResolutionResult:
    """Stream residual pairs through the gated single-token edit rule."""
    total_started = time.perf_counter()
    orthographic_config = settings.orthographic_resolution

    def product_path(override: Path | None, configured: Path) -> Path:
        path = override or configured
        return path if path.is_absolute() else root / path

    decisions_path = product_path(decisions_override, orthographic_config.pair_decisions_dataset)
    eligibility_path = product_path(
        eligibility_override, orthographic_config.employer_eligibility_dataset
    )
    features_path = product_path(features_override, orthographic_config.candidate_features_dataset)
    output_path = product_path(output_override, orthographic_config.output_dataset)
    metrics_path = product_path(None, orthographic_config.metrics_output)
    paths = (decisions_path, eligibility_path, features_path, output_path, metrics_path)
    if len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("Orthographic inputs, output, and metrics paths must be distinct")
    for label, path in (
        ("Pair-decision", decisions_path),
        ("Employer-eligibility", eligibility_path),
        ("Candidate-feature", features_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} dataset does not exist: {path}")

    decision_file = pq.ParquetFile(decisions_path)
    feature_file = pq.ParquetFile(features_path)
    if decision_file.schema_arrow != PAIR_DECISIONS_SCHEMA:
        raise ValueError("Pair-decision dataset does not match the required schema")
    if feature_file.schema_arrow != CANDIDATE_FEATURES_SCHEMA:
        raise ValueError("Candidate-feature dataset does not match the required schema")
    if decision_file.metadata.num_rows != feature_file.metadata.num_rows:
        raise ValueError("Pair-decision and candidate-feature row counts differ")

    orthographic_policy = build_orthographic_policy(
        corporate_suffix_aliases=settings.normalization.corporate_suffix_aliases,
        minimum_typo_token_length=settings.candidate_generation.minimum_typo_token_length,
        minimum_informative_token_length=(
            settings.candidate_generation.minimum_informative_token_length
        ),
    )
    employer_candidate_by_key, token_support = _load_employer_candidate_lookup(
        eligibility_path,
        batch_size=orthographic_config.batch_size,
        policy=orthographic_policy,
    )
    maximum_token_frequency = settings.candidate_generation.maximum_token_frequency

    target_context_signatures: set[tuple[str, ...]] = set()
    first_pass_rows = 0
    for (
        key_a,
        key_b,
        name_a,
        name_b,
        _,
        prior_status,
        numeric_relation,
    ) in _iter_aligned_orthographic_inputs(
        decision_file,
        feature_file,
        batch_size=orthographic_config.batch_size,
    ):
        first_pass_rows += 1
        if prior_status == AUTO_SAME:
            continue
        if key_a not in employer_candidate_by_key or key_b not in employer_candidate_by_key:
            raise ValueError("Residual pair endpoint is missing employer-eligibility evidence")
        prepared = prepare_orthographic_pair(
            name_a=name_a,
            name_b=name_b,
            numeric_relation=numeric_relation,
            employer_candidate_a=employer_candidate_by_key[key_a],
            employer_candidate_b=employer_candidate_by_key[key_b],
            policy=orthographic_policy,
        )
        if isinstance(prepared, OrthographicComparison):
            target_context_signatures.add(prepared.context_signature_tokens)
    if first_pass_rows != decision_file.metadata.num_rows:
        raise RuntimeError("Orthographic first pass failed row reconciliation")
    context_variants = _build_context_variant_index(
        eligibility_path,
        target_signatures=target_context_signatures,
        policy=orthographic_policy,
        batch_size=orthographic_config.batch_size,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(f"{output_path.suffix}.tmp")
    temporary.unlink(missing_ok=True)
    compression = (
        None
        if settings.processing.parquet_compression == "none"
        else settings.processing.parquet_compression
    )

    status_counts: Counter[str] = Counter()
    rule_counts: Counter[str] = Counter()
    abstention_reason_counts: Counter[str] = Counter()
    successful_numeric_relation_counts: Counter[str] = Counter()
    edit_operation_counts: Counter[str] = Counter()
    edit_location_counts: Counter[str] = Counter()
    successful_context_variant_counts: Counter[int] = Counter()
    successful_token_support_bands: Counter[str] = Counter()
    successful_minimum_shorter_length: int | None = None
    successful_maximum_longer_length: int | None = None
    successful_shorter_length_sum = 0
    successful_longer_length_sum = 0
    expected_digest = hashlib.sha256()
    pair_decisions_read = 0
    prior_auto_same_skipped = 0
    residual_pairs = 0
    eligible_employer_residual_pairs = 0
    writer: pq.ParquetWriter | None = None
    try:
        writer = pq.ParquetWriter(
            temporary,
            ORTHOGRAPHIC_PAIR_DECISIONS_SCHEMA,
            compression=compression,
            use_dictionary=True,
            write_statistics=True,
        )
        output_rows: list[dict[str, object]] = []
        for (
            key_a,
            key_b,
            name_a,
            name_b,
            blocking_methods,
            prior_status,
            numeric_relation,
        ) in _iter_aligned_orthographic_inputs(
            decision_file,
            feature_file,
            batch_size=orthographic_config.batch_size,
        ):
            pair_decisions_read += 1
            if prior_status == AUTO_SAME:
                prior_auto_same_skipped += 1
                continue

            residual_pairs += 1
            if key_a not in employer_candidate_by_key or key_b not in employer_candidate_by_key:
                raise ValueError("Residual pair endpoint is missing employer-eligibility evidence")
            employer_a = employer_candidate_by_key[key_a]
            employer_b = employer_candidate_by_key[key_b]
            if employer_a and employer_b:
                eligible_employer_residual_pairs += 1
            prepared = prepare_orthographic_pair(
                name_a=name_a,
                name_b=name_b,
                numeric_relation=numeric_relation,
                employer_candidate_a=employer_a,
                employer_candidate_b=employer_b,
                policy=orthographic_policy,
            )
            orthographic_decision = (
                prepared
                if isinstance(prepared, OrthographicDecision)
                else finalize_orthographic_comparison(
                    prepared,
                    token_support=token_support,
                    context_variants=context_variants,
                    maximum_token_frequency=maximum_token_frequency,
                )
            )
            status_counts[orthographic_decision.status] += 1
            rule_counts[orthographic_decision.rule] += 1
            if orthographic_decision.edit_location is not None:
                edit_location_counts[orthographic_decision.edit_location] += 1
            if orthographic_decision.status == STRONG_ORTHOGRAPHIC_EVIDENCE:
                successful_numeric_relation_counts[numeric_relation] += 1
                assert orthographic_decision.differing_token_a is not None
                assert orthographic_decision.differing_token_b is not None
                assert orthographic_decision.edit_operation is not None
                assert orthographic_decision.context_variant_count is not None
                assert orthographic_decision.token_support_a is not None
                assert orthographic_decision.token_support_b is not None
                lengths = (
                    len(orthographic_decision.differing_token_a),
                    len(orthographic_decision.differing_token_b),
                )
                shorter_length = min(lengths)
                longer_length = max(lengths)
                successful_minimum_shorter_length = (
                    shorter_length
                    if successful_minimum_shorter_length is None
                    else min(successful_minimum_shorter_length, shorter_length)
                )
                successful_maximum_longer_length = (
                    longer_length
                    if successful_maximum_longer_length is None
                    else max(successful_maximum_longer_length, longer_length)
                )
                successful_shorter_length_sum += shorter_length
                successful_longer_length_sum += longer_length
                edit_operation_counts[orthographic_decision.edit_operation] += 1
                successful_context_variant_counts[orthographic_decision.context_variant_count] += 1
                successful_token_support_bands[
                    _token_support_band(orthographic_decision.token_support_a)
                ] += 1
                successful_token_support_bands[
                    _token_support_band(orthographic_decision.token_support_b)
                ] += 1
            else:
                abstention_reason_counts[orthographic_decision.evidence] += 1

            output_rows.append(
                {
                    "key_a": key_a,
                    "key_b": key_b,
                    "name_a": name_a,
                    "name_b": name_b,
                    "blocking_methods": blocking_methods,
                    "orthographic_status": orthographic_decision.status,
                    "orthographic_rule": orthographic_decision.rule,
                    "orthographic_evidence": orthographic_decision.evidence,
                    "differing_token_a": orthographic_decision.differing_token_a,
                    "differing_token_b": orthographic_decision.differing_token_b,
                    "edit_operation": orthographic_decision.edit_operation,
                    "context_signature": orthographic_decision.context_signature,
                    "context_variant_count": (orthographic_decision.context_variant_count),
                    "token_support_a": orthographic_decision.token_support_a,
                    "token_support_b": orthographic_decision.token_support_b,
                }
            )
            _update_orthographic_digest(
                expected_digest,
                key_a=key_a,
                key_b=key_b,
                name_a=name_a,
                name_b=name_b,
                blocking_methods=blocking_methods,
                decision=orthographic_decision,
            )
            if len(output_rows) >= orthographic_config.batch_size:
                writer.write_table(
                    pa.Table.from_pylist(
                        output_rows,
                        schema=ORTHOGRAPHIC_PAIR_DECISIONS_SCHEMA,
                    )
                )
                output_rows.clear()
        if output_rows:
            writer.write_table(
                pa.Table.from_pylist(
                    output_rows,
                    schema=ORTHOGRAPHIC_PAIR_DECISIONS_SCHEMA,
                )
            )
    except Exception:
        if writer is not None:
            writer.close()
            writer = None
        temporary.unlink(missing_ok=True)
        raise
    finally:
        if writer is not None:
            writer.close()
    temporary.replace(output_path)

    output_file = pq.ParquetFile(output_path)
    if output_file.schema_arrow != ORTHOGRAPHIC_PAIR_DECISIONS_SCHEMA:
        raise RuntimeError("Orthographic output schema failed reconciliation")
    output_digest = hashlib.sha256()
    output_rows_reconciled = 0
    previous_output_pair: tuple[str, str] | None = None
    output_columns = (
        *_CANDIDATE_IDENTITY_COLUMNS,
        "orthographic_status",
        "orthographic_rule",
        "orthographic_evidence",
        "differing_token_a",
        "differing_token_b",
        "edit_operation",
        "context_signature",
        "context_variant_count",
        "token_support_a",
        "token_support_b",
    )
    for row in _iter_parquet_rows(
        output_file,
        batch_size=orthographic_config.batch_size,
        columns=output_columns,
    ):
        key_a, key_b, name_a, name_b, methods = _validated_candidate_row(row, previous_output_pair)
        previous_output_pair = (key_a, key_b)
        status = row.get("orthographic_status")
        rule = row.get("orthographic_rule")
        evidence = row.get("orthographic_evidence")
        differing_token_a = row.get("differing_token_a")
        differing_token_b = row.get("differing_token_b")
        edit_operation = row.get("edit_operation")
        context_signature = row.get("context_signature")
        context_variant_count = row.get("context_variant_count")
        token_support_a = row.get("token_support_a")
        token_support_b = row.get("token_support_b")
        if (
            status
            not in {
                STRONG_ORTHOGRAPHIC_EVIDENCE,
                NEEDS_FURTHER_RESOLUTION,
                NOT_ELIGIBLE_FOR_ORTHOGRAPHIC,
            }
            or rule
            not in {
                SINGLE_TOKEN_EDIT_EQUIVALENCE,
                NO_ORTHOGRAPHIC_EQUIVALENCE,
                NOT_ELIGIBLE_RULE,
            }
            or not isinstance(evidence, str)
            or not evidence
            or (differing_token_a is not None and not isinstance(differing_token_a, str))
            or (differing_token_b is not None and not isinstance(differing_token_b, str))
            or (
                edit_operation is not None
                and edit_operation not in {"insertion", "deletion", "substitution"}
            )
            or (context_signature is not None and not isinstance(context_signature, str))
            or (context_variant_count is not None and not isinstance(context_variant_count, int))
            or (token_support_a is not None and not isinstance(token_support_a, int))
            or (token_support_b is not None and not isinstance(token_support_b, int))
        ):
            raise RuntimeError("Orthographic output contains invalid decision values")
        assert isinstance(status, str)
        assert isinstance(rule, str)
        persisted_decision = OrthographicDecision(
            cast(OrthographicStatus, status),
            cast(OrthographicRule, rule),
            evidence,
            differing_token_a=differing_token_a,
            differing_token_b=differing_token_b,
            edit_operation=cast(EditOperation | None, edit_operation),
            context_signature=context_signature,
            context_variant_count=context_variant_count,
            token_support_a=token_support_a,
            token_support_b=token_support_b,
        )
        _update_orthographic_digest(
            output_digest,
            key_a=key_a,
            key_b=key_b,
            name_a=name_a,
            name_b=name_b,
            blocking_methods=methods,
            decision=persisted_decision,
        )
        output_rows_reconciled += 1

    output_row_count = output_file.metadata.num_rows
    status_total = sum(status_counts.values())
    counts_reconcile = (
        first_pass_rows
        == pair_decisions_read
        == decision_file.metadata.num_rows
        == feature_file.metadata.num_rows
        and pair_decisions_read == prior_auto_same_skipped + residual_pairs
        and residual_pairs == output_row_count == output_rows_reconciled == status_total
        and eligible_employer_residual_pairs
        == status_counts[STRONG_ORTHOGRAPHIC_EVIDENCE] + status_counts[NEEDS_FURTHER_RESOLUTION]
    )
    identities_reconcile = expected_digest.digest() == output_digest.digest()
    if not counts_reconcile or not identities_reconcile:
        raise RuntimeError("Orthographic output failed row or identity reconciliation")

    successful_count = status_counts[STRONG_ORTHOGRAPHIC_EVIDENCE]
    if successful_count:
        length_summary: dict[str, JsonValue] = {
            "pair_count": successful_count,
            "minimum_shorter_token_length": successful_minimum_shorter_length,
            "maximum_longer_token_length": successful_maximum_longer_length,
            "mean_shorter_token_length": round(successful_shorter_length_sum / successful_count, 6),
            "mean_longer_token_length": round(successful_longer_length_sum / successful_count, 6),
        }
    else:
        length_summary = {
            "pair_count": 0,
            "minimum_shorter_token_length": None,
            "maximum_longer_token_length": None,
            "mean_shorter_token_length": None,
            "mean_longer_token_length": None,
        }

    elapsed_seconds = time.perf_counter() - total_started
    status_order = (
        STRONG_ORTHOGRAPHIC_EVIDENCE,
        NEEDS_FURTHER_RESOLUTION,
        NOT_ELIGIBLE_FOR_ORTHOGRAPHIC,
    )
    rule_order = (
        SINGLE_TOKEN_EDIT_EQUIVALENCE,
        NO_ORTHOGRAPHIC_EQUIVALENCE,
        NOT_ELIGIBLE_RULE,
    )
    v2_1_guard_reasons = (
        "no_distinctive_exact_context",
        "multi_variant_context",
        "both_differing_tokens_established",
        "terminal_edit_requires_further_resolution",
    )
    support_band_order = ("1", "2-5", "6-10", "11-25", "26-50", "51-100", "101+")
    metrics: dict[str, JsonValue] = {
        "ruleset_version": orthographic_config.ruleset_version,
        "maximum_token_frequency": maximum_token_frequency,
        "v2_1_strong_orthographic_evidence_count": status_counts[STRONG_ORTHOGRAPHIC_EVIDENCE],
        "pair_decisions_read": pair_decisions_read,
        "candidate_feature_rows_read": pair_decisions_read,
        "prior_auto_same_pairs_skipped": prior_auto_same_skipped,
        "residual_pairs_read": residual_pairs,
        "eligible_employer_residual_pairs": eligible_employer_residual_pairs,
        "orthographic_status_counts": {status: status_counts[status] for status in status_order},
        "orthographic_rule_counts": {rule: rule_counts[rule] for rule in rule_order},
        "abstention_reason_counts": dict(sorted(abstention_reason_counts.items())),
        "v2_1_safety_abstention_counts": {
            reason: abstention_reason_counts[reason] for reason in v2_1_guard_reasons
        },
        "target_context_signatures_indexed": len(target_context_signatures),
        "successful_numeric_relation_distribution": {
            relation: successful_numeric_relation_counts[relation]
            for relation in ("none", "same", "one_sided", "conflict")
        },
        "successful_differing_token_length_summary": length_summary,
        "successful_edit_operation_counts": {
            operation: edit_operation_counts[operation]
            for operation in ("insertion", "deletion", "substitution")
        },
        "successful_context_variant_count_distribution": {
            str(count): successful_context_variant_counts[count]
            for count in sorted(successful_context_variant_counts)
        },
        "successful_token_support_distribution": {
            band: successful_token_support_bands[band] for band in support_band_order
        },
        "edit_location_distribution": {
            location: edit_location_counts[location] for location in ("beginning", "inside", "end")
        },
        "execution_runtime_seconds": round(elapsed_seconds, 6),
        "output_size_bytes": output_path.stat().st_size,
        "reconciliation": {
            "two_aligned_pair_passes_complete": True,
            "pair_decisions_and_features_aligned": True,
            "only_prior_residual_pairs_written": True,
            "one_output_row_per_prior_residual_pair": True,
            "persisted_identity_order_and_decisions_match": True,
            "status_counts_equal_rows_written": True,
            "status": "passed",
        },
    }
    _write_json(metrics_path, metrics)
    LOGGER.info(
        "Emitted strong orthographic evidence for %s of %s residual pairs; %s abstained and %s were ineligible",
        status_counts[STRONG_ORTHOGRAPHIC_EVIDENCE],
        residual_pairs,
        status_counts[NEEDS_FURTHER_RESOLUTION],
        status_counts[NOT_ELIGIBLE_FOR_ORTHOGRAPHIC],
    )
    return OrthographicResolutionResult(
        output_path=output_path,
        metrics_path=metrics_path,
        residual_pairs=residual_pairs,
        strong_orthographic_evidence_count=status_counts[STRONG_ORTHOGRAPHIC_EVIDENCE],
        needs_further_resolution_count=status_counts[NEEDS_FURTHER_RESOLUTION],
        not_eligible_count=status_counts[NOT_ELIGIBLE_FOR_ORTHOGRAPHIC],
        elapsed_seconds=elapsed_seconds,
    )


def _load_residual_profile_key_evidence(
    resolution_keys_path: Path,
    eligibility_path: Path,
    *,
    batch_size: int,
) -> tuple[dict[str, bool], dict[str, bool]]:
    """Load compact truncation and employer-compatibility evidence once per key."""
    keys_file = pq.ParquetFile(resolution_keys_path)
    eligibility_file = pq.ParquetFile(eligibility_path)
    if keys_file.schema_arrow != RESOLUTION_KEYS_SCHEMA:
        raise ValueError("Resolution-key dataset does not match the required schema")
    if eligibility_file.schema_arrow != EMPLOYER_ELIGIBILITY_SCHEMA:
        raise ValueError("Employer-eligibility dataset does not match the required schema")

    possible_truncation_by_key: dict[str, bool] = {}
    previous_key: str | None = None
    for row in _iter_parquet_rows(
        keys_file,
        batch_size=batch_size,
        columns=("resolution_key", "possible_truncation"),
    ):
        key = row.get("resolution_key")
        possible_truncation = row.get("possible_truncation")
        if not isinstance(key, str) or not key or not isinstance(possible_truncation, bool):
            raise ValueError("Resolution-key truncation evidence is invalid")
        if previous_key is not None and key <= previous_key:
            raise ValueError("Resolution keys must be unique and in ascending order")
        previous_key = key
        possible_truncation_by_key[key] = possible_truncation

    employer_candidate_by_key: dict[str, bool] = {}
    previous_key = None
    for row in _iter_parquet_rows(
        eligibility_file,
        batch_size=batch_size,
        columns=("resolution_key", "eligibility_status"),
    ):
        key = row.get("resolution_key")
        status = row.get("eligibility_status")
        if not isinstance(key, str) or not key or status not in _ELIGIBILITY_STATUSES:
            raise ValueError("Employer-eligibility evidence is invalid")
        if previous_key is not None and key <= previous_key:
            raise ValueError("Employer-eligibility keys must be unique and ascending")
        previous_key = key
        employer_candidate_by_key[key] = status == EMPLOYER_CANDIDATE

    if len(possible_truncation_by_key) != keys_file.metadata.num_rows:
        raise RuntimeError("Resolution-key evidence failed row reconciliation")
    if len(employer_candidate_by_key) != eligibility_file.metadata.num_rows:
        raise RuntimeError("Employer-eligibility evidence failed row reconciliation")
    if possible_truncation_by_key.keys() != employer_candidate_by_key.keys():
        raise ValueError("Resolution-key and employer-eligibility universes differ")
    return possible_truncation_by_key, employer_candidate_by_key


def _residual_profile_payload(
    *,
    key_a: str,
    key_b: str,
    name_a: str,
    name_b: str,
    profile: ResidualRelationshipProfile,
    prior_orthographic_evidence: str,
) -> dict[str, object]:
    return {
        "key_a": key_a,
        "key_b": key_b,
        "name_a": name_a,
        "name_b": name_b,
        "primary_family": profile.primary_family,
        "family_evidence": profile.family_evidence,
        "core_token_count_a": profile.core_token_count_a,
        "core_token_count_b": profile.core_token_count_b,
        "shared_exact_token_count": profile.shared_exact_token_count,
        "differing_token_positions": profile.differing_token_positions,
        "differing_token_count": profile.differing_token_count,
        "added_removed_token": profile.added_removed_token,
        "added_removed_token_count": profile.added_removed_token_count,
        "maximum_token_edit_distance": profile.maximum_token_edit_distance,
        "total_token_edit_distance": profile.total_token_edit_distance,
        "is_token_reorder": profile.is_token_reorder,
        "is_ordered_subsequence": profile.is_ordered_subsequence,
        "is_token_multiset_containment": profile.is_token_multiset_containment,
        "has_initialism_pattern": profile.has_initialism_pattern,
        "possible_truncation_a": profile.possible_truncation_a,
        "possible_truncation_b": profile.possible_truncation_b,
        "numeric_relation": profile.numeric_relation,
        "prior_orthographic_evidence": prior_orthographic_evidence,
    }


def _update_profile_digest(digest: hashlib._Hash, row: dict[str, object]) -> None:
    encoded = json.dumps(
        row,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    digest.update(len(encoded).to_bytes(8, byteorder="big"))
    digest.update(encoded)


def _weighted_integer_summary(values: Counter[int]) -> dict[str, JsonValue]:
    count = sum(values.values())
    if count == 0:
        return {
            "count": 0,
            "minimum": None,
            "median": None,
            "p95": None,
            "maximum": None,
            "mean": None,
        }

    def percentile(proportion: float) -> int:
        rank = max(1, int((proportion * count) + 0.999999999))
        cumulative = 0
        for value in sorted(values):
            cumulative += values[value]
            if cumulative >= rank:
                return value
        raise RuntimeError("Weighted percentile reconciliation failed")

    weighted_sum = sum(value * frequency for value, frequency in values.items())
    return {
        "count": count,
        "minimum": min(values),
        "median": percentile(0.5),
        "p95": percentile(0.95),
        "maximum": max(values),
        "mean": round(weighted_sum / count, 6),
    }


def profile_residual_relationships(
    root: Path,
    settings: Settings,
    *,
    orthographic_override: Path | None = None,
    features_override: Path | None = None,
    keys_override: Path | None = None,
    eligibility_override: Path | None = None,
    output_override: Path | None = None,
) -> ResidualProfileResult:
    """Stream and characterize orthographic residuals without identity decisions."""
    total_started = time.perf_counter()
    profile_config = settings.residual_profile

    def product_path(override: Path | None, configured: Path) -> Path:
        path = override or configured
        return path if path.is_absolute() else root / path

    orthographic_path = product_path(
        orthographic_override,
        profile_config.orthographic_decisions_dataset,
    )
    features_path = product_path(features_override, profile_config.candidate_features_dataset)
    keys_path = product_path(keys_override, profile_config.resolution_keys_dataset)
    eligibility_path = product_path(
        eligibility_override,
        profile_config.employer_eligibility_dataset,
    )
    output_path = product_path(output_override, profile_config.output_dataset)
    metrics_path = product_path(None, profile_config.metrics_output)
    paths = (
        orthographic_path,
        features_path,
        keys_path,
        eligibility_path,
        output_path,
        metrics_path,
    )
    if len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("Residual-profile inputs, output, and metrics paths must be distinct")
    for label, path in (
        ("Orthographic decision", orthographic_path),
        ("Candidate feature", features_path),
        ("Resolution key", keys_path),
        ("Employer eligibility", eligibility_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} dataset does not exist: {path}")

    orthographic_file = pq.ParquetFile(orthographic_path)
    feature_file = pq.ParquetFile(features_path)
    if orthographic_file.schema_arrow != ORTHOGRAPHIC_PAIR_DECISIONS_SCHEMA:
        raise ValueError("Orthographic decision dataset does not match the required schema")
    if feature_file.schema_arrow != CANDIDATE_FEATURES_SCHEMA:
        raise ValueError("Candidate-feature dataset does not match the required schema")

    possible_truncation_by_key, employer_candidate_by_key = _load_residual_profile_key_evidence(
        keys_path,
        eligibility_path,
        batch_size=profile_config.batch_size,
    )
    policy = build_orthographic_policy(
        corporate_suffix_aliases=settings.normalization.corporate_suffix_aliases,
        minimum_typo_token_length=settings.candidate_generation.minimum_typo_token_length,
        minimum_informative_token_length=(
            settings.candidate_generation.minimum_informative_token_length
        ),
    )
    source_truncation_boundaries = frozenset(
        settings.normalization.possible_truncation_content_lengths
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(f"{output_path.suffix}.tmp")
    temporary.unlink(missing_ok=True)
    compression = (
        None
        if settings.processing.parquet_compression == "none"
        else settings.processing.parquet_compression
    )

    orthographic_columns = (
        *_CANDIDATE_IDENTITY_COLUMNS,
        "orthographic_status",
        "orthographic_evidence",
    )
    feature_columns = (*_CANDIDATE_IDENTITY_COLUMNS, "numeric_relation")
    orthographic_rows = _iter_parquet_rows(
        orthographic_file,
        batch_size=profile_config.batch_size,
        columns=orthographic_columns,
    )
    feature_rows = _iter_parquet_rows(
        feature_file,
        batch_size=profile_config.batch_size,
        columns=feature_columns,
    )

    previous_orthographic_pair: tuple[str, str] | None = None
    previous_feature_pair: tuple[str, str] | None = None
    orthographic_rows_read = 0
    feature_rows_read = 0
    matched_pair_rows = 0
    skipped_strong = 0
    skipped_ineligible = 0
    profiled_rows = 0
    family_counts: Counter[str] = Counter()
    family_numeric_counts: Counter[tuple[str, str]] = Counter()
    core_token_delta_counts: Counter[int] = Counter()
    differing_token_counts: Counter[int] = Counter()
    maximum_edit_distance_counts: Counter[int] = Counter()
    total_edit_distance_counts: Counter[int] = Counter()
    expected_digest = hashlib.sha256()

    def next_feature() -> tuple[dict[str, object], tuple[str, str, str, str, list[str]]] | None:
        nonlocal feature_rows_read, previous_feature_pair
        try:
            row = next(feature_rows)
        except StopIteration:
            return None
        identity = _validated_candidate_row(row, previous_feature_pair)
        previous_feature_pair = (identity[0], identity[1])
        feature_rows_read += 1
        return row, identity

    current_feature = next_feature()
    writer: pq.ParquetWriter | None = None
    try:
        writer = pq.ParquetWriter(
            temporary,
            RESIDUAL_RELATIONSHIP_PROFILE_SCHEMA,
            compression=compression,
            use_dictionary=True,
            write_statistics=True,
        )
        output_rows: list[dict[str, object]] = []
        for orthographic_row in orthographic_rows:
            orthographic_identity = _validated_candidate_row(
                orthographic_row,
                previous_orthographic_pair,
            )
            key_a, key_b, name_a, name_b, _ = orthographic_identity
            previous_orthographic_pair = (key_a, key_b)
            orthographic_rows_read += 1
            orthographic_pair = (key_a, key_b)

            while (
                current_feature is not None
                and (current_feature[1][0], current_feature[1][1]) < orthographic_pair
            ):
                current_feature = next_feature()
            if current_feature is None:
                raise ValueError("Candidate features ended before orthographic decisions")
            feature_row, feature_identity = current_feature
            feature_pair = (feature_identity[0], feature_identity[1])
            if feature_pair != orthographic_pair or feature_identity != orthographic_identity:
                raise ValueError("Orthographic and feature pair identities are not aligned")
            matched_pair_rows += 1
            numeric_relation = _validated_numeric_relation(feature_row)
            current_feature = next_feature()

            status = orthographic_row.get("orthographic_status")
            prior_evidence = orthographic_row.get("orthographic_evidence")
            if not isinstance(prior_evidence, str) or not prior_evidence:
                raise ValueError("Orthographic evidence must be a nonblank string")
            if status == STRONG_ORTHOGRAPHIC_EVIDENCE:
                skipped_strong += 1
                continue
            if status == NOT_ELIGIBLE_FOR_ORTHOGRAPHIC:
                skipped_ineligible += 1
                continue
            if status != NEEDS_FURTHER_RESOLUTION:
                raise ValueError(f"Invalid orthographic status: {status!r}")
            if not employer_candidate_by_key.get(key_a, False) or not employer_candidate_by_key.get(
                key_b, False
            ):
                raise RuntimeError("Profiled residual is not employer-compatible on both sides")

            profile = profile_residual_relationship(
                key_a=key_a,
                key_b=key_b,
                name_a=name_a,
                name_b=name_b,
                numeric_relation=numeric_relation,
                possible_truncation_a=possible_truncation_by_key[key_a],
                possible_truncation_b=possible_truncation_by_key[key_b],
                source_truncation_boundaries=source_truncation_boundaries,
                prior_orthographic_evidence=prior_evidence,
                policy=policy,
            )
            payload = _residual_profile_payload(
                key_a=key_a,
                key_b=key_b,
                name_a=name_a,
                name_b=name_b,
                profile=profile,
                prior_orthographic_evidence=prior_evidence,
            )
            output_rows.append(payload)
            _update_profile_digest(expected_digest, payload)
            profiled_rows += 1
            family_counts[profile.primary_family] += 1
            family_numeric_counts[(profile.primary_family, numeric_relation)] += 1
            core_token_delta_counts[profile.core_token_count_b - profile.core_token_count_a] += 1
            differing_token_counts[profile.differing_token_count] += 1
            if profile.maximum_token_edit_distance is not None:
                maximum_edit_distance_counts[profile.maximum_token_edit_distance] += 1
            if profile.total_token_edit_distance is not None:
                total_edit_distance_counts[profile.total_token_edit_distance] += 1

            if len(output_rows) >= profile_config.batch_size:
                writer.write_table(
                    pa.Table.from_pylist(
                        output_rows,
                        schema=RESIDUAL_RELATIONSHIP_PROFILE_SCHEMA,
                    )
                )
                output_rows.clear()
        if output_rows:
            writer.write_table(
                pa.Table.from_pylist(
                    output_rows,
                    schema=RESIDUAL_RELATIONSHIP_PROFILE_SCHEMA,
                )
            )
        while current_feature is not None:
            current_feature = next_feature()
    except Exception:
        if writer is not None:
            writer.close()
            writer = None
        temporary.unlink(missing_ok=True)
        raise
    finally:
        if writer is not None:
            writer.close()
    temporary.replace(output_path)

    output_file = pq.ParquetFile(output_path)
    if output_file.schema_arrow != RESIDUAL_RELATIONSHIP_PROFILE_SCHEMA:
        raise RuntimeError("Residual-profile output schema failed reconciliation")
    output_digest = hashlib.sha256()
    output_rows_reconciled = 0
    previous_output_pair: tuple[str, str] | None = None
    for row in _iter_parquet_rows(
        output_file,
        batch_size=profile_config.batch_size,
        columns=tuple(RESIDUAL_RELATIONSHIP_PROFILE_SCHEMA.names),
    ):
        output_key_a = row.get("key_a")
        output_key_b = row.get("key_b")
        family = row.get("primary_family")
        output_numeric_relation = row.get("numeric_relation")
        evidence = row.get("family_evidence")
        if (
            not isinstance(output_key_a, str)
            or not isinstance(output_key_b, str)
            or output_key_a >= output_key_b
            or (
                previous_output_pair is not None
                and (output_key_a, output_key_b) <= previous_output_pair
            )
            or family not in PRIMARY_FAMILY_PRECEDENCE
            or output_numeric_relation not in {"none", "same", "one_sided", "conflict"}
            or not isinstance(evidence, str)
            or not evidence
        ):
            raise RuntimeError("Residual-profile output contains invalid values or order")
        previous_output_pair = (output_key_a, output_key_b)
        _update_profile_digest(output_digest, row)
        output_rows_reconciled += 1

    population_reconciles = (
        orthographic_rows_read == orthographic_file.metadata.num_rows
        and feature_rows_read == feature_file.metadata.num_rows
        and matched_pair_rows == orthographic_rows_read
        and orthographic_rows_read == profiled_rows + skipped_strong + skipped_ineligible
        and profiled_rows
        == output_file.metadata.num_rows
        == output_rows_reconciled
        == sum(family_counts.values())
    )
    digest_reconciles = expected_digest.digest() == output_digest.digest()
    if not population_reconciles or not digest_reconciles:
        raise RuntimeError("Residual-profile output failed population or digest reconciliation")

    elapsed_seconds = time.perf_counter() - total_started
    numeric_relations = ("none", "same", "one_sided", "conflict")
    family_metrics: dict[str, JsonValue] = {
        family: {
            "count": family_counts[family],
            "percentage": (
                round((family_counts[family] / profiled_rows) * 100, 6) if profiled_rows else 0.0
            ),
        }
        for family in PRIMARY_FAMILY_PRECEDENCE
    }
    family_numeric_cross_tab: dict[str, JsonValue] = {
        family: {
            relation: family_numeric_counts[(family, relation)] for relation in numeric_relations
        }
        for family in PRIMARY_FAMILY_PRECEDENCE
    }
    metrics: dict[str, JsonValue] = {
        "ruleset_version": profile_config.ruleset_version,
        "pair_rows_read": orthographic_rows_read,
        "candidate_feature_rows_read": feature_rows_read,
        "profiled_residual_rows": profiled_rows,
        "skipped_strong_orthographic_rows": skipped_strong,
        "skipped_ineligible_rows": skipped_ineligible,
        "primary_family_distribution": family_metrics,
        "primary_family_by_numeric_relation": family_numeric_cross_tab,
        "core_token_count_delta_distribution": {
            str(delta): core_token_delta_counts[delta] for delta in sorted(core_token_delta_counts)
        },
        "differing_token_count_summary": _weighted_integer_summary(differing_token_counts),
        "maximum_token_edit_distance_summary": _weighted_integer_summary(
            maximum_edit_distance_counts
        ),
        "total_token_edit_distance_summary": _weighted_integer_summary(total_edit_distance_counts),
        "truncation_family_count": family_counts["POSSIBLE_TRUNCATION_RELATIONSHIP"],
        "acronym_initialism_family_count": family_counts["ACRONYM_INITIALISM_PATTERN"],
        "containment_addition_removal_counts": {
            family: family_counts[family]
            for family in (
                "SINGLE_TOKEN_ADDITION_REMOVAL",
                "MULTI_TOKEN_ADDITION_REMOVAL",
                "EXACT_TOKEN_CONTAINMENT_NONALIGNED",
            )
        },
        "other_residual_count": family_counts["OTHER_RESIDUAL"],
        "execution_runtime_seconds": round(elapsed_seconds, 6),
        "output_size_bytes": output_path.stat().st_size,
        "reconciliation": {
            "orthographic_and_feature_pair_identities_aligned": True,
            "only_needs_further_resolution_rows_written": True,
            "one_primary_family_per_profiled_pair": True,
            "pair_order_preserved": True,
            "persisted_rows_match_generated_profiles": True,
            "status_counts_equal_orthographic_rows": True,
            "status": "passed",
        },
    }
    _write_json(metrics_path, metrics)
    LOGGER.info(
        "Profiled %s employer-compatible residual pairs into %s structural families",
        profiled_rows,
        sum(count > 0 for count in family_counts.values()),
    )
    return ResidualProfileResult(
        output_path=output_path,
        metrics_path=metrics_path,
        profiled_residual_rows=profiled_rows,
        skipped_strong_orthographic_rows=skipped_strong,
        skipped_ineligible_rows=skipped_ineligible,
        family_counts={family: family_counts[family] for family in PRIMARY_FAMILY_PRECEDENCE},
        elapsed_seconds=elapsed_seconds,
    )


def _load_employer_token_support(
    resolution_keys_path: Path,
    eligibility_path: Path,
    *,
    policy: OrthographicPolicy,
    batch_size: int,
) -> tuple[dict[str, int], int]:
    """Count each core token once per aligned employer-candidate resolution key."""
    keys_file = pq.ParquetFile(resolution_keys_path)
    eligibility_file = pq.ParquetFile(eligibility_path)
    if keys_file.schema_arrow != RESOLUTION_KEYS_SCHEMA:
        raise ValueError("Resolution-key dataset does not match the required schema")
    if eligibility_file.schema_arrow != EMPLOYER_ELIGIBILITY_SCHEMA:
        raise ValueError("Employer-eligibility dataset does not match the required schema")
    if keys_file.metadata.num_rows != eligibility_file.metadata.num_rows:
        raise ValueError("Resolution-key and employer-eligibility row counts differ")

    key_rows = _iter_parquet_rows(
        keys_file,
        batch_size=batch_size,
        columns=("resolution_key", "representative_name"),
    )
    eligibility_rows = _iter_parquet_rows(
        eligibility_file,
        batch_size=batch_size,
        columns=("resolution_key", "representative_name", "eligibility_status"),
    )
    support: Counter[str] = Counter()
    employer_keys = 0
    previous_key: str | None = None
    rows_read = 0
    for key_row, eligibility_row in zip(key_rows, eligibility_rows, strict=True):
        key = key_row.get("resolution_key")
        name = key_row.get("representative_name")
        eligibility_key = eligibility_row.get("resolution_key")
        eligibility_name = eligibility_row.get("representative_name")
        status = eligibility_row.get("eligibility_status")
        if (
            not isinstance(key, str)
            or not key
            or not isinstance(name, str)
            or not name
            or key != eligibility_key
            or name != eligibility_name
            or status not in _ELIGIBILITY_STATUSES
        ):
            raise ValueError("Resolution-key and eligibility evidence are not aligned")
        if previous_key is not None and key <= previous_key:
            raise ValueError("Employer evidence keys must be unique and ascending")
        previous_key = key
        rows_read += 1
        if status == EMPLOYER_CANDIDATE:
            employer_keys += 1
            support.update(employer_core_token_membership(name, policy))
    if rows_read != keys_file.metadata.num_rows:
        raise RuntimeError("Employer token-support input failed row reconciliation")
    return dict(support), employer_keys


def _distinctive_evidence_payload(
    *,
    key_a: str,
    key_b: str,
    name_a: str,
    name_b: str,
    primary_family: str,
    numeric_relation: str,
    evidence: DistinctiveNameEvidence,
) -> dict[str, object]:
    return {
        "key_a": key_a,
        "key_b": key_b,
        "name_a": name_a,
        "name_b": name_b,
        "shared_exact_tokens": evidence.shared_exact_tokens,
        "shared_distinctive_tokens": evidence.shared_distinctive_tokens,
        "shared_exact_token_count": evidence.shared_exact_token_count,
        "shared_distinctive_token_count": evidence.shared_distinctive_token_count,
        "shared_generic_token_count": evidence.shared_generic_token_count,
        "minimum_shared_token_support": evidence.minimum_shared_token_support,
        "maximum_shared_token_support": evidence.maximum_shared_token_support,
        "minimum_distinctive_token_support": evidence.minimum_distinctive_token_support,
        "maximum_distinctive_token_support": evidence.maximum_distinctive_token_support,
        "exact_coverage_a": evidence.exact_coverage_a,
        "exact_coverage_b": evidence.exact_coverage_b,
        "distinctive_coverage_a": evidence.distinctive_coverage_a,
        "distinctive_coverage_b": evidence.distinctive_coverage_b,
        "shorter_name_exact_coverage": evidence.shorter_name_exact_coverage,
        "longer_name_exact_coverage": evidence.longer_name_exact_coverage,
        "shorter_name_distinctive_coverage": evidence.shorter_name_distinctive_coverage,
        "longer_name_distinctive_coverage": evidence.longer_name_distinctive_coverage,
        "has_shared_exact_token": evidence.has_shared_exact_token,
        "has_shared_distinctive_token": evidence.has_shared_distinctive_token,
        "has_multiple_shared_distinctive_tokens": (evidence.has_multiple_shared_distinctive_tokens),
        "has_exact_overlap_without_distinctive_token": (
            evidence.has_exact_overlap_without_distinctive_token
        ),
        "primary_family": primary_family,
        "numeric_relation": numeric_relation,
    }


def _weighted_float_summary(values: Counter[float]) -> dict[str, JsonValue]:
    count = sum(values.values())
    if count == 0:
        return {
            "count": 0,
            "minimum": None,
            "median": None,
            "p95": None,
            "maximum": None,
            "mean": None,
        }

    def percentile(proportion: float) -> float:
        rank = max(1, int((proportion * count) + 0.999999999))
        cumulative = 0
        for value in sorted(values):
            cumulative += values[value]
            if cumulative >= rank:
                return value
        raise RuntimeError("Weighted coverage percentile reconciliation failed")

    weighted_sum = sum(value * frequency for value, frequency in values.items())
    return {
        "count": count,
        "minimum": min(values),
        "median": percentile(0.5),
        "p95": percentile(0.95),
        "maximum": max(values),
        "mean": round(weighted_sum / count, 6),
    }


def compute_distinctive_evidence(
    root: Path,
    settings: Settings,
    *,
    profile_override: Path | None = None,
    eligibility_override: Path | None = None,
    keys_override: Path | None = None,
    output_override: Path | None = None,
) -> DistinctiveEvidenceResult:
    """Stream residual pairs into exact token-rarity evidence without identity outcomes."""
    total_started = time.perf_counter()
    evidence_config = settings.distinctive_evidence

    def product_path(override: Path | None, configured: Path) -> Path:
        path = override or configured
        return path if path.is_absolute() else root / path

    profile_path = product_path(profile_override, evidence_config.residual_profile_dataset)
    eligibility_path = product_path(
        eligibility_override,
        evidence_config.employer_eligibility_dataset,
    )
    keys_path = product_path(keys_override, evidence_config.resolution_keys_dataset)
    output_path = product_path(output_override, evidence_config.output_dataset)
    metrics_path = product_path(None, evidence_config.metrics_output)
    paths = (profile_path, eligibility_path, keys_path, output_path, metrics_path)
    if len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("Distinctive-evidence inputs, output, and metrics must be distinct")
    for label, path in (
        ("Residual profile", profile_path),
        ("Employer eligibility", eligibility_path),
        ("Resolution key", keys_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} dataset does not exist: {path}")

    profile_file = pq.ParquetFile(profile_path)
    if profile_file.schema_arrow != RESIDUAL_RELATIONSHIP_PROFILE_SCHEMA:
        raise ValueError("Residual-profile dataset does not match the required schema")
    policy = build_orthographic_policy(
        corporate_suffix_aliases=settings.normalization.corporate_suffix_aliases,
        minimum_typo_token_length=settings.candidate_generation.minimum_typo_token_length,
        minimum_informative_token_length=(
            settings.candidate_generation.minimum_informative_token_length
        ),
    )
    token_support, employer_keys_used = _load_employer_token_support(
        keys_path,
        eligibility_path,
        policy=policy,
        batch_size=evidence_config.batch_size,
    )
    maximum_token_frequency = settings.candidate_generation.maximum_token_frequency

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(f"{output_path.suffix}.tmp")
    temporary.unlink(missing_ok=True)
    compression = (
        None
        if settings.processing.parquet_compression == "none"
        else settings.processing.parquet_compression
    )
    input_columns = (
        "key_a",
        "key_b",
        "name_a",
        "name_b",
        "primary_family",
        "numeric_relation",
    )

    rows_read = 0
    rows_written = 0
    pairs_with_zero_exact_overlap = 0
    pairs_with_exact_overlap = 0
    pairs_with_distinctive_overlap = 0
    pairs_with_multiple_distinctive = 0
    pairs_with_exact_overlap_without_distinctive = 0
    family_overlap_counts: Counter[tuple[str, str]] = Counter()
    shared_token_counts: Counter[int] = Counter()
    distinctive_token_counts: Counter[int] = Counter()
    coverage_counts: dict[str, Counter[float]] = {
        key: Counter()
        for key in (
            "exact_a",
            "exact_b",
            "exact_shorter",
            "exact_longer",
            "distinctive_a",
            "distinctive_b",
            "distinctive_shorter",
            "distinctive_longer",
        )
    }
    minimum_distinctive_support_bands: Counter[str] = Counter()
    expected_digest = hashlib.sha256()
    previous_pair: tuple[str, str] | None = None
    writer: pq.ParquetWriter | None = None
    try:
        writer = pq.ParquetWriter(
            temporary,
            DISTINCTIVE_NAME_EVIDENCE_SCHEMA,
            compression=compression,
            use_dictionary=True,
            write_statistics=True,
        )
        output_rows: list[dict[str, object]] = []
        for row in _iter_parquet_rows(
            profile_file,
            batch_size=evidence_config.batch_size,
            columns=input_columns,
        ):
            key_a = row.get("key_a")
            key_b = row.get("key_b")
            name_a = row.get("name_a")
            name_b = row.get("name_b")
            primary_family = row.get("primary_family")
            numeric_relation = row.get("numeric_relation")
            if (
                not isinstance(key_a, str)
                or not isinstance(key_b, str)
                or not isinstance(name_a, str)
                or not isinstance(name_b, str)
                or key_a >= key_b
                or (previous_pair is not None and (key_a, key_b) <= previous_pair)
                or primary_family not in PRIMARY_FAMILY_PRECEDENCE
                or numeric_relation not in {"none", "same", "one_sided", "conflict"}
            ):
                raise ValueError("Residual profile contains invalid identity, order, or context")
            previous_pair = (key_a, key_b)
            rows_read += 1
            evidence = compute_distinctive_name_evidence(
                name_a=name_a,
                name_b=name_b,
                token_support=token_support,
                maximum_token_frequency=maximum_token_frequency,
                policy=policy,
            )
            payload = _distinctive_evidence_payload(
                key_a=key_a,
                key_b=key_b,
                name_a=name_a,
                name_b=name_b,
                primary_family=primary_family,
                numeric_relation=numeric_relation,
                evidence=evidence,
            )
            output_rows.append(payload)
            _update_profile_digest(expected_digest, payload)
            rows_written += 1
            shared_token_counts[evidence.shared_exact_token_count] += 1
            distinctive_token_counts[evidence.shared_distinctive_token_count] += 1
            if evidence.has_shared_exact_token:
                pairs_with_exact_overlap += 1
            else:
                pairs_with_zero_exact_overlap += 1
            if evidence.has_shared_distinctive_token:
                pairs_with_distinctive_overlap += 1
            if evidence.has_multiple_shared_distinctive_tokens:
                pairs_with_multiple_distinctive += 1
            if evidence.has_exact_overlap_without_distinctive_token:
                pairs_with_exact_overlap_without_distinctive += 1
            overlap_category = (
                "two_or_more_distinctive"
                if evidence.shared_distinctive_token_count >= 2
                else "one_distinctive"
                if evidence.shared_distinctive_token_count == 1
                else "zero_distinctive"
            )
            family_overlap_counts[(primary_family, overlap_category)] += 1
            coverage_counts["exact_a"][evidence.exact_coverage_a] += 1
            coverage_counts["exact_b"][evidence.exact_coverage_b] += 1
            coverage_counts["exact_shorter"][evidence.shorter_name_exact_coverage] += 1
            coverage_counts["exact_longer"][evidence.longer_name_exact_coverage] += 1
            coverage_counts["distinctive_a"][evidence.distinctive_coverage_a] += 1
            coverage_counts["distinctive_b"][evidence.distinctive_coverage_b] += 1
            coverage_counts["distinctive_shorter"][evidence.shorter_name_distinctive_coverage] += 1
            coverage_counts["distinctive_longer"][evidence.longer_name_distinctive_coverage] += 1
            if evidence.minimum_distinctive_token_support is not None:
                minimum_distinctive_support_bands[
                    _token_support_band(evidence.minimum_distinctive_token_support)
                ] += 1

            if len(output_rows) >= evidence_config.batch_size:
                writer.write_table(
                    pa.Table.from_pylist(
                        output_rows,
                        schema=DISTINCTIVE_NAME_EVIDENCE_SCHEMA,
                    )
                )
                output_rows.clear()
        if output_rows:
            writer.write_table(
                pa.Table.from_pylist(
                    output_rows,
                    schema=DISTINCTIVE_NAME_EVIDENCE_SCHEMA,
                )
            )
    except Exception:
        if writer is not None:
            writer.close()
            writer = None
        temporary.unlink(missing_ok=True)
        raise
    finally:
        if writer is not None:
            writer.close()
    temporary.replace(output_path)

    output_file = pq.ParquetFile(output_path)
    if output_file.schema_arrow != DISTINCTIVE_NAME_EVIDENCE_SCHEMA:
        raise RuntimeError("Distinctive-evidence output schema failed reconciliation")
    output_digest = hashlib.sha256()
    output_rows_reconciled = 0
    previous_output_pair: tuple[str, str] | None = None
    for row in _iter_parquet_rows(
        output_file,
        batch_size=evidence_config.batch_size,
        columns=tuple(DISTINCTIVE_NAME_EVIDENCE_SCHEMA.names),
    ):
        output_key_a = row.get("key_a")
        output_key_b = row.get("key_b")
        if (
            not isinstance(output_key_a, str)
            or not isinstance(output_key_b, str)
            or output_key_a >= output_key_b
            or (
                previous_output_pair is not None
                and (output_key_a, output_key_b) <= previous_output_pair
            )
        ):
            raise RuntimeError("Distinctive-evidence output pair order is invalid")
        previous_output_pair = (output_key_a, output_key_b)
        _update_profile_digest(output_digest, row)
        output_rows_reconciled += 1

    counts_reconcile = (
        rows_read
        == profile_file.metadata.num_rows
        == rows_written
        == output_file.metadata.num_rows
        == output_rows_reconciled
        and rows_written == pairs_with_zero_exact_overlap + pairs_with_exact_overlap
        and rows_written == sum(shared_token_counts.values())
        and rows_written == sum(distinctive_token_counts.values())
    )
    digest_reconciles = expected_digest.digest() == output_digest.digest()
    if not counts_reconcile or not digest_reconciles:
        raise RuntimeError("Distinctive-evidence output failed reconciliation")

    elapsed_seconds = time.perf_counter() - total_started
    family_distribution: dict[str, JsonValue] = {
        family: {
            "zero_distinctive_overlap": family_overlap_counts[(family, "zero_distinctive")],
            "at_least_one_distinctive_token": (
                family_overlap_counts[(family, "one_distinctive")]
                + family_overlap_counts[(family, "two_or_more_distinctive")]
            ),
            "at_least_two_distinctive_tokens": family_overlap_counts[
                (family, "two_or_more_distinctive")
            ],
        }
        for family in PRIMARY_FAMILY_PRECEDENCE
    }
    coverage_metrics: dict[str, JsonValue] = {
        name: _weighted_float_summary(coverage_counts[name]) for name in coverage_counts
    }
    support_band_order = ("1", "2-5", "6-10", "11-25", "26-50", "51-100", "101+")
    metrics: dict[str, JsonValue] = {
        "ruleset_version": evidence_config.ruleset_version,
        "maximum_token_frequency": maximum_token_frequency,
        "profiled_pairs_read": rows_read,
        "output_rows": rows_written,
        "employer_keys_used_for_support": employer_keys_used,
        "unique_employer_tokens": len(token_support),
        "pairs_with_zero_exact_overlap": pairs_with_zero_exact_overlap,
        "pairs_with_exact_overlap": pairs_with_exact_overlap,
        "pairs_with_at_least_one_distinctive_token": pairs_with_distinctive_overlap,
        "pairs_with_at_least_two_distinctive_tokens": pairs_with_multiple_distinctive,
        "pairs_with_exact_overlap_without_distinctive_token": (
            pairs_with_exact_overlap_without_distinctive
        ),
        "distinctive_overlap_by_primary_family": family_distribution,
        "shared_token_count_summary": _weighted_integer_summary(shared_token_counts),
        "distinctive_token_count_summary": _weighted_integer_summary(distinctive_token_counts),
        "coverage_summaries": coverage_metrics,
        "minimum_distinctive_support_band_distribution": {
            band: minimum_distinctive_support_bands[band] for band in support_band_order
        },
        "execution_runtime_seconds": round(elapsed_seconds, 6),
        "output_size_bytes": output_path.stat().st_size,
        "reconciliation": {
            "one_output_row_per_profiled_pair": True,
            "pair_identity_and_order_preserved": True,
            "support_uses_aligned_resolution_key_eligibility_universe": True,
            "persisted_rows_match_computed_evidence": True,
            "overlap_counts_equal_output_rows": True,
            "status": "passed",
        },
    }
    _write_json(metrics_path, metrics)
    LOGGER.info(
        "Computed distinctive-name evidence for %s pairs; %s have exact overlap and %s have distinctive overlap",
        rows_written,
        pairs_with_exact_overlap,
        pairs_with_distinctive_overlap,
    )
    return DistinctiveEvidenceResult(
        output_path=output_path,
        metrics_path=metrics_path,
        output_rows=rows_written,
        pairs_with_exact_overlap=pairs_with_exact_overlap,
        pairs_with_distinctive_overlap=pairs_with_distinctive_overlap,
        pairs_with_multiple_distinctive_tokens=pairs_with_multiple_distinctive,
        elapsed_seconds=elapsed_seconds,
    )


def _validated_pair_identity(
    row: dict[str, object],
    previous_pair: tuple[str, str] | None,
    *,
    source: str,
) -> tuple[str, str, str, str]:
    values = tuple(row.get(column) for column in _CANDIDATE_IDENTITY_COLUMNS[:4])
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError(f"{source} pair identity must contain four nonblank strings")
    key_a, key_b, name_a, name_b = cast(tuple[str, str, str, str], values)
    pair = (key_a, key_b)
    if key_a >= key_b or (previous_pair is not None and pair <= previous_pair):
        raise ValueError(f"{source} pairs must be unique and in deterministic ascending order")
    return key_a, key_b, name_a, name_b


def _required_string(row: dict[str, object], column: str, *, source: str) -> str:
    value = row.get(column)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{source} has invalid {column}")
    return value


def _required_int(row: dict[str, object], column: str, *, source: str) -> int:
    value = row.get(column)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{source} has invalid {column}")
    return value


def _nullable_int(row: dict[str, object], column: str, *, source: str) -> int | None:
    value = row.get(column)
    if value is not None and (not isinstance(value, int) or isinstance(value, bool)):
        raise ValueError(f"{source} has invalid {column}")
    return value


def _required_float(row: dict[str, object], column: str, *, source: str) -> float:
    value = row.get(column)
    if not isinstance(value, (float, int)) or isinstance(value, bool):
        raise ValueError(f"{source} has invalid {column}")
    return float(value)


def _required_bool(row: dict[str, object], column: str, *, source: str) -> bool:
    value = row.get(column)
    if not isinstance(value, bool):
        raise ValueError(f"{source} has invalid {column}")
    return value


def _next_validated_candidate(
    rows: Iterator[dict[str, object]],
    previous_pair: tuple[str, str] | None,
) -> tuple[dict[str, object] | None, tuple[str, str, str, str, list[str]] | None]:
    row = next(rows, None)
    if row is None:
        return None, None
    return row, _validated_candidate_row(row, previous_pair)


def _multi_evidence_payload(
    *,
    identity: tuple[str, str, str, str],
    profile_row: dict[str, object],
    distinctive_row: dict[str, object],
    feature_row: dict[str, object],
    assessment: MultiEvidenceAssessment,
    prior_orthographic_evidence: str,
) -> dict[str, object]:
    key_a, key_b, name_a, name_b = identity
    payload: dict[str, object] = {
        "key_a": key_a,
        "key_b": key_b,
        "name_a": name_a,
        "name_b": name_b,
        "assessment_family": assessment.assessment_family,
        "assessment_evidence": assessment.assessment_evidence,
        "primary_family": _required_string(
            profile_row, "primary_family", source="Residual profile"
        ),
        "numeric_relation": _required_string(
            profile_row, "numeric_relation", source="Residual profile"
        ),
        "has_structural_exact_relation": assessment.has_structural_exact_relation,
        "has_orthographic_signal": assessment.has_orthographic_signal,
        "has_distinctive_exact_overlap": assessment.has_distinctive_exact_overlap,
        "has_multiple_distinctive_exact_overlap": (
            assessment.has_multiple_distinctive_exact_overlap
        ),
        "has_numeric_risk": assessment.has_numeric_risk,
        "has_zero_exact_overlap": assessment.has_zero_exact_overlap,
        "has_full_shorter_exact_coverage": assessment.has_full_shorter_exact_coverage,
        "has_full_shorter_distinctive_coverage": (assessment.has_full_shorter_distinctive_coverage),
        "prior_orthographic_evidence": prior_orthographic_evidence,
    }
    for column in (
        "core_token_count_a",
        "core_token_count_b",
        "shared_exact_token_count",
        "differing_token_count",
    ):
        payload[column] = _required_int(profile_row, column, source="Residual profile")
    for column in ("maximum_token_edit_distance", "total_token_edit_distance"):
        payload[column] = _nullable_int(profile_row, column, source="Residual profile")
    for column in (
        "is_token_reorder",
        "is_ordered_subsequence",
        "is_token_multiset_containment",
        "possible_truncation_a",
        "possible_truncation_b",
    ):
        payload[column] = _required_bool(profile_row, column, source="Residual profile")
    payload["shared_distinctive_token_count"] = _required_int(
        distinctive_row, "shared_distinctive_token_count", source="Distinctive evidence"
    )
    for column in ("minimum_distinctive_token_support", "maximum_distinctive_token_support"):
        payload[column] = _nullable_int(distinctive_row, column, source="Distinctive evidence")
    for column in (
        "exact_coverage_a",
        "exact_coverage_b",
        "distinctive_coverage_a",
        "distinctive_coverage_b",
        "shorter_name_exact_coverage",
        "longer_name_exact_coverage",
        "shorter_name_distinctive_coverage",
        "longer_name_distinctive_coverage",
    ):
        payload[column] = _required_float(distinctive_row, column, source="Distinctive evidence")
    for column in (
        "has_shared_distinctive_token",
        "has_multiple_shared_distinctive_tokens",
        "has_exact_overlap_without_distinctive_token",
    ):
        payload[column] = _required_bool(distinctive_row, column, source="Distinctive evidence")
    for column in _NUMERIC_FEATURE_COLUMNS:
        payload[column] = _required_float(feature_row, column, source="Candidate features")
    payload["same_first_token"] = _required_bool(
        feature_row, "same_first_token", source="Candidate features"
    )
    return payload


def assess_pair_evidence(
    root: Path,
    settings: Settings,
    *,
    profile_override: Path | None = None,
    distinctive_override: Path | None = None,
    features_override: Path | None = None,
    orthographic_override: Path | None = None,
    output_override: Path | None = None,
) -> MultiEvidenceAssessmentResult:
    """Stream-align approved evidence into one diagnostic row per residual pair."""
    total_started = time.perf_counter()
    config = settings.multi_evidence_assessment

    def product_path(override: Path | None, configured: Path) -> Path:
        path = override or configured
        return path if path.is_absolute() else root / path

    profile_path = product_path(profile_override, config.residual_profile_dataset)
    distinctive_path = product_path(distinctive_override, config.distinctive_evidence_dataset)
    feature_path = product_path(features_override, config.candidate_features_dataset)
    orthographic_path = product_path(orthographic_override, config.orthographic_decisions_dataset)
    output_path = product_path(output_override, config.output_dataset)
    metrics_path = product_path(None, config.metrics_output)
    paths = (
        profile_path,
        distinctive_path,
        feature_path,
        orthographic_path,
        output_path,
        metrics_path,
    )
    if len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("Multi-evidence inputs, output, and metrics must be distinct")
    for label, path in (
        ("Residual profile", profile_path),
        ("Distinctive evidence", distinctive_path),
        ("Candidate features", feature_path),
        ("Orthographic decisions", orthographic_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} dataset does not exist: {path}")

    profile_file = pq.ParquetFile(profile_path)
    distinctive_file = pq.ParquetFile(distinctive_path)
    feature_file = pq.ParquetFile(feature_path)
    orthographic_file = pq.ParquetFile(orthographic_path)
    expected_schemas = (
        (profile_file, RESIDUAL_RELATIONSHIP_PROFILE_SCHEMA, "Residual profile"),
        (distinctive_file, DISTINCTIVE_NAME_EVIDENCE_SCHEMA, "Distinctive evidence"),
        (feature_file, CANDIDATE_FEATURES_SCHEMA, "Candidate features"),
        (orthographic_file, ORTHOGRAPHIC_PAIR_DECISIONS_SCHEMA, "Orthographic decisions"),
    )
    for parquet_file, schema, label in expected_schemas:
        if parquet_file.schema_arrow != schema:
            raise ValueError(f"{label} dataset does not match the required schema")

    profile_columns = (
        "key_a",
        "key_b",
        "name_a",
        "name_b",
        "primary_family",
        "core_token_count_a",
        "core_token_count_b",
        "shared_exact_token_count",
        "differing_token_count",
        "maximum_token_edit_distance",
        "total_token_edit_distance",
        "is_token_reorder",
        "is_ordered_subsequence",
        "is_token_multiset_containment",
        "possible_truncation_a",
        "possible_truncation_b",
        "numeric_relation",
        "prior_orthographic_evidence",
    )
    distinctive_columns = (
        "key_a",
        "key_b",
        "name_a",
        "name_b",
        "shared_exact_token_count",
        "shared_distinctive_token_count",
        "minimum_distinctive_token_support",
        "maximum_distinctive_token_support",
        "exact_coverage_a",
        "exact_coverage_b",
        "distinctive_coverage_a",
        "distinctive_coverage_b",
        "shorter_name_exact_coverage",
        "longer_name_exact_coverage",
        "shorter_name_distinctive_coverage",
        "longer_name_distinctive_coverage",
        "has_shared_exact_token",
        "has_shared_distinctive_token",
        "has_multiple_shared_distinctive_tokens",
        "has_exact_overlap_without_distinctive_token",
        "primary_family",
        "numeric_relation",
    )
    feature_columns = (
        *_CANDIDATE_IDENTITY_COLUMNS,
        *_NUMERIC_FEATURE_COLUMNS,
        "same_first_token",
        "numeric_relation",
    )
    orthographic_columns = (
        *_CANDIDATE_IDENTITY_COLUMNS,
        "orthographic_status",
        "orthographic_evidence",
    )
    profile_rows = _iter_parquet_rows(
        profile_file, batch_size=config.batch_size, columns=profile_columns
    )
    distinctive_rows = _iter_parquet_rows(
        distinctive_file, batch_size=config.batch_size, columns=distinctive_columns
    )
    feature_rows = _iter_parquet_rows(
        feature_file, batch_size=config.batch_size, columns=feature_columns
    )
    orthographic_rows = _iter_parquet_rows(
        orthographic_file, batch_size=config.batch_size, columns=orthographic_columns
    )

    feature_row, feature_identity = _next_validated_candidate(feature_rows, None)
    orthographic_row, orthographic_identity = _next_validated_candidate(orthographic_rows, None)
    feature_rows_read = int(feature_row is not None)
    orthographic_rows_read = int(orthographic_row is not None)
    previous_profile_pair: tuple[str, str] | None = None
    previous_distinctive_pair: tuple[str, str] | None = None
    previous_feature_pair = (
        (feature_identity[0], feature_identity[1]) if feature_identity is not None else None
    )
    previous_orthographic_pair = (
        (orthographic_identity[0], orthographic_identity[1])
        if orthographic_identity is not None
        else None
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(f"{output_path.suffix}.tmp")
    temporary.unlink(missing_ok=True)
    compression = (
        None
        if settings.processing.parquet_compression == "none"
        else settings.processing.parquet_compression
    )
    writer: pq.ParquetWriter | None = None
    rows_written = 0
    distinctive_rows_aligned = 0
    family_counts: Counter[str] = Counter()
    family_by_primary: Counter[tuple[str, str]] = Counter()
    family_by_numeric: Counter[tuple[str, str]] = Counter()
    flag_counts: Counter[str] = Counter()
    cross_tab_counts: Counter[str] = Counter()
    zero_by_primary: Counter[str] = Counter()
    expected_digest = hashlib.sha256()
    output_batch: list[dict[str, object]] = []
    flag_names = (
        "has_structural_exact_relation",
        "has_orthographic_signal",
        "has_distinctive_exact_overlap",
        "has_multiple_distinctive_exact_overlap",
        "has_numeric_risk",
        "has_zero_exact_overlap",
        "has_full_shorter_exact_coverage",
        "has_full_shorter_distinctive_coverage",
    )
    try:
        writer = pq.ParquetWriter(
            temporary,
            MULTI_EVIDENCE_ASSESSMENT_SCHEMA,
            compression=compression,
            use_dictionary=True,
            write_statistics=True,
        )
        for profile_row, distinctive_row in zip(profile_rows, distinctive_rows, strict=True):
            profile_identity = _validated_pair_identity(
                profile_row, previous_profile_pair, source="Residual profile"
            )
            distinctive_identity = _validated_pair_identity(
                distinctive_row, previous_distinctive_pair, source="Distinctive evidence"
            )
            pair = (profile_identity[0], profile_identity[1])
            previous_profile_pair = pair
            previous_distinctive_pair = (distinctive_identity[0], distinctive_identity[1])
            if profile_identity != distinctive_identity:
                raise ValueError("Residual-profile and distinctive-evidence identities differ")
            distinctive_rows_aligned += 1

            while (
                feature_identity is not None and (feature_identity[0], feature_identity[1]) < pair
            ):
                feature_row, feature_identity = _next_validated_candidate(
                    feature_rows, previous_feature_pair
                )
                feature_rows_read += int(feature_row is not None)
                previous_feature_pair = (
                    (feature_identity[0], feature_identity[1])
                    if feature_identity is not None
                    else previous_feature_pair
                )
            if feature_identity is None or (feature_identity[0], feature_identity[1]) != pair:
                raise ValueError(f"Candidate features are missing residual pair {pair!r}")
            if feature_identity[:4] != profile_identity or feature_row is None:
                raise ValueError("Residual-profile and candidate-feature identities differ")

            while (
                orthographic_identity is not None
                and (orthographic_identity[0], orthographic_identity[1]) < pair
            ):
                orthographic_row, orthographic_identity = _next_validated_candidate(
                    orthographic_rows, previous_orthographic_pair
                )
                orthographic_rows_read += int(orthographic_row is not None)
                previous_orthographic_pair = (
                    (orthographic_identity[0], orthographic_identity[1])
                    if orthographic_identity is not None
                    else previous_orthographic_pair
                )
            if (
                orthographic_identity is None
                or (orthographic_identity[0], orthographic_identity[1]) != pair
            ):
                raise ValueError(f"Orthographic decisions are missing residual pair {pair!r}")
            if orthographic_identity[:4] != profile_identity or orthographic_row is None:
                raise ValueError("Residual-profile and orthographic identities differ")
            if feature_identity != orthographic_identity:
                raise ValueError("Candidate-feature and orthographic retrieval evidence differ")
            if orthographic_row.get("orthographic_status") != NEEDS_FURTHER_RESOLUTION:
                raise ValueError("Every residual pair must have NEEDS_FURTHER_RESOLUTION status")

            primary_family_raw = profile_row.get("primary_family")
            numeric_relation_raw = profile_row.get("numeric_relation")
            if primary_family_raw not in PRIMARY_FAMILY_PRECEDENCE:
                raise ValueError("Residual profile has an unknown primary family")
            if numeric_relation_raw not in {"none", "same", "one_sided", "conflict"}:
                raise ValueError("Residual profile has an invalid numeric relation")
            if distinctive_row.get("primary_family") != primary_family_raw:
                raise ValueError(
                    "Distinctive evidence primary family differs from residual profile"
                )
            if distinctive_row.get("numeric_relation") != numeric_relation_raw:
                raise ValueError(
                    "Distinctive evidence numeric relation differs from residual profile"
                )
            if feature_row.get("numeric_relation") != numeric_relation_raw:
                raise ValueError("Candidate-feature numeric relation differs from residual profile")
            profile_shared_exact = _required_int(
                profile_row, "shared_exact_token_count", source="Residual profile"
            )
            distinctive_shared_exact = _required_int(
                distinctive_row, "shared_exact_token_count", source="Distinctive evidence"
            )
            if profile_shared_exact != distinctive_shared_exact:
                raise ValueError("Residual and distinctive shared-exact-token counts differ")
            prior_evidence = _required_string(
                profile_row, "prior_orthographic_evidence", source="Residual profile"
            )
            orthographic_evidence = _required_string(
                orthographic_row, "orthographic_evidence", source="Orthographic decisions"
            )
            if prior_evidence != orthographic_evidence:
                raise ValueError("Residual and orthographic evidence strings differ")

            shared_distinctive_count = _required_int(
                distinctive_row,
                "shared_distinctive_token_count",
                source="Distinctive evidence",
            )
            has_distinctive = _required_bool(
                distinctive_row, "has_shared_distinctive_token", source="Distinctive evidence"
            )
            has_multiple_distinctive = _required_bool(
                distinctive_row,
                "has_multiple_shared_distinctive_tokens",
                source="Distinctive evidence",
            )
            has_shared_exact = _required_bool(
                distinctive_row, "has_shared_exact_token", source="Distinctive evidence"
            )
            exact_without_distinctive = _required_bool(
                distinctive_row,
                "has_exact_overlap_without_distinctive_token",
                source="Distinctive evidence",
            )
            if has_shared_exact != (profile_shared_exact > 0) or exact_without_distinctive != (
                profile_shared_exact > 0 and not has_distinctive
            ):
                raise ValueError("Distinctive evidence overlap flags are internally inconsistent")

            assessment = assess_multi_evidence(
                primary_family=primary_family_raw,
                numeric_relation=cast("NumericRelation", numeric_relation_raw),
                shared_exact_token_count=profile_shared_exact,
                shared_distinctive_token_count=shared_distinctive_count,
                has_shared_distinctive_token=has_distinctive,
                has_multiple_shared_distinctive_tokens=has_multiple_distinctive,
                shorter_name_exact_coverage=_required_float(
                    distinctive_row,
                    "shorter_name_exact_coverage",
                    source="Distinctive evidence",
                ),
                shorter_name_distinctive_coverage=_required_float(
                    distinctive_row,
                    "shorter_name_distinctive_coverage",
                    source="Distinctive evidence",
                ),
                prior_orthographic_evidence=prior_evidence,
            )
            if assessment.assessment_family not in ACTIVE_ASSESSMENT_FAMILIES:
                raise RuntimeError("Assessment produced an inactive or unknown family")
            payload = _multi_evidence_payload(
                identity=profile_identity,
                profile_row=profile_row,
                distinctive_row=distinctive_row,
                feature_row=feature_row,
                assessment=assessment,
                prior_orthographic_evidence=prior_evidence,
            )
            output_batch.append(payload)
            _update_profile_digest(expected_digest, payload)
            rows_written += 1
            family = assessment.assessment_family
            primary_family = cast(str, primary_family_raw)
            numeric_relation = numeric_relation_raw
            family_counts[family] += 1
            family_by_primary[(family, primary_family)] += 1
            family_by_numeric[(family, numeric_relation)] += 1
            for flag_name in flag_names:
                if payload[flag_name] is True:
                    flag_counts[flag_name] += 1
            if assessment.has_structural_exact_relation and has_distinctive:
                cross_tab_counts["structural_and_distinctive"] += 1
            if assessment.has_orthographic_signal and has_distinctive:
                cross_tab_counts["orthographic_and_distinctive"] += 1
            if assessment.has_numeric_risk and has_distinctive:
                cross_tab_counts["numeric_risk_and_distinctive"] += 1
            if assessment.has_zero_exact_overlap:
                zero_by_primary[primary_family] += 1

            feature_row, feature_identity = _next_validated_candidate(
                feature_rows, previous_feature_pair
            )
            feature_rows_read += int(feature_row is not None)
            previous_feature_pair = (
                (feature_identity[0], feature_identity[1])
                if feature_identity is not None
                else previous_feature_pair
            )
            orthographic_row, orthographic_identity = _next_validated_candidate(
                orthographic_rows, previous_orthographic_pair
            )
            orthographic_rows_read += int(orthographic_row is not None)
            previous_orthographic_pair = (
                (orthographic_identity[0], orthographic_identity[1])
                if orthographic_identity is not None
                else previous_orthographic_pair
            )

            if len(output_batch) >= config.batch_size:
                writer.write_table(
                    pa.Table.from_pylist(output_batch, schema=MULTI_EVIDENCE_ASSESSMENT_SCHEMA)
                )
                output_batch.clear()
        if output_batch:
            writer.write_table(
                pa.Table.from_pylist(output_batch, schema=MULTI_EVIDENCE_ASSESSMENT_SCHEMA)
            )
        while feature_identity is not None:
            feature_row, feature_identity = _next_validated_candidate(
                feature_rows, previous_feature_pair
            )
            feature_rows_read += int(feature_row is not None)
            previous_feature_pair = (
                (feature_identity[0], feature_identity[1])
                if feature_identity is not None
                else previous_feature_pair
            )
        while orthographic_identity is not None:
            orthographic_row, orthographic_identity = _next_validated_candidate(
                orthographic_rows, previous_orthographic_pair
            )
            orthographic_rows_read += int(orthographic_row is not None)
            previous_orthographic_pair = (
                (orthographic_identity[0], orthographic_identity[1])
                if orthographic_identity is not None
                else previous_orthographic_pair
            )
        if feature_rows_read != feature_file.metadata.num_rows:
            raise RuntimeError("Candidate-feature input failed row reconciliation")
        if orthographic_rows_read != orthographic_file.metadata.num_rows:
            raise RuntimeError("Orthographic input failed row reconciliation")
    except Exception:
        if writer is not None:
            writer.close()
            writer = None
        temporary.unlink(missing_ok=True)
        raise
    finally:
        if writer is not None:
            writer.close()

    temporary.replace(output_path)
    output_file = pq.ParquetFile(output_path)
    if output_file.schema_arrow != MULTI_EVIDENCE_ASSESSMENT_SCHEMA:
        raise RuntimeError("Multi-evidence output schema failed reconciliation")
    output_digest = hashlib.sha256()
    output_rows_reconciled = 0
    previous_output_pair: tuple[str, str] | None = None
    for row in _iter_parquet_rows(
        output_file,
        batch_size=config.batch_size,
        columns=tuple(MULTI_EVIDENCE_ASSESSMENT_SCHEMA.names),
    ):
        identity = _validated_pair_identity(
            row, previous_output_pair, source="Multi-evidence output"
        )
        previous_output_pair = (identity[0], identity[1])
        if row.get("assessment_family") not in ACTIVE_ASSESSMENT_FAMILIES:
            raise RuntimeError("Persisted assessment family is inactive or unknown")
        _update_profile_digest(output_digest, row)
        output_rows_reconciled += 1

    counts_reconcile = (
        rows_written
        == profile_file.metadata.num_rows
        == distinctive_file.metadata.num_rows
        == distinctive_rows_aligned
        == output_file.metadata.num_rows
        == output_rows_reconciled
        == sum(family_counts.values())
    )
    digest_reconciles = expected_digest.digest() == output_digest.digest()
    if not counts_reconcile or not digest_reconciles:
        raise RuntimeError("Multi-evidence assessment failed output reconciliation")

    elapsed_seconds = time.perf_counter() - total_started
    family_distribution: dict[str, JsonValue] = {
        family: {
            "count": family_counts[family],
            "percentage": round(100.0 * family_counts[family] / rows_written, 6)
            if rows_written
            else 0.0,
        }
        for family in ASSESSMENT_FAMILY_PRECEDENCE
    }
    by_primary: dict[str, JsonValue] = {
        family: {
            primary: family_by_primary[(family, primary)] for primary in PRIMARY_FAMILY_PRECEDENCE
        }
        for family in ASSESSMENT_FAMILY_PRECEDENCE
    }
    numeric_order = ("none", "same", "one_sided", "conflict")
    by_numeric: dict[str, JsonValue] = {
        family: {relation: family_by_numeric[(family, relation)] for relation in numeric_order}
        for family in ASSESSMENT_FAMILY_PRECEDENCE
    }
    metrics: dict[str, JsonValue] = {
        "ruleset_version": config.ruleset_version,
        "input_residual_rows": profile_file.metadata.num_rows,
        "distinctive_rows_aligned": distinctive_rows_aligned,
        "candidate_feature_rows_read": feature_rows_read,
        "orthographic_rows_read": orthographic_rows_read,
        "output_rows": rows_written,
        "assessment_family_distribution": family_distribution,
        "assessment_family_by_primary_family": by_primary,
        "assessment_family_by_numeric_relation": by_numeric,
        "evidence_flag_counts": {name: flag_counts[name] for name in flag_names},
        "evidence_cross_tabs": {
            "structural_and_distinctive": cross_tab_counts["structural_and_distinctive"],
            "orthographic_and_distinctive": cross_tab_counts["orthographic_and_distinctive"],
            "numeric_risk_and_distinctive": cross_tab_counts["numeric_risk_and_distinctive"],
            "zero_exact_overlap_by_primary_family": {
                family: zero_by_primary[family] for family in PRIMARY_FAMILY_PRECEDENCE
            },
        },
        "high_lexical_family_active": False,
        "lexical_feature_summary_source": str(feature_path),
        "execution_runtime_seconds": round(elapsed_seconds, 6),
        "output_size_bytes": output_path.stat().st_size,
        "reconciliation": {
            "one_output_row_per_residual_pair": True,
            "residual_and_distinctive_identity_aligned": True,
            "candidate_features_aligned_for_every_residual_pair": True,
            "orthographic_needs_further_resolution_invariant": True,
            "pair_order_preserved": True,
            "persisted_rows_match_generated_digest": True,
            "status": "passed",
        },
    }
    _write_json(metrics_path, metrics)
    LOGGER.info("Assessed evidence convergence for %s residual pairs", rows_written)
    return MultiEvidenceAssessmentResult(
        output_path=output_path,
        metrics_path=metrics_path,
        output_rows=rows_written,
        family_counts={family: family_counts[family] for family in ASSESSMENT_FAMILY_PRECEDENCE},
        elapsed_seconds=elapsed_seconds,
    )
