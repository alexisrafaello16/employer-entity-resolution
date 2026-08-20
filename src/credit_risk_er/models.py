"""Small typed contracts shared by the preprocessing modules."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]


@dataclass(frozen=True, slots=True)
class SourceMetadata:
    path: Path
    sha256: str
    size_bytes: int
    sheet_name: str
    source_column: str
    max_row: int

    @property
    def data_rows(self) -> int:
        return self.max_row - 1


@dataclass(frozen=True, slots=True)
class SourceRow:
    source_row_number: int
    nombre_original: str | None


@dataclass(frozen=True, slots=True)
class PreprocessResult:
    output_path: Path
    metrics_path: Path
    manifest_path: Path
    row_count: int
    processing_fingerprint: str
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class ResolutionResult:
    output_path: Path
    metrics_path: Path
    row_count: int
    resolved_count: int
    unresolved_count: int
    entity_count: int
    alias_count: int
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class CandidateGenerationResult:
    resolution_keys_path: Path
    candidate_pairs_path: Path
    metrics_path: Path
    unresolved_rows: int
    eligible_rows: int
    unique_keys: int
    candidate_pairs: int
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class CandidateScoringResult:
    output_path: Path
    metrics_path: Path
    candidate_pairs: int
    feature_rows: int
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class PairDecisionResult:
    output_path: Path
    metrics_path: Path
    candidate_pairs: int
    decision_rows: int
    auto_same_count: int
    needs_further_resolution_count: int
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class EmployerEligibilityResult:
    output_path: Path
    metrics_path: Path
    resolution_keys: int
    rows_written: int
    employer_candidate_count: int
    address_count: int
    non_employer_status_count: int
    ambiguous_count: int
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class OrthographicResolutionResult:
    output_path: Path
    metrics_path: Path
    residual_pairs: int
    strong_orthographic_evidence_count: int
    needs_further_resolution_count: int
    not_eligible_count: int
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class ResidualProfileResult:
    output_path: Path
    metrics_path: Path
    profiled_residual_rows: int
    skipped_strong_orthographic_rows: int
    skipped_ineligible_rows: int
    family_counts: dict[str, int]
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class DistinctiveEvidenceResult:
    output_path: Path
    metrics_path: Path
    output_rows: int
    pairs_with_exact_overlap: int
    pairs_with_distinctive_overlap: int
    pairs_with_multiple_distinctive_tokens: int
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class MultiEvidenceAssessmentResult:
    output_path: Path
    metrics_path: Path
    output_rows: int
    family_counts: dict[str, int]
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class FinalizationResult:
    parquet_output_path: Path
    csv_output_path: Path
    metrics_path: Path
    top_keys_path: Path
    row_count: int
    public_enriched_rows: int
    outcome_counts: dict[str, int]
    elapsed_seconds: float
