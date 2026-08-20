"""Strict product configuration and deterministic fingerprints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    """Reject undocumented configuration keys and prevent runtime mutation."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class SourceConfig(StrictModel):
    workbook: Path
    sheet_name: str
    column: str
    expected_sha256: str = Field(pattern=r"^[A-Fa-f0-9]{64}$")
    expected_sheet_names: tuple[str, ...]


class NormalizationConfig(StrictModel):
    ruleset_version: str = Field(min_length=1)
    possible_truncation_content_lengths: tuple[int, ...]
    trailing_numeric_candidate_max_digits: int = Field(gt=0)
    corporate_suffix_aliases: dict[str, tuple[str, ...]]


class AddressSignalConfig(StrictModel):
    explicit_tokens: tuple[str, ...]
    contextual_tokens: tuple[str, ...]
    intersection_markers: tuple[str, ...]
    moderate_score: int = Field(gt=0)
    strong_score: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_threshold_order(self) -> AddressSignalConfig:
        if self.strong_score <= self.moderate_score:
            raise ValueError("strong_score must be greater than moderate_score")
        return self


class OccupationSignalConfig(StrictModel):
    status_phrases: tuple[str, ...]


class OrganizationSignalConfig(StrictModel):
    corporate_suffix_tokens: tuple[str, ...]
    organization_tokens: tuple[str, ...]


class RecordTypingConfig(StrictModel):
    ruleset_version: str = Field(min_length=1)
    address: AddressSignalConfig
    occupation: OccupationSignalConfig
    organization: OrganizationSignalConfig


class ProcessingConfig(StrictModel):
    batch_size: int = Field(gt=0)
    output_dataset: Path
    metrics_file: Path
    manifest_file: Path
    parquet_compression: Literal["zstd", "snappy", "gzip", "none"] = "zstd"


class ReferenceDataConfig(StrictModel):
    employer_master: Path
    employer_aliases: Path


class ResolutionConfig(StrictModel):
    output_dataset: Path
    metrics_output: Path


class CandidateGenerationConfig(StrictModel):
    resolution_keys_output: Path
    candidate_pairs_output: Path
    metrics_output: Path
    maximum_block_size: int = Field(gt=1)
    minimum_truncation_prefix_length: int = Field(gt=0)
    minimum_informative_token_length: int = Field(gt=1)
    maximum_token_frequency: int = Field(gt=1)
    prefix_signature_length: int = Field(gt=1)
    minimum_typo_token_length: int = Field(gt=1)
    maximum_typo_signature_frequency: int = Field(gt=1)
    maximum_typo_context_frequency: int = Field(gt=1)


class CandidateScoringConfig(StrictModel):
    input_dataset: Path
    output_dataset: Path
    metrics_output: Path
    batch_size: int = Field(gt=0)


class PairDecisionConfig(StrictModel):
    input_dataset: Path
    resolution_keys_dataset: Path
    output_dataset: Path
    metrics_output: Path
    batch_size: int = Field(gt=0)
    minimum_whitespace_compact_length: int = Field(ge=5)


class EmployerEligibilityConfig(StrictModel):
    ruleset_version: str = Field(min_length=1)
    preprocessed_dataset: Path
    resolution_keys_dataset: Path
    output_dataset: Path
    metrics_output: Path
    batch_size: int = Field(gt=0)
    structural_address_road_type_tokens: tuple[str, ...] = Field(min_length=1)


class OrthographicResolutionConfig(StrictModel):
    ruleset_version: str = Field(min_length=1)
    pair_decisions_dataset: Path
    employer_eligibility_dataset: Path
    candidate_features_dataset: Path
    output_dataset: Path
    metrics_output: Path
    batch_size: int = Field(gt=0)


class ResidualProfileConfig(StrictModel):
    ruleset_version: str = Field(min_length=1)
    orthographic_decisions_dataset: Path
    candidate_features_dataset: Path
    resolution_keys_dataset: Path
    employer_eligibility_dataset: Path
    output_dataset: Path
    metrics_output: Path
    batch_size: int = Field(gt=0)


class DistinctiveEvidenceConfig(StrictModel):
    ruleset_version: str = Field(min_length=1)
    residual_profile_dataset: Path
    employer_eligibility_dataset: Path
    resolution_keys_dataset: Path
    output_dataset: Path
    metrics_output: Path
    batch_size: int = Field(gt=0)


class MultiEvidenceAssessmentConfig(StrictModel):
    ruleset_version: str = Field(min_length=1)
    residual_profile_dataset: Path
    distinctive_evidence_dataset: Path
    candidate_features_dataset: Path
    orthographic_decisions_dataset: Path
    output_dataset: Path
    metrics_output: Path
    batch_size: int = Field(gt=0)


class FinalizationConfig(StrictModel):
    ruleset_version: str = Field(min_length=1)
    preprocessed_dataset: Path
    resolution_keys_dataset: Path
    pair_decisions_dataset: Path
    employer_eligibility_dataset: Path
    public_enrichment_csv: Path
    parquet_output: Path
    csv_output: Path
    metrics_output: Path
    top_keys_output: Path
    top_key_limit: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    sector_keyword_rules: dict[str, tuple[str, ...]] = Field(min_length=1)


class EvaluationConfig(StrictModel):
    pair_sample_size: int = Field(gt=0)
    blocking_miss_sample_size: int = Field(gt=0)
    random_seed: int
    output_directory: Path


class Settings(StrictModel):
    source: SourceConfig
    normalization: NormalizationConfig
    record_typing: RecordTypingConfig
    processing: ProcessingConfig
    reference_data: ReferenceDataConfig
    resolution: ResolutionConfig
    candidate_generation: CandidateGenerationConfig
    candidate_scoring: CandidateScoringConfig
    pair_decision: PairDecisionConfig
    employer_eligibility: EmployerEligibilityConfig
    orthographic_resolution: OrthographicResolutionConfig
    residual_profile: ResidualProfileConfig
    distinctive_evidence: DistinctiveEvidenceConfig
    multi_evidence_assessment: MultiEvidenceAssessmentConfig
    finalization: FinalizationConfig
    evaluation: EvaluationConfig


def load_settings(config_path: Path) -> Settings:
    """Load YAML into the strict product configuration model."""
    parsed = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return Settings.model_validate(parsed)


def configuration_fingerprint(settings: Settings, *, source_path: Path, output_path: Path) -> str:
    """Fingerprint effective settings, including command-line path overrides."""
    payload = settings.model_dump(mode="json")
    payload["effective_paths"] = {
        "source": str(source_path.resolve()),
        "output": str(output_path.resolve()),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()
