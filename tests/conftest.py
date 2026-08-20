"""Shared test fixtures for compact product tests."""

from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path

import pytest
from openpyxl import Workbook

from credit_risk_er.config import Settings
from credit_risk_er.ingestion import sha256_file


@pytest.fixture
def workbook_factory() -> Callable[..., Path]:
    def create(
        path: Path,
        values: list[str | None],
        *,
        header: str = "nombre_original",
        sheet_name: str = "Sheet1",
        extra_sheet: bool = False,
    ) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        workbook = Workbook()
        sheet = workbook.active
        sheet.title = sheet_name
        sheet.append([header])
        for value in values:
            sheet.append([value])
        if extra_sheet:
            workbook.create_sheet("Extra")
        workbook.save(path)
        workbook.close()
        return path

    return create


@pytest.fixture
def reference_factory() -> Callable[..., tuple[Path, Path]]:
    def create(
        directory: Path,
        *,
        master_rows: list[tuple[str, str]],
        alias_rows: list[tuple[str, str]],
    ) -> tuple[Path, Path]:
        directory.mkdir(parents=True, exist_ok=True)
        master_path = directory / "employer_master.csv"
        aliases_path = directory / "employer_aliases.csv"
        with master_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["entity_id", "canonical_name"])
            writer.writerows(master_rows)
        with aliases_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["entity_id", "alias_name"])
            writer.writerows(alias_rows)
        return master_path, aliases_path

    return create


def settings_for(root: Path, workbook: Path) -> Settings:
    return Settings.model_validate(
        {
            "source": {
                "workbook": str(workbook),
                "sheet_name": "Sheet1",
                "column": "nombre_original",
                "expected_sha256": sha256_file(workbook),
                "expected_sheet_names": ["Sheet1"],
            },
            "normalization": {
                "ruleset_version": "1.0.0",
                "possible_truncation_content_lengths": [30],
                "trailing_numeric_candidate_max_digits": 1,
                "corporate_suffix_aliases": {
                    "SA": ["S A", "SA"],
                    "SAS": ["S A S", "SAS"],
                    "INC": ["INC"],
                },
            },
            "record_typing": {
                "ruleset_version": "1.0.0",
                "address": {
                    "explicit_tokens": ["CALLE", "AVENIDA", "PH"],
                    "contextual_tokens": ["VIA", "PLAZA"],
                    "intersection_markers": ["ESQUINA", "CON CALLE"],
                    "moderate_score": 2,
                    "strong_score": 4,
                },
                "occupation": {"status_phrases": ["ESTUDIANTE", "AMA DE CASA", "INDEPENDIENTE"]},
                "organization": {
                    "corporate_suffix_tokens": ["SA", "SAS", "INC"],
                    "organization_tokens": ["EMPRESA", "UNIVERSIDAD", "GRUPO"],
                },
            },
            "processing": {
                "batch_size": 3,
                "output_dataset": str(root / "data/processed/preprocessed_employers.parquet"),
                "metrics_file": str(root / "data/processed/preprocessing_metrics.json"),
                "manifest_file": str(root / "data/processed/run_manifest.json"),
                "parquet_compression": "zstd",
            },
            "reference_data": {
                "employer_master": str(root / "data/reference/employer_master.csv"),
                "employer_aliases": str(root / "data/reference/employer_aliases.csv"),
            },
            "resolution": {
                "output_dataset": str(root / "data/processed/resolved_employers.parquet"),
                "metrics_output": str(root / "data/processed/resolution_metrics.json"),
            },
            "candidate_generation": {
                "resolution_keys_output": str(root / "data/processed/resolution_keys.parquet"),
                "candidate_pairs_output": str(root / "data/processed/candidate_pairs.parquet"),
                "metrics_output": str(root / "data/processed/candidate_generation_metrics.json"),
                "maximum_block_size": 10,
                "minimum_truncation_prefix_length": 10,
                "minimum_informative_token_length": 5,
                "maximum_token_frequency": 10,
                "prefix_signature_length": 6,
                "minimum_typo_token_length": 6,
                "maximum_typo_signature_frequency": 10,
                "maximum_typo_context_frequency": 20,
            },
            "candidate_scoring": {
                "input_dataset": str(root / "data/processed/candidate_pairs.parquet"),
                "output_dataset": str(root / "data/processed/candidate_pair_features.parquet"),
                "metrics_output": str(root / "data/processed/candidate_scoring_metrics.json"),
                "batch_size": 2,
            },
            "pair_decision": {
                "input_dataset": str(root / "data/processed/candidate_pair_features.parquet"),
                "resolution_keys_dataset": str(root / "data/processed/resolution_keys.parquet"),
                "output_dataset": str(root / "data/processed/pair_resolution_decisions.parquet"),
                "metrics_output": str(root / "data/processed/pair_decision_metrics.json"),
                "batch_size": 2,
                "minimum_whitespace_compact_length": 5,
            },
            "employer_eligibility": {
                "ruleset_version": "1.0.0",
                "preprocessed_dataset": str(root / "data/processed/preprocessed_employers.parquet"),
                "resolution_keys_dataset": str(root / "data/processed/resolution_keys.parquet"),
                "output_dataset": str(root / "data/processed/employer_eligibility.parquet"),
                "metrics_output": str(root / "data/processed/employer_eligibility_metrics.json"),
                "batch_size": 2,
                "structural_address_road_type_tokens": [
                    "ST",
                    "STREET",
                    "RD",
                    "ROAD",
                ],
            },
            "orthographic_resolution": {
                "ruleset_version": "2.1.0",
                "pair_decisions_dataset": str(
                    root / "data/processed/pair_resolution_decisions.parquet"
                ),
                "employer_eligibility_dataset": str(
                    root / "data/processed/employer_eligibility.parquet"
                ),
                "candidate_features_dataset": str(
                    root / "data/processed/candidate_pair_features.parquet"
                ),
                "output_dataset": str(root / "data/processed/orthographic_pair_decisions.parquet"),
                "metrics_output": str(root / "data/processed/orthographic_resolution_metrics.json"),
                "batch_size": 2,
            },
            "residual_profile": {
                "ruleset_version": "1.0.0",
                "orthographic_decisions_dataset": str(
                    root / "data/processed/orthographic_pair_decisions.parquet"
                ),
                "candidate_features_dataset": str(
                    root / "data/processed/candidate_pair_features.parquet"
                ),
                "resolution_keys_dataset": str(root / "data/processed/resolution_keys.parquet"),
                "employer_eligibility_dataset": str(
                    root / "data/processed/employer_eligibility.parquet"
                ),
                "output_dataset": str(
                    root / "data/processed/residual_relationship_profile.parquet"
                ),
                "metrics_output": str(
                    root / "data/processed/residual_relationship_profile_metrics.json"
                ),
                "batch_size": 2,
            },
            "distinctive_evidence": {
                "ruleset_version": "1.0.0",
                "residual_profile_dataset": str(
                    root / "data/processed/residual_relationship_profile.parquet"
                ),
                "employer_eligibility_dataset": str(
                    root / "data/processed/employer_eligibility.parquet"
                ),
                "resolution_keys_dataset": str(root / "data/processed/resolution_keys.parquet"),
                "output_dataset": str(root / "data/processed/distinctive_name_evidence.parquet"),
                "metrics_output": str(
                    root / "data/processed/distinctive_name_evidence_metrics.json"
                ),
                "batch_size": 2,
            },
            "multi_evidence_assessment": {
                "ruleset_version": "1.0.0",
                "residual_profile_dataset": str(
                    root / "data/processed/residual_relationship_profile.parquet"
                ),
                "distinctive_evidence_dataset": str(
                    root / "data/processed/distinctive_name_evidence.parquet"
                ),
                "candidate_features_dataset": str(
                    root / "data/processed/candidate_pair_features.parquet"
                ),
                "orthographic_decisions_dataset": str(
                    root / "data/processed/orthographic_pair_decisions.parquet"
                ),
                "output_dataset": str(root / "data/processed/multi_evidence_assessment.parquet"),
                "metrics_output": str(
                    root / "data/processed/multi_evidence_assessment_metrics.json"
                ),
                "batch_size": 2,
            },
            "finalization": {
                "ruleset_version": "1.1.0",
                "preprocessed_dataset": str(root / "data/processed/preprocessed_employers.parquet"),
                "resolution_keys_dataset": str(root / "data/processed/resolution_keys.parquet"),
                "pair_decisions_dataset": str(
                    root / "data/processed/pair_resolution_decisions.parquet"
                ),
                "employer_eligibility_dataset": str(
                    root / "data/processed/employer_eligibility.parquet"
                ),
                "public_enrichment_csv": str(
                    root / "data/reference/public_employer_enrichment.csv"
                ),
                "parquet_output": str(root / "output/employer_resolution_final.parquet"),
                "csv_output": str(root / "output/employer_resolution_final.csv"),
                "metrics_output": str(root / "output/finalization_metrics.json"),
                "top_keys_output": str(root / "output/top_employer_resolution_keys.csv"),
                "top_key_limit": 40,
                "batch_size": 2,
                "sector_keyword_rules": {"Educación": ["COLEGIO", "ESCUELA"]},
            },
            "evaluation": {
                "pair_sample_size": 24,
                "blocking_miss_sample_size": 12,
                "random_seed": 20260813,
                "output_directory": str(root / "data/evaluation"),
            },
        }
    )
