"""Configuration contract tests."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from credit_risk_er.config import Settings, configuration_fingerprint
from tests.conftest import settings_for


def test_unknown_configuration_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        Settings.model_validate({"unknown": True})


def test_effective_configuration_fingerprint_is_stable(tmp_path: Path) -> None:
    source = tmp_path / "source.xlsx"
    source.write_bytes(b"not-opened-in-this-test")
    settings = settings_for(tmp_path, source)
    first = configuration_fingerprint(
        settings, source_path=source, output_path=tmp_path / "out.parquet"
    )
    second = configuration_fingerprint(
        settings, source_path=source, output_path=tmp_path / "out.parquet"
    )
    changed = configuration_fingerprint(
        settings, source_path=source, output_path=tmp_path / "other.parquet"
    )
    assert first == second
    assert first != changed
    assert len(first) == 64


def test_reference_and_resolution_paths_are_explicit(tmp_path: Path) -> None:
    source = tmp_path / "source.xlsx"
    source.write_bytes(b"not-opened-in-this-test")
    settings = settings_for(tmp_path, source)
    assert settings.reference_data.employer_master.name == "employer_master.csv"
    assert settings.reference_data.employer_aliases.name == "employer_aliases.csv"
    assert settings.resolution.output_dataset.name == "resolved_employers.parquet"
    assert settings.resolution.metrics_output.name == "resolution_metrics.json"
    assert settings.candidate_generation.resolution_keys_output.name == "resolution_keys.parquet"
    assert settings.candidate_generation.candidate_pairs_output.name == "candidate_pairs.parquet"
    assert settings.candidate_generation.minimum_typo_token_length == 6
    assert settings.candidate_generation.maximum_typo_signature_frequency == 10
    assert settings.candidate_generation.maximum_typo_context_frequency == 20
    assert settings.candidate_scoring.input_dataset.name == "candidate_pairs.parquet"
    assert settings.candidate_scoring.output_dataset.name == "candidate_pair_features.parquet"
    assert settings.pair_decision.input_dataset.name == "candidate_pair_features.parquet"
    assert settings.pair_decision.resolution_keys_dataset.name == "resolution_keys.parquet"
    assert settings.pair_decision.output_dataset.name == "pair_resolution_decisions.parquet"
    assert settings.pair_decision.metrics_output.name == "pair_decision_metrics.json"
    assert settings.pair_decision.minimum_whitespace_compact_length == 5
    assert settings.employer_eligibility.preprocessed_dataset.name == (
        "preprocessed_employers.parquet"
    )
    assert settings.employer_eligibility.ruleset_version == "1.0.0"
    assert settings.employer_eligibility.resolution_keys_dataset.name == "resolution_keys.parquet"
    assert settings.employer_eligibility.output_dataset.name == "employer_eligibility.parquet"
    assert settings.employer_eligibility.metrics_output.name == "employer_eligibility_metrics.json"
    assert settings.employer_eligibility.structural_address_road_type_tokens == (
        "ST",
        "STREET",
        "RD",
        "ROAD",
    )
    assert settings.orthographic_resolution.pair_decisions_dataset.name == (
        "pair_resolution_decisions.parquet"
    )
    assert settings.orthographic_resolution.ruleset_version == "2.1.0"
    assert settings.orthographic_resolution.employer_eligibility_dataset.name == (
        "employer_eligibility.parquet"
    )
    assert settings.orthographic_resolution.candidate_features_dataset.name == (
        "candidate_pair_features.parquet"
    )
    assert settings.orthographic_resolution.output_dataset.name == (
        "orthographic_pair_decisions.parquet"
    )
    assert settings.orthographic_resolution.metrics_output.name == (
        "orthographic_resolution_metrics.json"
    )
    assert settings.residual_profile.ruleset_version == "1.0.0"
    assert settings.residual_profile.orthographic_decisions_dataset.name == (
        "orthographic_pair_decisions.parquet"
    )
    assert settings.residual_profile.candidate_features_dataset.name == (
        "candidate_pair_features.parquet"
    )
    assert settings.residual_profile.resolution_keys_dataset.name == ("resolution_keys.parquet")
    assert settings.residual_profile.employer_eligibility_dataset.name == (
        "employer_eligibility.parquet"
    )
    assert settings.residual_profile.output_dataset.name == (
        "residual_relationship_profile.parquet"
    )
    assert settings.residual_profile.metrics_output.name == (
        "residual_relationship_profile_metrics.json"
    )
    assert settings.distinctive_evidence.ruleset_version == "1.0.0"
    assert settings.distinctive_evidence.residual_profile_dataset.name == (
        "residual_relationship_profile.parquet"
    )
    assert settings.distinctive_evidence.employer_eligibility_dataset.name == (
        "employer_eligibility.parquet"
    )
    assert settings.distinctive_evidence.resolution_keys_dataset.name == ("resolution_keys.parquet")
    assert settings.distinctive_evidence.output_dataset.name == (
        "distinctive_name_evidence.parquet"
    )
    assert settings.distinctive_evidence.metrics_output.name == (
        "distinctive_name_evidence_metrics.json"
    )
    assert settings.multi_evidence_assessment.ruleset_version == "1.0.0"
    assert settings.multi_evidence_assessment.residual_profile_dataset.name == (
        "residual_relationship_profile.parquet"
    )
    assert settings.multi_evidence_assessment.distinctive_evidence_dataset.name == (
        "distinctive_name_evidence.parquet"
    )
    assert settings.multi_evidence_assessment.candidate_features_dataset.name == (
        "candidate_pair_features.parquet"
    )
    assert settings.multi_evidence_assessment.orthographic_decisions_dataset.name == (
        "orthographic_pair_decisions.parquet"
    )
    assert settings.multi_evidence_assessment.output_dataset.name == (
        "multi_evidence_assessment.parquet"
    )
    assert settings.multi_evidence_assessment.metrics_output.name == (
        "multi_evidence_assessment_metrics.json"
    )
    assert settings.finalization.ruleset_version == "1.1.0"
    assert settings.finalization.preprocessed_dataset.name == "preprocessed_employers.parquet"
    assert settings.finalization.resolution_keys_dataset.name == "resolution_keys.parquet"
    assert settings.finalization.pair_decisions_dataset.name == (
        "pair_resolution_decisions.parquet"
    )
    assert settings.finalization.employer_eligibility_dataset.name == (
        "employer_eligibility.parquet"
    )
    assert settings.finalization.public_enrichment_csv.name == ("public_employer_enrichment.csv")
    assert settings.finalization.parquet_output.name == "employer_resolution_final.parquet"
    assert settings.finalization.csv_output.name == "employer_resolution_final.csv"
    assert settings.finalization.metrics_output.name == "finalization_metrics.json"
    assert settings.finalization.top_keys_output.name == "top_employer_resolution_keys.csv"
    assert settings.finalization.sector_keyword_rules["Educación"] == ("COLEGIO", "ESCUELA")
    assert settings.evaluation.output_directory.name == "evaluation"
    assert settings.evaluation.random_seed == 20260813
