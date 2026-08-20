"""Human-facing command-line entry point."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from credit_risk_er.canonical_quality import assess_corporate_master_quality
from credit_risk_er.config import load_settings
from credit_risk_er.corporate_master import build_corporate_master
from credit_risk_er.evaluation import build_evaluation_sample
from credit_risk_er.finalization import finalize_employers
from credit_risk_er.pipeline import (
    assess_pair_evidence,
    classify_employer_eligibility,
    compute_distinctive_evidence,
    decide_candidate_pairs,
    generate_candidates,
    preprocess,
    profile_residual_relationships,
    resolve_employers,
    resolve_orthographic_pairs,
    score_candidate_pairs,
)
from credit_risk_er.profiling import create_evaluation_sample
from credit_risk_er.reference_promotion import promote_references

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "preprocessed_employers.parquet"
DEFAULT_SAMPLE_PATH = PROJECT_ROOT / "data" / "processed" / "evaluation_sample.parquet"


def _config_project_root(config_path: Path) -> Path:
    """Resolve configured relative paths from the config directory's project."""
    parent = config_path.resolve().parent
    return parent.parent if parent.name.casefold() == "config" else parent


def _explicit_path(path: Path | None) -> Path | None:
    """Resolve explicit CLI paths with ordinary shell-relative semantics."""
    return path.expanduser().resolve() if path is not None else None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="credit-risk-er",
        description="Prepare employer records for later entity resolution.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    preprocess_parser = subcommands.add_parser(
        "preprocess", help="Create the compact internal preprocessing dataset."
    )
    preprocess_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    preprocess_parser.add_argument("--input", type=Path)
    preprocess_parser.add_argument("--output", type=Path)

    resolve_parser = subcommands.add_parser(
        "resolve", help="Resolve employers by exact match to validated reference knowledge."
    )
    resolve_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    resolve_parser.add_argument("--input", type=Path, help="Preprocessed Parquet dataset.")
    resolve_parser.add_argument("--output", type=Path, help="Resolved Parquet dataset.")
    resolve_parser.add_argument("--master", type=Path, help="Employer Master CSV.")
    resolve_parser.add_argument("--aliases", type=Path, help="Validated aliases CSV.")

    candidates_parser = subcommands.add_parser(
        "candidates",
        help="Generate bounded candidate pairs for unresolved employer discovery.",
    )
    candidates_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    candidates_parser.add_argument("--input", type=Path, help="Resolved Parquet dataset.")
    candidates_parser.add_argument(
        "--keys-output", type=Path, help="Unique resolution-key Parquet dataset."
    )
    candidates_parser.add_argument(
        "--pairs-output", type=Path, help="Candidate-pair Parquet dataset."
    )

    scoring_parser = subcommands.add_parser(
        "score-candidates",
        help="Compute lexical and structural evidence for existing candidate pairs.",
    )
    scoring_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    scoring_parser.add_argument("--input", type=Path, help="Candidate-pair Parquet dataset.")
    scoring_parser.add_argument("--output", type=Path, help="Candidate-feature Parquet dataset.")

    decision_parser = subcommands.add_parser(
        "decide-pairs",
        help="Apply high-precision deterministic decisions and abstain on all other pairs.",
    )
    decision_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    decision_parser.add_argument("--input", type=Path, help="Candidate-feature Parquet dataset.")
    decision_parser.add_argument("--keys", type=Path, help="Resolution-key metadata dataset.")
    decision_parser.add_argument("--output", type=Path, help="Pair-decision Parquet dataset.")

    eligibility_parser = subcommands.add_parser(
        "classify-eligibility",
        help="Classify normalized keys as employer, address, non-employer, or ambiguous.",
    )
    eligibility_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    eligibility_parser.add_argument(
        "--preprocessed", type=Path, help="Preprocessed employer-evidence dataset."
    )
    eligibility_parser.add_argument("--keys", type=Path, help="Resolution-key metadata dataset.")
    eligibility_parser.add_argument(
        "--output", type=Path, help="Employer-eligibility Parquet dataset."
    )

    orthographic_parser = subcommands.add_parser(
        "resolve-orthographic",
        help="Generate strong one-edit evidence for employer-compatible residual pairs.",
    )
    orthographic_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    orthographic_parser.add_argument(
        "--decisions", type=Path, help="Deterministic pair-decision dataset."
    )
    orthographic_parser.add_argument(
        "--eligibility", type=Path, help="Employer-eligibility dataset."
    )
    orthographic_parser.add_argument("--features", type=Path, help="Candidate-feature dataset.")
    orthographic_parser.add_argument(
        "--output", type=Path, help="Orthographic decision Parquet dataset."
    )

    residual_profile_parser = subcommands.add_parser(
        "profile-residuals",
        help="Characterize unresolved pair structure without deciding identity.",
    )
    residual_profile_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    residual_profile_parser.add_argument(
        "--orthographic", type=Path, help="Orthographic evidence dataset."
    )
    residual_profile_parser.add_argument("--features", type=Path, help="Candidate-feature dataset.")
    residual_profile_parser.add_argument("--keys", type=Path, help="Resolution-key dataset.")
    residual_profile_parser.add_argument(
        "--eligibility", type=Path, help="Employer-eligibility dataset."
    )
    residual_profile_parser.add_argument(
        "--output", type=Path, help="Residual relationship profile dataset."
    )

    distinctive_parser = subcommands.add_parser(
        "compute-distinctive-evidence",
        help="Compute exact token-rarity evidence without deciding identity.",
    )
    distinctive_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    distinctive_parser.add_argument(
        "--profile", type=Path, help="Residual relationship profile dataset."
    )
    distinctive_parser.add_argument(
        "--eligibility", type=Path, help="Employer-eligibility dataset."
    )
    distinctive_parser.add_argument("--keys", type=Path, help="Resolution-key dataset.")
    distinctive_parser.add_argument(
        "--output", type=Path, help="Distinctive-name evidence dataset."
    )

    assessment_parser = subcommands.add_parser(
        "assess-evidence",
        help="Integrate approved residual evidence into diagnostic convergence families.",
    )
    assessment_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    assessment_parser.add_argument("--profile", type=Path, help="Residual profile dataset.")
    assessment_parser.add_argument(
        "--distinctive", type=Path, help="Distinctive-name evidence dataset."
    )
    assessment_parser.add_argument("--features", type=Path, help="Candidate-feature dataset.")
    assessment_parser.add_argument(
        "--orthographic", type=Path, help="Orthographic decision dataset."
    )
    assessment_parser.add_argument("--output", type=Path, help="Multi-evidence assessment dataset.")

    finalization_parser = subcommands.add_parser(
        "finalize",
        help="Build the conservative canonical employer and sector dataset.",
    )
    finalization_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    finalization_parser.add_argument("--preprocessed", type=Path)
    finalization_parser.add_argument("--keys", type=Path)
    finalization_parser.add_argument("--decisions", type=Path)
    finalization_parser.add_argument("--eligibility", type=Path)
    finalization_parser.add_argument("--enrichment", type=Path)
    finalization_parser.add_argument("--parquet-output", type=Path)
    finalization_parser.add_argument("--csv-output", type=Path)

    corporate_master_parser = subcommands.add_parser(
        "build-corporate-master",
        help="Materialize the reusable corporate employer master and alias dictionary.",
    )
    corporate_master_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    corporate_master_parser.add_argument(
        "--final-dataset",
        type=Path,
        help="Final employer-resolution Parquet dataset.",
    )
    corporate_master_parser.add_argument(
        "--keys",
        type=Path,
        help="Resolution-key metadata Parquet dataset.",
    )

    canonical_quality_parser = subcommands.add_parser(
        "canonical-quality",
        help="Assess canonical employer-name quality before persistent promotion.",
    )
    canonical_quality_parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
    )
    canonical_quality_parser.add_argument(
        "--master",
        type=Path,
        help="Corporate master Parquet dataset.",
    )

    promotion_parser = subcommands.add_parser(
        "promote-references",
        help="Promote quality-approved entities into reusable exact-match references.",
    )
    promotion_parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
    )
    promotion_parser.add_argument(
        "--master",
        type=Path,
        help="Corporate master Parquet dataset.",
    )
    promotion_parser.add_argument(
        "--aliases",
        type=Path,
        help="Corporate aliases Parquet dataset.",
    )
    promotion_parser.add_argument(
        "--quality",
        type=Path,
        help="Canonical-name quality Parquet dataset.",
    )
    promotion_parser.add_argument(
        "--reference-master",
        type=Path,
        help="Persistent Employer Master CSV.",
    )
    promotion_parser.add_argument(
        "--reference-aliases",
        type=Path,
        help="Persistent employer aliases CSV.",
    )

    review_parser = subcommands.add_parser(
        "build-evaluation-sample",
        help="Create deterministic, initially unlabeled human-review CSV files.",
    )
    review_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    review_parser.add_argument(
        "--output-directory", type=Path, help="Directory for both human-review CSV files."
    )

    sample_parser = subcommands.add_parser(
        "evaluation-sample",
        help="Separately create a deterministic development/evaluation sample.",
    )
    sample_parser.add_argument("--input", type=Path, default=DEFAULT_DATASET_PATH)
    sample_parser.add_argument("--output", type=Path, default=DEFAULT_SAMPLE_PATH)
    sample_parser.add_argument("--seed", type=int, default=20260813)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run a product command and return a shell-compatible status code."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    arguments = _parser().parse_args(argv)

    try:
        if arguments.command == "preprocess":
            config_path = arguments.config.expanduser().resolve()
            settings = load_settings(config_path)
            preprocess_result = preprocess(
                _config_project_root(config_path),
                settings,
                source_override=_explicit_path(arguments.input),
                output_override=_explicit_path(arguments.output),
            )
            print(f"Completed: {preprocess_result.row_count:,} rows")
            print(f"Dataset: {preprocess_result.output_path}")
            print(f"Manifest: {preprocess_result.manifest_path}")
            print(f"Runtime: {preprocess_result.elapsed_seconds:.2f} seconds")
            return 0

        if arguments.command == "resolve":
            config_path = arguments.config.expanduser().resolve()
            settings = load_settings(config_path)
            resolution_result = resolve_employers(
                _config_project_root(config_path),
                settings,
                input_override=_explicit_path(arguments.input),
                output_override=_explicit_path(arguments.output),
                master_override=_explicit_path(arguments.master),
                aliases_override=_explicit_path(arguments.aliases),
            )
            print(f"Completed: {resolution_result.row_count:,} rows")
            print(f"Employer Master: {resolution_result.entity_count:,} entities")
            print(f"Validated aliases: {resolution_result.alias_count:,}")
            print(f"Resolved exact: {resolution_result.resolved_count:,}")
            print(f"Unresolved: {resolution_result.unresolved_count:,}")
            print(f"Dataset: {resolution_result.output_path}")
            print(f"Metrics: {resolution_result.metrics_path}")
            print(f"Runtime: {resolution_result.elapsed_seconds:.2f} seconds")
            return 0

        if arguments.command == "candidates":
            config_path = arguments.config.expanduser().resolve()
            settings = load_settings(config_path)
            candidate_result = generate_candidates(
                _config_project_root(config_path),
                settings,
                input_override=_explicit_path(arguments.input),
                keys_output_override=_explicit_path(arguments.keys_output),
                pairs_output_override=_explicit_path(arguments.pairs_output),
            )
            print(f"Unresolved rows: {candidate_result.unresolved_rows:,}")
            print(f"Eligible rows: {candidate_result.eligible_rows:,}")
            print(f"Unique resolution keys: {candidate_result.unique_keys:,}")
            print(f"Candidate pairs: {candidate_result.candidate_pairs:,}")
            print(f"Resolution keys: {candidate_result.resolution_keys_path}")
            print(f"Candidate pairs dataset: {candidate_result.candidate_pairs_path}")
            print(f"Metrics: {candidate_result.metrics_path}")
            print(f"Runtime: {candidate_result.elapsed_seconds:.2f} seconds")
            return 0

        if arguments.command == "score-candidates":
            config_path = arguments.config.expanduser().resolve()
            settings = load_settings(config_path)
            scoring_result = score_candidate_pairs(
                _config_project_root(config_path),
                settings,
                input_override=_explicit_path(arguments.input),
                output_override=_explicit_path(arguments.output),
            )
            print(f"Candidate pairs read: {scoring_result.candidate_pairs:,}")
            print(f"Feature rows written: {scoring_result.feature_rows:,}")
            print(f"Feature dataset: {scoring_result.output_path}")
            print(f"Metrics: {scoring_result.metrics_path}")
            print(f"Runtime: {scoring_result.elapsed_seconds:.2f} seconds")
            return 0

        if arguments.command == "decide-pairs":
            config_path = arguments.config.expanduser().resolve()
            settings = load_settings(config_path)
            decision_result = decide_candidate_pairs(
                _config_project_root(config_path),
                settings,
                input_override=_explicit_path(arguments.input),
                keys_override=_explicit_path(arguments.keys),
                output_override=_explicit_path(arguments.output),
            )
            print(f"Candidate pairs read: {decision_result.candidate_pairs:,}")
            print(f"Decision rows written: {decision_result.decision_rows:,}")
            print(f"AUTO_SAME: {decision_result.auto_same_count:,}")
            print(f"NEEDS_FURTHER_RESOLUTION: {decision_result.needs_further_resolution_count:,}")
            print(f"Decision dataset: {decision_result.output_path}")
            print(f"Metrics: {decision_result.metrics_path}")
            print(f"Runtime: {decision_result.elapsed_seconds:.2f} seconds")
            return 0

        if arguments.command == "classify-eligibility":
            config_path = arguments.config.expanduser().resolve()
            settings = load_settings(config_path)
            eligibility_result = classify_employer_eligibility(
                _config_project_root(config_path),
                settings,
                preprocessed_override=_explicit_path(arguments.preprocessed),
                keys_override=_explicit_path(arguments.keys),
                output_override=_explicit_path(arguments.output),
            )
            print(f"Resolution keys: {eligibility_result.resolution_keys:,}")
            print(f"EMPLOYER_CANDIDATE: {eligibility_result.employer_candidate_count:,}")
            print(f"ADDRESS: {eligibility_result.address_count:,}")
            print(f"NON_EMPLOYER_STATUS: {eligibility_result.non_employer_status_count:,}")
            print(f"AMBIGUOUS: {eligibility_result.ambiguous_count:,}")
            print(f"Dataset: {eligibility_result.output_path}")
            print(f"Metrics: {eligibility_result.metrics_path}")
            print(f"Runtime: {eligibility_result.elapsed_seconds:.2f} seconds")
            return 0

        if arguments.command == "resolve-orthographic":
            config_path = arguments.config.expanduser().resolve()
            settings = load_settings(config_path)
            orthographic_result = resolve_orthographic_pairs(
                _config_project_root(config_path),
                settings,
                decisions_override=_explicit_path(arguments.decisions),
                eligibility_override=_explicit_path(arguments.eligibility),
                features_override=_explicit_path(arguments.features),
                output_override=_explicit_path(arguments.output),
            )
            print(f"Residual pairs: {orthographic_result.residual_pairs:,}")
            print(
                "STRONG_ORTHOGRAPHIC_EVIDENCE: "
                f"{orthographic_result.strong_orthographic_evidence_count:,}"
            )
            print(
                f"NEEDS_FURTHER_RESOLUTION: "
                f"{orthographic_result.needs_further_resolution_count:,}"
            )
            print(
                f"NOT_ELIGIBLE_FOR_ORTHOGRAPHIC: "
                f"{orthographic_result.not_eligible_count:,}"
            )
            print(f"Dataset: {orthographic_result.output_path}")
            print(f"Metrics: {orthographic_result.metrics_path}")
            print(f"Runtime: {orthographic_result.elapsed_seconds:.2f} seconds")
            return 0

        if arguments.command == "profile-residuals":
            config_path = arguments.config.expanduser().resolve()
            settings = load_settings(config_path)
            profile_result = profile_residual_relationships(
                _config_project_root(config_path),
                settings,
                orthographic_override=_explicit_path(arguments.orthographic),
                features_override=_explicit_path(arguments.features),
                keys_override=_explicit_path(arguments.keys),
                eligibility_override=_explicit_path(arguments.eligibility),
                output_override=_explicit_path(arguments.output),
            )
            print(f"Profiled residual pairs: {profile_result.profiled_residual_rows:,}")
            for family, count in profile_result.family_counts.items():
                print(f"  {family}: {count:,}")
            print(
                "Skipped STRONG_ORTHOGRAPHIC_EVIDENCE: "
                f"{profile_result.skipped_strong_orthographic_rows:,}"
            )
            print(f"Skipped ineligible: {profile_result.skipped_ineligible_rows:,}")
            print(f"Dataset: {profile_result.output_path}")
            print(f"Metrics: {profile_result.metrics_path}")
            print(f"Runtime: {profile_result.elapsed_seconds:.2f} seconds")
            return 0

        if arguments.command == "compute-distinctive-evidence":
            config_path = arguments.config.expanduser().resolve()
            settings = load_settings(config_path)
            evidence_result = compute_distinctive_evidence(
                _config_project_root(config_path),
                settings,
                profile_override=_explicit_path(arguments.profile),
                eligibility_override=_explicit_path(arguments.eligibility),
                keys_override=_explicit_path(arguments.keys),
                output_override=_explicit_path(arguments.output),
            )
            print(f"Profiled pairs: {evidence_result.output_rows:,}")
            print(f"Pairs with exact overlap: {evidence_result.pairs_with_exact_overlap:,}")
            print(
                "Pairs with distinctive overlap: "
                f"{evidence_result.pairs_with_distinctive_overlap:,}"
            )
            print(
                "Pairs with multiple distinctive tokens: "
                f"{evidence_result.pairs_with_multiple_distinctive_tokens:,}"
            )
            print(f"Dataset: {evidence_result.output_path}")
            print(f"Metrics: {evidence_result.metrics_path}")
            print(f"Runtime: {evidence_result.elapsed_seconds:.2f} seconds")
            return 0

        if arguments.command == "assess-evidence":
            config_path = arguments.config.expanduser().resolve()
            settings = load_settings(config_path)
            assessment_result = assess_pair_evidence(
                _config_project_root(config_path),
                settings,
                profile_override=_explicit_path(arguments.profile),
                distinctive_override=_explicit_path(arguments.distinctive),
                features_override=_explicit_path(arguments.features),
                orthographic_override=_explicit_path(arguments.orthographic),
                output_override=_explicit_path(arguments.output),
            )
            print(f"Residual pairs assessed: {assessment_result.output_rows:,}")
            for family, count in assessment_result.family_counts.items():
                print(f"  {family}: {count:,}")
            print(f"Dataset: {assessment_result.output_path}")
            print(f"Metrics: {assessment_result.metrics_path}")
            print(f"Runtime: {assessment_result.elapsed_seconds:.2f} seconds")
            return 0

        if arguments.command == "finalize":
            config_path = arguments.config.expanduser().resolve()
            settings = load_settings(config_path)
            finalization_result = finalize_employers(
                _config_project_root(config_path),
                settings,
                preprocessed_override=_explicit_path(arguments.preprocessed),
                keys_override=_explicit_path(arguments.keys),
                decisions_override=_explicit_path(arguments.decisions),
                eligibility_override=_explicit_path(arguments.eligibility),
                enrichment_override=_explicit_path(arguments.enrichment),
                parquet_output_override=_explicit_path(arguments.parquet_output),
                csv_output_override=_explicit_path(arguments.csv_output),
            )
            print(f"Final rows: {finalization_result.row_count:,}")
            for outcome, count in finalization_result.outcome_counts.items():
                print(f"  {outcome}: {count:,}")
            print(f"Publicly enriched rows: {finalization_result.public_enriched_rows:,}")
            print(f"Parquet: {finalization_result.parquet_output_path}")
            print(f"CSV: {finalization_result.csv_output_path}")
            print(f"Top employer keys: {finalization_result.top_keys_path}")
            print(f"Metrics: {finalization_result.metrics_path}")
            print(f"Runtime: {finalization_result.elapsed_seconds:.2f} seconds")
            return 0

        if arguments.command == "build-corporate-master":
            config_path = arguments.config.expanduser().resolve()
            corporate_master_result = build_corporate_master(
                _config_project_root(config_path),
                final_dataset_override=_explicit_path(arguments.final_dataset),
                resolution_keys_override=_explicit_path(arguments.keys),
            )
            print(f"Corporate entities: {corporate_master_result.entity_count:,}")
            print(f"Aliases: {corporate_master_result.alias_count:,}")
            print(
                "Source rows represented: "
                f"{corporate_master_result.source_row_count:,}"
            )
            print(
                "Publicly validated entities: "
                f"{corporate_master_result.public_validated_entity_count:,}"
            )
            print(f"Master Parquet: {corporate_master_result.master_parquet_path}")
            print(f"Master CSV: {corporate_master_result.master_csv_path}")
            print(f"Aliases Parquet: {corporate_master_result.aliases_parquet_path}")
            print(f"Aliases CSV: {corporate_master_result.aliases_csv_path}")
            print(f"Metrics: {corporate_master_result.metrics_path}")
            return 0

        if arguments.command == "canonical-quality":
            config_path = arguments.config.expanduser().resolve()
            quality_result = assess_corporate_master_quality(
                _config_project_root(config_path),
                master_override=_explicit_path(arguments.master),
            )
            print(f"Corporate entities assessed: {quality_result.entity_count:,}")
            print(f"PUBLIC_VALIDATED: {quality_result.public_validated_count:,}")
            print(f"ACCEPTABLE: {quality_result.acceptable_count:,}")
            print(f"SUSPICIOUS: {quality_result.suspicious_count:,}")
            print(f"NOT_PROMOTABLE: {quality_result.not_promotable_count:,}")
            print(f"Promotion eligible: {quality_result.promotion_eligible_count:,}")
            print(f"Promotion blocked: {quality_result.promotion_blocked_count:,}")
            print(f"Trailing numeric: {quality_result.trailing_numeric_count:,}")
            print(
                "Legal suffix + trailing numeric: "
                f"{quality_result.legal_suffix_trailing_numeric_count:,}"
            )
            print(
                "Other trailing numeric: "
                f"{quality_result.other_trailing_numeric_count:,}"
            )
            print(f"Invalid characters: {quality_result.invalid_character_count:,}")
            print(f"Parquet: {quality_result.parquet_path}")
            print(f"CSV: {quality_result.csv_path}")
            print(f"Metrics: {quality_result.metrics_path}")
            return 0

        if arguments.command == "promote-references":
            config_path = arguments.config.expanduser().resolve()
            promotion_result = promote_references(
                _config_project_root(config_path),
                master_override=_explicit_path(arguments.master),
                aliases_override=_explicit_path(arguments.aliases),
                quality_override=_explicit_path(arguments.quality),
                reference_master_override=_explicit_path(arguments.reference_master),
                reference_aliases_override=_explicit_path(arguments.reference_aliases),
            )
            print(f"Eligible entities: {promotion_result.eligible_entity_count:,}")
            print(f"New entities promoted: {promotion_result.promoted_entity_count:,}")
            print(f"New aliases promoted: {promotion_result.promoted_alias_count:,}")
            print(f"Existing entities: {promotion_result.existing_entity_count:,}")
            print(f"Existing aliases: {promotion_result.existing_alias_count:,}")
            print(f"Final reference entities: {promotion_result.final_entity_count:,}")
            print(f"Final reference aliases: {promotion_result.final_alias_count:,}")
            print(f"Reference Master: {promotion_result.master_path}")
            print(f"Reference aliases: {promotion_result.aliases_path}")
            print(f"Metrics: {promotion_result.metrics_path}")
            return 0

        if arguments.command == "build-evaluation-sample":
            config_path = arguments.config.expanduser().resolve()
            settings = load_settings(config_path)
            evaluation_result = build_evaluation_sample(
                _config_project_root(config_path),
                settings,
                output_directory_override=_explicit_path(arguments.output_directory),
            )
            print(f"Pair review sample: {evaluation_result.pair_sample_size:,} rows")
            for stratum, count in evaluation_result.pair_stratum_counts.items():
                print(f"  {stratum}: {count:,}")
            print(f"Zero-candidate population: {evaluation_result.zero_candidate_population:,}")
            print(f"Blocking-miss audit: {evaluation_result.blocking_miss_sample_size:,} rows")
            print(f"Pair review CSV: {evaluation_result.pair_review_path}")
            print(f"Blocking-miss CSV: {evaluation_result.blocking_miss_path}")
            return 0

        sample_count = create_evaluation_sample(
            arguments.input,
            arguments.output,
            random_seed=arguments.seed,
        )
        print(f"Evaluation sample: {sample_count:,} rows")
        print(f"Dataset: {arguments.output}")
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        logging.error("%s", error)
        return 1
