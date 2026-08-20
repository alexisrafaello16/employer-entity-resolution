"""Deterministic, initially unlabeled human-evaluation dataset tests."""

from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from credit_risk_er.cli import main
from credit_risk_er.config import Settings
from credit_risk_er.evaluation import (
    AUDIT_OUTPUT_COLUMNS,
    PAIR_OUTPUT_COLUMNS,
    PAIR_STRATUM_PRIORITY,
    AuditKey,
    EvaluationResult,
    PairEvidence,
    assign_pair_stratum,
    build_evaluation_sample,
    derive_zero_candidate_keys,
    sample_blocking_misses,
    sample_review_pairs,
)
from credit_risk_er.pipeline import (
    CANDIDATE_FEATURES_SCHEMA,
    CANDIDATE_PAIRS_SCHEMA,
    RESOLUTION_KEYS_SCHEMA,
)
from tests.conftest import settings_for


def _evidence(
    *,
    key: str = "PAIR01",
    name_a: str = "ALPHA COMPANY",
    name_b: str = "BETA COMPANY",
    methods: tuple[str, ...] = ("prefix_signature",),
    char_ratio: float = 70.0,
    token_set_ratio: float = 70.0,
    partial_ratio: float = 70.0,
    length_ratio: float = 1.0,
    numeric_relation: str = "none",
) -> PairEvidence:
    return PairEvidence(
        key_a=f"{key} A",
        key_b=f"{key} B",
        name_a=name_a,
        name_b=name_b,
        blocking_methods=methods,
        blocking_method_count=len(methods),
        char_ratio=char_ratio,
        token_sort_ratio=char_ratio,
        token_set_ratio=token_set_ratio,
        partial_ratio=partial_ratio,
        length_ratio=length_ratio,
        common_prefix_ratio=0.0,
        token_jaccard=0.25,
        same_first_token=False,
        numeric_relation=numeric_relation,
    )


def _feature_row(pair: PairEvidence) -> dict[str, object]:
    return {
        "key_a": pair.key_a,
        "key_b": pair.key_b,
        "name_a": pair.name_a,
        "name_b": pair.name_b,
        "blocking_methods": list(pair.blocking_methods),
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
    }


def _stratum_pairs() -> list[PairEvidence]:
    return [
        _evidence(key="PAIR01", numeric_relation="conflict"),
        _evidence(key="PAIR02", methods=("trailing_numeric",), numeric_relation="one_sided"),
        _evidence(key="PAIR03", token_set_ratio=98.0, char_ratio=70.0),
        _evidence(key="PAIR04", partial_ratio=100.0, length_ratio=0.7),
        _evidence(key="PAIR05", methods=("relaxed_key", "prefix_signature")),
        _evidence(key="PAIR06", methods=("truncation_prefix",)),
        _evidence(key="PAIR07", methods=("informative_token",)),
        _evidence(key="PAIR08", name_a="ALPHA", name_b="BETA"),
        _evidence(key="PAIR09", char_ratio=50.0),
        _evidence(key="PAIR10", char_ratio=98.0),
        _evidence(key="PAIR11", char_ratio=93.0),
        _evidence(key="PAIR12", char_ratio=85.0),
    ]


@pytest.mark.parametrize(
    ("pair", "expected"),
    [
        (_evidence(numeric_relation="conflict"), "numeric_conflict"),
        (
            _evidence(methods=("trailing_numeric",), numeric_relation="one_sided"),
            "trailing_numeric",
        ),
        (
            _evidence(token_set_ratio=100.0, char_ratio=60.0),
            "high_token_set_low_char",
        ),
        (_evidence(char_ratio=50.0), "low_similarity_retrieved"),
    ],
)
def test_explicit_pair_stratum_behavior(pair: PairEvidence, expected: str) -> None:
    assert assign_pair_stratum(pair) == expected


def test_pair_sampling_is_deterministic_unique_and_covers_available_strata(
    tmp_path: Path,
) -> None:
    path = tmp_path / "features.parquet"
    rows = [_feature_row(pair) for pair in _stratum_pairs()]
    pq.write_table(pa.Table.from_pylist(rows, schema=CANDIDATE_FEATURES_SCHEMA), path)

    first = sample_review_pairs(path, sample_size=12, random_seed=20260813, batch_size=3)
    second = sample_review_pairs(path, sample_size=12, random_seed=20260813, batch_size=5)

    assert first == second
    identities = [(pair.key_a, pair.key_b) for _, pair in first]
    assert len(identities) == len(set(identities)) == 12
    assert Counter(stratum for stratum, _ in first) == Counter(
        {stratum: 1 for stratum in PAIR_STRATUM_PRIORITY}
    )


def _resolution_row(key: str, *, truncated: bool = False) -> dict[str, object]:
    return {
        "resolution_key": key,
        "representative_name": key,
        "relaxed_key": key,
        "source_row_frequency": 1,
        "representative_record_id": f"record-{key}",
        "representative_source_row_number": 2,
        "representative_route": "employer_resolution_candidate",
        "trailing_numeric_candidate": None,
        "possible_truncation": truncated,
        "token_count": len(key.split()),
    }


def test_zero_candidate_derivation_excludes_every_key_with_a_pair(tmp_path: Path) -> None:
    resolution_path = tmp_path / "keys.parquet"
    pairs_path = tmp_path / "pairs.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [_resolution_row(key) for key in ("ALPHA", "BETA", "DELTA", "GAMMA")],
            schema=RESOLUTION_KEYS_SCHEMA,
        ),
        resolution_path,
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "key_a": "ALPHA",
                    "key_b": "BETA",
                    "name_a": "ALPHA",
                    "name_b": "BETA",
                    "blocking_methods": ["prefix_signature"],
                }
            ],
            schema=CANDIDATE_PAIRS_SCHEMA,
        ),
        pairs_path,
    )

    zero = derive_zero_candidate_keys(resolution_path, pairs_path, batch_size=2)
    assert [item.resolution_key for item in zero] == ["DELTA", "GAMMA"]


def test_blocking_miss_sampling_is_seed_reproducible() -> None:
    zero = [
        AuditKey(
            resolution_key=f"EMPLOYER {index}",
            representative_name=f"EMPLOYER {index}",
            source_row_frequency=2 if index % 5 == 0 else 1,
            representative_route=(
                "ambiguous_review_candidate" if index % 7 == 0 else "employer_resolution_candidate"
            ),
            possible_truncation=index % 11 == 0,
            token_count=2,
        )
        for index in range(60)
    ]
    first = sample_blocking_misses(zero, sample_size=18, random_seed=42)
    second = sample_blocking_misses(zero, sample_size=18, random_seed=42)
    assert first == second
    assert len({item.resolution_key for _, item in first}) == len(first) == 18


def _write_integration_inputs(settings: Settings) -> None:
    feature_rows = [_feature_row(pair) for pair in _stratum_pairs()]
    feature_path = settings.candidate_scoring.output_dataset
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(feature_rows, schema=CANDIDATE_FEATURES_SCHEMA), feature_path
    )
    candidate_pair = {
        "key_a": "KNOWN A",
        "key_b": "KNOWN B",
        "name_a": "KNOWN A",
        "name_b": "KNOWN B",
        "blocking_methods": ["prefix_signature"],
    }
    candidate_path = settings.candidate_generation.candidate_pairs_output
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist([candidate_pair], schema=CANDIDATE_PAIRS_SCHEMA), candidate_path
    )
    resolution_rows = [_resolution_row("KNOWN A"), _resolution_row("KNOWN B")]
    resolution_rows.extend(_resolution_row(f"ZERO EMPLOYER {index}") for index in range(20))
    resolution_path = settings.candidate_generation.resolution_keys_output
    pq.write_table(
        pa.Table.from_pylist(resolution_rows, schema=RESOLUTION_KEYS_SCHEMA), resolution_path
    )


def test_build_evaluation_csvs_are_human_readable_and_initially_unlabeled(
    tmp_path: Path,
    workbook_factory: Callable[..., Path],
) -> None:
    workbook = workbook_factory(tmp_path / "input/source.xlsx", ["unused"])
    settings = settings_for(tmp_path, workbook)
    settings = settings.model_copy(
        update={
            "evaluation": settings.evaluation.model_copy(
                update={"pair_sample_size": 12, "blocking_miss_sample_size": 6}
            )
        }
    )
    _write_integration_inputs(settings)

    result = build_evaluation_sample(tmp_path, settings)

    assert result.pair_sample_size == 12
    assert result.duplicate_pairs == 0
    assert result.nonblank_review_labels == 0
    assert result.blocking_miss_sample_size == 6
    assert result.zero_candidate_population == 20
    assert result.pair_review_path.read_bytes().startswith(b"\xef\xbb\xbf")
    with result.pair_review_path.open(encoding="utf-8-sig", newline="") as stream:
        pair_rows = list(csv.DictReader(stream))
    with result.blocking_miss_path.open(encoding="utf-8-sig", newline="") as stream:
        audit_rows = list(csv.DictReader(stream))
    assert tuple(pair_rows[0]) == PAIR_OUTPUT_COLUMNS
    assert tuple(audit_rows[0]) == AUDIT_OUTPUT_COLUMNS
    assert all(not row["review_label"] and not row["review_notes"] for row in pair_rows)
    assert all(
        not row["possible_blocking_miss"]
        and not row["suspected_variant"]
        and not row["review_notes"]
        for row in audit_rows
    )
    assert {row["resolution_key"] for row in audit_rows}.isdisjoint({"KNOWN A", "KNOWN B"})


def test_build_evaluation_sample_cli_smoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = EvaluationResult(
        pair_review_path=tmp_path / "pair_review_sample.csv",
        blocking_miss_path=tmp_path / "blocking_miss_audit.csv",
        pair_sample_size=12,
        pair_stratum_counts={"numeric_conflict": 1},
        duplicate_pairs=0,
        nonblank_review_labels=0,
        blocking_miss_sample_size=6,
        zero_candidate_population=20,
    )
    monkeypatch.setattr("credit_risk_er.cli.load_settings", lambda _: object())
    monkeypatch.setattr(
        "credit_risk_er.cli.build_evaluation_sample", lambda *args, **kwargs: result
    )
    exit_code = main(["build-evaluation-sample"])
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Pair review sample: 12 rows" in output
    assert "Blocking-miss audit: 6 rows" in output
