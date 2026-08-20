"""Deterministic normalization tests."""

from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from credit_risk_er.config import NormalizationConfig
from credit_risk_er.normalization import normalize_employer
from tests.conftest import settings_for


@pytest.fixture
def normalization_config(tmp_path: Path) -> NormalizationConfig:
    from openpyxl import Workbook

    path = tmp_path / "source.xlsx"
    workbook = Workbook()
    workbook.save(path)
    workbook.close()
    return settings_for(tmp_path, path).normalization


def test_strict_and_relaxed_representations(
    normalization_config: NormalizationConfig,
) -> None:
    result = normalize_employer("'  Café—Norte, S A  ", normalization_config)
    assert result.strict == "CAFÉ-NORTE, S A"
    assert result.relaxed == "CAFÉ NORTE SA"


def test_originally_blank_and_null_are_distinct_inputs(
    normalization_config: NormalizationConfig,
) -> None:
    assert normalize_employer(None, normalization_config).strict is None
    assert normalize_employer("'", normalization_config).strict == ""


@pytest.mark.parametrize(
    ("value", "has_token", "candidate"),
    [("EMPRESA 7", True, "EMPRESA"), ("EMPRESA 77", True, None), ("7 EMPRESA", False, None)],
)
def test_trailing_numeric_candidate(
    normalization_config: NormalizationConfig,
    value: str,
    has_token: bool,
    candidate: str | None,
) -> None:
    result = normalize_employer(value, normalization_config)
    assert result.has_trailing_numeric_token is has_token
    assert result.trailing_numeric_candidate == candidate


def test_possible_truncation_is_only_a_signal(
    normalization_config: NormalizationConfig,
) -> None:
    result = normalize_employer("A" * 30, normalization_config)
    assert result.strict == "A" * 30
    assert result.possible_truncation is True


@given(st.one_of(st.none(), st.text(max_size=100)))
def test_normalization_is_deterministic_and_idempotent(value: str | None) -> None:
    config = NormalizationConfig(
        ruleset_version="1",
        possible_truncation_content_lengths=(30,),
        trailing_numeric_candidate_max_digits=1,
        corporate_suffix_aliases={"SA": ("S A", "SA")},
    )
    first = normalize_employer(value, config)
    assert first == normalize_employer(value, config)
