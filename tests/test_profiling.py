"""Explicit profiling and separately invoked sampling tests."""

from collections.abc import Callable
from pathlib import Path
from typing import cast

import pyarrow.parquet as pq

from credit_risk_er.pipeline import preprocess
from credit_risk_er.profiling import create_evaluation_sample, profile_dataset
from tests.conftest import settings_for


def test_profile_and_sample_are_explicit_and_deterministic(
    tmp_path: Path, workbook_factory: Callable[..., Path]
) -> None:
    values = [
        "ACME",
        "CALLE 10",
        "ESTUDIANTE",
        "EMPRESA SA",
        "123",
        None,
        "ABC",
        "UNIVERSIDAD ESTUDIANTE",
    ]
    workbook = workbook_factory(tmp_path / "source.xlsx", values)
    result = preprocess(tmp_path, settings_for(tmp_path, workbook))
    first = tmp_path / "sample-a.parquet"
    second = tmp_path / "sample-b.parquet"
    quotas = {"blank": 2, "ordinary-employer-like": 2, "address-candidate": 2}

    first_count = create_evaluation_sample(result.output_path, first, quotas=quotas)
    second_count = create_evaluation_sample(result.output_path, second, quotas=quotas)

    assert first_count == second_count
    assert pq.read_table(first).equals(pq.read_table(second))
    routes = cast(dict[str, int], profile_dataset(result.output_path)["routes"])
    assert set(routes) == {
        "address_candidate",
        "ambiguous_review_candidate",
        "blank_candidate",
        "employer_resolution_candidate",
        "non_employer_status_candidate",
    }
