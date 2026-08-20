"""Immutable source-contract tests."""

from collections.abc import Callable
from pathlib import Path

import pytest

from credit_risk_er.ingestion import (
    SourceContractError,
    deterministic_record_id,
    iter_source_batches,
    sha256_file,
    validate_source,
)
from tests.conftest import settings_for


def test_validation_and_iteration_preserve_values(
    tmp_path: Path, workbook_factory: Callable[..., Path]
) -> None:
    values = ["  ACME  ", None, "Café S.A.", "ACME"]
    workbook = workbook_factory(tmp_path / "source.xlsx", values)
    original_hash = sha256_file(workbook)
    settings = settings_for(tmp_path, workbook)

    metadata = validate_source(workbook, settings.source)
    observed = [row for batch in iter_source_batches(metadata, 2) for row in batch]

    assert [row.nombre_original for row in observed] == values
    assert [row.source_row_number for row in observed] == [2, 3, 4, 5]
    assert sha256_file(workbook) == original_hash


def test_source_fingerprint_mismatch_is_rejected(
    tmp_path: Path, workbook_factory: Callable[..., Path]
) -> None:
    workbook = workbook_factory(tmp_path / "source.xlsx", ["ACME"])
    settings = settings_for(tmp_path, workbook)
    invalid_source = settings.source.model_copy(update={"expected_sha256": "0" * 64})
    with pytest.raises(SourceContractError, match="SHA-256 mismatch"):
        validate_source(workbook, invalid_source)


@pytest.mark.parametrize(
    ("header", "extra_sheet", "message"),
    [
        ("Nombre original", False, "Source column mismatch"),
        ("nombre_original", True, "Workbook sheets differ"),
    ],
)
def test_schema_mismatch_is_rejected(
    tmp_path: Path,
    workbook_factory: Callable[..., Path],
    header: str,
    extra_sheet: bool,
    message: str,
) -> None:
    workbook = workbook_factory(
        tmp_path / "source.xlsx", ["ACME"], header=header, extra_sheet=extra_sheet
    )
    settings = settings_for(tmp_path, workbook)
    with pytest.raises(SourceContractError, match=message):
        validate_source(workbook, settings.source)


def test_record_ids_are_deterministic_and_row_specific() -> None:
    first = deterministic_record_id("A" * 64, "Sheet1", 2)
    assert first == deterministic_record_id("A" * 64, "Sheet1", 2)
    assert first != deterministic_record_id("A" * 64, "Sheet1", 3)
    assert len(first) == 64
