"""Employer Master and validated alias contract tests."""

from collections.abc import Callable
from pathlib import Path

import pytest

from credit_risk_er.config import NormalizationConfig
from credit_risk_er.employer_master import ReferenceDataError, load_employer_knowledge
from credit_risk_er.normalization import normalize_employer
from tests.conftest import settings_for


def _normalization(tmp_path: Path) -> NormalizationConfig:
    source = tmp_path / "source.xlsx"
    source.write_bytes(b"not-read")
    return settings_for(tmp_path, source).normalization


def test_empty_header_only_reference_data_is_valid(
    tmp_path: Path, reference_factory: Callable[..., tuple[Path, Path]]
) -> None:
    master, aliases = reference_factory(tmp_path, master_rows=[], alias_rows=[])
    knowledge = load_employer_knowledge(master, aliases, _normalization(tmp_path))
    assert knowledge.entity_count == 0
    assert knowledge.alias_count == 0
    assert knowledge.exact_index == {}


def test_valid_reference_data_builds_canonical_and_alias_index(
    tmp_path: Path, reference_factory: Callable[..., tuple[Path, Path]]
) -> None:
    master, aliases = reference_factory(
        tmp_path,
        master_rows=[("EMP-000001", "BANCO GENERAL, S.A.")],
        alias_rows=[("EMP-000001", "BANCO GENERAL")],
    )
    config = _normalization(tmp_path)
    knowledge = load_employer_knowledge(master, aliases, config)
    canonical_key = normalize_employer("BANCO GENERAL, S.A.", config).strict
    alias_key = normalize_employer("BANCO GENERAL", config).strict
    assert canonical_key is not None and alias_key is not None
    assert knowledge.exact_index[canonical_key].match_source == "canonical"
    assert knowledge.exact_index[alias_key].match_source == "alias"


def test_alias_referencing_missing_entity_fails(
    tmp_path: Path, reference_factory: Callable[..., tuple[Path, Path]]
) -> None:
    master, aliases = reference_factory(
        tmp_path,
        master_rows=[],
        alias_rows=[("EMP-000001", "UNKNOWN")],
    )
    with pytest.raises(ReferenceDataError, match="references missing entity"):
        load_employer_knowledge(master, aliases, _normalization(tmp_path))


def test_duplicate_entity_with_different_canonical_names_fails(
    tmp_path: Path, reference_factory: Callable[..., tuple[Path, Path]]
) -> None:
    master, aliases = reference_factory(
        tmp_path,
        master_rows=[("EMP-000001", "ACME"), ("EMP-000001", "OTHER ACME")],
        alias_rows=[],
    )
    with pytest.raises(ReferenceDataError, match="multiple canonical names"):
        load_employer_knowledge(master, aliases, _normalization(tmp_path))


def test_strict_normalized_alias_conflict_across_entities_fails(
    tmp_path: Path, reference_factory: Callable[..., tuple[Path, Path]]
) -> None:
    master, aliases = reference_factory(
        tmp_path,
        master_rows=[("EMP-000001", "FIRST"), ("EMP-000002", "SECOND")],
        alias_rows=[("EMP-000001", "Global  Bank"), ("EMP-000002", "GLOBAL BANK")],
    )
    with pytest.raises(ReferenceDataError, match="Ambiguous strict-normalized key"):
        load_employer_knowledge(master, aliases, _normalization(tmp_path))


def test_canonical_conflict_after_normalization_fails(
    tmp_path: Path, reference_factory: Callable[..., tuple[Path, Path]]
) -> None:
    master, aliases = reference_factory(
        tmp_path,
        master_rows=[("EMP-000001", "Acme"), ("EMP-000002", " ACME ")],
        alias_rows=[],
    )
    with pytest.raises(ReferenceDataError, match="Ambiguous strict-normalized key"):
        load_employer_knowledge(master, aliases, _normalization(tmp_path))


def test_duplicate_identical_alias_relationship_is_deduplicated(
    tmp_path: Path, reference_factory: Callable[..., tuple[Path, Path]]
) -> None:
    master, aliases = reference_factory(
        tmp_path,
        master_rows=[("EMP-000001", "ACME")],
        alias_rows=[("EMP-000001", "ACME CORP"), ("EMP-000001", "ACME CORP")],
    )
    knowledge = load_employer_knowledge(master, aliases, _normalization(tmp_path))
    assert knowledge.alias_count == 1


@pytest.mark.parametrize(
    ("master_rows", "alias_rows", "message"),
    [
        ([("", "ACME")], [], "Blank entity_id"),
        ([("BAD-1", "ACME")], [], "Invalid entity_id"),
        ([("EMP-000001", "")], [], "Blank canonical_name"),
        ([("EMP-000001", "ACME")], [("EMP-000001", "")], "Blank alias_name"),
    ],
)
def test_blank_and_invalid_reference_values_fail(
    tmp_path: Path,
    reference_factory: Callable[..., tuple[Path, Path]],
    master_rows: list[tuple[str, str]],
    alias_rows: list[tuple[str, str]],
    message: str,
) -> None:
    master, aliases = reference_factory(tmp_path, master_rows=master_rows, alias_rows=alias_rows)
    with pytest.raises(ReferenceDataError, match=message):
        load_employer_knowledge(master, aliases, _normalization(tmp_path))


def test_missing_reference_file_fails(tmp_path: Path) -> None:
    with pytest.raises(ReferenceDataError, match="does not exist"):
        load_employer_knowledge(
            tmp_path / "missing-master.csv",
            tmp_path / "missing-aliases.csv",
            _normalization(tmp_path),
        )


def test_reference_columns_must_match_contract(tmp_path: Path) -> None:
    master = tmp_path / "employer_master.csv"
    aliases = tmp_path / "employer_aliases.csv"
    master.write_text("entity_id,name\n", encoding="utf-8")
    aliases.write_text("entity_id,alias_name\n", encoding="utf-8")
    with pytest.raises(ReferenceDataError, match="Invalid columns"):
        load_employer_knowledge(master, aliases, _normalization(tmp_path))
