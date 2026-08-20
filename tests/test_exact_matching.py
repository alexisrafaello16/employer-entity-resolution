"""Exact matching behavior against validated employer knowledge."""

from collections.abc import Callable
from pathlib import Path

from credit_risk_er.employer_master import load_employer_knowledge
from credit_risk_er.matching.exact import resolve_exact
from credit_risk_er.normalization import normalize_employer
from tests.conftest import settings_for


def test_canonical_alias_unknown_blank_and_numeric_protection(
    tmp_path: Path, reference_factory: Callable[..., tuple[Path, Path]]
) -> None:
    source = tmp_path / "source.xlsx"
    source.write_bytes(b"not-read")
    config = settings_for(tmp_path, source).normalization
    master, aliases = reference_factory(
        tmp_path / "reference",
        master_rows=[
            ("EMP-000001", "BANCO GENERAL"),
            ("EMP-000002", "STUDIO 507"),
        ],
        alias_rows=[("EMP-000001", "BANCO GRAL")],
    )
    knowledge = load_employer_knowledge(master, aliases, config)

    canonical = resolve_exact(
        normalize_employer("banco general", config).strict, knowledge.exact_index
    )
    alias = resolve_exact(normalize_employer("BANCO GRAL", config).strict, knowledge.exact_index)
    unknown = resolve_exact(normalize_employer("UNKNOWN", config).strict, knowledge.exact_index)
    trailing = resolve_exact(
        normalize_employer("BANCO GENERAL 4", config).strict, knowledge.exact_index
    )
    studio = resolve_exact(normalize_employer("STUDIO 507", config).strict, knowledge.exact_index)
    blank = resolve_exact(None, knowledge.exact_index)

    assert canonical.resolution_method == "exact_canonical"
    assert canonical.canonical_name == "BANCO GENERAL"
    assert alias.resolution_method == "exact_alias"
    assert unknown.resolution_status == "unresolved"
    assert unknown.canonical_name is None
    assert trailing.resolution_status == "unresolved"
    assert studio.resolution_status == "resolved"
    assert studio.canonical_name == "STUDIO 507"
    assert blank.resolution_status == "unresolved"
