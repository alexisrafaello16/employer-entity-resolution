"""Validated persistent employer knowledge and exact lookup construction."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

from credit_risk_er.config import NormalizationConfig
from credit_risk_er.normalization import normalize_employer

type MatchSource = Literal["canonical", "alias"]

ENTITY_ID_PATTERN: Final = re.compile(
    r"^(?:EMP-\d{6}|PUB-\d{6}|ENT-[0-9A-F]{16})$"
)

ENTITY_ID_FORMAT_DESCRIPTION: Final = (
    "EMP- followed by six digits, "
    "PUB- followed by six digits, "
    "or ENT- followed by sixteen uppercase hexadecimal characters"
)

MASTER_COLUMNS: Final = (
    "entity_id",
    "canonical_name",
)

ALIAS_COLUMNS: Final = (
    "entity_id",
    "alias_name",
)


class ReferenceDataError(ValueError):
    """Raised when validated employer knowledge is missing or ambiguous."""


@dataclass(frozen=True, slots=True)
class EmployerEntity:
    """One reusable canonical employer entity."""

    entity_id: str
    canonical_name: str


@dataclass(frozen=True, slots=True)
class ExactLookupEntry:
    """One strict-normalized exact-match lookup entry."""

    entity_id: str
    canonical_name: str
    match_source: MatchSource


@dataclass(frozen=True, slots=True)
class EmployerKnowledge:
    """Validated entities, deduplicated aliases, and strict exact-match index."""

    entities: dict[str, EmployerEntity]
    aliases: frozenset[tuple[str, str]]
    exact_index: dict[str, ExactLookupEntry]

    @property
    def entity_count(self) -> int:
        """Return the number of canonical entities."""
        return len(self.entities)

    @property
    def alias_count(self) -> int:
        """Return the number of unique entity-alias relations."""
        return len(self.aliases)


def _read_rows(
    path: Path,
    expected_columns: tuple[str, ...],
) -> list[dict[str, str]]:
    if not path.is_file():
        raise ReferenceDataError(
            f"Required reference file does not exist: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as stream:
        reader = csv.DictReader(stream)
        observed = tuple(
            reader.fieldnames or ()
        )

        if observed != expected_columns:
            raise ReferenceDataError(
                f"Invalid columns in {path.name}: "
                f"expected {expected_columns}, "
                f"observed {observed}"
            )

        return [
            dict(row)
            for row in reader
        ]


def _required_value(
    row: dict[str, str],
    field: str,
    path: Path,
    row_number: int,
) -> str:
    value = (
        row.get(field) or ""
    ).strip()

    if not value:
        raise ReferenceDataError(
            f"Blank {field} in {path.name} "
            f"at CSV row {row_number}"
        )

    return value


def _entity_id(
    row: dict[str, str],
    path: Path,
    row_number: int,
) -> str:
    entity_id = _required_value(
        row,
        "entity_id",
        path,
        row_number,
    )

    if (
        ENTITY_ID_PATTERN.fullmatch(
            entity_id
        )
        is None
    ):
        raise ReferenceDataError(
            f"Invalid entity_id {entity_id!r} "
            f"in {path.name} at CSV row {row_number}; "
            f"expected {ENTITY_ID_FORMAT_DESCRIPTION}"
        )

    return entity_id


def _strict_key(
    value: str,
    config: NormalizationConfig,
    *,
    context: str,
) -> str:
    key = normalize_employer(
        value,
        config,
    ).strict

    if not key:
        raise ReferenceDataError(
            f"{context} has no nonblank "
            "strict-normalized lookup key"
        )

    return key


def _add_lookup(
    index: dict[str, ExactLookupEntry],
    key: str,
    entry: ExactLookupEntry,
) -> None:
    existing = index.get(
        key
    )

    if existing is None:
        index[key] = entry
        return

    if (
        existing.entity_id
        != entry.entity_id
    ):
        raise ReferenceDataError(
            f"Ambiguous strict-normalized key "
            f"{key!r}: maps to both "
            f"{existing.entity_id} and "
            f"{entry.entity_id}"
        )

    # Canonical knowledge takes precedence when an alias
    # repeats the same strict-normalized key for the same entity.
    if entry.match_source == "canonical":
        index[key] = entry


def load_employer_knowledge(
    master_path: Path,
    aliases_path: Path,
    normalization: NormalizationConfig,
) -> EmployerKnowledge:
    """
    Load, validate, deduplicate, and index reusable employer knowledge.

    Supported persistent entity-ID namespaces are:

    - EMP-NNNNNN: curated or legacy reference entities.
    - PUB-NNNNNN: publicly validated entities.
    - ENT-XXXXXXXXXXXXXXXX: deterministic generated entities,
      where X is an uppercase hexadecimal character.

    Every canonical name and alias is strict-normalized before being
    inserted into the exact-match index. A normalized key mapping to
    more than one entity is rejected rather than resolved ambiguously.
    """

    master_rows = _read_rows(
        master_path,
        MASTER_COLUMNS,
    )

    alias_rows = _read_rows(
        aliases_path,
        ALIAS_COLUMNS,
    )

    entities: dict[
        str,
        EmployerEntity,
    ] = {}

    index: dict[
        str,
        ExactLookupEntry,
    ] = {}

    for (
        row_number,
        row,
    ) in enumerate(
        master_rows,
        start=2,
    ):
        entity_id = _entity_id(
            row,
            master_path,
            row_number,
        )

        canonical_name = _required_value(
            row,
            "canonical_name",
            master_path,
            row_number,
        )

        existing = entities.get(
            entity_id
        )

        if (
            existing is not None
            and existing.canonical_name
            != canonical_name
        ):
            raise ReferenceDataError(
                f"Entity {entity_id} maps to "
                "multiple canonical names in "
                f"{master_path.name}"
            )

        entity = EmployerEntity(
            entity_id=entity_id,
            canonical_name=canonical_name,
        )

        entities[
            entity_id
        ] = entity

        key = _strict_key(
            canonical_name,
            normalization,
            context=(
                f"Canonical name for {entity_id}"
            ),
        )

        _add_lookup(
            index,
            key,
            ExactLookupEntry(
                entity_id=entity_id,
                canonical_name=canonical_name,
                match_source="canonical",
            ),
        )

    aliases: set[
        tuple[str, str]
    ] = set()

    for (
        row_number,
        row,
    ) in enumerate(
        alias_rows,
        start=2,
    ):
        entity_id = _entity_id(
            row,
            aliases_path,
            row_number,
        )

        alias_name = _required_value(
            row,
            "alias_name",
            aliases_path,
            row_number,
        )

        alias_entity = entities.get(
            entity_id
        )

        if alias_entity is None:
            raise ReferenceDataError(
                f"Alias at {aliases_path.name} "
                f"row {row_number} references "
                "missing entity "
                f"{entity_id}"
            )

        relation = (
            entity_id,
            alias_name,
        )

        if relation in aliases:
            continue

        aliases.add(
            relation
        )

        key = _strict_key(
            alias_name,
            normalization,
            context=(
                f"Alias for {entity_id}"
            ),
        )

        _add_lookup(
            index,
            key,
            ExactLookupEntry(
                entity_id=entity_id,
                canonical_name=(
                    alias_entity.canonical_name
                ),
                match_source="alias",
            ),
        )

    return EmployerKnowledge(
        entities=entities,
        aliases=frozenset(
            aliases
        ),
        exact_index=index,
    )