"""Conservative final employer canonicalization and business-ready export."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Final, cast
from urllib.parse import urlparse

import pyarrow as pa
import pyarrow.parquet as pq

from credit_risk_er.config import Settings
from credit_risk_er.ingestion import sha256_file
from credit_risk_er.models import FinalizationResult, JsonValue

LOGGER = logging.getLogger(__name__)

EMPLOYER_CANDIDATE: Final = "EMPLOYER_CANDIDATE"
ADDRESS: Final = "ADDRESS"
AUTO_SAME: Final = "AUTO_SAME"

EXPLICIT_MISSING_INFORMATION_VALUES: Final = frozenset(
    {
        "SIN INFORMACION",
        "NO TIENE",
        "NO APLICA",
        "NOAPLICA",
        "NA",
        "N A",
        "NONE",
        "NINGUNO",
        "NINGUNA",
        "DESCONOCIDO",
    }
)

LEGAL_FORM_SUFFIXES: Final = (
    ("S", "A", "S"),
    ("SAS",),
    ("S", "A"),
    ("SA",),
    ("INC",),
    ("CORP",),
    ("LLC",),
    ("LTDA",),
    ("CIA",),
)
LEGAL_FORM_CANONICAL: Final = {
    ("S", "A", "S"): "SAS",
    ("SAS",): "SAS",
    ("S", "A"): "SA",
    ("SA",): "SA",
    ("INC",): "INC",
    ("CORP",): "CORP",
    ("LLC",): "LLC",
    ("LTDA",): "LTDA",
    ("CIA",): "CIA",
}

SCORE_PUBLIC_VALIDATED: Final = 100
SCORE_DETERMINISTIC_AUTO_SAME: Final = 92
SCORE_NORMALIZED_EMPLOYER: Final = 60
SCORE_ADDRESS_CLASSIFICATION: Final = 100
SCORE_EXPLICIT_MISSING_INFORMATION: Final = 100
SCORE_PRECISION_FIRST_ABSTENTION: Final = 40

PUBLIC_ENRICHMENT_COLUMNS: Final = (
    "public_entity_id",
    "resolution_key",
    "canonical_name",
    "sector",
    "source_url",
    "validated_on",
)
PUBLIC_ENTITY_ID_PATTERN: Final = re.compile(r"^PUB-\d{6}$")

FINAL_SCHEMA = pa.schema(
    [
        pa.field("record_id", pa.string(), nullable=False),
        pa.field("source_row_number", pa.int64(), nullable=False),
        pa.field("nombre_original", pa.string(), nullable=True),
        pa.field("nombre_normalizado", pa.string(), nullable=True),
        pa.field("resolution_key", pa.string(), nullable=True),
        pa.field("entity_id", pa.string(), nullable=True),
        pa.field("nombre_propuesto", pa.string(), nullable=False),
        pa.field("sector_propuesto", pa.string(), nullable=False),
        pa.field("metodo_sector", pa.string(), nullable=False),
        pa.field("sector_evidence", pa.string(), nullable=True),
        pa.field("resultado_final", pa.string(), nullable=False),
        pa.field("confianza_resolucion", pa.string(), nullable=False),
        pa.field("score_confianza_resolucion", pa.int32(), nullable=False),
        pa.field("confianza_sector", pa.string(), nullable=False),
        pa.field("metodo_resolucion", pa.string(), nullable=False),
        pa.field("explicacion", pa.string(), nullable=False),
        pa.field("eligibility_status", pa.string(), nullable=True),
        pa.field("eligibility_rule", pa.string(), nullable=True),
        pa.field("eligibility_evidence", pa.string(), nullable=True),
        pa.field("canonical_resolution_key", pa.string(), nullable=True),
        pa.field("component_key_count", pa.int32(), nullable=False),
        pa.field("component_source_row_count", pa.int64(), nullable=False),
        pa.field("decision_rules", pa.string(), nullable=True),
        pa.field("public_entity_id", pa.string(), nullable=True),
        pa.field("public_source_url", pa.string(), nullable=True),
        pa.field("public_validation_date", pa.string(), nullable=True),
        pa.field("route", pa.string(), nullable=False),
        pa.field("route_reason", pa.string(), nullable=False),
    ]
)
FINAL_COLUMNS: Final = tuple(field.name for field in FINAL_SCHEMA)


@dataclass(frozen=True, slots=True)
class KeyMetadata:
    """Fields used to select and explain one component representative."""

    resolution_key: str
    representative_name: str
    source_row_frequency: int
    possible_truncation: bool
    token_count: int
    representative_route: str


@dataclass(frozen=True, slots=True)
class EligibilityMetadata:
    """Automatic employer-compatibility outcome for one resolution key."""

    status: str
    rule: str
    evidence: str


@dataclass(frozen=True, slots=True)
class PublicEnrichment:
    """One manually approved resolution key linked to a public entity."""

    public_entity_id: str
    resolution_key: str
    canonical_name: str
    sector: str
    source_url: str
    validated_on: str


@dataclass(frozen=True, slots=True)
class PublicEntityDefinition:
    """Shared public attributes for all approved keys of one entity."""

    public_entity_id: str
    canonical_name: str
    sector: str
    source_url: str
    validated_on: str


@dataclass(frozen=True, slots=True)
class ComponentSummary:
    """Canonical resolution result shared by every key in one component."""

    entity_id: str
    canonical_resolution_key: str
    canonical_name: str
    sector: str
    sector_method: str
    sector_evidence: str | None
    sector_confidence: str
    key_count: int
    source_row_count: int
    decision_rules: tuple[str, ...]
    public_entity_id: str | None
    public_source_url: str | None
    public_validation_date: str | None


@dataclass(frozen=True, slots=True)
class ExactReferenceRow:
    """One row resolved early against persistent exact-match knowledge."""

    record_id: str
    source_row_number: int
    nombre_original: str | None
    nombre_normalizado: str
    route: str
    route_reason: str
    entity_id: str
    canonical_name: str
    resolution_method: str
    resolution_reason: str


@dataclass(frozen=True, slots=True)
class ExactReferenceReuse:
    """Validated exact-reference rows and deterministic entity summaries."""

    by_record_id: dict[str, ExactReferenceRow]
    component_by_entity_id: dict[str, ComponentSummary]
    total_rows: int
    unresolved_rows: int
    canonical_rows: int
    alias_rows: int
    source_contract_digest: str


class UnionFind:
    """Deterministic components with legal-form safeguards for automatic unions."""

    def __init__(self, values: set[str]) -> None:
        self._parent = {value: value for value in values}
        self._size = dict.fromkeys(values, 1)
        self._legal_forms = {value: _legal_forms(value) for value in values}

    def find(self, value: str) -> str:
        parent = self._parent
        if value not in parent:
            raise ValueError(f"Unknown employer-compatible resolution key: {value!r}")
        root = value
        while parent[root] != root:
            root = parent[root]
        while parent[value] != value:
            next_value = parent[value]
            parent[value] = root
            value = next_value
        return root

    def legal_forms(self, value: str) -> frozenset[str]:
        return self._legal_forms[self.find(value)]

    def has_legal_form_conflict(self, left: str, right: str) -> bool:
        left_forms = self.legal_forms(left)
        right_forms = self.legal_forms(right)
        return bool(left_forms and right_forms and left_forms.isdisjoint(right_forms))

    def union(self, left: str, right: str, *, enforce_legal_compatibility: bool = True) -> bool:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return False
        if enforce_legal_compatibility and self.has_legal_form_conflict(left_root, right_root):
            return False
        if self._size[left_root] < self._size[right_root]:
            left_root, right_root = right_root, left_root
        self._parent[right_root] = left_root
        self._size[left_root] += self._size[right_root]
        self._legal_forms[left_root] = frozenset(
            self._legal_forms[left_root] | self._legal_forms[right_root]
        )
        del self._legal_forms[right_root]
        return True


def _product_path(root: Path, override: Path | None, configured: Path) -> Path:
    path = override or configured
    return path if path.is_absolute() else root / path


def _required_string(row: dict[str, object], column: str, *, source: str) -> str:
    value = row.get(column)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{source} contains a blank or invalid {column}")
    return value


def _required_int(row: dict[str, object], column: str, *, source: str) -> int:
    value = row.get(column)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{source} contains an invalid {column}")
    return value


def _required_bool(row: dict[str, object], column: str, *, source: str) -> bool:
    value = row.get(column)
    if not isinstance(value, bool):
        raise ValueError(f"{source} contains an invalid {column}")
    return value


def _require_parquet_columns(path: Path, required: tuple[str, ...]) -> pq.ParquetFile:
    if not path.is_file():
        raise FileNotFoundError(f"Required finalization input does not exist: {path}")
    parquet = pq.ParquetFile(path)
    observed = set(parquet.schema_arrow.names)
    missing = sorted(set(required) - observed)
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")
    return parquet


def _load_key_metadata(path: Path) -> dict[str, KeyMetadata]:
    columns = (
        "resolution_key",
        "representative_name",
        "source_row_frequency",
        "possible_truncation",
        "token_count",
        "representative_route",
    )
    parquet = _require_parquet_columns(path, columns)
    result: dict[str, KeyMetadata] = {}
    for batch in parquet.iter_batches(columns=list(columns)):
        for raw_row in batch.to_pylist():
            row = cast(dict[str, object], raw_row)
            key = _required_string(row, "resolution_key", source=path.name)
            frequency = _required_int(row, "source_row_frequency", source=path.name)
            if frequency <= 0:
                raise ValueError(f"{path.name} contains non-positive source_row_frequency")
            metadata = KeyMetadata(
                resolution_key=key,
                representative_name=_required_string(row, "representative_name", source=path.name),
                source_row_frequency=frequency,
                possible_truncation=_required_bool(row, "possible_truncation", source=path.name),
                token_count=_required_int(row, "token_count", source=path.name),
                representative_route=_required_string(
                    row, "representative_route", source=path.name
                ),
            )
            if key in result:
                raise ValueError(f"Duplicate resolution_key in {path.name}: {key!r}")
            result[key] = metadata
    return result


def _load_eligibility(path: Path) -> dict[str, EligibilityMetadata]:
    columns = (
        "resolution_key",
        "eligibility_status",
        "eligibility_rule",
        "eligibility_evidence",
    )
    parquet = _require_parquet_columns(path, columns)
    result: dict[str, EligibilityMetadata] = {}
    for batch in parquet.iter_batches(columns=list(columns)):
        for raw_row in batch.to_pylist():
            row = cast(dict[str, object], raw_row)
            key = _required_string(row, "resolution_key", source=path.name)
            if key in result:
                raise ValueError(f"Duplicate resolution_key in {path.name}: {key!r}")
            result[key] = EligibilityMetadata(
                status=_required_string(row, "eligibility_status", source=path.name),
                rule=_required_string(row, "eligibility_rule", source=path.name),
                evidence=_required_string(row, "eligibility_evidence", source=path.name),
            )
    return result


def load_public_enrichments(path: Path) -> tuple[PublicEnrichment, ...]:
    """Validate the small, versioned catalogue of publicly confirmed employers."""
    if not path.is_file():
        raise FileNotFoundError(f"Public enrichment CSV does not exist: {path}")
    enrichments: list[PublicEnrichment] = []
    seen_keys: set[str] = set()
    definitions: dict[str, tuple[str, str, str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        observed = tuple(reader.fieldnames or ())
        if observed != PUBLIC_ENRICHMENT_COLUMNS:
            raise ValueError(
                f"Invalid columns in {path.name}: expected {PUBLIC_ENRICHMENT_COLUMNS}, "
                f"observed {observed}"
            )
        for row_number, raw_row in enumerate(reader, start=2):
            values = {column: (raw_row.get(column) or "").strip() for column in observed}
            blank_columns = [column for column, value in values.items() if not value]
            if blank_columns:
                raise ValueError(
                    f"Blank public enrichment fields at {path.name} row {row_number}: "
                    f"{blank_columns}"
                )
            public_entity_id = values["public_entity_id"]
            if PUBLIC_ENTITY_ID_PATTERN.fullmatch(public_entity_id) is None:
                raise ValueError(
                    f"Invalid public_entity_id at {path.name} row {row_number}: "
                    f"{public_entity_id!r}"
                )
            resolution_key = values["resolution_key"]
            if resolution_key in seen_keys:
                raise ValueError(f"Duplicate public enrichment resolution_key: {resolution_key!r}")
            seen_keys.add(resolution_key)
            parsed_url = urlparse(values["source_url"])
            if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
                raise ValueError(f"Invalid public source URL at {path.name} row {row_number}")
            try:
                date.fromisoformat(values["validated_on"])
            except ValueError as error:
                raise ValueError(
                    f"Invalid ISO validation date at {path.name} row {row_number}"
                ) from error
            definition = (
                values["canonical_name"],
                values["sector"],
                values["source_url"],
                values["validated_on"],
            )
            existing = definitions.get(public_entity_id)
            if existing is not None and existing != definition:
                raise ValueError(
                    f"Conflicting public attributes for {public_entity_id} in {path.name}"
                )
            definitions[public_entity_id] = definition
            enrichments.append(
                PublicEnrichment(
                    public_entity_id=public_entity_id,
                    resolution_key=resolution_key,
                    canonical_name=values["canonical_name"],
                    sector=values["sector"],
                    source_url=values["source_url"],
                    validated_on=values["validated_on"],
                )
            )
    return tuple(enrichments)


def _legal_forms(value: str) -> frozenset[str]:
    """Return normalized terminal legal-form evidence from one resolution key."""
    tokens = tuple(re.findall(r"[A-Z0-9]+", value.upper()))
    for suffix in LEGAL_FORM_SUFFIXES:
        if len(tokens) >= len(suffix) and tokens[-len(suffix) :] == suffix:
            return frozenset({LEGAL_FORM_CANONICAL[suffix]})
    return frozenset()


def _is_explicit_missing_information(normalized_name: str | None) -> bool:
    if normalized_name is None:
        return False
    return " ".join(normalized_name.upper().split()) in EXPLICIT_MISSING_INFORMATION_VALUES


def _canonical_rank(metadata: KeyMetadata) -> tuple[int, int, int, int, int, int, str]:
    key = metadata.resolution_key
    tokens = key.split()
    has_terminal_legal_suffix = int(bool(_legal_forms(key)))
    single_letter_tokens = sum(len(token) == 1 for token in tokens)
    return (
        int(not metadata.possible_truncation),
        metadata.source_row_frequency,
        has_terminal_legal_suffix,
        -single_letter_tokens,
        metadata.token_count,
        len(metadata.representative_name),
        metadata.resolution_key,
    )


def _stable_component_id(keys: list[str]) -> str:
    payload = "\0".join(sorted(keys)).encode("utf-8")
    return f"ENT-{hashlib.sha256(payload).hexdigest()[:16].upper()}"


def _lineage_payload(record_id: str, source_row_number: int, nombre_original: str | None) -> bytes:
    payload = [record_id, source_row_number, nombre_original]
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"


def _source_contract_payload(
    record_id: str,
    source_row_number: int,
    nombre_original: str | None,
    nombre_normalizado: str | None,
    route: str,
    route_reason: str,
) -> bytes:
    payload = [
        record_id,
        source_row_number,
        nombre_original,
        nombre_normalizado,
        route,
        route_reason,
    ]
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"


def _write_json(path: Path, payload: dict[str, JsonValue]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.unlink(missing_ok=True)
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _json_int_dict(values: dict[str, int]) -> dict[str, JsonValue]:
    return {key: value for key, value in values.items()}


def _public_definitions(
    enrichments: tuple[PublicEnrichment, ...],
) -> dict[str, PublicEntityDefinition]:
    definitions: dict[str, PublicEntityDefinition] = {}
    for item in enrichments:
        definitions[item.public_entity_id] = PublicEntityDefinition(
            public_entity_id=item.public_entity_id,
            canonical_name=item.canonical_name,
            sector=item.sector,
            source_url=item.source_url,
            validated_on=item.validated_on,
        )
    return definitions


def infer_sector(
    canonical_name: str,
    sector_keyword_rules: dict[str, tuple[str, ...]],
) -> tuple[str, str, str | None]:
    """Infer a broad sector from exact configured tokens, in configured precedence."""
    tokens = set(re.findall(r"[A-Z0-9]+", canonical_name.upper()))
    for sector, configured_keywords in sector_keyword_rules.items():
        normalized_sector = sector.strip()
        if not normalized_sector:
            raise ValueError("Finalization sector taxonomy contains a blank sector")
        keywords = {keyword.strip().upper() for keyword in configured_keywords}
        if not keywords or "" in keywords:
            raise ValueError(f"Finalization sector taxonomy contains blank keywords for {sector!r}")
        matches = sorted(tokens & keywords)
        if matches:
            return normalized_sector, "keyword_taxonomy", "|".join(matches)
    return "No determinado", "not_determined", None


def _load_exact_reference_reuse(
    path: Path,
    *,
    enrichments: tuple[PublicEnrichment, ...],
    sector_keyword_rules: dict[str, tuple[str, ...]],
    batch_size: int,
) -> ExactReferenceReuse:
    columns = (
        "record_id",
        "source_row_number",
        "nombre_original",
        "nombre_normalizado",
        "route",
        "route_reason",
        "entity_id",
        "canonical_name",
        "resolution_status",
        "resolution_method",
        "resolution_reason",
    )
    parquet = _require_parquet_columns(path, columns)
    by_record_id: dict[str, ExactReferenceRow] = {}
    observed_record_ids: set[str] = set()
    canonical_name_by_entity: dict[str, str] = {}
    keys_by_entity: dict[str, set[str]] = defaultdict(set)
    source_rows_by_entity: Counter[str] = Counter()
    reasons_by_entity: dict[str, set[str]] = defaultdict(set)
    source_contract_digest = hashlib.sha256()
    unresolved_rows = 0
    canonical_rows = 0
    alias_rows = 0

    for batch in parquet.iter_batches(columns=list(columns), batch_size=batch_size):
        for raw_row in batch.to_pylist():
            row = cast(dict[str, object], raw_row)
            record_id = _required_string(row, "record_id", source=path.name)
            if record_id in observed_record_ids:
                raise ValueError(
                    f"Duplicate record_id in {path.name}: {record_id!r}"
                )
            observed_record_ids.add(record_id)
            source_row_number = _required_int(
                row,
                "source_row_number",
                source=path.name,
            )
            original_value = row.get("nombre_original")
            if original_value is not None and not isinstance(original_value, str):
                raise ValueError(f"{path.name} contains an invalid nombre_original")
            normalized_value = row.get("nombre_normalizado")
            if normalized_value is not None and not isinstance(normalized_value, str):
                raise ValueError(f"{path.name} contains an invalid nombre_normalizado")
            route = _required_string(row, "route", source=path.name)
            route_reason = _required_string(row, "route_reason", source=path.name)
            resolution_status = _required_string(
                row,
                "resolution_status",
                source=path.name,
            )
            resolution_reason = _required_string(
                row,
                "resolution_reason",
                source=path.name,
            )
            entity_id_value = row.get("entity_id")
            canonical_name_value = row.get("canonical_name")
            resolution_method_value = row.get("resolution_method")

            source_contract_digest.update(
                _source_contract_payload(
                    record_id,
                    source_row_number,
                    original_value,
                    normalized_value,
                    route,
                    route_reason,
                )
            )

            if resolution_status == "unresolved":
                unresolved_rows += 1
                if entity_id_value is not None:
                    raise ValueError(
                        "unresolved reference row must not contain entity_id"
                    )
                if canonical_name_value is not None:
                    raise ValueError(
                        "unresolved reference row must not contain canonical_name"
                    )
                if resolution_method_value is not None:
                    raise ValueError(
                        "unresolved reference row must not contain resolution_method"
                    )
                continue

            if resolution_status != "resolved":
                raise ValueError(
                    f"Unsupported resolution_status in {path.name}: "
                    f"{resolution_status!r}"
                )
            if not isinstance(entity_id_value, str) or not entity_id_value:
                raise ValueError(
                    "resolved exact reference row must contain a nonblank entity_id"
                )
            if not isinstance(canonical_name_value, str) or not canonical_name_value:
                raise ValueError(
                    "resolved exact reference row must contain a nonblank canonical_name"
                )
            if not isinstance(normalized_value, str) or not normalized_value:
                raise ValueError(
                    "resolved exact reference row must contain a nonblank "
                    "nombre_normalizado"
                )
            if resolution_method_value not in {"exact_canonical", "exact_alias"}:
                raise ValueError(
                    "resolved exact reference row has unsupported resolution_method: "
                    f"{resolution_method_value!r}"
                )

            resolution_method = resolution_method_value
            exact_row = ExactReferenceRow(
                record_id=record_id,
                source_row_number=source_row_number,
                nombre_original=original_value,
                nombre_normalizado=normalized_value,
                route=route,
                route_reason=route_reason,
                entity_id=entity_id_value,
                canonical_name=canonical_name_value,
                resolution_method=resolution_method,
                resolution_reason=resolution_reason,
            )
            by_record_id[record_id] = exact_row
            canonical_rows += int(resolution_method == "exact_canonical")
            alias_rows += int(resolution_method == "exact_alias")

            existing_name = canonical_name_by_entity.get(entity_id_value)
            if existing_name is not None and existing_name != canonical_name_value:
                raise ValueError(
                    f"Exact-reference entity {entity_id_value!r} has conflicting "
                    "canonical names"
                )
            canonical_name_by_entity[entity_id_value] = canonical_name_value
            keys_by_entity[entity_id_value].add(normalized_value)
            source_rows_by_entity[entity_id_value] += 1
            reasons_by_entity[entity_id_value].add(resolution_reason)

    if len(observed_record_ids) != parquet.metadata.num_rows:
        raise RuntimeError("Resolved-reference row count failed reconciliation")
    if canonical_rows + alias_rows + unresolved_rows != parquet.metadata.num_rows:
        raise RuntimeError("Resolved-reference status counts failed reconciliation")

    public_definitions = _public_definitions(enrichments)
    component_by_entity_id: dict[str, ComponentSummary] = {}
    for entity_id, canonical_name in canonical_name_by_entity.items():
        public_definition = public_definitions.get(entity_id)
        if (
            public_definition is not None
            and public_definition.canonical_name != canonical_name
        ):
            raise ValueError(
                f"Exact-reference entity {entity_id!r} conflicts with public "
                "canonical_name"
            )

        if public_definition is not None:
            sector = public_definition.sector
            sector_method = "public_source"
            sector_evidence: str | None = public_definition.source_url
            sector_confidence = "ALTA"
        else:
            sector, sector_method, sector_evidence = infer_sector(
                canonical_name,
                sector_keyword_rules,
            )
            sector_confidence = (
                "MEDIA" if sector_method == "keyword_taxonomy" else "NO_DETERMINADA"
            )

        component_by_entity_id[entity_id] = ComponentSummary(
            entity_id=entity_id,
            canonical_resolution_key=canonical_name,
            canonical_name=canonical_name,
            sector=sector,
            sector_method=sector_method,
            sector_evidence=sector_evidence,
            sector_confidence=sector_confidence,
            key_count=len(keys_by_entity[entity_id]),
            source_row_count=source_rows_by_entity[entity_id],
            decision_rules=tuple(sorted(reasons_by_entity[entity_id])),
            public_entity_id=(
                public_definition.public_entity_id
                if public_definition is not None
                else None
            ),
            public_source_url=(
                public_definition.source_url
                if public_definition is not None
                else None
            ),
            public_validation_date=(
                public_definition.validated_on
                if public_definition is not None
                else None
            ),
        )

    return ExactReferenceReuse(
        by_record_id=by_record_id,
        component_by_entity_id=component_by_entity_id,
        total_rows=parquet.metadata.num_rows,
        unresolved_rows=unresolved_rows,
        canonical_rows=canonical_rows,
        alias_rows=alias_rows,
        source_contract_digest=source_contract_digest.hexdigest(),
    )


def _build_components(
    *,
    key_metadata: dict[str, KeyMetadata],
    eligibility: dict[str, EligibilityMetadata],
    decisions_path: Path,
    enrichments: tuple[PublicEnrichment, ...],
    exact_public_entity_ids: set[str],
    sector_keyword_rules: dict[str, tuple[str, ...]],
    batch_size: int,
) -> tuple[
    dict[str, ComponentSummary],
    dict[str, int],
    dict[str, int],
    dict[str, int],
]:
    employer_keys = {
        key
        for key, item in eligibility.items()
        if item.status == EMPLOYER_CANDIDATE
        and not _is_explicit_missing_information(key)
    }
    union_find = UnionFind(employer_keys)
    accepted_edges: list[tuple[str, str, str]] = []
    decision_rule_counts: Counter[str] = Counter()
    endpoint_combination_counts: Counter[str] = Counter()
    auto_same_rows = 0
    employer_compatible_auto_same_rows = 0
    accepted_auto_same_rows = 0
    blocked_legal_form_conflict_auto_same = 0
    excluded_explicit_missing_auto_same = 0

    decision_columns = ("key_a", "key_b", "decision_status", "decision_rule")
    decisions = _require_parquet_columns(decisions_path, decision_columns)
    for batch in decisions.iter_batches(columns=list(decision_columns), batch_size=batch_size):
        for raw_row in batch.to_pylist():
            row = cast(dict[str, object], raw_row)
            if _required_string(row, "decision_status", source=decisions_path.name) != AUTO_SAME:
                continue
            auto_same_rows += 1
            key_a = _required_string(row, "key_a", source=decisions_path.name)
            key_b = _required_string(row, "key_b", source=decisions_path.name)
            if key_a not in eligibility or key_b not in eligibility:
                raise ValueError("AUTO_SAME references a key missing from employer eligibility")
            status_a = eligibility[key_a].status
            status_b = eligibility[key_b].status
            endpoint_combination_counts["|".join(sorted((status_a, status_b)))] += 1
            if status_a != EMPLOYER_CANDIDATE or status_b != EMPLOYER_CANDIDATE:
                continue
            if key_a not in employer_keys or key_b not in employer_keys:
                excluded_explicit_missing_auto_same += 1
                continue
            employer_compatible_auto_same_rows += 1
            rule = _required_string(row, "decision_rule", source=decisions_path.name)
            if union_find.has_legal_form_conflict(key_a, key_b):
                blocked_legal_form_conflict_auto_same += 1
                continue
            union_find.union(key_a, key_b)
            accepted_edges.append((key_a, key_b, rule))
            decision_rule_counts[rule] += 1
            accepted_auto_same_rows += 1

    candidate_enrichments: list[PublicEnrichment] = []
    keys_by_public_entity: dict[str, list[str]] = defaultdict(list)
    for item in enrichments:
        key = item.resolution_key
        if key not in key_metadata:
            if item.public_entity_id in exact_public_entity_ids:
                continue
            raise ValueError(
                f"Public enrichment references an unknown resolution key: {key!r}"
            )
        item_eligibility = eligibility.get(key)
        if item_eligibility is None or item_eligibility.status != EMPLOYER_CANDIDATE:
            raise ValueError(f"Public enrichment key is not EMPLOYER_CANDIDATE: {key!r}")
        candidate_enrichments.append(item)
        keys_by_public_entity[item.public_entity_id].append(key)

    public_reference_links = 0
    public_reference_new_unions = 0
    for keys in keys_by_public_entity.values():
        anchor = keys[0]
        for key in keys[1:]:
            public_reference_links += 1
            public_reference_new_unions += int(
                union_find.union(anchor, key, enforce_legal_compatibility=False)
            )

    component_members: dict[str, list[str]] = defaultdict(list)
    for key in employer_keys:
        component_members[union_find.find(key)].append(key)

    rules_by_root: dict[str, set[str]] = defaultdict(set)
    for key_a, _key_b, rule in accepted_edges:
        rules_by_root[union_find.find(key_a)].add(rule)

    public_ids_by_root: dict[str, set[str]] = defaultdict(set)
    for item in candidate_enrichments:
        public_ids_by_root[union_find.find(item.resolution_key)].add(item.public_entity_id)
    for root, public_ids in public_ids_by_root.items():
        if len(public_ids) > 1:
            raise ValueError(
                "A deterministic component contains conflicting public entity IDs: "
                f"root={root!r}, ids={sorted(public_ids)}"
            )

    definitions = _public_definitions(enrichments)
    component_by_key: dict[str, ComponentSummary] = {}
    for root, members in component_members.items():
        members.sort()
        canonical_key = max(members, key=lambda key: _canonical_rank(key_metadata[key]))
        public_ids = public_ids_by_root.get(root, set())
        public_id = next(iter(public_ids)) if public_ids else None
        public_definition = definitions.get(public_id) if public_id is not None else None
        canonical_name = (
            public_definition.canonical_name
            if public_definition is not None
            else key_metadata[canonical_key].representative_name
        )
        sector_evidence: str | None
        if public_definition is not None:
            sector = public_definition.sector
            sector_method = "public_source"
            sector_evidence = public_definition.source_url
            sector_confidence = "ALTA"
        else:
            sector, sector_method, sector_evidence = infer_sector(
                canonical_name, sector_keyword_rules
            )
            sector_confidence = "MEDIA" if sector_method == "keyword_taxonomy" else "NO_DETERMINADA"
        summary = ComponentSummary(
            entity_id=public_id or _stable_component_id(members),
            canonical_resolution_key=canonical_key,
            canonical_name=canonical_name,
            sector=sector,
            sector_method=sector_method,
            sector_evidence=sector_evidence,
            sector_confidence=sector_confidence,
            key_count=len(members),
            source_row_count=sum(key_metadata[key].source_row_frequency for key in members),
            decision_rules=tuple(sorted(rules_by_root.get(root, set()))),
            public_entity_id=public_id,
            public_source_url=(
                public_definition.source_url if public_definition is not None else None
            ),
            public_validation_date=(
                public_definition.validated_on if public_definition is not None else None
            ),
        )
        for key in members:
            component_by_key[key] = summary

    component_identity_counts = {
        "employer_keys": len(employer_keys),
        "components": len(component_members),
        "multi_key_components": sum(len(members) > 1 for members in component_members.values()),
        "keys_in_multi_key_components": sum(
            len(members) for members in component_members.values() if len(members) > 1
        ),
        "maximum_component_keys": max(
            (len(members) for members in component_members.values()), default=0
        ),
        "maximum_component_source_rows": max(
            (summary.source_row_count for summary in component_by_key.values()), default=0
        ),
        "public_entities": len(definitions),
        "public_resolution_keys": len(candidate_enrichments),
        "public_reference_links": public_reference_links,
        "public_reference_new_unions": public_reference_new_unions,
    }
    decision_counts = {
        "auto_same_rows": auto_same_rows,
        "employer_compatible_auto_same": employer_compatible_auto_same_rows,
        "accepted_employer_compatible_auto_same": accepted_auto_same_rows,
        "blocked_legal_form_conflict_auto_same": blocked_legal_form_conflict_auto_same,
        "excluded_explicit_missing_auto_same": excluded_explicit_missing_auto_same,
        "excluded_non_employer_compatible_auto_same": (
            auto_same_rows
            - employer_compatible_auto_same_rows
            - excluded_explicit_missing_auto_same
        ),
        **{f"accepted_rule::{rule}": count for rule, count in sorted(decision_rule_counts.items())},
    }
    return (
        component_by_key,
        component_identity_counts,
        decision_counts,
        dict(sorted(endpoint_combination_counts.items())),
    )


def _write_top_keys(
    *,
    path: Path,
    key_metadata: dict[str, KeyMetadata],
    eligibility: dict[str, EligibilityMetadata],
    component_by_key: dict[str, ComponentSummary],
    limit: int,
) -> None:
    fieldnames = (
        "resolution_key",
        "representative_name",
        "source_row_frequency",
        "representative_route",
        "entity_id",
        "canonical_name",
        "sector",
        "sector_method",
        "sector_evidence",
        "component_key_count",
        "component_source_row_count",
        "public_source_url",
    )
    employer_metadata = [
        metadata
        for key, metadata in key_metadata.items()
        if eligibility[key].status == EMPLOYER_CANDIDATE and key in component_by_key
    ]
    employer_metadata.sort(key=lambda item: (-item.source_row_frequency, item.resolution_key))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.unlink(missing_ok=True)
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for metadata in employer_metadata[:limit]:
            component = component_by_key[metadata.resolution_key]
            writer.writerow(
                {
                    "resolution_key": metadata.resolution_key,
                    "representative_name": metadata.representative_name,
                    "source_row_frequency": metadata.source_row_frequency,
                    "representative_route": metadata.representative_route,
                    "entity_id": component.entity_id,
                    "canonical_name": component.canonical_name,
                    "sector": component.sector,
                    "sector_method": component.sector_method,
                    "sector_evidence": component.sector_evidence,
                    "component_key_count": component.key_count,
                    "component_source_row_count": component.source_row_count,
                    "public_source_url": component.public_source_url,
                }
            )
    temporary.replace(path)


def _exact_reference_business_fields(
    exact_row: ExactReferenceRow,
    component: ComponentSummary,
) -> dict[str, object]:
    public_validated = component.public_entity_id is not None
    decision_rules = "|".join(component.decision_rules) or None
    method_suffix = (
        "canonical" if exact_row.resolution_method == "exact_canonical" else "alias"
    )
    if public_validated:
        explanation = (
            "Identidad reutilizada por referencia exacta persistente y enriquecida "
            "con la fuente pública validada."
        )
    else:
        explanation = (
            "Identidad reutilizada por referencia exacta persistente. "
            + (
                f"Sector inferido por palabras clave exactas: {component.sector_evidence}."
                if component.sector_method == "keyword_taxonomy"
                else "Sector no asignado sin evidencia suficiente."
            )
        )
    return {
        "resolution_key": exact_row.nombre_normalizado,
        "entity_id": component.entity_id,
        "nombre_propuesto": component.canonical_name,
        "sector_propuesto": component.sector,
        "metodo_sector": component.sector_method,
        "sector_evidence": component.sector_evidence,
        "resultado_final": (
            "EMPRESA_VALIDADA_PUBLICAMENTE"
            if public_validated
            else "EMPRESA_CANONICALIZADA_AUTO"
        ),
        "confianza_resolucion": "ALTA",
        "score_confianza_resolucion": (
            SCORE_PUBLIC_VALIDATED
            if public_validated
            else SCORE_DETERMINISTIC_AUTO_SAME
        ),
        "confianza_sector": component.sector_confidence,
        "metodo_resolucion": f"exact_reference_{method_suffix}",
        "explicacion": explanation,
        "canonical_resolution_key": component.canonical_resolution_key,
        "component_key_count": component.key_count,
        "component_source_row_count": component.source_row_count,
        "decision_rules": decision_rules,
        "public_entity_id": component.public_entity_id,
        "public_source_url": component.public_source_url,
        "public_validation_date": component.public_validation_date,
    }


def _business_fields(
    *,
    normalized_name: str | None,
    route: str,
    eligibility: EligibilityMetadata | None,
    component_by_key: dict[str, ComponentSummary],
) -> dict[str, object]:
    resolution_key = normalized_name if normalized_name in component_by_key else None
    status = eligibility.status if eligibility is not None else None
    if _is_explicit_missing_information(normalized_name):
        return {
            "resolution_key": normalized_name if eligibility is not None else None,
            "entity_id": None,
            "nombre_propuesto": "No identificable - Falta informacion",
            "sector_propuesto": "No aplica",
            "metodo_sector": "not_applicable",
            "sector_evidence": None,
            "resultado_final": "NO_IDENTIFICABLE_FALTA_INFORMACION",
            "confianza_resolucion": "ALTA",
            "score_confianza_resolucion": SCORE_EXPLICIT_MISSING_INFORMATION,
            "confianza_sector": "NO_APLICA",
            "metodo_resolucion": "explicit_missing_information",
            "explicacion": (
                "El valor normalizado corresponde explícitamente a ausencia de información "
                "y no se interpreta como empleador."
            ),
            "canonical_resolution_key": None,
            "component_key_count": 0,
            "component_source_row_count": 0,
            "decision_rules": None,
            "public_entity_id": None,
            "public_source_url": None,
            "public_validation_date": None,
        }
    if route == "address_candidate" or status == ADDRESS:
        confidence = "ALTA" if status == ADDRESS else "MEDIA"
        return {
            "resolution_key": normalized_name if eligibility is not None else None,
            "entity_id": None,
            "nombre_propuesto": "No identificable - Direcciones",
            "sector_propuesto": "No aplica",
            "metodo_sector": "not_applicable",
            "sector_evidence": None,
            "resultado_final": "NO_IDENTIFICABLE_DIRECCION",
            "confianza_resolucion": confidence,
            "score_confianza_resolucion": SCORE_ADDRESS_CLASSIFICATION,
            "confianza_sector": "NO_APLICA",
            "metodo_resolucion": "record_typing_address",
            "explicacion": (
                "El registro contiene evidencia de dirección y no se atribuye a una empresa."
            ),
            "canonical_resolution_key": None,
            "component_key_count": 0,
            "component_source_row_count": 0,
            "decision_rules": None,
            "public_entity_id": None,
            "public_source_url": None,
            "public_validation_date": None,
        }
    if status == EMPLOYER_CANDIDATE and resolution_key is not None:
        component = component_by_key[resolution_key]
        decision_rules = "|".join(component.decision_rules) or None
        if component.public_entity_id is not None:
            return {
                "resolution_key": resolution_key,
                "entity_id": component.entity_id,
                "nombre_propuesto": component.canonical_name,
                "sector_propuesto": component.sector,
                "metodo_sector": component.sector_method,
                "sector_evidence": component.sector_evidence,
                "resultado_final": "EMPRESA_VALIDADA_PUBLICAMENTE",
                "confianza_resolucion": "ALTA",
                "score_confianza_resolucion": SCORE_PUBLIC_VALIDATED,
                "confianza_sector": "ALTA",
                "metodo_resolucion": "public_reference_and_deterministic_component",
                "explicacion": (
                    "Identidad y sector validados en fuente pública; propagación limitada a "
                    "claves aprobadas y equivalencias AUTO_SAME."
                ),
                "canonical_resolution_key": component.canonical_resolution_key,
                "component_key_count": component.key_count,
                "component_source_row_count": component.source_row_count,
                "decision_rules": decision_rules,
                "public_entity_id": component.public_entity_id,
                "public_source_url": component.public_source_url,
                "public_validation_date": component.public_validation_date,
            }
        if component.key_count > 1 and component.decision_rules:
            return {
                "resolution_key": resolution_key,
                "entity_id": component.entity_id,
                "nombre_propuesto": component.canonical_name,
                "sector_propuesto": component.sector,
                "metodo_sector": component.sector_method,
                "sector_evidence": component.sector_evidence,
                "resultado_final": "EMPRESA_CANONICALIZADA_AUTO",
                "confianza_resolucion": "ALTA",
                "score_confianza_resolucion": SCORE_DETERMINISTIC_AUTO_SAME,
                "confianza_sector": component.sector_confidence,
                "metodo_resolucion": "deterministic_auto_same_component",
                "explicacion": (
                    "Equivalencia textual determinista AUTO_SAME. "
                    + (
                        f"Sector inferido por palabras clave exactas: {component.sector_evidence}."
                        if component.sector_method == "keyword_taxonomy"
                        else "Sector no asignado sin evidencia suficiente."
                    )
                ),
                "canonical_resolution_key": component.canonical_resolution_key,
                "component_key_count": component.key_count,
                "component_source_row_count": component.source_row_count,
                "decision_rules": decision_rules,
                "public_entity_id": None,
                "public_source_url": None,
                "public_validation_date": None,
            }
        return {
            "resolution_key": resolution_key,
            "entity_id": component.entity_id,
            "nombre_propuesto": component.canonical_name,
            "sector_propuesto": component.sector,
            "metodo_sector": component.sector_method,
            "sector_evidence": component.sector_evidence,
            "resultado_final": "EMPRESA_NORMALIZADA_SIN_VALIDACION_PUBLICA",
            "confianza_resolucion": "MEDIA",
            "score_confianza_resolucion": SCORE_NORMALIZED_EMPLOYER,
            "confianza_sector": component.sector_confidence,
            "metodo_resolucion": "normalized_employer_key",
            "explicacion": (
                "Se conserva la clave normalizada sin enlace determinista. "
                + (
                    f"Sector inferido por palabras clave exactas: {component.sector_evidence}."
                    if component.sector_method == "keyword_taxonomy"
                    else "Sector no asignado sin evidencia suficiente."
                )
            ),
            "canonical_resolution_key": component.canonical_resolution_key,
            "component_key_count": component.key_count,
            "component_source_row_count": component.source_row_count,
            "decision_rules": decision_rules,
            "public_entity_id": None,
            "public_source_url": None,
            "public_validation_date": None,
        }
    return {
        "resolution_key": normalized_name if eligibility is not None else None,
        "entity_id": None,
        "nombre_propuesto": "No identificable - Falta informacion",
        "sector_propuesto": "No aplica",
        "metodo_sector": "not_applicable",
        "sector_evidence": None,
        "resultado_final": "NO_IDENTIFICABLE_FALTA_INFORMACION",
        "confianza_resolucion": "BAJA",
        "score_confianza_resolucion": SCORE_PRECISION_FIRST_ABSTENTION,
        "confianza_sector": "NO_APLICA",
        "metodo_resolucion": "precision_first_abstention",
        "explicacion": (
            "La evidencia no permite identificar una empresa con precisión suficiente; "
            "el proceso se abstiene."
        ),
        "canonical_resolution_key": None,
        "component_key_count": 0,
        "component_source_row_count": 0,
        "decision_rules": None,
        "public_entity_id": None,
        "public_source_url": None,
        "public_validation_date": None,
    }


def _reconcile_final_output(
    path: Path, *, expected_rows: int, expected_lineage_digest: str
) -> None:
    parquet = pq.ParquetFile(path)
    if parquet.metadata.num_rows != expected_rows or parquet.schema_arrow != FINAL_SCHEMA:
        raise RuntimeError("Final Parquet metadata failed reconciliation")
    output_digest = hashlib.sha256()
    columns = [
        "record_id",
        "source_row_number",
        "nombre_original",
        "nombre_propuesto",
        "sector_propuesto",
        "resultado_final",
        "score_confianza_resolucion",
        "entity_id",
        "canonical_resolution_key",
    ]
    for batch in parquet.iter_batches(columns=columns):
        for raw_row in batch.to_pylist():
            row = cast(dict[str, object], raw_row)
            record_id = _required_string(row, "record_id", source=path.name)
            source_row_number = _required_int(row, "source_row_number", source=path.name)
            original_value = row.get("nombre_original")
            if original_value is not None and not isinstance(original_value, str):
                raise ValueError(f"{path.name} contains an invalid nombre_original")
            _required_string(row, "nombre_propuesto", source=path.name)
            _required_string(row, "sector_propuesto", source=path.name)
            outcome = _required_string(row, "resultado_final", source=path.name)
            score = _required_int(row, "score_confianza_resolucion", source=path.name)
            if not 0 <= score <= 100:
                raise RuntimeError("Final output contains an out-of-range confidence score")
            if outcome.startswith("NO_IDENTIFICABLE_") and (
                row.get("entity_id") is not None
                or row.get("canonical_resolution_key") is not None
            ):
                raise RuntimeError(
                    "Non-identifiable output must not contain entity or canonical identifiers"
                )
            output_digest.update(_lineage_payload(record_id, source_row_number, original_value))
    if output_digest.hexdigest() != expected_lineage_digest:
        raise RuntimeError("Final output failed source-lineage reconciliation")


def finalize_employers(
    root: Path,
    settings: Settings,
    *,
    preprocessed_override: Path | None = None,
    resolved_override: Path | None = None,
    keys_override: Path | None = None,
    decisions_override: Path | None = None,
    eligibility_override: Path | None = None,
    enrichment_override: Path | None = None,
    parquet_output_override: Path | None = None,
    csv_output_override: Path | None = None,
) -> FinalizationResult:
    """Build a full row-preserving final dataset without reopening upstream stages."""
    started = time.perf_counter()
    config = settings.finalization
    preprocessed_path = _product_path(root, preprocessed_override, config.preprocessed_dataset)
    resolved_path = _product_path(
        root,
        resolved_override,
        settings.resolution.output_dataset,
    )
    keys_path = _product_path(root, keys_override, config.resolution_keys_dataset)
    decisions_path = _product_path(root, decisions_override, config.pair_decisions_dataset)
    eligibility_path = _product_path(
        root, eligibility_override, config.employer_eligibility_dataset
    )
    enrichment_path = _product_path(root, enrichment_override, config.public_enrichment_csv)
    parquet_output_path = _product_path(root, parquet_output_override, config.parquet_output)
    csv_output_path = _product_path(root, csv_output_override, config.csv_output)
    metrics_path = _product_path(root, None, config.metrics_output)
    top_keys_path = _product_path(root, None, config.top_keys_output)

    input_paths = (
        preprocessed_path,
        resolved_path,
        keys_path,
        decisions_path,
        eligibility_path,
    )
    resolved_outputs = {parquet_output_path.resolve(), csv_output_path.resolve()}
    if any(path.resolve() in resolved_outputs for path in input_paths):
        raise ValueError("Final outputs must not overwrite finalization inputs")
    if parquet_output_path.resolve() == csv_output_path.resolve():
        raise ValueError("Parquet and CSV final outputs must be different files")

    input_columns = (
        "record_id",
        "source_row_number",
        "nombre_original",
        "nombre_normalizado",
        "route",
        "route_reason",
    )
    preprocessed = _require_parquet_columns(preprocessed_path, input_columns)
    enrichments = load_public_enrichments(enrichment_path)

    exact_reuse: ExactReferenceReuse | None = None
    if resolved_path.is_file():
        exact_reuse = _load_exact_reference_reuse(
            resolved_path,
            enrichments=enrichments,
            sector_keyword_rules=config.sector_keyword_rules,
            batch_size=config.batch_size,
        )
        if exact_reuse.total_rows != preprocessed.metadata.num_rows:
            raise ValueError(
                "Resolved and preprocessed row counts differ: "
                f"resolved={exact_reuse.total_rows}, "
                f"preprocessed={preprocessed.metadata.num_rows}"
            )
    elif resolved_override is not None:
        raise FileNotFoundError(
            f"Required finalization input does not exist: {resolved_path}"
        )

    no_unresolved_population = (
        exact_reuse is not None and exact_reuse.unresolved_rows == 0
    )
    keys_file = pq.ParquetFile(keys_path) if keys_path.is_file() else None
    eligibility_file = (
        pq.ParquetFile(eligibility_path) if eligibility_path.is_file() else None
    )
    if keys_file is None:
        raise FileNotFoundError(
            f"Required finalization input does not exist: {keys_path}"
        )
    if eligibility_file is None:
        raise FileNotFoundError(
            f"Required finalization input does not exist: {eligibility_path}"
        )

    if (
        no_unresolved_population
        and keys_file.metadata.num_rows == 0
        and eligibility_file.metadata.num_rows == 0
    ):
        key_metadata: dict[str, KeyMetadata] = {}
        eligibility: dict[str, EligibilityMetadata] = {}
    else:
        key_metadata = _load_key_metadata(keys_path)
        eligibility = _load_eligibility(eligibility_path)

    if set(key_metadata) != set(eligibility):
        missing_eligibility = len(set(key_metadata) - set(eligibility))
        missing_keys = len(set(eligibility) - set(key_metadata))
        raise ValueError(
            "Resolution-key and eligibility universes differ: "
            f"missing_eligibility={missing_eligibility}, missing_keys={missing_keys}"
        )
    (
        component_by_key,
        component_counts,
        decision_counts,
        endpoint_combination_counts,
    ) = _build_components(
        key_metadata=key_metadata,
        eligibility=eligibility,
        decisions_path=decisions_path,
        enrichments=enrichments,
        exact_public_entity_ids=(
            set(exact_reuse.component_by_entity_id)
            if exact_reuse is not None
            else set()
        ),
        sector_keyword_rules=config.sector_keyword_rules,
        batch_size=config.batch_size,
    )

    _write_top_keys(
        path=top_keys_path,
        key_metadata=key_metadata,
        eligibility=eligibility,
        component_by_key=component_by_key,
        limit=config.top_key_limit,
    )

    parquet_output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_parquet = parquet_output_path.with_suffix(f"{parquet_output_path.suffix}.tmp")
    temporary_csv = csv_output_path.with_suffix(f"{csv_output_path.suffix}.tmp")
    temporary_parquet.unlink(missing_ok=True)
    temporary_csv.unlink(missing_ok=True)
    compression = (
        None
        if settings.processing.parquet_compression == "none"
        else settings.processing.parquet_compression
    )
    parquet_writer: pq.ParquetWriter | None = None
    row_count = 0
    lineage_digest = hashlib.sha256()
    source_contract_digest = hashlib.sha256()
    exact_reference_rows_written = 0
    outcome_counts: Counter[str] = Counter()
    method_counts: Counter[str] = Counter()
    resolution_confidence_counts: Counter[str] = Counter()
    resolution_score_counts: Counter[int] = Counter()
    sector_confidence_counts: Counter[str] = Counter()
    sector_method_counts: Counter[str] = Counter()
    sector_counts: Counter[str] = Counter()
    public_enriched_rows = 0
    try:
        parquet_writer = pq.ParquetWriter(
            temporary_parquet,
            FINAL_SCHEMA,
            compression=compression,
            use_dictionary=True,
            write_statistics=True,
        )
        with temporary_csv.open("w", encoding="utf-8-sig", newline="") as csv_stream:
            csv_writer = csv.DictWriter(csv_stream, fieldnames=FINAL_COLUMNS)
            csv_writer.writeheader()
            for batch in preprocessed.iter_batches(
                columns=list(input_columns), batch_size=config.batch_size
            ):
                output_rows: list[dict[str, object]] = []
                for raw_row in batch.to_pylist():
                    row = cast(dict[str, object], raw_row)
                    record_id = _required_string(row, "record_id", source=preprocessed_path.name)
                    source_row_number = _required_int(
                        row, "source_row_number", source=preprocessed_path.name
                    )
                    normalized_value = row.get("nombre_normalizado")
                    if normalized_value is not None and not isinstance(normalized_value, str):
                        raise ValueError("Invalid nombre_normalizado in preprocessed input")
                    original_value = row.get("nombre_original")
                    if original_value is not None and not isinstance(original_value, str):
                        raise ValueError("Invalid nombre_original in preprocessed input")
                    route = _required_string(row, "route", source=preprocessed_path.name)
                    route_reason = _required_string(
                        row, "route_reason", source=preprocessed_path.name
                    )
                    eligibility_item = (
                        eligibility.get(normalized_value) if normalized_value is not None else None
                    )
                    exact_row = (
                        exact_reuse.by_record_id.get(record_id)
                        if exact_reuse is not None
                        else None
                    )
                    if exact_row is not None:
                        assert exact_reuse is not None
                        if (
                            exact_row.source_row_number != source_row_number
                            or exact_row.nombre_original != original_value
                            or exact_row.nombre_normalizado != normalized_value
                            or exact_row.route != route
                            or exact_row.route_reason != route_reason
                        ):
                            raise ValueError(
                                "Resolved exact reference row does not match the "
                                f"preprocessed source contract for record_id={record_id!r}"
                            )
                        component = exact_reuse.component_by_entity_id[
                            exact_row.entity_id
                        ]
                        business = _exact_reference_business_fields(
                            exact_row,
                            component,
                        )
                        exact_reference_rows_written += 1
                    else:
                        business = _business_fields(
                            normalized_name=normalized_value,
                            route=route,
                            eligibility=eligibility_item,
                            component_by_key=component_by_key,
                        )
                    output_row: dict[str, object] = {
                        "record_id": record_id,
                        "source_row_number": source_row_number,
                        "nombre_original": original_value,
                        "nombre_normalizado": normalized_value,
                        **business,
                        "eligibility_status": (
                            eligibility_item.status if eligibility_item is not None else None
                        ),
                        "eligibility_rule": (
                            eligibility_item.rule if eligibility_item is not None else None
                        ),
                        "eligibility_evidence": (
                            eligibility_item.evidence if eligibility_item is not None else None
                        ),
                        "route": route,
                        "route_reason": route_reason,
                    }
                    output_rows.append(output_row)
                    outcome_counts[cast(str, business["resultado_final"])] += 1
                    method_counts[cast(str, business["metodo_resolucion"])] += 1
                    resolution_confidence_counts[cast(str, business["confianza_resolucion"])] += 1
                    resolution_score_counts[cast(int, business["score_confianza_resolucion"])] += 1
                    sector_confidence_counts[cast(str, business["confianza_sector"])] += 1
                    sector_method_counts[cast(str, business["metodo_sector"])] += 1
                    sector_counts[cast(str, business["sector_propuesto"])] += 1
                    public_enriched_rows += int(business["public_entity_id"] is not None)
                    lineage_digest.update(
                        _lineage_payload(record_id, source_row_number, original_value)
                    )
                    source_contract_digest.update(
                        _source_contract_payload(
                            record_id,
                            source_row_number,
                            original_value,
                            normalized_value,
                            route,
                            route_reason,
                        )
                    )
                output_batch = pa.RecordBatch.from_pylist(output_rows, schema=FINAL_SCHEMA)
                parquet_writer.write_batch(output_batch)
                csv_writer.writerows(output_rows)
                row_count += len(output_rows)
    except Exception:
        if parquet_writer is not None:
            parquet_writer.close()
            parquet_writer = None
        temporary_parquet.unlink(missing_ok=True)
        temporary_csv.unlink(missing_ok=True)
        raise
    finally:
        if parquet_writer is not None:
            parquet_writer.close()

    if row_count != preprocessed.metadata.num_rows:
        temporary_parquet.unlink(missing_ok=True)
        temporary_csv.unlink(missing_ok=True)
        raise RuntimeError(
            f"Final row reconciliation failed: wrote {row_count}, "
            f"expected {preprocessed.metadata.num_rows}"
        )
    if exact_reuse is not None:
        if exact_reference_rows_written != len(exact_reuse.by_record_id):
            temporary_parquet.unlink(missing_ok=True)
            temporary_csv.unlink(missing_ok=True)
            raise RuntimeError(
                "Exact-reference rows failed final-output reconciliation"
            )
        if source_contract_digest.hexdigest() != exact_reuse.source_contract_digest:
            temporary_parquet.unlink(missing_ok=True)
            temporary_csv.unlink(missing_ok=True)
            raise RuntimeError(
                "Resolved employers failed preprocessing source-contract reconciliation"
            )
    try:
        _reconcile_final_output(
            temporary_parquet,
            expected_rows=row_count,
            expected_lineage_digest=lineage_digest.hexdigest(),
        )
    except Exception:
        temporary_parquet.unlink(missing_ok=True)
        temporary_csv.unlink(missing_ok=True)
        raise
    temporary_parquet.replace(parquet_output_path)
    temporary_csv.replace(csv_output_path)

    elapsed_seconds = time.perf_counter() - started
    pair_decision_metrics = _json_int_dict(decision_counts)
    pair_decision_metrics["auto_same_endpoint_eligibility"] = _json_int_dict(
        endpoint_combination_counts
    )
    input_hashes: dict[str, JsonValue] = {
        "preprocessed": sha256_file(preprocessed_path),
        "resolution_keys": sha256_file(keys_path),
        "pair_decisions": sha256_file(decisions_path),
        "employer_eligibility": sha256_file(eligibility_path),
        "public_enrichment": sha256_file(enrichment_path),
    }
    if exact_reuse is not None:
        input_hashes["resolved_employers"] = sha256_file(resolved_path)

    exact_reference_metrics: dict[str, JsonValue] = {
        "resolved_dataset_present": exact_reuse is not None,
        "resolved_dataset_rows": exact_reuse.total_rows if exact_reuse is not None else 0,
        "unresolved_rows": exact_reuse.unresolved_rows if exact_reuse is not None else 0,
        "exact_reference_rows": (
            len(exact_reuse.by_record_id) if exact_reuse is not None else 0
        ),
        "exact_reference_canonical_rows": (
            exact_reuse.canonical_rows if exact_reuse is not None else 0
        ),
        "exact_reference_alias_rows": (
            exact_reuse.alias_rows if exact_reuse is not None else 0
        ),
        "exact_reference_entity_count": (
            len(exact_reuse.component_by_entity_id) if exact_reuse is not None else 0
        ),
    }
    metrics: dict[str, JsonValue] = {
        "ruleset_version": config.ruleset_version,
        "rows_processed": row_count,
        "outcome_counts": dict(sorted(outcome_counts.items())),
        "resolution_method_counts": dict(sorted(method_counts.items())),
        "resolution_confidence_counts": dict(sorted(resolution_confidence_counts.items())),
        "resolution_score_counts": {
            str(score): count for score, count in sorted(resolution_score_counts.items())
        },
        "sector_confidence_counts": dict(sorted(sector_confidence_counts.items())),
        "sector_method_counts": dict(sorted(sector_method_counts.items())),
        "sector_counts": dict(sorted(sector_counts.items())),
        "identity_components": _json_int_dict(component_counts),
        "pair_decisions": pair_decision_metrics,
        "incremental_reference_reuse": exact_reference_metrics,
        "public_enrichment": {
            "catalogue_rows": len(enrichments),
            "source_rows_enriched": public_enriched_rows,
            "coverage_percent": round(100 * public_enriched_rows / row_count, 6),
            "catalogue_path": str(enrichment_path.resolve()),
        },
        "reconciliation": {
            "input_rows": preprocessed.metadata.num_rows,
            "parquet_output_rows": row_count,
            "csv_output_rows": row_count,
            "row_count_matches": True,
            "row_order_preserved": True,
            "record_id_preserved": True,
            "source_row_number_preserved": True,
            "nombre_original_preserved": True,
        },
        "input_sha256": input_hashes,
        "outputs": {
            "parquet": str(parquet_output_path.resolve()),
            "csv": str(csv_output_path.resolve()),
            "top_employer_keys": str(top_keys_path.resolve()),
        },
        "execution_timing": {"total_seconds": round(elapsed_seconds, 6)},
    }
    _write_json(metrics_path, metrics)
    LOGGER.info(
        "Finalized %s source rows into %s business outcomes",
        row_count,
        len(outcome_counts),
    )
    return FinalizationResult(
        parquet_output_path=parquet_output_path,
        csv_output_path=csv_output_path,
        metrics_path=metrics_path,
        top_keys_path=top_keys_path,
        row_count=row_count,
        public_enriched_rows=public_enriched_rows,
        outcome_counts=dict(sorted(outcome_counts.items())),
        elapsed_seconds=elapsed_seconds,
    )
