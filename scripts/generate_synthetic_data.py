#!/usr/bin/env python3
"""Generate a deterministic synthetic employer-name workbook for the public demo.

The generator intentionally reproduces common entity-resolution pathologies without
using any private source records: aliases, legal-suffix variation, typos, truncation,
addresses, occupations/statuses, missing values, numeric variants, and similar names.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import re
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from openpyxl import Workbook

DEFAULT_ROWS = 5_000
DEFAULT_SEED = 20260818
FIXED_TIME = datetime(2026, 1, 1, tzinfo=UTC)
ZIP_TIME = (2026, 1, 1, 0, 0, 0)

EMPLOYERS = [
    "NORTHSTAR LOGISTICS SA",
    "BLUE HARBOR FOODS INC",
    "SUMMIT DATA SERVICES LLC",
    "GREENLINE CONSTRUCTION CORP",
    "PACIFIC CREST MEDICAL GROUP",
    "ORBITAL TELECOM SERVICES SA",
    "CEDAR POINT CONSULTING LLC",
    "AURORA INDUSTRIAL SYSTEMS INC",
    "RIVERSTONE FINANCIAL SERVICES",
    "HORIZON EDUCATION NETWORK",
    "SILVER OAK ENERGY SA",
    "METROVALE PROPERTY GROUP",
    "BRIGHTPATH SOFTWARE LABS",
    "IRONWOOD ENGINEERING CORP",
    "SUNRIDGE AGRICULTURAL EXPORTS",
    "CLEARWATER SECURITY SERVICES",
    "REDWOOD MARITIME LOGISTICS",
    "SKYBRIDGE TRAVEL SERVICES",
    "GOLDEN FIELD DISTRIBUTION SA",
    "PIONEER AUTOMOTIVE PARTS INC",
    "NORTHSTAR LOGISTICS GROUP SA",
    "BLUE HARBOR FOOD SERVICES INC",
    "SUMMIT DATA SYSTEMS LLC",
    "GREENLINE ENGINEERING CORP",
    "PACIFIC CREST HEALTH SERVICES",
    "ORBITAL COMMUNICATIONS SA",
    "CEDAR POINT LEGAL CONSULTING",
    "AURORA INDUSTRIAL SOLUTIONS",
    "RIVERSTONE CREDIT SERVICES",
    "HORIZON LEARNING NETWORK",
]

STATUS_VALUES = [
    "INDEPENDIENTE", "AMA DE CASA", "DESEMPLEADO", "ESTUDIANTE",
    "JUBILADO", "PENSIONADO", "NO LABORA", "DEPENDIENTE ECONOMICO",
]

ADDRESSES = [
    "CALLE 50 EDIFICIO TORRE NORTE",
    "AVENIDA CENTRAL PH METROPOLIS",
    "VIA PRINCIPAL EDIF 12",
    "CALLE 10 RESIDENCIAL LOS PINOS",
    "AVENIDA 5 PLAZA CENTRAL",
]

AMBIGUOUS = [
    "SERVICIOS GENERALES", "NEGOCIO FAMILIAR", "EMPRESA PRIVADA",
    "VENTAS", "COMERCIO", "TALLER", "CONSULTORIA", "CONTRATISTA",
]

MISSING = [None, "", "N/A", "UNKNOWN", "SIN INFORMACION", "-", "0"]
LEGAL_SUFFIXES = [" SA", " S A", " INC", " CORP", " LLC", " LTDA", ""]


def _drop_legal_suffix(name: str) -> str:
    parts = name.split()
    if parts and parts[-1] in {"SA", "INC", "CORP", "LLC", "LTDA"}:
        return " ".join(parts[:-1])
    return name


def _abbreviate(name: str, rng: random.Random) -> str:
    replacements = {
        "LOGISTICS": "LOGIST",
        "SERVICES": "SRVCS",
        "CONSTRUCTION": "CONST",
        "CONSULTING": "CONSULT",
        "INDUSTRIAL": "IND",
        "FINANCIAL": "FIN",
        "EDUCATION": "EDU",
        "SOFTWARE": "SFTWR",
        "ENGINEERING": "ENG",
        "DISTRIBUTION": "DIST",
        "AUTOMOTIVE": "AUTO",
        "COMMUNICATIONS": "COMM",
        "AGRICULTURAL": "AGRO",
    }
    candidates = [k for k in replacements if k in name]
    if not candidates:
        return name
    token = rng.choice(candidates)
    return name.replace(token, replacements[token])


def _single_typo(name: str, rng: random.Random) -> str:
    positions = [i for i, ch in enumerate(name) if ch.isalpha()]
    if len(positions) < 4:
        return name
    i = rng.choice(positions[1:-1])
    mode = rng.choice(["delete", "substitute", "transpose"])
    if mode == "delete":
        return name[:i] + name[i + 1 :]
    if mode == "transpose" and i + 1 < len(name) and name[i + 1].isalpha():
        return name[:i] + name[i + 1] + name[i] + name[i + 2 :]
    replacement = rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    return name[:i] + replacement + name[i + 1 :]


def _numeric_variant(name: str, rng: random.Random) -> str:
    return f"{_drop_legal_suffix(name)} {rng.randint(1, 9)}"


def _variant(name: str, rng: random.Random) -> str:
    mode = rng.choices(
        ["exact", "suffix", "abbrev", "typo", "truncate", "numeric", "spacing", "lower"],
        weights=[20, 15, 13, 18, 9, 7, 10, 8],
        k=1,
    )[0]
    if mode == "exact":
        return name
    if mode == "suffix":
        base = _drop_legal_suffix(name)
        return base + rng.choice(LEGAL_SUFFIXES)
    if mode == "abbrev":
        return _abbreviate(name, rng)
    if mode == "typo":
        return _single_typo(name, rng)
    if mode == "truncate":
        return name[:30]
    if mode == "numeric":
        return _numeric_variant(name, rng)
    if mode == "spacing":
        return name.replace(" ", rng.choice(["  ", " ", "-"]))
    return name.lower().title()


def generate_values(rows: int, seed: int) -> list[str | None]:
    rng = random.Random(seed)
    values: list[str | None] = []
    for _ in range(rows):
        category = rng.choices(
            ["employer", "status", "address", "ambiguous", "missing"],
            weights=[78, 7, 5, 6, 4],
            k=1,
        )[0]
        if category == "employer":
            values.append(_variant(rng.choice(EMPLOYERS), rng))
        elif category == "status":
            values.append(rng.choice(STATUS_VALUES))
        elif category == "address":
            values.append(rng.choice(ADDRESSES))
        elif category == "ambiguous":
            values.append(rng.choice(AMBIGUOUS))
        else:
            values.append(rng.choice(MISSING))
    return values


def _save_deterministic_xlsx(values: list[str | None], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp_dir:
        raw = Path(tmp_dir) / "raw.xlsx"
        wb = Workbook()
        wb.properties.creator = "Portfolio synthetic data generator"
        wb.properties.lastModifiedBy = "Portfolio synthetic data generator"
        wb.properties.created = FIXED_TIME
        wb.properties.modified = FIXED_TIME
        ws = wb.active
        ws.title = "Sheet1"
        ws.append(["nombre_original"])
        for value in values:
            ws.append([value])
        wb.save(raw)
        wb.close()

        with zipfile.ZipFile(raw, "r") as src, zipfile.ZipFile(
            output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as dst:
            for name in sorted(src.namelist()):
                info = zipfile.ZipInfo(name, ZIP_TIME)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o600 << 16
                data = src.read(name)
                if name == "docProps/core.xml":
                    data = re.sub(
                        rb"(<dcterms:(?:created|modified)[^>]*>).*?(</dcterms:(?:created|modified)>)",
                        rb"\g<1>2026-01-01T00:00:00Z\g<2>",
                        data,
                    )
                dst.writestr(info, data)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def write_preview(values: list[str | None], output: Path, limit: int = 100) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["nombre_original"])
        for value in values[:limit]:
            writer.writerow(["" if value is None else value])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--output", type=Path, default=Path("data/sample/synthetic_employers.xlsx")
    )
    parser.add_argument(
        "--preview", type=Path, default=Path("data/sample/synthetic_employers_preview.csv")
    )
    args = parser.parse_args()

    values = generate_values(args.rows, args.seed)
    _save_deterministic_xlsx(values, args.output)
    write_preview(values, args.preview)
    print(f"Generated {len(values):,} synthetic rows: {args.output}")
    print(f"SHA-256: {sha256_file(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
