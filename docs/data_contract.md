# Public Data Contract

## Source

The portfolio edition uses only the committed synthetic workbook:

`data/sample/synthetic_employers.xlsx`

Expected source contract:

| Property | Value |
|---|---|
| Workbook | `data/sample/synthetic_employers.xlsx` |
| Sheet | `Sheet1` |
| Required column | `nombre_original` |
| Default rows | 5,000 synthetic records |
| SHA-256 | `0036A9305EBBB3469659A2922BD4E5859712127789CEC312D83BBFCC97D49E2C` |

The fingerprint is validated before processing. The source workbook is opened read-only and is not modified by the pipeline.

## Reference data

`data/reference/employer_master.csv`

```text
entity_id,canonical_name
```

`data/reference/employer_aliases.csv`

```text
entity_id,alias_name
```

Both files contain fictional seed entities only.

`data/reference/public_employer_enrichment.csv` is intentionally schema-only in the public demo. No fictional employer is represented as publicly verified.

## Generated data

Generated artifacts are written under `data/processed/`, `data/evaluation/`, and `output/`. These directories are Git-ignored except for `.gitkeep` markers.

## Regeneration

```bash
python scripts/generate_synthetic_data.py
```

Default generation is deterministic. Custom `--rows` or `--seed` values produce a different fingerprint; update `source.expected_sha256` in `config/config.yaml` with the printed SHA-256 before preprocessing.

## Privacy boundary

The private source workbook, original full outputs, private reference master, review samples, and selection/evaluation documents are deliberately excluded from this repository.
