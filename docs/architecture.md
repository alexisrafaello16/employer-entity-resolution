# Architecture & Design Notes

## System boundaries

The pipeline accepts one free-text employer field, validates its source contract, and produces an auditable entity-resolution dataset plus reusable master-data artifacts. Each stage has a deliberately narrow responsibility.

```mermaid
flowchart TD
  S[Source workbook] --> P[Preprocess / normalize]
  P --> T[Record typing]
  T --> X[Exact reference resolution]
  X -->|unresolved| C[Candidate generation]
  C --> F[Candidate features]
  F --> D[Deterministic decisions]
  D --> E[Employer eligibility]
  E --> O[Orthographic evidence]
  O --> R[Residual profile]
  R --> V[Distinctive evidence]
  V --> M[Multi-evidence assessment]
  X -->|resolved| Z[Finalization]
  M --> Z
  Z --> CM[Corporate Master]
  CM --> Q[Canonical quality gate]
  Q --> PR[Reference promotion]
```

## Raw records, aliases, and canonical entities

```mermaid
erDiagram
    SOURCE_RECORD }o--|| ALIAS : normalizes_to
    ALIAS }o--|| CANONICAL_ENTITY : represents
    CANONICAL_ENTITY ||--o{ PROMOTED_REFERENCE : may_publish

    SOURCE_RECORD {
      int source_row
      string nombre_original
    }
    ALIAS {
      string alias_name
      string entity_id
    }
    CANONICAL_ENTITY {
      string entity_id
      string canonical_name
      string quality_status
    }
    PROMOTED_REFERENCE {
      string resolution_key
      string entity_id
    }
```

## Design principles

1. **Precision before forced coverage.** Unsupported records abstain.
2. **Evidence before decision.** Feature computation does not silently establish identity.
3. **Bounded candidate generation.** Expensive pair evaluation is limited to plausible candidates.
4. **Persistent knowledge is gated.** The master is reusable only after canonical quality checks.
5. **Determinism.** Configuration, stable ordering, fingerprints, metrics, and explicit rules support reproducibility.
6. **Incrementality.** Promoted references can short-circuit later runs through exact resolution.

## Scaling path

The current implementation is a batch Python pipeline using Parquet for intermediate artifacts. At higher scale, the same logical stages can be moved to Spark, SQL/warehouse execution, or distributed batch infrastructure. Candidate blocking should remain partitionable, and persistent master/reference data should be versioned and quality-gated independently from transactional runs.
