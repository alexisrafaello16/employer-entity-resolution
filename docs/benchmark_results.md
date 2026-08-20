# Benchmark Results: How to Read the Numbers

The README includes sanitized aggregate metrics measured on a larger private dataset. They are included to demonstrate engineering scale, not to provide a reproducible public benchmark.

## Initial run

- 323,001 source records processed.
- 2,195,333 candidate pairs generated after bounded candidate generation.
- 91,634 rows finalized as automatically canonicalized employers.
- 218,012 rows retained as normalized employers without public validation.
- 83 rows received selective public validation.
- 2,255 rows were classified as non-identifiable address-like values.
- 11,017 rows were retained as non-identifiable due to insufficient information.

## Corporate Master

- 257,711 canonical entities.
- 308,748 aliases.
- 309,729 source rows represented.
- 16 publicly validated entities.

The high canonical-entity count is consistent with the system's conservative merge policy: uncertain similarities are not collapsed merely to increase deduplication coverage.

## Canonical quality gate

- 16 `PUBLIC_VALIDATED`.
- 38,632 `ACCEPTABLE`.
- 1,496 `SUSPICIOUS`.
- 217,567 `NOT_PROMOTABLE`.

## Reference promotion

- 38,648 entities promoted.
- 88,151 aliases promoted.

Promotion is deliberately narrower than master creation. An entity can exist in the master without being trusted as reusable reference knowledge.

## Second run

- 88,690 records resolved by exact reference matching before residual candidate matching.
- 234,311 records remained unresolved at that stage.
- Candidate pairs fell from 2,195,333 to 1,593,662.
- Reduction: 601,671 candidate pairs, approximately 27.4%.
- Approximately 27.46% of the source rows were resolved early through persistent reference knowledge.

## Software quality snapshot

- 445 tests passed.
- Ruff: all checks passed.
- mypy: success.
- 26 source files analyzed.

These figures belong to the original execution snapshot and should not be interpreted as public synthetic-demo performance.
