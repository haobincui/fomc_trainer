# TOTALSL overlay source notes

## Source scope

- Parent release: `dataset/processed/retrain_v2/analysis_base_full_v7_automated_v3_20260804`
- Parent dataset: `dataset/processed/retrain_v2/analysis_base_full_v7_automated_v3_20260804/analysis_grpo`
- Parent manifest SHA-256: `e0dd631097b0a79a7592599161c801b493866e4c48ae59935dd4a1017fbf2c52`
- Train SHA-256: `da32684c3001d8ac908a99c20dcc5d2397a7351e57f3b9ea15ede63990f2b4bd`
- Validation (`eval.jsonl`) SHA-256: `fdbf39b642714b549da18fd360510b9b050b20e7512fa795c44bd955db51be09`
- Test SHA-256: `8e0e89fe89d271b6d6862f5059431e5ad12d03646a4d8f1ab5e304c84b939a20`

## Definition and transformation

An affected evidence entry is an object in `provided_data.evidence` whose `series_id` is exactly `TOTALSL` and whose `units` is exactly `Millions of Dollars`. The overlay changes only this `units` value to `Billions of Dollars`. It then replaces the one exact embedded copy of the old `provided_data` in `prompt` with the synchronized new value.

The overlay does not change sample IDs, row order, meeting/observation dates, values, evidence IDs, source SHA-256 fields, topics, or any non-TOTALSL evidence. Parent files remain read-only.

## Expected population

| Split | Rows | Affected rows | TOTALSL entries |
|---|---:|---:|---:|
| train | 493 | 30 | 46 |
| validation (`eval`) | 199 | 12 | 12 |
| test | 190 | 13 | 13 |
| total | 882 | 55 | 71 |

Unchanged rows expected to remain byte-for-byte identical: 827.

## Evidence boundary

This is a deterministic metadata repair, not a statistical estimate and not a semantic rewrite of any model target. The audit proves the declared field-level and byte-level invariants; it does not independently re-download or revalidate the upstream TOTALSL series.

## Visualization decision

The report uses one minimal bar chart to show how the 55 affected rows are distributed across train, validation and test. The exact row and evidence-entry counts remain in the accompanying audit table, which is the authoritative representation for verification.
