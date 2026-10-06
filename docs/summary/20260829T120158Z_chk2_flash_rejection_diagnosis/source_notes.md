# Source notes

## Scope

The primary diagnosis covers the 1,710 machine-gate non-passes among 3,198 prepared candidates. The 1,679 preparation exclusions from the original 4,877 source rows are reported separately and are not mixed into the machine-rejection denominator.

## Reconciliation

- Prepared: 3,198
- Machine pass / promoted: 1,488
- Machine rejection ledger: 1,710
- Generation gate rejection: 1,488
- Verification gate rejection: 222
- 1,488 + 1,710 = 3,198; 1,488 + 222 = 1,710

## Interpretation boundaries

- Reason-family counts are row-level and non-exclusive.
- Error codes record rules that fired; they do not by themselves prove why the model produced the error.
- Paragraph-length quartiles are equal-row buckets after sorting by whitespace-delimited word count. Tied word counts can straddle adjacent buckets.
- The promoted release is machine-screen-only after removal of the human-review requirement.

## Packaging

`artifact.json` is the canonical structured report. `report.html` is a self-contained local rendering. The environment had no Node executable, so the packaged portable-report builder could not be run; the static fallback is validated separately in `delivery_receipt.json`.
