# chk3 checkpoint selection — source notes

Generated at `2026-08-10T22:31:01Z`.

## Source authority

- The 318-step training curve is read from the completed run's `loss_history.jsonl`; its SHA-256 is recorded in `analysis_snapshot.json`.
- Probe results are the sealed/redacted `results.jsonl` files for the frozen three-case short/medium/long suite. All candidates use the same sample manifest, tokenizer path, greedy decoding, 3,072-token cap, and chk1 baseline.
- `analysis_snapshot.json` was produced by the executed `fomc_trainer` notebook and is the normalized data source for the report.

## Decision rule

1. Reject a checkpoint if any probe fails EOS, cap, boundary, final-answer, number/date preservation, periodic-tail, or repetition gates.
2. Among survivors, require eval loss within `0.001` of the best observed eval loss.
3. Prefer lower repetition; use the earlier checkpoint only as a final tie-breaker.

This rule selects checkpoint 250. Checkpoint 230 is rejected because its long case omitted four source dates. Checkpoint 250 is also the global eval-loss minimum.

## Limits

- The generation gate contains only three deterministic, redacted cases; it is a regression test, not a semantic certification set.
- It verifies deterministic number/date preservation and structural behavior, but does not independently judge all nonnumeric facts, causal claims, or prose quality.
- This is a standalone direct chk1-to-chk3 experiment and is not automatically promotable into the canonical chk2-to-chk3 DAG.
