# chk3 residual targeted repair plan

## Objective

Repair only the 17 chk3 rows still missing after repeated generic `--resume`
runs. Preserve the 2,055 accepted caches and continue to validate every new
response with the unchanged chk3 v2 target validator.

## Implementation

- Add `jobs/generation/repair_chk3_sft_targets.py`; do not modify
  `generate_chk3_sft_targets.py`, because its code hash binds the existing
  accepted cache.
- Verify the on-disk v2 prompt contract against the unchanged main generator,
  load all 2,055 accepted entries, and select only rows without an accepted
  current-contract cache entry.
- Use `deepseek-v4-pro`, thinking enabled, high reasoning effort, JSON output,
  concurrency 8, and at most three targeted attempts per residual row.
- Require a concise 100--350 word fidelity plan in native reasoning and a
  single compact Minutes paragraph. Supply an explicit occurrence-level list
  of source numeric quantities and a normalized date set. Require every
  listed quantity occurrence, prohibit any extra number, and instruct the
  teacher to prioritize a complete answer before the token limit.
- Save targeted-repair contract and provenance separately. A response enters
  the main accepted cache only after the original number/date/attribution/meta
  and 64/2,400/4,096-token gates pass. Mark the accepted payload with the
  targeted-repair contract hash and attempt number.
- Rematerialize the main `teacher_responses`, `sft`, `manifests`, failures, and
  summary from the union of original and repaired accepted caches.

## Verification and execution

- Add unit tests for quantity/date inventory rendering, concise prompt rules,
  contract binding, and compatibility of targeted accepted payloads with the
  original cache validator.
- Run ruff, focused pytest, and a dry run in the `fomc_trainer` environment.
- Start the real targeted repair detached in the background and monitor until
  it completes. Do not silently accept, delete, or truncate a failed row.
