# chk4 teacher target failure repair

## Observed failure

- Gold-blind meeting brief acquisition stopped at 114/128.
- Fourteen missing briefs consisted of six empty/invalid JSON responses and
  eight numeric surface-form conversions.
- The original acquisition contract and 114 accepted cache entries were
  byte-for-byte aligned, so changing the original validator would have
  invalidated all accepted cache keys.
- The first Decision-SFT target run exposed a separate semantic defect:
  DeepSeek native `reasoning_content` mostly contained instruction processing,
  canonical-label repetition, and output-format discussion. Even accepted
  entries were unsuitable as student reasoning targets.

## Implemented repair

1. `repair_chk4_meeting_briefs.py` performs a targeted retry of only missing
   brief rows. It requires verbatim numeric/date surface forms, short native
   reasoning, full rejected-response provenance, the original validator, and
   the original accepted-cache key.
2. The brief repair completed 14/14 with exactly fourteen new API requests.
   The immutable brief release now has 128/128 accepted meetings.
3. Decision target contract v1 is retained at
   `output/data/retrain_v2/chk4/deepseek_v4_pro_v1` for audit but is forbidden
   as training input because native provider reasoning leaks teacher-only task
   information.
4. Decision target contract v2 writes an explicit teacher content object:

   ```json
   {
     "reasoning": "<grounded qualitative policy rationale>",
     "direction": "hold",
     "magnitude_bp": 0
   }
   ```

   The student response remains:

   ```text
   <validated content.reasoning>
   </think>
   <locally serialized canonical decision JSON>
   ```

   Native `reasoning_content` is retained only in provider provenance and is
   never copied into training data.
5. V2 uses a new output root and contract/cache schemas, so no v1 target can be
   silently mixed into v2:

   - teacher release: `deepseek_v4_pro_v2`
   - training release: `training_data_v2`

6. The full pipeline shell now preserves/rechecks all 128 meeting briefs,
   generates v2 targets, invokes a targeted target retry after any incomplete
   normal pass, revalidates all caches, and only then materializes SFT/GRPO
   rows.

## Verification before restart

- `fomc_trainer` Python compilation passed.
- chk4 generation tests: 10 passed.
- V2 dry-run prepared 128 meetings with split counts 102/13/13.
- Prompt token audit: min 321, max 969, mean 585.015625; no truncation.
- V1 target directory remains preserved.

## Final run result

- Meeting briefs: 128/128 accepted.
- V2 Decision-SFT targets: 128/128 accepted, zero final failures.
- Offline final-response QA: exactly one `</think>` per row, nonempty rationale,
  final JSON equal to manifest gold, and zero matches for the audited
  teacher/canonical/gold/validation/prompt/schema meta-leak patterns.
- Unique release counts: train 102, validation 13, test 13.
- Direction-balanced physical train rows: 141 for SFT and 141 for GRPO;
  validation and sealed test remain unrepeated at 13 each.
- End-to-end shell reached `chk4 DeepSeek pipeline complete` and wrote
  `output/data/retrain_v2/chk4/pipeline_v1/complete.json`.
