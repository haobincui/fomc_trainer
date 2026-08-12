# Source and transformation notes

## Question and cutoff

This snapshot answers: what is the current chk2 reward pattern, why did recent training attempts exit, and is the active process actually continuing the latest checkpoint? The operational cutoff is `2026-08-10T18:31:36Z`.

## Canonical reward stitching

The 40-step trajectory is reconstructed without replayed or incomplete groups:

- step 1: the first eight candidate receipts from `chk2_clean_v2_cp200_deepseek_high_totalsl_v1_20260810`;
- steps 2-4: the first 24 candidate receipts from `chk2_clean_v2_cp200_deepseek_low_totalsl_v1_resume_cp1_20260810`;
- steps 5-40: the first 288 candidate receipts from `chk2_clean_v2_cp200_deepseek_low600_totalsl_v1_resume_cp4_20260810`.

The four candidate receipts written for the failed next step in the second and third runs are excluded. The currently active replay of steps 5 onward is also excluded from the canonical trend.

## Exit classification

All three logs terminate with `JudgeInfrastructureError: DeepSeek judge failed after 2 attempts: error_class=_RetryableProviderResponseError`. There are no CUDA OOM records and no runtime-safety threshold violations. The internal error class is raised only after the Responses API returns an object that fails a response-contract check, such as incomplete status, missing visible or hidden output, missing reasoning-token usage, or invalid strict JSON/schema. The current logger discards the final in-memory attempt details on failure, so the exact subtype cannot be recovered from these logs.

## Chart map

- `reward_trend_chart`: line chart, optimizer step grain, one reward series, blue single-root palette, supports the conclusion that reward is noisy but the final ten-step window recovered.
- Exit events and candidate diagnostics use tables because exact lookup and failure classification matter more than shape.

## Limitations

Step reward compares different prompts and is not a fixed validation set. The three runs also change Judge reasoning effort or timeout, so the stitched series is an operational training trajectory rather than a controlled reward-model experiment. The active process may change after the cutoff.
