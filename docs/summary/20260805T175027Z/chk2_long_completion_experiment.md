# chk2 long-completion experiment

UTC date: 2026-08-05

## Outcome

Increasing the policy completion budget did not restore the native DeepSeek
reasoning boundary. Both long-output runs were stopped because continuing them
would not produce a useful chk2 model.

The current versioned v3 source contract is left at the largest measured A30
profile:

- policy prompt: 2,560 tokens;
- policy completion: 4,096 tokens;
- policy maximum sequence: 6,656 tokens;
- Judge maximum context: 12,288 tokens;
- Judge candidate reserve: 4,352 tokens;
- GPU0: Qwen3.5-9B Judge;
- GPU1: 4-bit chk1 GRPO policy.

The 12,288-token Judge started successfully with 72,864 KV-cache tokens and
the policy reached a measured GPU1 peak of 21,601 MiB without OOM. This is the
largest practical profile under the 22 GiB A30 safety target.

## v15: 2,560 completion tokens

Run ID:
`retrain_v2_full_v7_automated_v15_reward_v3_long2560_20260805`

The first optimizer step completed after 744.78 seconds:

- reward mean: 0.20148;
- reward standard deviation: 0.03965;
- 8/8 candidates had no `</think>`;
- completion mean/min/max: exactly 2,560 tokens;
- clipped ratio: 1.0;
- loss: 0.0;
- gradient norm: 0.0.

The run was stopped after checkpoint-1 because all rewards were masked by
`mask_truncated_completions=true`.

## v16: 4,096 completion tokens

Run ID:
`retrain_v2_full_v7_automated_v16_reward_v3_long4096_20260805`

The first four candidates were generated and judged successfully:

- reward range: 0.15324 to 0.17410;
- Judge requests successful on attempt 1: 4/4;
- `</think>` present: 0/4;
- plain-answer fallback: 4/4;
- Judge rubric scores: zero for all five dimensions on all four candidates;
- completion character range: 9,200 to 9,685;
- GPU1 measured peak: 21,601 MiB;
- no CUDA OOM or Judge context overflow.

The run was stopped before spending another full optimizer step. Its outputs
had already passed the reasoning lengths of matching chk1 SFT targets without
emitting the boundary.

## Root-cause evidence

The chk1 raw SFT files contain 1,744 responses. Every response contains exactly
one `</think>`, no response contains `<answer>`, and every boundary has nonempty
reasoning and answer text. The DeepSeek chat template supplies the opening
`<think>` in the rendered prompt. A reproduction of the SFT prompt-completion
mask confirmed that `</think>` is part of the effective completion loss.

The train-set reasoning length through `</think>` has median 1,797 tokens and
P95 3,395 tokens. For the 2015-06-17 examples sampled by v16, locally matching
SFT targets reached `</think>` after 1,186 and 2,559 tokens. The policy still did
not emit the boundary within 4,096 generated tokens. Therefore the active
failure is not an incorrectly named `<answer>` tag and is no longer explained
by the completion limit alone: the merged chk1 does not reliably reproduce the
boundary learned from its target files.

## Required next action

Do not resume v15 or v16 as formal chk2 training. First create and validate a
short format-repair SFT stage from the existing local chk1 data, with a compact
reasoning target and an explicit `</think>` followed by the original final
answer. Only after standalone generation demonstrates terminated responses
should chk2 GRPO be restarted. Disabling truncated-completion masking would
make gradients nonzero, but would optimize long unfinished candidates and is
not an acceptable substitute for repairing chk1.

## Validation

- DAG, reward, token-gate, execution-receipt and launcher tests: 252 passed.
- Background launcher, Judge health/attestation, preflight-cache and runtime
  safety tests: 37 passed.
- Both GPUs were returned to 14 MiB idle memory after stopping v16.
