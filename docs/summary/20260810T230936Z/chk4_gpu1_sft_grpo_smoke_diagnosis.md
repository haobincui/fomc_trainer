# chk4 GPU1 Decision-SFT / GRPO smoke diagnosis

Generated at: `2026-08-10T23:09:36Z`

## Outcome

The first GPU1-only Decision-SFT run completed successfully, but its merged
model did not satisfy the output contract strongly enough to seed GRPO.  The
GRPO run was therefore stopped after two complete checkpoints rather than
allowed to continue with zero gradients.

GPU0 was not used or interrupted by either chk4 training stage.

## SFT run

- Parent: the separately attested chk1 checkpoint-200 merge scoped to chk4.
- Device topology: physical GPU1 only; one process; effective optimizer batch
  `1 * 1 * 8 = 8`.
- Configuration SHA-256:
  `6e43b01a0ee438722a9075e78b27b4390c44514a81e65b6f0ef3b54590be9cef`.
- Schedule: three epochs, 54 optimizer steps, peak learning rate `1e-6`.
- Runtime: 426.17 seconds.
- Overall train loss: `2.0088246222`.
- Eval loss: `2.0737204552` at step 10, `2.0024948120` at step 50,
  and `2.0025002956` at the final step.
- No OOM, CUDA, NCCL, or non-finite error occurred.

The loss improved by only about 3.4 percent and token accuracy remained near
0.55.  This was underfitting for a target whose response has a median length of
118 tokens and must end with `</think>` followed by bare JSON.

## GRPO smoke

- Device topology: physical GPU1 only; one process; effective optimizer batch
  8; `num_generations=4`; `generation_batch_size=8`.
- Configuration SHA-256:
  `0a3a13ba8546b048a4da555f93551f2b7e80592fcf51f576dcbeef49d35d8894`.
- Completed checkpoints: 1 and 2.
- Peak observed GPU1 memory: approximately 12.1 GiB; no OOM.
- Step 1: mean completion length `496.25`, clipped ratio `0.75`, reward
  `0`, reward standard deviation `0`, loss `0`, gradient norm `0`.
- Step 2: mean completion length `512`, clipped ratio `1.0`, reward `0`,
  reward standard deviation `0`, loss `0`, gradient norm `0`.

Across the 16 saved completions:

- 14 were truncated at 512 tokens before producing a usable final JSON;
- two produced the correct semantic `hold/0` decision after `</think>`, but
  wrapped it in a Markdown JSON fence, which correctly failed the strict bare
  JSON contract;
- all 16 therefore had `prediction=null`, `format_valid=false`, reward zero,
  and advantage zero.

With `mask_truncated_completions=true`, this run had no useful policy gradient
and could not recover by continuing.  It was deliberately stopped after the
complete checkpoint-2 write.

## Evidence

- SFT trainer state:
  `output/training/retrain_v2/chk4_from_chk1_cp200_sft_grpo_core_v3_20260810/adapters/chk4_sft/trainer_state.json`
  (`1b08871aeb04688987865a167a83e969be600de5055eebfb757206e29694fb7a`).
- GRPO checkpoint-2 trainer state:
  `output/training/retrain_v2/chk4_from_chk1_cp200_sft_grpo_core_v3_20260810/adapters/chk4_grpo/checkpoint-2/trainer_state.json`
  (`2fc5f4db2934997914de07b9981080bfdde54076645d2543cc996e9a2c52ede0`).
- Step-1 completions:
  `output/training/retrain_v2/chk4_from_chk1_cp200_sft_grpo_core_v3_20260810/adapters/chk4_grpo/completions/completions_00001.parquet`
  (`a24dc073b78d941ac6ee0b16bbf7189220137d028bfe45fa260fd23b69f1790b`).
- Step-2 completions:
  `output/training/retrain_v2/chk4_from_chk1_cp200_sft_grpo_core_v3_20260810/adapters/chk4_grpo/completions/completions_00002.parquet`
  (`335f0137e2ab51e9c780783138c8102f35f5caa25e31beb771c866f5cdfa9ddf`).
- Per-completion reward log:
  `output/training/retrain_v2/chk4_from_chk1_cp200_sft_grpo_core_v3_20260810/adapters/chk4_grpo/reward.jsonl`
  (`bad66382978079a4d0878e961574d1fc76740c402596e274dafcfb20c3826d4e`).
- Step reward history:
  `output/training/retrain_v2/chk4_from_chk1_cp200_sft_grpo_core_v3_20260810/adapters/chk4_grpo/reward_history.jsonl`
  (`0d076feaeff5b67a7a08e2660034be9226b3878ce11ce0e5039bf0ff4eea9d9b`).
- Step loss history:
  `output/training/retrain_v2/chk4_from_chk1_cp200_sft_grpo_core_v3_20260810/adapters/chk4_grpo/loss_history.jsonl`
  (`25dbc3c0e5c30334bf50a5f9986e5b8b867fc867af9e912ed23e143e60581f6d`).

## Corrective rollout

The failed artifacts remain immutable and will not be resumed.  A separate
warm-fix profile will use a fresh SFT output with `learning_rate=5e-6`,
`max_steps=12`, two warmup steps, and the same data, parent, LoRA, seed, and
effective batch.  It must pass a validation-only generation gate before merge:

- one `</think>` followed by exact-key, unfenced JSON;
- finite output and EOS termination;
- no strict periodic tail or severe n-gram repetition;
- low clipping and non-zero reward/group variance under GRPO-like sampling.

Validation exact magnitude is reported but is not a hard gate because the
sealed training split does not contain the validation-only 50/75 bp hike
magnitudes.  If the SFT gate passes, GRPO will start from that isolated merge
with a 1024-token completion ceiling and the strict reward unchanged.
