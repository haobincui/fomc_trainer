# chk4 24-step Decision-SFT checkpoint pilot diagnosis

## Technical summary

The 24-step Decision-SFT run completed successfully on one A30 exposed as `CUDA_VISIBLE_DEVICES=1`, and its token-level validation loss improved monotonically from `2.011529` at step 4 to `1.728163` at step 24. That optimization result did **not** translate into a checkpoint that was safe to promote to Decision-GRPO.

All three audited checkpoints failed the same sealed, train-only, stratified rollout contract. `checkpoint-10` and `checkpoint-24` had zero sampled reward, zero sampled reward variance, and zero correct sampled directions for both `hike` and `cut`. `checkpoint-20` was the closest candidate: its `hike` group contained one correct sampled decision, but its `cut` group still contained no correct direction. The sole nonzero `cut` reward at checkpoint 20 was `0.05`, the strict-JSON format component for a semantically wrong `hold/0` prediction.

The stable pattern is a decision collapse toward `hold`: every checkpoint got `3/4` sampled `hold` cases correct, while sampled `hike` correctness was `0/4`, `1/4`, and `0/4` at steps 10, 20, and 24, and sampled `cut` correctness was `0/4` at all three checkpoints. This is not explained by truncation alone: checkpoint 10's sampled `cut` group had `0/4` caps and `4/4` complete fenced decisions, yet all four were wrong.

No checkpoint passed the promotion gate, so no 24-step SFT checkpoint was selected or merged. The configured `merged/chk4_sft` source for GRPO does not exist, and neither a GRPO adapter directory nor a GRPO merged directory was created. Decision-GRPO was therefore correctly left unstarted.

## Scope and comparison contract

This report compares the same three SFT checkpoints against the same immutable pilot manifest and generation settings. It does not reproduce completion text.

| Item | Fixed contract |
|---|---|
| Source split | `decision_grpo/train.jsonl` only |
| Stratification | one sealed prompt for each of `hold`, `hike`, and `cut` |
| Cases per direction | one greedy plus four sampled generations |
| Total per checkpoint | 15 generations |
| Sampling | temperature `0.7`, top-p `0.9` |
| Completion budget | 1,024 tokens |
| Evaluator | `decision_dense_v3` plus pilot fail-closed EOS/cap rule |
| Hardware | GPU1 only; pilot runtime peak reserved memory `7.4 GiB` |
| Shared sample-manifest SHA-256 | `1894525fa76fc947b31e78226d4b96f9d556e715ef7e8deb9e24f27f9ef3f4a2` |

The pilot is intentionally a pre-GRPO safety diagnostic, not an unbiased model-quality estimate. It contains one fixed train prompt per direction and four stochastic draws per prompt.

## SFT optimized token loss but stopped after 24 optimizer steps

The SFT configuration used QLoRA (`r=32`, alpha `64`), NF4 4-bit loading, BF16 compute, per-device batch size `1`, gradient accumulation `8`, cosine scheduling, two warmup steps, learning rate `1e-5`, completion-only loss, and `max_length=3072`. Although `num_train_epochs=3` remains in the config, `max_steps=24` controlled termination; the recorded terminal epoch was `1.340426`.

The run completed without OOM or non-finite loss. Its aggregate train loss was `1.809128` over 24 optimizer steps and its final evaluation loss was `1.728163` on 13 validation rows.

| Eval step | Eval loss | Eval token accuracy |
|---:|---:|---:|
| 4 | 2.011529 | 0.554902 |
| 8 | 1.875813 | 0.562162 |
| 12 | 1.791024 | 0.572828 |
| 16 | 1.748666 | 0.580836 |
| 20 | 1.731787 | 0.581489 |
| 24 | 1.728163 | 0.586800 |

For checkpoint comparison, the train loss is the logged loss at that optimizer step. Evaluation ran every four steps, so checkpoint 10 has no same-step evaluation; its last available evaluation is step 8.

| Checkpoint | Epoch | Step train loss | Eval reference | Eval loss | Adapter directory SHA-256 |
|---:|---:|---:|---:|---:|---|
| 10 | 0.567376 | 1.7470 | step 8 | 1.875813 | `3a3affdfd5679ef2bfc268ee140b081220a996cd146888b6a456ff8b18e9e545` |
| 20 | 1.113475 | 1.6696 | step 20 | 1.731787 | `7e09eb7d9d00069c6fb2ec6f0ddb338ddbdcd5f5b700a772fb3263c86b91a310` |
| 24 | 1.340426 | 1.6323 | step 24 | 1.728163 | `5daba41322edebe6e5fadfaa9342a79cbb0f317b6cc5fb1990fcdb25ec0bd12f` |

The continuing loss improvement between steps 20 and 24 was small (`-0.003625` eval loss) and coincided with worse directional behavior: checkpoint 20 retained one correct sampled `hike`, while checkpoint 24 retained none. Token cross-entropy is therefore not a sufficient checkpoint-selection metric for this decision task.

## `decision_dense_v3` scoring contract

The evaluator first requires exactly one `</think>` boundary, nonempty reasoning, and a schema-exact decision with exactly `direction` and `magnitude_bp`. It accepts either native plain JSON or one exact lowercase `json` Markdown fence. Invalid, empty, malformed, or schema-invalid answers receive zero.

For a parsed prediction, define:

```text
direction_ok = 1[predicted direction == target direction]
magnitude_score = direction_ok * max(0, 1 - abs(predicted_bp - target_bp) / 100)
exact = 1[predicted direction and magnitude equal target]

semantic = 0.45 * direction_ok + 0.30 * magnitude_score + 0.20 * exact

strict JSON reward = 0.05 + semantic
fenced JSON reward = 0.25 * semantic
invalid reward = 0
```

The pilot additionally forces reward to zero if generation does not hit EOS or reaches the 1,024-token cap. Consequently, a fully correct strict answer scores `1.0`, a fully correct fenced answer scores `0.2375`, and a schema-valid strict but directionally wrong answer scores only the `0.05` format component. This last case explains checkpoint 20's nonzero `cut` reward without any correct `cut` prediction.

## Every checkpoint failed the promotion gate

The gate is evaluated on the four sampled generations for each direction:

- at least one nonzero reward per direction;
- at least one correct direction per direction;
- reward population standard deviation greater than zero per direction;
- overall cap rate no greater than `0.25`;
- overall exactly-one-boundary rate at least `0.75`;
- zero strict-periodic tails.

All three checkpoints passed the overall cap, boundary, and periodic-tail conditions. Their failures came from direction-specific semantic signal.

### Overall 15-case results

| Checkpoint | Reward mean | Reward std | Direction correct | Nonzero | Cap | One boundary | Strict JSON | Fenced JSON | Gate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 10 | 0.063333 | 0.105026 | 4/15 | 4/15 | 2/15 | 13/15 | 0/15 | 10/15 | failed |
| 20 | 0.082500 | 0.110284 | 5/15 | 6/15 | 3/15 | 13/15 | 1/15 | 11/15 | failed |
| 24 | 0.114167 | 0.254782 | 4/15 | 4/15 | 3/15 | 12/15 | 1/15 | 9/15 | failed |

Checkpoint 24's highest aggregate reward is driven by one strict, correct sampled `hold` answer worth `1.0`; it is not evidence of broader decision competence.

### Sampled four-generation results by target direction

`Strict` below means accepted by the v3 parser as schema-exact native JSON, not merely text that begins with a brace.

| Checkpoint | Target | Reward mean | Reward std | Nonzero | Direction correct | Cap | One boundary | Strict | Fenced |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | hold | 0.178125 | 0.102841 | 3/4 | 3/4 | 0/4 | 4/4 | 0/4 | 3/4 |
| 10 | hike | 0.000000 | 0.000000 | 0/4 | 0/4 | 1/4 | 3/4 | 0/4 | 2/4 |
| 10 | cut | 0.000000 | 0.000000 | 0/4 | 0/4 | 0/4 | 4/4 | 0/4 | 4/4 |
| 20 | hold | 0.178125 | 0.102841 | 3/4 | 3/4 | 1/4 | 3/4 | 0/4 | 3/4 |
| 20 | hike | 0.059375 | 0.102841 | 1/4 | 1/4 | 0/4 | 4/4 | 0/4 | 4/4 |
| 20 | cut | 0.012500 | 0.021651 | 1/4 | 0/4 | 2/4 | 3/4 | 1/4 | 1/4 |
| 24 | hold | 0.368750 | 0.377129 | 3/4 | 3/4 | 1/4 | 3/4 | 1/4 | 2/4 |
| 24 | hike | 0.000000 | 0.000000 | 0/4 | 0/4 | 1/4 | 3/4 | 0/4 | 2/4 |
| 24 | cut | 0.000000 | 0.000000 | 0/4 | 0/4 | 1/4 | 3/4 | 0/4 | 2/4 |

### Exact gate failures

| Checkpoint | Failure reasons |
|---:|---|
| 10 | `hike`: no sampled nonzero reward, no sampled correct direction, zero sampled reward variance; `cut`: the same three failures |
| 20 | `cut`: no sampled correct direction |
| 24 | `hike`: no sampled nonzero reward, no sampled correct direction, zero sampled reward variance; `cut`: the same three failures |

Checkpoint 20 is the least-bad diagnostic checkpoint, but it still does not satisfy the minimum semantic contract. Its one nonzero sampled `cut` score is format-only and would not establish a correct policy direction.

## The dominant failure is a stable `hold` bias, not only formatting or length

Three checks narrow the diagnosis:

1. **The bias survives checkpoint selection.** Sampled `hold` direction accuracy stays at `3/4` for steps 10, 20, and 24. Sampled `cut` accuracy is `0/4` at every checkpoint, while sampled `hike` accuracy is only briefly `1/4` at step 20.
2. **Truncation is not sufficient to explain the failure.** Every checkpoint's overall cap rate is within the `25%` gate. More decisively, checkpoint 10's sampled `cut` group has no caps, four valid boundaries, four EOS terminations, and four recoverable fenced decisions, but zero correct directions.
3. **Fenced formatting is not sufficient to explain the failure.** V3 deliberately recovers fenced decisions at a semantic discount. A correct fenced prediction still produces `0.2375`; the zero rewards in complete fenced `hike` and `cut` cases therefore represent semantic direction errors, usually `hold/0`, rather than a strict-format rejection.

The release itself records limited directional coverage. Its 141 physical training rows contain `89 hold`, `36 hike`, and `16 cut` targets (`63.1%`, `25.5%`, and `11.3%`). At the unique-meeting level, the imbalance is larger: `89 hold`, `9 hike`, and `4 cut` among 102 unique rows (`87.3%`, `8.8%`, and `3.9%`). Minority actions are repeated four times in the physical training set, but the number of distinct minority contexts remains small. The observed collapse is consistent with that coverage limit, though this three-prompt pilot cannot by itself prove causality.

## Why Decision-GRPO was not started

The GRPO config uses `scale_rewards: group` with four generations. A sampled group with zero reward variance has no within-group advantage signal. Checkpoints 10 and 24 would therefore supply no useful signal for both `hike` and `cut`. Checkpoint 20 supplies variance for `cut`, but the only nonzero reward is the `0.05` format component on a wrong direction; promoting it risks reinforcing output shape without establishing the missing decision behavior.

The fail-closed handoff state is observable on disk:

- no checkpoint has `quality_status: passed`;
- no checkpoint-selection or SFT-merge receipt exists for this 24-step branch;
- configured source `merged/chk4_sft` is absent;
- `adapters/chk4_grpo` is absent;
- `merged/chk4_grpo` is absent;
- no GRPO process was active at the final audit.

Therefore the correct state is `SFT complete / checkpoint promotion blocked / merge not run / GRPO not started`.

## Limitations and robustness

- The pilot uses train-only prompts to test whether the warm start can produce a learnable reward distribution. It is not a held-out performance estimate.
- Each direction is represented by one fixed lower-median-length prompt. Four sampled generations quantify seed sensitivity for that prompt but not prompt-population uncertainty.
- The checkpoints share prompts, seeds, generation settings, evaluator, parent model, and dataset hashes. This makes checkpoint comparison controlled, but the absolute rates retain high sampling uncertainty.
- Loss and rollout metrics measure different objectives. The monotonic validation-loss reduction establishes token-level optimization, not policy-direction calibration.
- No completion text is reproduced in this report or its companion JSON; detailed text remains only in the sealed `results.jsonl` evidence.

## Recommended next steps

1. Do not merge or start GRPO from checkpoints 10, 20, or 24 under the current release.
2. Repair the Decision-SFT signal before adding policy optimization: increase distinct `hike` and `cut` contexts or use a separately versioned direction-balanced release while retaining target-blind, point-in-time inputs.
3. Repeat the exact same 15-case stratified pilot. Require all three sampled direction groups to meet nonzero, correct-direction, and positive-variance gates before merge.
4. Treat checkpoint 20 only as the closest diagnostic reference, not as an approved GRPO parent.
5. After a passing pilot, create an immutable selection receipt, merge that exact adapter, attest the merged payload, and only then authorize the fresh GRPO output directory.

## Further questions

- Does a direction-balanced but unique-context-preserving SFT release remove the `hold` collapse without increasing target leakage?
- How much of the remaining fenced-output rate comes from the base/checkpoint chat behavior versus the 141 SFT targets, which themselves use plain JSON?
- Once the three-prompt safety pilot passes, does the same behavior hold across a larger stratified train replay and sealed validation prompts?

## Evidence ledger

The companion structured record is `chk4_steps24_sft_checkpoint_pilot_diagnosis.json`. All hashes below were recomputed from the local files after the pilots completed.

| Evidence | SHA-256 |
|---|---|
| SFT config `configs/retrain_v2/chk4_decision_sft_from_chk1_cp200_warm_fix_lr1e5_steps24_v3_20260811.yaml` | `77dbf1a67530c32d74d2c5f8d1e1336e9d0efe9b1c9a783454a110719b78c6df` |
| SFT resolved runtime | `e20fc57c1c59eb9dadaa154d6a9fcf1ce1555b4988b19dbfc9102d8627cf11ac` |
| SFT trainer state | `2e7bc68c84174d235f45a0778d974d2301a9744eb98ad9f809271d3e69513754` |
| SFT loss history | `ffda2dd6bfaad56bc494b77d5c9580e4247c484cce312f81b740cb63981aef14` |
| SFT log | `5ffadc8a985fcb706b2bca462a9d2ff0ccd05975496f08c24b55507965537986` |
| Dataset release manifest | `8b05dce09bcb0a0b27ee240da1e5d730be93921ce19e650a3b3e8b2689adb893` |
| Pilot sample manifest | `1894525fa76fc947b31e78226d4b96f9d556e715ef7e8deb9e24f27f9ef3f4a2` |
| GRPO config | `f88aa1f9fe5f6845e4a8f92408e3df7d5437aabd124e356dcfb6d9f8bb46f2be` |
| Reward v3 source | `186f76d17b43e978e73ffe3601f015961f24d2498ebb866da78fe7414a8289bc` |
| Stratified pilot source | `2ca85b6dd697eac2d95ac6a14645fac255b4e203967ff417b8450d62ff83225d` |
| cp10 launch / results / summary | `3e8be970e7e9a578606916b4925214403f03e06b35875ca8e1a8e922e4f1f733` / `4b3a47244bfbb29a7e3d54499766e5daf51d36f982faf653cb9a88c81b748650` / `2a1f9cc067cde69a16348723ed648b35e1a3882259ac4ff8f097ffc00e0072c8` |
| cp20 launch / results / summary | `06eaf4e023eb78625682d00dc0018d9968b63551cfacb91dc1f5d3d6551fead3` / `365e1b329952ed9ade183dbc3ffc6f68645c0d36b8805be1bcf553684d499fd9` / `a5a079e1345755e49e76ff7b05e687cabde9601f96551e53193c263b5c99d081` |
| cp24 launch / results / summary | `07807b3113505632da3d1803b22526fd5a2c08b007c8fadac5cde116a80de20d` / `86c014ea7a9b4bf9787488c42dddfb40d38bb5c78f210f6167c53d5b909acde0` / `fc9ca86ef6270eeb5518a595e6f6faac990a4567186716f9c4c2e9a39908ae6c` |

