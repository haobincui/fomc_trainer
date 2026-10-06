# chk1 checkpoint-selection probe update

## Decision

Retain `checkpoint-200` as the selected chk1 checkpoint. The evidence supports
calling it a **defensible representative of the converged plateau**, not a
statistically unique optimum.

## Comparison contract

Only checkpoints from the same clean-v2, learning-rate `1e-6`, 255-step chk1
trajectory are compared. The original four probe points (`cp80`, `cp150`,
`cp200`, and `cp255`) are supplemented here with `cp90` and `cp170` under the
same frozen contract:

- one manifest with six unique prompts and two additional greedy decodes;
- identical prompt, case, seed, tokenizer, 4-bit, SDPA, and 3,072-token
  generation settings;
- a hard gate requiring finite output, EOS and contract-valid rates of 1.0,
  cap, periodic-tail, and catastrophic-case rates of 0.0, mean 4-gram
  repetition increase over chk0 no greater than 0.10, and every case increase
  no greater than 0.20;
- validation loss is the ranking metric after the hard gate is passed.

The two new probes are retrospective robustness checks. They do not make the
historical selection preregistered.

## Results

All six checkpoints completed all eight cases, had EOS and contract-valid
rates of 1.0, and had zero capped, periodic-tail, or catastrophic cases. All six
also passed the frozen chk0-relative repetition thresholds.

| checkpoint | epoch | validation loss | loss delta vs cp200 | validation mean token accuracy | mean 4-gram repetition | mean delta vs chk0 | maximum case delta vs chk0 | hard gate | interpretation |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---|
| 80 | 0.945 | 1.769117 | +0.083018 | 0.594792 | 0.121932 | +0.039213 | +0.171121 | pass | Stable generation, but early in the fitting trajectory |
| 90 | 1.059 | 1.754564 | +0.068465 | 0.595915 | 0.114604 | +0.031885 | +0.133506 | pass | One-epoch neighbourhood remains materially above the loss plateau |
| 150 | 1.768 | 1.698782 | +0.012683 | 0.601087 | 0.136407 | +0.053688 | +0.118595 | pass | Still improving |
| 170 | 2.000 | 1.688037 | +0.001938 | 0.602320 | 0.160549 | +0.077830 | +0.172363 | pass | Near the plateau, with higher repetition than cp200 |
| **200** | **2.355** | **1.686099** | **0.000000** | **0.602685** | **0.134664** | **+0.051945** | **+0.144158** | **pass** | **Selected: lowest recorded validation loss and no behaviour-gate failure** |
| 255 | 3.000 | 1.686131 | +0.000032 | not recorded | 0.140996 | +0.058276 | +0.171121 | pass | No additional validation-loss benefit after 55 more steps |

`checkpoint-200` has the lowest validation loss among all 26 recorded
evaluations in the run, not only among the six generation-probed points.
Training through `checkpoint-255` adds 55 optimizer steps while changing the
loss by only `+0.00003242` and increasing mean 4-gram repetition from
`0.134664` to `0.140996`. The early checkpoints are behaviourally valid but
have appreciably higher validation loss. This makes `checkpoint-200` a
reasonable loss/stability/compute trade-off.

## What this evidence does not establish

- The eight generation cases comprise six unique prompts plus two greedy
  repeats, so they are a diagnostic gate rather than an inferential test.
- The probe includes examples from the model-selection data and therefore is
  not an untouched final test set.
- The run uses one training seed. The tiny loss differences within the late
  plateau do not establish statistically significant superiority over nearby
  checkpoints.
- The probe tests termination, output-contract validity, and degeneration; it
  does not by itself establish factual grounding, directional correctness, or
  human preference.

For a stronger publication claim, compare a small representative sweep
(`cp90`, `cp170`, `cp200`, and `cp255`) on a sealed meeting-level holdout with
paired meeting-cluster uncertainty. That would test generalization; it is not
needed merely to justify `cp200` as the internal checkpoint-selection choice.

## Suggested paper wording

> Checkpoint 200 was selected from the clean-v2 learning-rate 1e-6 trajectory
> using a two-stage rule: candidates first had to pass a fixed generation
> degeneration screen, after which validation loss was used for ranking. In a
> six-point retrospective sweep, every checkpoint passed the screen, while
> checkpoint 200 achieved the lowest validation loss among all 26 recorded
> evaluations (1.686099 at epoch 2.355). Continuing to checkpoint 255 changed
> validation loss by only +0.000032 and modestly increased mean 4-gram
> repetition (0.1347 to 0.1410). We therefore treat checkpoint 200 as a
> defensible representative of the converged plateau rather than as a
> statistically unique optimum.

## Evidence files

- Frozen sample manifest:
  `docs/summary/20260809T234411Z/chk1_sft_degeneration_samples.json`
  (`sha256=3675e72983f6f235015b80460c1895f662bb8dbfe3fdd28bb1ac4634ecf65246`)
- Original trajectory summary:
  `docs/summary/20260810T123000Z/chk1_checkpoint_selection/loss_snapshot.jsonl`
- New cp90 result and comparison:
  `chk1_lr1e6_cp90_probe/summary.json`,
  `chk1_lr1e6_cp90_vs_chk0_comparison.json`
- New cp170 result and comparison:
  `chk1_lr1e6_cp170_probe/summary.json`,
  `chk1_lr1e6_cp170_vs_chk0_comparison.json`
- The six result files share the same ordered case-key digest:
  `362bf0b582817069ec8ec5e126f880d6dd9b6f99f52411eab323d46fb7a110d6`.

