# chk0 / chk1 / chk3 checkpoint-250 generation comparison

This directory records the evaluation-only handoff for the user-selected standalone chk3 checkpoint-250.

## Scope

- New generation leg: `eval-chk3-direct-chk1-cp200-sft-cp250`.
- Frozen baselines: the existing sealed chk0 and clean-v2 chk1 checkpoint-200 generations.
- Frozen test: 33 meeting-section rows from the Chapter 2 raw D-1 evidence-to-Minutes stress benchmark.
- Decoding: temperature 0, top-p 1, max 8,192 new tokens, max model length 24,576, deterministic per-sample seeds.
- Device: only physical GPU0 is used for the new generation leg.
- Intermediate output: every two rows are appended and fsynced under the run's `.partial` directory with an atomic progress state.

## Model lineage

The checkpoint-250 LoRA adapter was merged non-destructively into the selected chk1 checkpoint-200 merged parent. The exact verifier proved all 291 model tensors: 224 adapted tensors reconstructed exactly, 67 unchanged tensors remained exact, 448 adapter tensors were consumed, and mismatch count was zero.

## Claim boundary

This is an out-of-task end-to-end stress comparison. The chk3 branch was trained on analysis-to-Minutes-style inputs, whereas this frozen benchmark supplies raw D-1 evidence directly. The direct chk1-to-chk3 branch is standalone and non-promotable under the canonical chk2-to-chk3 DAG.

Existing chk0/chk1 generations are reused only through their original sealed, progress-bound manifests. They are not copied or rebound to the chk3 single-leg checkpoint manifest.

## Primary evidence

- `authorization.json`
- `chk3_cp250_exact_merge_lineage.json`
- `chk3_checkpoint_manifest.json`
- `three_leg_lineage_manifest.json`
- generation log and `.partial` state under `output/evaluation/main/checkpoint_generation/retrain_v2_chk0_chk1_chk3_cp250_20260810_v1`
