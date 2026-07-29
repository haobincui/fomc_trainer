# `process_fomc_report`

This package contains the active prompt-generation pipeline and the QA master rebuild utility.

Active files:

- `build_qa_master.py`
- `training_strategy.md`
- `generate_prompt_and_response/run_generate_prompt_pipeline.py`
- `generate_prompt_and_response/algo/`
- `generate_prompt_and_response/templates/`
- `generate_prompt_and_response/input_data/`

Historical downloads, old outputs, and legacy code were moved to `archive/`.

## Main Commands

Run the four compatibility entrypoints in order:

```bash
./run/generate_input/chk1.sh
./run/generate_input/chk2.sh
./run/generate_input/chk3.sh
./run/generate_input/chk4.sh
```

The active response serialization mode is controlled by `response_template` in `configs/main/prompt_pipeline.yaml`.

## Shell Wrappers

`run/generate_input/` now only keeps the core shell entrypoints:

- public scripts: `chk1.sh`, `chk2.sh`, `chk3.sh`, `chk4.sh`
- internal helper: `_run_common.sh`

High-level behavior:

- `chk1.sh` drives the `analysis_sft` data build, including QA master export, input normalization, labeling, prompts, teacher responses, and train jsonl output
- `chk2.sh` drives the `analysis_grpo` data build and requires completed `analysis_sft` outputs
- `chk3.sh` drives the `minutes_alignment` data build, including rewrite prompts, rewrite teacher responses, and final train jsonl; it requires completed `analysis_sft` teacher artifacts
- `chk4.sh` drives the `decision` data build and writes both `decision_sft` and `decision_grpo` train jsonl outputs

The shell entrypoints activate `FOMC_TRAINER_CONDA_ENV` and default to `configs/main/prompt_pipeline.yaml`. Override via:

```bash
FOMC_INPUT_CONFIG=configs/main/prompt_pipeline.yaml \
FOMC_INPUT_SCOPE=after_2009 \
FOMC_INPUT_PROFILE=compat \
./run/generate_input/chk4.sh
```

For lower-level debugging, call the Python modules directly instead of extra shell wrappers.

## Active Output Roots

- `dataset/processed/input_sources/`
- `dataset/processed/pipeline/analysis_sft/`
- `dataset/processed/pipeline/analysis_grpo/`
- `dataset/processed/pipeline/minutes_alignment/`
- `dataset/processed/pipeline/decision/`
- `dataset/processed/pipeline/audit/`
- `dataset/processed/train/`
- `dataset/processed/manifests/`

## Input Policy

- `dataset/raw_data/` only stores upstream raw data.
- reusable derived inputs live under `dataset/processed/input_sources/`.
- prompt pipeline audits and intermediates live under `dataset/processed/pipeline/`.
