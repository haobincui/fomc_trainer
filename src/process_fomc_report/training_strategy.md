# Prompt And Training Strategy

This document describes the active processed-data workflow only.

## Directory Contract

- raw source files stay in `dataset/raw_data/`
- reusable derived inputs stay in `dataset/processed/input_sources/`
- prompt-pipeline intermediates stay in `dataset/processed/pipeline/`
- canonical trainable datasets stay in `dataset/processed/train/`
- split manifests stay in `dataset/processed/manifests/`
- training artifacts stay in `output/training/main/`
- evaluation artifacts stay in `output/evaluation/main/`

## Prompt Pipeline Stages

```text
normalize_input_sources
label_html
merge_labels
analysis_sft_teacher_prompts
analysis_sft_teacher_responses
analysis_sft_prompts
analysis_sft_dataset
analysis_grpo_prompts
analysis_grpo_dataset
minutes_alignment_prompts
minutes_alignment_teacher_responses
minutes_alignment_dataset
decision_prompts
decision_dataset
```

## Recommended Data Build Order

Run the four checkpoint shells in order:

```bash
./run/generate_input/chk1.sh
./run/generate_input/chk2.sh
./run/generate_input/chk3.sh
./run/generate_input/chk4.sh
```

Behavior notes:

- `chk1.sh` builds `analysis_sft` inputs, prompts, teacher outputs, and train datasets; it auto-runs `label_html.sh` if `dataset/processed/pipeline/analysis_sft/labeled/after_2009` is missing or empty
- `chk2.sh` only builds `analysis_grpo` artifacts and expects completed `analysis_sft` outputs
- `chk3.sh` builds `minutes_alignment` prompts, rewrite teacher responses, and final train jsonl; it expects completed `analysis_sft` teacher artifacts
- `chk4.sh` only builds `decision_sft` and `decision_grpo` artifacts

`configs/main/prompt_pipeline.yaml` controls response serialization via `response_template`.

## Low-Level Stage Wrappers

Use the Python module entrypoint directly for debugging or partial rebuilds:

```bash
python -m process_fomc_report.generate_prompt_and_response.run_generate_prompt_pipeline --help
python -m process_fomc_report.generate_prompt_and_response.run_generate_prompt_pipeline \
  --config configs/main/prompt_pipeline.yaml \
  --scope after_2009 \
  --profile compat \
  --stage normalize_input_sources
```

When you need lower-level CLI flags such as `--input-file`, `--build`, or `--dataset-kind`, call the Python modules directly:

```bash
python -m process_fomc_report.generate_prompt_and_response.algo.label_minutes_sections --input-file path/to/file.xlsx
python -m process_fomc_report.generate_prompt_and_response.algo.build_minutes_rewrite_dataset --build prompts
python -m process_fomc_report.generate_prompt_and_response.algo.assemble_analysis_training_data --dataset-kind analysis_grpo --profile strict
```

## Training Order

```text
Backbone-0
-> analysis_sft
-> analysis_grpo
-> minutes_alignment_sft
-> decision_sft
-> decision_grpo
```

Train and merge with:

```bash
python -m jobs.main.run_pipeline train analysis_sft
python -m jobs.main.run_pipeline merge analysis_sft
```

Repeat for the remaining stages.

## Notes

- prompt builders write train-ready jsonl directly into `dataset/processed/train/`.
- `jobs.main.build_datasets` and `jobs.main.audit_splits` own the canonical split manifests under `dataset/processed/manifests/`.
- all historical prompt runs and legacy datasets are archived under `archive/`.
