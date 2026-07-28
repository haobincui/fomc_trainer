# Project Structure

## Active Tree

```text
src/
  open_r1/
  create_prompt/
  process_fomc_report/
jobs/
  main/
  train/
  eval/
  generation/
  models/
configs/
  main/
dataset/
  raw_data/
  processed/main/
output/
  training/main/
  evaluation/main/
metadata/
  main/
archive/
  code/
  configs/
  data/
  scripts/
  docs/
```

## Rules

- `dataset/raw_data/` only stores raw upstream files.
- `dataset/processed/main/` stores all derived inputs, prompts, datasets, and manifests.
- `output/training/` stores checkpoints, adapters, and merged models.
- `output/evaluation/` stores model outputs and evaluation reports.
- `archive/` stores deprecated assets and must remain read-only for active workflows.
