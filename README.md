# FOMC Trainer

The current provenance-controlled retraining workflow is `retrain_v2`; its executable
runbook is [`run/retrain_v2/README.md`](run/retrain_v2/README.md). The older `main`
workflow remains available for historical compatibility, but it is not the chk0→chk4
training entry point.

Core conventions:

- `dataset/raw_data/` contains only raw upstream data.
- `dataset/processed/main/` contains all derived inputs, prompts, intermediate artifacts, training sets, manifests, and evaluation inputs.
- `output/training/main/` contains training outputs.
- `output/evaluation/main/` contains evaluation outputs.
- `archive/` contains historical code, data, scripts, and documentation; it is not an active entry point.

## Repository Layout

```text
fomc_trainer/
├── src/
│   ├── open_r1/
│   ├── create_prompt/
│   └── process_fomc_report/
├── jobs/
│   ├── main/
│   ├── train/
│   ├── eval/
│   ├── generation/
│   └── models/
├── configs/
│   └── main/
├── dataset/
│   ├── raw_data/
│   └── processed/main/
├── output/
│   ├── training/main/
│   └── evaluation/main/
├── metadata/
│   └── main/
└── archive/
```

Active configuration files:

- `configs/main/analysis_sft.yaml`
- `configs/main/analysis_grpo.yaml`
- `configs/main/minutes_alignment_sft.yaml`
- `configs/main/decision_sft.yaml`
- `configs/main/decision_grpo.yaml`
- `configs/main/prompt_pipeline.yaml`

## Environment Setup

```bash
run/setup_retrain_v2_envs.sh --train --skip-checks
run/check_retrain_v2_envs.sh --train --skip-gpu
```

Retrain-v2 training and data code always runs in the `fomc_trainer` conda environment.
The setup command pins Python 3.10.9 and the training lock, removes incompatible legacy
packages, and writes the exact environment freeze. Do not install this repository with
the legacy editable dependency set and do not install the old FlashAttention 2.5.6 pin.
Qwen3.5-9B serving remains isolated in `fomc_judge_v2`.

After installation, verify the retraining entry points:

```bash
conda run -n fomc_trainer python -m jobs.retrain_v2.chk1.canonical_workflow --help
conda run -n fomc_trainer python -m jobs.retrain_v2.build_base_release --help
run/retrain_v2/pipeline.sh --help
run/retrain_v2/stage.sh --help
```

## Recommended Entry Points

Prefer these two entry points:

- Unified training and evaluation entry point: `python -m jobs.main.run_pipeline`
- Data-generation entry points: `./run/generate_input/chk1.sh` through `./run/generate_input/chk4.sh`

## Data Path Conventions

Frequently used active paths:

- QA master data: `dataset/processed/main/pipeline/final/qa/`
- Rewrite data: `dataset/processed/main/pipeline/final/rewrite/`
- Decision data: `dataset/processed/main/pipeline/final/decision/`
- Canonical training sets: `dataset/processed/main/datasets/`
- Split manifests: `dataset/processed/main/manifests/`
- Prompt and evaluation inputs: `dataset/processed/main/evaluation_inputs/`
- Adapter outputs: `output/training/main/adapters/`
- Merged-model outputs: `output/training/main/merged/`
- Evaluation outputs: `output/evaluation/main/`

## Running the Main Workflow

### 1. Run the input-building compatibility scripts

Run them in order:

```bash
./run/generate_input/chk1.sh
./run/generate_input/chk2.sh
./run/generate_input/chk3.sh
./run/generate_input/chk4.sh
```

Behavior:

- `chk1.sh` rebuilds the QA master, normalizes input sources, and automatically runs `label_html` if `dataset/processed/main/pipeline/labeled/after_2009` is missing or empty.
- `chk2.sh` generates only the `analysis_grpo` data and requires `chk1` to have completed.
- `chk3.sh` generates only the `minutes_alignment` data and requires `chk1` to have completed.
- `chk4.sh` generates only the `decision` data and synchronizes `dataset/processed/main/datasets/` at the end.

The `chkN.sh` names identify data-building stages. Their execution
dependencies do not define model-checkpoint ancestry. In the intended
analysis-to-Minutes design, model weights follow
`chk-0 -> chk-1 -> chk-2 -> chk-3`: `chk-2` is GRPO initialized from
`chk-1`, and `chk-3` is Minutes SFT initialized from `chk-2`.

Defaults:

- `FOMC_INPUT_CONFIG=configs/main/prompt_pipeline.yaml`
- `FOMC_INPUT_SCOPE=after_2009`
- `FOMC_INPUT_PROFILE=compat`

Example:

```bash
FOMC_INPUT_PROFILE=strict ./run/generate_input/chk1.sh
```

### 2. Low-level prompt-pipeline CLI

For debugging or running one stage only, invoke the Python module directly:

```bash
python -m process_fomc_report.generate_prompt_and_response.run_generate_prompt_pipeline --help
python -m process_fomc_report.generate_prompt_and_response.run_generate_prompt_pipeline \
  --config configs/main/prompt_pipeline.yaml \
  --scope after_2009 \
  --profile compat \
  --stage chk3_prompts
```

### 3. Additional canonical dataset and split CLI

The lower-level `jobs.main` commands remain available:

```bash
python -m jobs.main.run_pipeline build-datasets
python -m jobs.main.run_pipeline audit
```

## Training CLI

### Operational entry point for the active configurations

The active configuration set preserves historical executable forks and does
not, as currently written, instantiate every edge of the intended linear
`chk-0 -> chk-1 -> chk-2 -> chk-3` design. In particular, the retained
Minutes-SFT configurations initialize from analysis SFT. The
provenance-controlled four-artifact evaluation uses frozen archived artifacts
and does not run any of these training commands.

Train one stage:

```bash
python -m jobs.main.run_pipeline train analysis_sft
python -m jobs.main.run_pipeline train analysis_grpo
python -m jobs.main.run_pipeline train minutes_alignment_sft
python -m jobs.main.run_pipeline train decision_sft
python -m jobs.main.run_pipeline train decision_grpo
```

Merge one stage:

```bash
python -m jobs.main.run_pipeline merge analysis_sft
python -m jobs.main.run_pipeline merge analysis_grpo
python -m jobs.main.run_pipeline merge minutes_alignment_sft
python -m jobs.main.run_pipeline merge decision_sft
python -m jobs.main.run_pipeline merge decision_grpo
```

Train or merge all stages:

```bash
python -m jobs.main.run_pipeline train all
python -m jobs.main.run_pipeline merge all
```

Preview the commands without executing them:

```bash
python -m jobs.main.run_pipeline train analysis_sft --dry-run
python -m jobs.main.run_pipeline merge analysis_sft --dry-run
```

### Low-level training entry points

SFT:

```bash
accelerate launch --config_file configs/accelerate/zero3.yaml \
  -m jobs.train.train_sft \
  --config configs/main/analysis_sft.yaml
```

GRPO:

```bash
accelerate launch --config_file configs/accelerate/zero2.yaml \
  -m jobs.train.train_grpo \
  --config configs/main/analysis_grpo.yaml
```

### Background scripts for `chk2`

`chk2` is split into two fixed steps:

1. Start the local Gemma 12B judge on GPU `0`.
2. Start `analysis_grpo` training on GPU `1`.

Standard startup order:

```bash
./run/judge_chk2.sh
./run/chk2.sh
```

Default judge configuration:

```bash
configs/main/judge_chk2.yaml
```

Defaults:

- Judge endpoint: `http://127.0.0.1:8000/v1/chat/completions`
- Judge model: `models/gemma-3-12b-it`
- Judge logs: `logs/judge/`
- Training logs: `logs/train/`

Check whether the judge is running:

```bash
curl http://127.0.0.1:8000/v1/models
```

Merge directly:

```bash
python -m jobs.merge_model --config configs/main/analysis_sft.yaml
```

Alternatively, specify the paths manually:

```bash
python -m jobs.merge_model \
  --base-model models/DeepSeek-R1-Distill-Llama-8B \
  --adapter-path output/training/main/adapters/analysis_sft \
  --merged-path output/training/main/merged/analysis_sft
```

## Evaluation CLI

### 1. Generate held-out minutes

Recommended entry point:

```bash
python -m jobs.main.run_pipeline generate-minutes \
  --model output/training/main/merged/minutes_alignment_sft \
  --input dataset/processed/main/datasets/minutes_alignment/test.jsonl \
  --output-dir output/evaluation/main/generated_minutes \
  --start-index 0 \
  --end-index 10
```

Low-level entry point:

```bash
python -m jobs.generation.synthetic_generation stage2-full \
  --model output/training/main/merged/minutes_alignment_sft \
  --input dataset/processed/main/datasets/minutes_alignment/test.jsonl \
  --output-dir output/evaluation/main/generated_minutes \
  --start-index 0 \
  --end-index 10
```

### 2. Text-similarity evaluation

Recommended entry point:

```bash
python -m jobs.main.run_pipeline eval-text-similarity \
  --baseline-file output/evaluation/main/generated_minutes/run_a.jsonl \
  --aligned-file output/evaluation/main/generated_minutes/run_b.jsonl
```

Low-level entry point:

```bash
python -m jobs.main.eval_text_similarity \
  --baseline-file output/evaluation/main/generated_minutes/run_a.jsonl \
  --aligned-file output/evaluation/main/generated_minutes/run_b.jsonl \
  --embedding-model-path output/training/main/merged/analysis_sft \
  --bertscore-model bert-base-uncased \
  --output-json output/evaluation/main/text_similarity.json
```

### 3. Leave-one-out masking

The primary metric is the paired signed delta:

```text
delta = cos(full_output, target) - cos(masked_output, target)
      = masked_distance - full_distance
```

The new generation-only canonical workflow neither trains nor merges models. It
uses the API-key-free ALFRED Graph CSV endpoint to freeze one previous-day
information set for each meeting:

```text
information_as_of_date = requested_vintage_date
                       = meeting_date - 1 calendar day
observation_date <= information_as_of_date
```

All same-day data are excluded; the workflow does not distinguish among 08:30,
13:59, and 14:00 releases. “Previous-day data” means the historical vintage
snapshot available as of D−1; it does not require every observation to have
occurred exactly on D−1. The source registry, raw CSV files, request parameters
and hashes, the `13 × 26 = 338`-row ledger, generation artifacts, and final
release manifest form a replayable provenance chain. The complete background
entry point is:

```bash
./run/generate_loo_end_to_end.sh all
```

By default, this script starts in the background with `nohup` and immediately
prints the PID, log path, and workflow directory. Replace `all` with `pilot` or
`formal` as needed, or set `LOO_FOREGROUND=1` for debugging. Before launch, it
resolves physical GPU `1` with `nvidia-smi`, exposes numeric device `1` to
every background worker, and verifies that logical `cuda:0` has GPU `1`'s
recorded UUID. Numeric visibility is required by the pinned vLLM runtime. A
caller-supplied `CUDA_VISIBLE_DEVICES` cannot override this canonical policy.
It also checks PyTorch, Transformers, and vLLM. On the current host,
if `LOO_PYTHON` is unset, it uses the available
`~/.conda/envs/llama_factory` environment to avoid the incomplete PyTorch
namespace in the base Python installation.

Before the first Minutes prompt is generated, the inner launcher verifies the
complete runtime model and tokenizer directory trees against the sealed,
externally pinned checkpoint provenance manifest. It requires the
`eval-minutes-sft-from-chk1` artifact to be marked usable, checks both manifest
digests and both runtime artifact digests, and requires the recovered model to
differ from the historical overwrite reference. A merely existing directory
is not sufficient. The legacy `llama_sft_synthetic_20250526` directory and any
unregistered `LOO_MINUTES_MODEL` or `LOO_MINUTES_TOKENIZER` override fail before
generation.

The workflow neither reads nor requires `FRED_API_KEY`. Network concurrency is
limited to 1–2 requests. The 26 vintages are fetched per series in
`12 + 12 + 2` request batches, with strict validation of response headers,
vintage dates, the cutoff date for non-null values, and raw-file hashes. The
legacy `synthetic_text` analysis contains material whose D−1 availability
cannot be established; it is used only for source mapping, and none of its
values or text are reused. See
`docs/summary/20260728T091012Z/canonical_loo_generation_implementation.md`
for the complete input schemas, populations, seeds, directories, and claim
boundaries. After all four generation cells finish, the launcher validates
every file, requires the same-seed full baselines from the deletion and neutral
arms to match exactly, and only then writes the population and workflow release
manifests.

Compute may be split by intervention indicator without changing the full
baseline. In `pilot` mode, set `LOO_INTERVENTION_INDICATORS` to a
comma-separated subset of canonical IDs. Analysis and the full Minutes prompt
still retain all 26 indicator blocks; only the requested deletion and neutral
cells are generated, and `None` is included automatically. The result is a
`partial_complete` `intervention_shard_manifest.json`, not a standalone
canonical release. Legacy seven-indicator outputs cannot fill omitted cells
because their populations, prompts, and decoding policies differ.

Minutes generation remains fail-closed by default. For an operational partial
shard, `LOO_MINUTES_TOKEN_LIMIT_POLICY=exclude` retains any token-limit row as
raw provenance, marks it `excluded_token_limit_finish`, and continues with
later artifacts. These rows are never eligible for LOO scoring. A shard with
such exclusions is sealed as
`partial_complete_with_generation_exclusions`, explicitly marked
non-canonical, and reports every affected sample and effective denominator.

Indicator analysis uses a frozen 8,192-token technical generation limit while
its dedicated system prompt requests a compact answer within 4,096 tokens. The
raw completion is always retained unchanged. Before Minutes prompt
construction, a deterministic projection requires exactly one `</think>`
delimiter, non-empty reasoning before it, and a non-empty final answer after
it. Only that final answer becomes the `minutes_analysis` evidence field; no
second model, summarization, fallback, or truncation is allowed. The projection
manifest binds every raw row, extracted answer, tokenizer token count, and
output hash. The prompt builder then applies the real Minutes chat template and
requires `prompt_tokens + 8,192 <= 16,384` for every full, deletion, and neutral
row before writing any prompt artifacts.

At most two otherwise valid technical token-limit finishes may be retained in
raw indicator analysis, with each exception recorded by sample ID; a third
exception fails the stage. This bounded exception applies only to indicator
analysis. Input truncation, malformed projections, empty or missing rows,
context overflow and unknown finish reasons remain fatal. Minutes token-limit
finishes are also fatal under the default policy; the explicit partial-shard
exclusion policy described above records and excludes them without weakening
any other safety check.

The input directory must contain `None_masked_*.jsonl` files (full prompts) and
indicator-specific `*_masked_*.jsonl` files for the same sample set. Each row
should retain `sample_id`; legacy files must at least provide `meeting_date`,
`section_name`, and a non-overwritten `source_index/index`. Canonical generation
also requires a frozen intervention roster. The default
`configs/main/leave_one_out_roster.json` defines the 26 indicators and the
`after_2009` context used in the historical experiment. Before rebuilding
prompts, confirm that this universe still matches the current research design.

The generator requires each masked prompt to be derivable from its matching full
prompt through exactly one contiguous block deletion, and all
indicator × context cells must have identical sample-key sets. The roster's
`indicator_markers` also verify that the first semantic line of the deleted
block identifies the corresponding indicator. Deletion boundaries must cover
complete lines or blocks, indicator deletion spans must not overlap, and two
indicators must not produce the same masked prompt for a sample. This repository
does not retain the historical masking prompts or
`dataset/processed/main/manifests/analysis_minutes_split.json`; therefore, run
the following filter command only after both the prompts and frozen split
manifest have been recovered or rebuilt:

```bash
python -m jobs.main.run_pipeline filter-mask-prompts \
  --input-folder dataset/processed/main/evaluation_inputs/source_prompts/mask_indicator/after_2009 \
  --output-folder dataset/processed/main/evaluation_inputs/mask_prompts_test \
  --split test
```

Then generate the masking outputs:

```bash
python -m jobs.main.run_pipeline generate-masking \
  --model output/training/main/merged/minutes_alignment_sft \
  --input-folder dataset/processed/main/evaluation_inputs/mask_prompts_test \
  --output-dir output/evaluation/main/leave_one_out_masking/generated \
  --roster-file configs/main/leave_one_out_roster.json \
  --simulation-step 1 \
  --seed 20260728 \
  --temperature 0 \
  --top-p 1
```

Equivalent low-level entry point:

```bash
python -m jobs.generation.mask_generation \
  --input-folder dataset/processed/main/evaluation_inputs/mask_prompts_test \
  --model output/training/main/merged/minutes_alignment_sft \
  --roster-file configs/main/leave_one_out_roster.json \
  --simulation-step 1 \
  --seed 20260728 \
  --temperature 0 \
  --top-p 1 \
  --output-dir output/evaluation/main/leave_one_out_masking/generated
```

New output files record `replicate_id`; a per-row seed derived from
`replicate_seed + sample_id` and independent of batch, order, and indicator;
model path; decoding parameters; original `source_index`; generation position;
input-file SHA-256; and expected row count. They also produce
`intervention_manifest.json` and `generation_manifest.json`. The former stores
the frozen indicator/context universe, per-row full and masked prompt hashes,
the deleted-block hash, indicator markers, and block-boundary evidence. The
latter references and validates the intervention manifest while inventorying
the complete generation output.

The generation model must be a locally readable file or directory. The
generator computes a SHA-256 inventory over the complete checkpoint, tokenizer,
and configuration tree. Every output row also stores its source-prompt hash;
resume and evaluation operations bind `row key + prompt + meeting + section`
back to the intervention manifest row by row. Existing files are reused only
when the input hashes, complete row set, model, tokenizer, batch size, all
decoding parameters, and masking strategy match. Partial or stale outputs are
never silently treated as complete.

The evaluator validates these artifacts against the generation manifest, but
the Chapter 2 release manifest must itself record the manifest's SHA-256 to
provide an external frozen anchor. For stochastic-decoding robustness analysis,
increase `--simulation-step` and set `--temperature` explicitly. Full and
masked files must share the same replicate, row seed, generation position, and
decoding configuration. A shared seed is reproducibility metadata; by itself,
it does not prove that the underlying inference engine used an identical random
number stream for two different prompts.

To measure internal output sensitivity, use the paired full output as the
target:

```bash
python -m jobs.main.run_pipeline eval-masking synthetic \
  --input-folder output/evaluation/main/leave_one_out_masking/generated \
  --output-file output/evaluation/main/leave_one_out_masking/internal_summary.jsonl \
  --embedding-model-path models/all-mpnet-base-v2-pinned
```

To measure the external alignment contribution, use the same frozen actual
Minutes as the target:

```bash
python -m jobs.main.run_pipeline eval-masking actual \
  --input-folder output/evaluation/main/leave_one_out_masking/generated \
  --reference-file path/to/frozen_actual_minutes.jsonl \
  --reference-text-field response \
  --output-file output/evaluation/main/leave_one_out_masking/external_summary.jsonl \
  --embedding-model-path models/all-mpnet-base-v2-pinned
```

By default, actual-Minutes records are uniquely matched on `meeting_date` and
`section_name`. If the fields differ, pass `--reference-key` multiple times. If
a section is intentionally split across multiple rows, explicitly use
`--reference-duplicate-policy concatenate`. The retained historical candidate
file
`archive/data/dataset_20260421/raw_data/archive/synthetic_text_20250518.jsonl`
uses the `reference` field. A read-only audit found exact agreement in meeting,
section, and actual-Minutes text for all 3,481 observed records shared with
`../synthetic_text/synthetic_text/output/synthetic_text_reason/`
`synthetic_text_20250520_reason.jsonl`. The candidate contains 697 source
records, whereas the legacy file covers 696, omits source index `153`, and
duplicates replicate ID `0-1`. The candidate may therefore serve as the
actual-Minutes reference for a new experiment, but it cannot be described as
the byte-identical frozen input to the legacy LOO run. Before using it in the
main chapter, record its path, SHA-256
`0edc4c44ce13870933a461d3507cc24c40e792667e54bda97a0150b1ebc3ac06`,
schema, and 697-row universe in the Chapter 2 release manifest, and pass
`--reference-text-field reference`.

The corresponding low-level entry point is:

```bash
python -m jobs.eval.eval_leave_one_out \
  --input-folder output/evaluation/main/leave_one_out_masking/generated \
  --target-mode actual-minutes \
  --reference-file archive/data/dataset_20260421/raw_data/archive/synthetic_text_20250518.jsonl \
  --reference-text-field reference \
  --output-file output/evaluation/main/leave_one_out_masking/external_summary.jsonl \
  --embedding-model-path models/all-mpnet-base-v2-pinned
```

In addition to the summary, the evaluator automatically writes `.rows.jsonl`,
`.exclusions.jsonl`, and `.audit.json`. Canonical runs require and validate the
generation manifest's file set, row counts, and SHA-256 hashes by default. Use
`--allow-missing-generation-manifest` only for explicit audits of legacy files.
Statistical summaries first average samples within `meeting × replicate`, then
give replicates equal weight when aggregating to the meeting level. They use
meetings as clusters for bootstrap inference and report Holm-adjusted p-values
over all indicator–section–context cells in one run. The historical
`jobs.eval.eval_mask test1`–`test4` commands are legacy-only rescoring or
aggregation entry points and must not be used for new canonical LOO results.

The embedding model must also be stored as a local immutable snapshot. The
evaluator computes an inventory SHA-256 over the complete model directory and
records it in every scored row and audit. Optionally use
`--embedding-model-sha256 <expected-sha256>` to verify the expected hash. A
mutable Hub alias, such as a model name without a pinned revision or snapshot,
is not admissible for canonical evaluation.

The paper's historical descriptive tables can be mechanically reconstructed
from the retained aggregate workbooks:

```bash
python -m jobs.eval.recover_legacy_loo \
  --input-xlsx archive/code/reformat_shapley_result/shapley_result_with_diff.xlsx \
  --output-csv docs/Chapter2/Chapter2Results/loo_external_legacy_descriptive.csv \
  --output-tex docs/Chapter2/Chapter2Results/loo_external_legacy_descriptive.tex \
  --internal-input-xlsx archive/code/reformat_shapley_result/shapley_filter.xlsx \
  --internal-output-csv docs/Chapter2/Chapter2Results/loo_internal_legacy_descriptive.csv \
  --internal-output-tex docs/Chapter2/Chapter2Results/loo_internal_legacy_descriptive.tex
```

The sibling repository `../synthetic_text/synthetic_text` retains 2,660
row-level full/masked pairs from `ft_20250330`, but this is only a seven-indicator
pilot and cannot replace the chapter's 25/26-indicator results. The legacy
script used stochastic decoding without a seed and also rewrote the prompt
template during masking. None of the 2,660 generated pairs satisfies the
canonical intervention requirement that the masked prompt differ from the full
prompt only by deletion of one contiguous indicator block. For Outlook and
Policy Action samples, all seven indicators even use identical masked prompts.
The stored `cos` value is only `cos(full_output, masked_output)`.

These rows may be used only in a separately labeled legacy prompt-perturbation
diagnostic or provenance audit. They must not be combined with the main chapter
table or used for formal significance inference. For complete file-level
findings and hashes, see
`docs/summary/20260728T091012Z/leave_one_out_output_reuse_audit.md`
and its machine-readable inventory.

For an actual-Minutes target-relative rescore of this seven-indicator pilot,
reuse the existing full and masked outputs; no text regeneration is required.
The background entry point is:

```bash
./run/eval_legacy_7_indicator_pilot.sh
```

This task always computes
`delta = cos(full_output, target) - cos(masked_output, target)`. The historical
generation actually used the `output` field from
`../synthetic_text/synthetic_text/data/training/input_qa.json` as the target.
`input_qa_tagged.json` is used only to recover meeting and section metadata and
cannot replace the scoring target. By default, rescoring uses the local
`models/DeepSeek-R1-Distill-Llama-8B` as a newly frozen encoder, with
`embedding batch size=1` and no truncation. Override these settings with
`EMBEDDING_MODEL_PATH`, `EMBEDDING_BATCH_SIZE`,
`LEGACY_7_INDICATOR_MAX_TOKENS`, and `CUDA_VISIBLE_DEVICES`.

The script prints its PID and writes logs under `logs/eval/`. Results are saved
to
`output/evaluation/main/legacy_7_indicator_pilot/actual_minutes_rescore/`.
Row-level results, indicator summaries, indicator-section summaries,
meeting-level results, five truncation exclusions, and the complete audit are
stored separately as CSV or JSONL files; `results.md` provides a directly
readable result table. Treat the files as a complete result set only when
`run_status.json` reports `mode=full_rescore` and `status=complete`. A rerun
invalidates the previous completion marker and score artifacts before starting.
Embeddings are cached by unique text, model fingerprint, and explicit scoring
implementation version, so an interrupted task can resume with the same
command. To validate only the inputs before launch:

```bash
LEGACY_7_INDICATOR_VALIDATE_ONLY=1 \
  ./run/eval_legacy_7_indicator_pilot.sh
```

### 4. Decision-baseline evaluation

Recommended entry point:

```bash
python -m jobs.main.run_pipeline eval-decision-baselines \
  --prediction-file output/evaluation/main/decision/backbone_predictions.jsonl \
  --prediction-file output/evaluation/main/decision/decision_grpo_predictions.jsonl
```

Low-level entry point:

```bash
python -m jobs.main.eval_decision_baselines \
  --dataset-root dataset/processed/main/datasets/decision_grpo \
  --market-baseline dataset/external/market_baselines/market_implied_baseline.csv \
  --prediction-file output/evaluation/main/decision/backbone_predictions.jsonl \
  --prediction-file output/evaluation/main/decision/decision_grpo_predictions.jsonl \
  --output-json output/evaluation/main/decision_baselines.json
```

### 5. Evaluate decision-generation results separately

```bash
python -m jobs.eval.eval_decision \
  --input output/evaluation/main/decision/decision_predictions.jsonl \
  --output output/evaluation/main/decision/decision_predictions_scored.jsonl \
  --rate-change-map dataset/processed/main/input_sources/rate_change_map.json
```

### 6. Normalize the market baseline

```bash
python -m jobs.main.fetch_market_baseline \
  --source-file path/to/market_baseline.csv \
  --output-file dataset/external/market_baselines/market_implied_baseline.csv \
  --coverage-report dataset/external/market_baselines/coverage_report.json \
  --reference-file dataset/processed/main/datasets/decision_grpo/test.jsonl
```

## Cleanup CLI

Inspect the inventory:

```bash
python -m jobs.main.cleanup_generated --inventory metadata/main/legacy_artifact_inventory.json
```

Execute the cleanup:

```bash
python -m jobs.main.run_pipeline cleanup-generated --execute
```

Or invoke the lower-level command directly:

```bash
python -m jobs.main.cleanup_generated \
  --inventory metadata/main/legacy_artifact_inventory.json \
  --execute
```

## Common Help Commands

```bash
python -m jobs.main.run_pipeline --help
python -m jobs.main.build_datasets --help
python -m jobs.main.audit_splits --help
python -m jobs.main.eval_text_similarity --help
python -m jobs.main.eval_decision_baselines --help
python -m jobs.main.fetch_market_baseline --help
python -m jobs.merge_model --help
python -m jobs.eval.eval_decision --help
python -m jobs.eval.eval_mask --help
python -m jobs.generation.synthetic_generation --help
python -m jobs.generation.data_to_analysis_generation --help
python -m jobs.generation.mask_generation --help
python -m process_fomc_report.build_qa_master --help
python -m process_fomc_report.generate_prompt_and_response.run_generate_prompt_pipeline --help
```

## Archive

- `archive/code/`: legacy code
- `archive/data/`: historical data and outputs
- `archive/configs/`: legacy configuration
- `archive/scripts/`: legacy shell entry points
- `archive/docs/`: historical plans and documentation

Active workflows must not write to `archive/`.




## Retrain-v2 setup and training

Do not use the former `train_llama`, `vllm_env`, `start_vllm.sh`, or `run_grpo.sh`
instructions for the rebuilt checkpoints. Use the locked environments and stage launchers
documented in [`run/retrain_v2/README.md`](run/retrain_v2/README.md).

## Config Files

Accelerate configs: configs/accelerate/*.yaml

SFT configs: configs/sft/sft_*.yaml
GRPO configs: configs/grpo/grpo_*.yaml


## Merged Model

# save model
```
source activate
```


```
<YOUR_ACCESS_TOKEN>
```
