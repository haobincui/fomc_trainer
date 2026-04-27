# FOMC Trainer

`fomc_trainer` 现在只保留一条活跃主线：`main`。

核心约束：

- `dataset/raw_data/` 只放原始上游数据
- `dataset/processed/main/` 放所有派生输入、prompt、中间产物、训练集、manifests、评估输入
- `output/training/main/` 放训练输出
- `output/evaluation/main/` 放评估输出
- `archive/` 放历史代码、历史数据、旧脚本、旧文档，不再作为活跃入口

## 目录结构

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

当前活跃配置文件：

- `configs/main/analysis_sft.yaml`
- `configs/main/analysis_grpo.yaml`
- `configs/main/minutes_alignment_sft.yaml`
- `configs/main/decision_sft.yaml`
- `configs/main/decision_grpo.yaml`
- `configs/main/prompt_pipeline.yaml`

## 环境安装

```bash
conda create -n fomc_trainer python=3.10
conda activate fomc_trainer

pip install -r requirements.txt
pip install -e .
pip install flash-attn==2.5.6 --no-build-isolation
```

安装完成后，先确认几个主入口可见：

```bash
python -m jobs.main.run_pipeline --help
python -m jobs.main.build_datasets --help
python -m process_fomc_report.build_qa_master --help
python -m process_fomc_report.generate_prompt_and_response.run_generate_prompt_pipeline --help
```

## 推荐入口

推荐优先使用这两个入口：

- 训练与评估总入口：`python -m jobs.main.run_pipeline`
- 数据生成主入口：`./run/generate_input/chk1.sh` 到 `./run/generate_input/chk4.sh`

## 数据路径约定

常用活跃路径：

- QA 主数据：`dataset/processed/main/pipeline/final/qa/`
- rewrite 数据：`dataset/processed/main/pipeline/final/rewrite/`
- decision 数据：`dataset/processed/main/pipeline/final/decision/`
- canonical 训练集：`dataset/processed/main/datasets/`
- split manifests：`dataset/processed/main/manifests/`
- prompt/eval 输入：`dataset/processed/main/evaluation_inputs/`
- adapter 输出：`output/training/main/adapters/`
- merged model 输出：`output/training/main/merged/`
- 评估输出：`output/evaluation/main/`

## 一条主线怎么跑

### 1. 运行四个 checkpoint 数据脚本

按顺序执行：

```bash
./run/generate_input/chk1.sh
./run/generate_input/chk2.sh
./run/generate_input/chk3.sh
./run/generate_input/chk4.sh
```

行为约定：

- `chk1.sh` 会重建 QA master、标准化 input sources，并在 `dataset/processed/main/pipeline/labeled/after_2009` 缺失或为空时自动补跑 `label_html`
- `chk2.sh` 只生成 `analysis_grpo` 对应的数据，要求 `chk1` 已完成
- `chk3.sh` 只生成 `minutes_alignment` 对应的数据，要求 `chk1` 已完成
- `chk4.sh` 只生成 `decision` 对应的数据，并在最后同步 `dataset/processed/main/datasets/`

默认参数：

- `FOMC_INPUT_CONFIG=configs/main/prompt_pipeline.yaml`
- `FOMC_INPUT_SCOPE=after_2009`
- `FOMC_INPUT_PROFILE=compat`

例如：

```bash
FOMC_INPUT_PROFILE=strict ./run/generate_input/chk1.sh
```

### 2. 低层 prompt pipeline CLI

调试或只跑单个阶段时，直接使用 Python 模块入口：

```bash
python -m process_fomc_report.generate_prompt_and_response.run_generate_prompt_pipeline --help
python -m process_fomc_report.generate_prompt_and_response.run_generate_prompt_pipeline \
  --config configs/main/prompt_pipeline.yaml \
  --scope after_2009 \
  --profile compat \
  --stage chk3_prompts
```

### 3. 额外的 canonical dataset / split CLI

如果你要直接走 `jobs.main` 侧的底层命令，仍然保留：

```bash
python -m jobs.main.run_pipeline build-datasets
python -m jobs.main.run_pipeline audit
```

## 训练 CLI

### 推荐入口

单阶段训练：

```bash
python -m jobs.main.run_pipeline train analysis_sft
python -m jobs.main.run_pipeline train analysis_grpo
python -m jobs.main.run_pipeline train minutes_alignment_sft
python -m jobs.main.run_pipeline train decision_sft
python -m jobs.main.run_pipeline train decision_grpo
```

单阶段 merge：

```bash
python -m jobs.main.run_pipeline merge analysis_sft
python -m jobs.main.run_pipeline merge analysis_grpo
python -m jobs.main.run_pipeline merge minutes_alignment_sft
python -m jobs.main.run_pipeline merge decision_sft
python -m jobs.main.run_pipeline merge decision_grpo
```

全量训练或全量 merge：

```bash
python -m jobs.main.run_pipeline train all
python -m jobs.main.run_pipeline merge all
```

只看实际会执行的命令，不真正运行：

```bash
python -m jobs.main.run_pipeline train analysis_sft --dry-run
python -m jobs.main.run_pipeline merge analysis_sft --dry-run
```

### 底层训练入口

SFT：

```bash
accelerate launch --config_file configs/accelerate/zero3.yaml \
  -m jobs.train.train_sft \
  --config configs/main/analysis_sft.yaml
```

GRPO：

```bash
accelerate launch --config_file configs/accelerate/zero2.yaml \
  -m jobs.train.train_grpo \
  --config configs/main/analysis_grpo.yaml
```

### `chk2` 后台运行脚本

`chk2` 现在固定拆成两步：

1. 先在 GPU `0` 启动本地 Gemma 12B judge
2. 再在 GPU `1` 启动 `analysis_grpo` 训练

标准启动顺序：

```bash
./run/judge_chk2.sh
./run/chk2.sh
```

judge 默认配置文件：

```bash
configs/main/judge_chk2.yaml
```

约定如下：

- judge 服务地址：`http://127.0.0.1:8000/v1/chat/completions`
- judge 模型名：`models/gemma-3-12b-it`
- judge 日志目录：`logs/judge/`
- 训练日志目录：`logs/train/`

检查 judge 是否起来：

```bash
curl http://127.0.0.1:8000/v1/models
```

直接 merge：

```bash
python -m jobs.merge_model --config configs/main/analysis_sft.yaml
```

或者手动指定路径：

```bash
python -m jobs.merge_model \
  --base-model models/DeepSeek-R1-Distill-Llama-8B \
  --adapter-path output/training/main/adapters/analysis_sft \
  --merged-path output/training/main/merged/analysis_sft
```

## 评估 CLI

### 1. 生成 held-out minutes

推荐入口：

```bash
python -m jobs.main.run_pipeline generate-minutes \
  --model output/training/main/merged/minutes_alignment_sft \
  --input dataset/processed/main/datasets/minutes_alignment/test.jsonl \
  --output-dir output/evaluation/main/generated_minutes \
  --start-index 0 \
  --end-index 10
```

底层入口：

```bash
python -m jobs.generation.synthetic_generation stage2-full \
  --model output/training/main/merged/minutes_alignment_sft \
  --input dataset/processed/main/datasets/minutes_alignment/test.jsonl \
  --output-dir output/evaluation/main/generated_minutes \
  --start-index 0 \
  --end-index 10
```

### 2. 文本相似度评估

推荐入口：

```bash
python -m jobs.main.run_pipeline eval-text-similarity \
  --baseline-file output/evaluation/main/generated_minutes/run_a.jsonl \
  --aligned-file output/evaluation/main/generated_minutes/run_b.jsonl
```

底层入口：

```bash
python -m jobs.main.eval_text_similarity \
  --baseline-file output/evaluation/main/generated_minutes/run_a.jsonl \
  --aligned-file output/evaluation/main/generated_minutes/run_b.jsonl \
  --embedding-model-path output/training/main/merged/analysis_sft \
  --bertscore-model bert-base-uncased \
  --output-json output/evaluation/main/text_similarity.json
```

### 3. leave-one-out masking

先按 split 过滤 prompt：

```bash
python -m jobs.main.run_pipeline filter-mask-prompts \
  --input-folder dataset/processed/main/evaluation_inputs/source_prompts/mask_indicator/after_2009 \
  --output-folder dataset/processed/main/evaluation_inputs/mask_prompts_test \
  --split test
```

再生成 masking 输出：

```bash
python -m jobs.main.run_pipeline generate-masking \
  --model output/training/main/merged/minutes_alignment_sft \
  --input-folder dataset/processed/main/evaluation_inputs/mask_prompts_test \
  --output-dir output/evaluation/main/leave_one_out_masking/generated \
  --simulation-step 5
```

等价底层入口：

```bash
python -m jobs.generation.mask_generation \
  --input-folder dataset/processed/main/evaluation_inputs/mask_prompts_test \
  --model output/training/main/merged/minutes_alignment_sft \
  --simulation-step 5 \
  --output-dir output/evaluation/main/leave_one_out_masking/generated
```

计算 synthetic target：

```bash
python -m jobs.main.run_pipeline eval-masking synthetic \
  --input-folder output/evaluation/main/leave_one_out_masking/generated \
  --output-file output/evaluation/main/leave_one_out_masking/synthetic_target.jsonl
```

计算 actual target：

```bash
python -m jobs.main.run_pipeline eval-masking actual \
  --input-folder output/evaluation/main/leave_one_out_masking/generated \
  --output-file output/evaluation/main/leave_one_out_masking/actual_target.jsonl
```

对应底层入口：

```bash
python -m jobs.eval.eval_mask test3 \
  --input-folder output/evaluation/main/leave_one_out_masking/generated \
  --output-file output/evaluation/main/leave_one_out_masking/synthetic_target.jsonl

python -m jobs.eval.eval_mask test4 \
  --input-folder output/evaluation/main/leave_one_out_masking/generated \
  --output-file output/evaluation/main/leave_one_out_masking/actual_target.jsonl
```

### 4. decision baseline 评估

推荐入口：

```bash
python -m jobs.main.run_pipeline eval-decision-baselines \
  --prediction-file output/evaluation/main/decision/backbone_predictions.jsonl \
  --prediction-file output/evaluation/main/decision/decision_grpo_predictions.jsonl
```

底层入口：

```bash
python -m jobs.main.eval_decision_baselines \
  --dataset-root dataset/processed/main/datasets/decision_grpo \
  --market-baseline dataset/external/market_baselines/market_implied_baseline.csv \
  --prediction-file output/evaluation/main/decision/backbone_predictions.jsonl \
  --prediction-file output/evaluation/main/decision/decision_grpo_predictions.jsonl \
  --output-json output/evaluation/main/decision_baselines.json
```

### 5. 单独评估 decision 生成结果

```bash
python -m jobs.eval.eval_decision \
  --input output/evaluation/main/decision/decision_predictions.jsonl \
  --output output/evaluation/main/decision/decision_predictions_scored.jsonl \
  --rate-change-map dataset/processed/main/input_sources/rate_change_map.json
```

### 6. 市场基线标准化

```bash
python -m jobs.main.fetch_market_baseline \
  --source-file path/to/market_baseline.csv \
  --output-file dataset/external/market_baselines/market_implied_baseline.csv \
  --coverage-report dataset/external/market_baselines/coverage_report.json \
  --reference-file dataset/processed/main/datasets/decision_grpo/test.jsonl
```

## 清理 CLI

查看 inventory：

```bash
python -m jobs.main.cleanup_generated --inventory metadata/main/legacy_artifact_inventory.json
```

真正执行清理：

```bash
python -m jobs.main.run_pipeline cleanup-generated --execute
```

或者直接调用：

```bash
python -m jobs.main.cleanup_generated \
  --inventory metadata/main/legacy_artifact_inventory.json \
  --execute
```

## 常见帮助命令

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

## Archive 说明

- `archive/code/`：旧代码
- `archive/data/`：历史数据和历史输出
- `archive/configs/`：旧配置
- `archive/scripts/`：旧 shell 入口
- `archive/docs/`：旧规划和旧说明文档

活跃流程不要再写入 `archive/`。




## Setup
```bash
conda create -n train_llama python=3.10

# for mac and windows
# conda activate fomc_trainer

# for linux
source activate train_llama

pip install -e.[dev]
pip install flash-attn==2.5.6 --no-build-isolation
```


## setup for judge model (GRPO)
```bash
conda create -n vllm_env python=3.10
# for mac and windows
# conda activate vllm_env

# for linux
source activate vllm_env

pip install "bitsandbytes>=0.43.0"
pip install "peft>=0.14.0"
pip install vllm
```


## run training
```bash
# SFT
./run/train_llama/chk1.sh


# GRPO
## start the judge model
bash start_vllm.sh

## run training
bash run_grpo.sh
```

## Config Files

Accelerate configs: configs/accelerate/*.yaml

SFT configs: configs/sft/sft_*.yaml
GRPO configs: configs/grpo/grpo_*.yaml


## merged model

# save model
```
source activate 