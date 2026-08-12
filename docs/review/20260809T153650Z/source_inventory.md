# Chapter 2 审阅来源清单

审阅截止：2026-08-09 15:51 UTC  
用途：记录 `chapter2_current_implementation_review.md` 的事实源、优先级、测试和状态边界。

## 1. 文章文件

- `docs/Chapter2/chapter2.tex`
- `docs/Chapter2/sections/intro.tex`
- `docs/Chapter2/sections/literature_review.tex`
- `docs/Chapter2/sections/dataset_construction.tex`
- `docs/Chapter2/sections/model_training.tex`
- `docs/Chapter2/chapter2_outline.md`
- `docs/Chapter2/chapter2_review.md`
- `docs/Chapter2/archive/chapter2_revision_notes.md`

审阅时前四个主要 TeX 文件存在用户未提交修改。本次只读审阅，没有覆盖或整理这些修改。

## 2. 当前数据 release

### chk1/chk2 canonical base

- `dataset/processed/retrain_v2/analysis_base_full_v7_automated_v3_20260804/base_release_manifest.json`
- `dataset/processed/retrain_v2/analysis_base_full_v7_automated_v3_20260804/analysis_sft/{train,eval,test}.jsonl`
- `dataset/processed/retrain_v2/analysis_base_full_v7_automated_v3_20260804/analysis_grpo/{train,eval,test}.jsonl`
- `dataset/processed/retrain_v2/analysis_base_full_v7_automated_v3_20260804/audits/*.json`
- `docs/summary/20260728T091012Z/chapter2_sample_flow_ledger.md`
- `output/data/retrain_v2/chk1/canonical_releases/chk1_full_v7_automated_v2_20260804/audit/quality_report.json`
- `output/data/retrain_v2/chk1/generation_full_v7/generation_handoff.json`
- `output/data/retrain_v2/chk1/canonical_releases/chk1_full_v7_automated_v2_20260804/source/generation/generation_handoff.json`

### chk1 reasoning compression

- `dataset/processed/retrain_v2/chk1_reasoning_compressed_flash_max_v1_20260805/release_manifest.json`
- `output/data/retrain_v2/chk1/reasoning_compression_flash_max_v1_20260805/run_manifest.json`

### chk3

- `dataset/processed/retrain_v2/chk3_minutes_clean_v3_20260805/release_manifest.json`
- `dataset/processed/retrain_v2/chk3_minutes_clean_v3_20260805/chk3_minutes_sft.template.yaml`
- `docs/summary/20260805T123700Z/chk3_training_data_technical_report.md`

### chk4

- `output/data/retrain_v2/chk4/deepseek_v4_pro_v2/summary.json`
- `output/data/retrain_v2/chk4/training_data_v2/summary.json`
- `docs/summary/20260804T221321Z/chk4_decision_sft_grpo_implementation.md`

## 3. 配置与实现

- `configs/retrain_v2/chk1_analysis_sft_compressed_flash_max_v1_20260805.yaml`
- `configs/retrain_v2/chk2_analysis_grpo_compressed_chk1_v1_20260805.yaml`
- `configs/retrain_v2/chk3_minutes_sft.yaml`
- `configs/retrain_v2/chk4_decision_grpo.yaml`
- `configs/retrain_v2/dag_reward_v3.yaml`
- `src/open_r1/structured_response.py`
- `src/open_r1/trainer/rewards/reward_funcs/analysis_reward_v3.py`
- `src/open_r1/trainer/rewards/reward_funcs/decision_reward_v2.py`
- `jobs/retrain_v2/chk1/contracts.py`

当前 YAML 仅表示当前模板。若与具体 run 的 `resolved_runtime_config.json` 冲突，以后者为准。

## 4. 模型、训练与 lineage

### chk0

- `models/DeepSeek-R1-Distill-Llama-8B/config.json`
- `models/DeepSeek-R1-Distill-Llama-8B/README.md`

### chk1

- `output/training/retrain_v2/chk1_compressed_flash_max_v1_20260805/adapters/chk1/resolved_runtime_config.json`
- `output/training/retrain_v2/chk1_compressed_flash_max_v1_20260805/adapters/chk1/train_results.json`
- `output/training/retrain_v2/chk1_compressed_flash_max_v1_20260805/adapters/chk1/eval_results.json`

### chk2 fresh run

- `output/training/retrain_v2/chk2_compressed_chk1_v1_reward_v3_long4096_fresh_20260807/adapters/chk2/resolved_runtime_config.json`
- `output/training/retrain_v2/chk2_compressed_chk1_v1_reward_v3_long4096_fresh_20260807/adapters/chk2/checkpoint-183/trainer_state.json`
- `output/training/retrain_v2/chk2_compressed_chk1_v1_reward_v3_long4096_fresh_20260807/adapters/chk2/reward_history.jsonl`
- `output/training/retrain_v2/chk2_compressed_chk1_v1_reward_v3_long4096_fresh_20260807/adapters/chk2/loss_history.jsonl`
- `output/training/retrain_v2/chk2_compressed_chk1_v1_reward_v3_long4096_fresh_20260807/adapters/chk2/reward.jsonl`
- `output/training/retrain_v2/chk2_compressed_chk1_v1_reward_v3_long4096_fresh_20260807/adapters/chk2/runtime_safety.jsonl`
- `docs/summary/20260809T112950Z/analysis_snapshot.json`

### cp150 selected evaluation candidate

- `docs/summary/20260809T152718Z/chk2_checkpoint150_merge_eval/checkpoint_manifest.json`
- `docs/summary/20260809T152718Z/chk2_checkpoint150_merge_eval/lineage_manifest.json`
- `docs/summary/20260809T152718Z/chk2_checkpoint150_merge_eval/chk1_exact_merge_lineage.json`
- `docs/summary/20260809T152718Z/chk2_checkpoint150_merge_eval/chk2_cp150_exact_merge_lineage.json`

审阅截止时，该目录只有 manifest/merge lineage 文件，没有可用于结论的 completed evaluation results。

## 5. 自动化验证

环境：`fomc_trainer`

- `tests/test_retrain_v2_rewards.py`
- `tests/test_structured_response.py`
- `tests/test_retrain_v2_dag.py`
- `tests/test_retrain_v2_execution_receipt.py`

结果：241 passed，20.98 秒。

审阅时检测到的包版本：

- PyTorch 2.10.0+cu128
- Transformers 4.57.6
- TRL 1.2.0
- PEFT 0.15.2
- Accelerate 1.4.0
- bitsandbytes 0.48.2

## 6. 状态边界

- 当前 chk2 fresh run 的 durable state 是 183/247，并非完成到 247。
- checkpoint-150 是已合并的 evaluation candidate，不是经 task-matched validation 证明的最终最佳 chk2。
- 当前 chk3 release 为 `dag_bindable=false`，没有绑定 cp150 的正式训练模型。
- 当前 chk4 只有数据和配置准备，没有 current-chain 训练完成模型。
- pre-2009 chk4 supplement 尚未形成可绑定主结果的 release。
- 审阅截止时 cp150 common-test generation 正在运行；不将运行中产物作为证据。

## 7. 已知审阅限制

- 本次是仓库内静态/运行状态审计，没有进行外部文献检索或引用真实性的逐项网络核验。
- 没有对所有训练 completion 做人工事实审核；reward 诊断来自现有日志。
- 当前环境未发现 `latexmk`/`pdflatex`，没有实际编译 Chapter 2 PDF。
- 本次未改动 Chapter 2、模型、数据、配置、训练或评估进程。
