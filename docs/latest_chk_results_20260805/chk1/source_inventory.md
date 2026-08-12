# chk1 报告来源清单

所有路径均相对于仓库根目录。报告没有访问网络，也没有读取或复制明文 API key。

## 数据获取与准入

| 来源 | 用途 |
|---|---|
| `output/data/retrain_v2/chk1/generation_full_v7/generation_handoff.json` | 生成状态、selected/accepted/excluded、DeepSeek 合同、并发和 provenance |
| `output/data/retrain_v2/chk1/generation_full_v7/audit/exclusions.jsonl` | 两条 `invalid_generator_result` 的 sample、topic 和 meeting |
| `output/logs/retrain_v2/chk1_deepseek/20260803T223506Z_all.log` | 全量获取完成的运行日志 |
| `output/data/retrain_v2/chk1/canonical_releases/chk1_full_v7_automated_v2_20260804/audit/quality_report.json` | 2,072 accepted、45 excluded、topic/meeting/准入不变量 |
| `output/data/retrain_v2/chk1/canonical_releases/chk1_full_v7_automated_v2_20260804/audit/exclusions.jsonl` | Canonical 排除明细 |
| `output/data/retrain_v2/chk1/canonical_releases/chk1_full_v7_automated_v2_20260804/sft/{train,eval,test}.jsonl` | Canonical 1683/199/190 行数复算 |

## 基础数据发布与质量门

| 来源 | 用途 |
|---|---|
| `dataset/processed/retrain_v2/analysis_base_full_v7_automated_v3_20260804/base_release_manifest.json` | 发布 ID、数据集指纹、教师/代码/tokenizer provenance、8 项审计索引 |
| `dataset/processed/retrain_v2/analysis_base_full_v7_automated_v3_20260804/analysis_sft/{train,eval,test}.jsonl` | 最终 SFT 1355/199/190、响应完整性复算 |
| `dataset/processed/retrain_v2/analysis_base_full_v7_automated_v3_20260804/analysis_grpo/{train,eval,test}.jsonl` | 后续 GRPO 493/199/190 的分桶对账 |
| `dataset/processed/retrain_v2/analysis_base_full_v7_automated_v3_20260804/audits/split_integrity.json` | 1190 SFT-only、328 GRPO-only、165 shared；无跨 split meeting/sample 重叠 |
| `dataset/processed/retrain_v2/analysis_base_full_v7_automated_v3_20260804/audits/token_budget.json` | SFT 3072 prompt / 7168 total 合同及最大观测 |
| `dataset/processed/retrain_v2/analysis_base_full_v7_automated_v3_20260804/audits/{schema,reference_leakage,point_in_time,target_consistency,encoding,teacher_grounding}.json` | 其余数据质量门状态与细节 |
| `dataset/processed/retrain_v2/analysis_base_full_v7_automated_v3_20260804/provenance/build_base_release.py` | 稳定哈希 70/20/10 分桶算法及 SFT/GRPO 发布逻辑 |

## 训练配置、结果与模型

| 来源 | 用途 |
|---|---|
| `configs/retrain_v2/chk1_analysis_sft.yaml` | chk1 SFT 配置模板 |
| `output/training/retrain_v2/retrain_v2_full_v7_automated_v5_20260804/resolved_configs/chk1.yaml` | 本次运行实际解析配置 |
| `output/training/retrain_v2/retrain_v2_full_v7_automated_v5_20260804/adapters/chk1/resolved_runtime_config.json` | 模型、量化、LoRA、GPU 和 world size 的运行时快照 |
| `output/training/retrain_v2/retrain_v2_full_v7_automated_v5_20260804/adapters/chk1/all_results.json` | train/eval loss、样本数、runtime 和吞吐 |
| `output/training/retrain_v2/retrain_v2_full_v7_automated_v5_20260804/adapters/chk1/trainer_state.json` | global step、训练曲线、epoch 验证指标 |
| `output/training/retrain_v2/retrain_v2_full_v7_automated_v5_20260804/run_manifest.json` | `sealed` 状态、数据绑定、parent/adapter/merged 指纹 |
| `output/training/retrain_v2/retrain_v2_full_v7_automated_v5_20260804/receipts/chk1.training.json` | 训练 lineage receipt |
| `output/training/retrain_v2/retrain_v2_full_v7_automated_v5_20260804/receipts/chk1.merge.json` | 合并 lineage receipt 和 shard 指纹 |
| `output/training/retrain_v2/retrain_v2_full_v7_automated_v5_20260804/merged/chk1/merge_attestation.json` | 合并方法、LoRA 前后结构和等价性声明边界 |
| `output/training/retrain_v2/retrain_v2_full_v7_automated_v{6,7,8,9}_20260804/imports/chk1.json` | 后续运行导入同一 v5 merged SHA 的证明 |

## 报告内数字的口径

- DeepSeek retrieval rate = `generation.accepted_count / generation.selected_count`。
- Canonical accepted = canonical SFT train + eval + test。
- SFT train = `sft_only + shared = 1190 + 165 = 1355`。
- GRPO train = `grpo_only + shared = 328 + 165 = 493`；此处只用于解释分桶，不代表 chk2 已在本报告范围内训练。
- 最终 eval 指标取 `trainer_state.json` 的 epoch 2 / step 170 记录，并与 `all_results.json` 对账。
- 模型完成以 `run_manifest.stages.chk1.status == sealed` 和 merged artifact SHA 为准。
