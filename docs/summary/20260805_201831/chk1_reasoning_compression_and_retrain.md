# chk1 Reasoning 压缩、审计与重训交接

时间：2026-08-05 20:18 UTC

## 结论

- `deepseek-v4-flash` Max Thinking 压缩已完成，共处理 1,744 条。
- 语义审计与定向修复后，1,743 条通过；最终仍有 1 条明确错误，已按用户指示从训练集排除。
- 已发布新的不可变 SFT 数据 release：
  `dataset/processed/retrain_v2/chk1_reasoning_compressed_flash_max_v1_20260805`
- 已在 `fomc_trainer` 环境使用 GPU0、GPU1 启动 chk0 -> chk1 双卡 QLoRA SFT。
- 截至本文记录时训练已稳定推进到至少 `3/170`，首步 loss 为 `2.0371`，未出现 OOM、NCCL failure 或非有限 loss。

## 压缩与审计

压缩工作目录：

`output/data/retrain_v2/chk1/reasoning_compression_flash_max_v1_20260805`

压缩请求合同：

- model: `deepseek-v4-flash`
- API: Responses API
- `reasoning.effort`: `max`
- `max_output_tokens`: 32,768
- 并发上限：1,000
- 可见 compressed reasoning 的 chk0 tokenizer 范围：512–2,400 tokens
- final answer 保持原文，不允许改写；隐藏 reasoning 不进入训练 target

语义审计按 source-only 约束执行。失败数经过定向修复从 333 降至 47、14、3；用户要求的最后一次重试后剩余 1 条。最终审计摘要：

`output/data/retrain_v2/chk1/reasoning_compression_flash_max_v1_20260805/semantic_audit_v1/audit_summary.json`

最终失败并排除的样本：

- split/line: `train:1327`
- sample key: `ea1c81fb6d059e9433dd489d288832bc359836aff2a63902bc83d8796024ec31`
- 主要问题：把 `4279.31192 million` 错写为 `$4,279 billion`，数值高 1,000 倍
- 次要问题：从储蓄率延伸出 source 不支持的因果与预测性表述

## 已发布数据

Release manifest：

`dataset/processed/retrain_v2/chk1_reasoning_compressed_flash_max_v1_20260805/release_manifest.json`

发布计数：

- train: 1,354
- eval: 199
- test: 190
- total: 1,743

发布数据的完整 chat token 审计最大值为 4,180，没有样本超过训练配置的 `max_length=4608`，也没有发现不稳定的 prompt prefix。

## chk1 训练

配置：

`configs/retrain_v2/chk1_analysis_sft_compressed_flash_max_v1_20260805.yaml`

关键参数：

- base/chk0: `models/DeepSeek-R1-Distill-Llama-8B`
- 4-bit NF4 QLoRA，BF16
- LoRA rank/alpha: 32/64
- 2 epochs
- batch: 每卡 1，gradient accumulation 8，全局 batch 16
- max length: 4,608
- optimizer: paged AdamW 8-bit
- 总优化步：170

运行入口：tmux session `chk1_compressed_v1`

日志：

`output/training/retrain_v2/chk1_compressed_flash_max_v1_20260805/logs/chk1_train.log`

adapter 输出：

`output/training/retrain_v2/chk1_compressed_flash_max_v1_20260805/adapters/chk1`

计划 merged 输出：

`output/training/retrain_v2/chk1_compressed_flash_max_v1_20260805/merged/chk1`

启动后观测：

- 两个 distributed rank 正常进入训练
- 首步：loss `2.0371`，grad norm `1.8671875`，mean token accuracy `0.5687864`
- 训练期间总显存约为 GPU0 18.5 GiB、GPU1 11.9 GiB；GPU0 数值包含用户已存在的其他任务
- 首步 75 秒，后续约 52–58 秒/step；在 GPU0 资源竞争持续不变时，初始 ETA 约 2.7–3.5 小时

训练仍在后台运行。可用以下命令查看：

```bash
tmux attach -t chk1_compressed_v1
tail -f output/training/retrain_v2/chk1_compressed_flash_max_v1_20260805/logs/chk1_train.log
```
