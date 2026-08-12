# chk1 clean-v2：LR 1e-5 十步对照试验

## 目标

在不改变数据、样本顺序、随机种子、LoRA、batch、优化器、warmup、训练步数和生成门禁的前提下，只将 chk1 SFT 的峰值学习率从 `5e-5` 降到 `1e-5`，验证十步后是否仍出现无 `</think>`、不停止和无限重复退化。

## 固定条件

- 从原始 chk0 `models/DeepSeek-R1-Distill-Llama-8B` 重新开始，不续接旧 smoke。
- 使用相同 clean-v2 candidate 与 chk1-only semantic override 收据。
- 使用两张 A30、DDP、4-bit NF4 QLoRA、LoRA r32/alpha64、相同 target modules。
- 保持 `seed=42`、`data_seed=42`、global effective batch 16、`max_steps=10`。
- 保持 `warmup_steps=9`，使 LR 轨迹相对旧 smoke 仅按 1/5 缩放。
- 保持 completion-only loss、`max_length=4608`、单 BOS/EOS 数据路径和每步 checkpoint 保存。

## 唯一配置差异

1. `learning_rate: 5e-5 -> 1e-5`
2. adapter 输出改到新的隔离目录。
3. merged-model 目标改到新的隔离目录。

## 验证流程

1. 解析新旧 YAML 并证明除上述三项外配置完全一致；确认新输出目录不存在、GPU 空闲。
2. 在 `fomc_trainer` 环境用 GPU0+GPU1 训练 10 steps，检查 OOM、非有限 loss/grad、数据绑定和 checkpoint 完整性。
3. 对 checkpoint-10/root adapter 运行与旧 smoke 相同的 8-case generation probe：相同样本清单、seed、`temperature=0.6`、`top_p=0.95`、4-bit、SDPA 和 `max_new_tokens=3072`。
4. 与冻结 chk0 baseline 和旧 LR=5e-5 smoke 比较：EOS、`</think>`、非空 answer、是否触顶、严格周期尾及全文/尾部 4-gram 重复率。
5. 任一输出触顶、缺少 boundary/answer、严格周期尾，或重复率达到既有灾难阈值，则判定 LR-only 修复未通过；本轮不自动启动正式训练。

## 输出

- 新配置：`configs/retrain_v2/chk1_analysis_sft_clean_v2_override_smoke10_lr1e5_20260810.yaml`
- 训练目录：`output/training/retrain_v2/chk1_clean_v2_override_smoke10_lr1e5_20260810/`
- 证据与结论：本目录下的训练日志、probe 工件、对比收据和最终结果文档。

