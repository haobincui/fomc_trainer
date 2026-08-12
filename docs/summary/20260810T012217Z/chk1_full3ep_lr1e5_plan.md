# chk1 clean-v2：LR 1e-5 三轮正式训练计划

## 目标

基于已通过 10-step 退化门禁的配置，从原始 chk0 重新开始一轮独立 chk1 SFT：使用两张 A30、训练 3 epochs、每 10 steps 保存完整 checkpoint、每 10 steps 计算 eval loss。

## 固定训练合同

- 数据保持为 clean-v2 candidate，继续使用已授权的 chk1-only semantic override；不授权进入 chk2/chk3/chk4。
- 峰值学习率固定为 `1e-5`，`warmup_ratio=0.05`；3 epochs 预计 255 steps，对应约 13 个 warmup steps。
- 保持 4-bit NF4 QLoRA、r32/alpha64、相同七类 target modules、paged AdamW 8-bit、cosine scheduler、global effective batch 16、seed/data_seed 42。
- `num_train_epochs=3`、`max_steps=-1`，预计约 255 optimizer steps。
- `eval_steps=10`；延续既有保留策略，每步保存后仅保留“所有整十 checkpoint + 最新 3 个”，每个保留点均含完整 adapter、optimizer、scheduler、RNG 和 trainer state。
- `save_steps=1`、`save_total_limit=null`，由版本化 retention callback 最终保留 `10/20/.../250` 和 `253/254/255`，避免丢失最终可精确恢复状态。

## Token 预算

- 将训练配置 `max_length` 从 4608 提高到 7168，适配两张 24 GB A30 的单卡 QLoRA 副本。
- 该字段是每条样本的 prompt+target 总上限，不是新生成 target 的长度；sealed candidate 当前真实最大序列约 4019 tokens，因此本次仍应为 0 截断，且实际显存主要由真实 batch 长度决定。
- per-device batch 保持 1，gradient accumulation 保持 8；两卡 DDP 不做显存池化。

## 启动与运行边界

- 新输出目录必须在启动前不存在，防止训练器自动恢复旧 checkpoint；从 chk0 fresh start。
- 使用 `fomc_trainer` 环境和 GPU0+GPU1。
- 启动后至少确认配置解析、world size=2、数据绑定、模型加载和首个 optimizer step 正常；正式训练随后后台持续运行。
- 本轮不会复用 smoke adapter，也不会自动进入 chk2。
