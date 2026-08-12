# chk1 稳定 checkpoint 选择与 chk2 启动计划

## 目标

基于 full3ep chk1 的完整 train/eval 历史和固定行为门禁，定位兼顾收敛与生成稳定性的 checkpoint。只有最终候选完整通过退化测试后，才将其精确合并为新的 chk1 模型并启动全新的 chk2 reward-v3 训练。

## 选择原则

- loss 只用于确定收敛区间和候选优先级，不单独授权模型进入 chk2。
- 行为门禁固定复用此前 8-case manifest、6 个 sampling case、2 个 greedy case、相同 seed、4-bit SDPA 和 `max_new_tokens=3072`。
- 通过必须同时满足：8/8 finite、8/8 EOS、8/8 contract-valid、0 completion cap、0 catastrophic repetition、0 strict periodic tail；相对 chk0 每 case 重复率增幅不超过 0.20、均值增幅不超过 0.10、合同率不得下降。
- 若候选失败，继续向更早 checkpoint 回溯；不以最低 eval loss 覆盖行为失败。

## 执行顺序

1. 导出每 10 steps 的 train/eval loss、LR、epoch 和 checkpoint 完整性表。
2. 使用两张空闲 A30 并行测试中期 checkpoint，快速包围最后稳定区间；再在边界内按十步粒度定位。
3. 对最后稳定候选重新核对完整 8-case summary、严格 chk0 compare、adapter SHA 和对应 loss。
4. 将选定 adapter 与原始 chk0 精确 merge 到全新不可覆盖目录；验证 tensor/index、父模型、adapter 和 merged fingerprint。
5. 基于现有 `grounded_analysis_v3` 创建 chk2-only fresh 配置，绑定新 merged chk1、独立输出目录和执行收据。
6. 启动 Qwen3.5-9B Judge（GPU0）和 chk2 QLoRA policy（GPU1）；确认 Judge health、首个 GRPO step、reward 有限、无 OOM/context overflow 后交付运行状态。

## 安全边界

- checkpoint-255 已确认退化，禁止进入 chk2。
- 不覆盖任何历史 adapter、merged model、配置、reward 日志或训练目录。
- chk2 必须 fresh start，不自动恢复旧 GRPO checkpoint。
- 若没有任何 full3ep checkpoint 通过门禁，则停止在选择阶段，不启动 chk2。
