# checkpoint-200 合并与 DeepSeek Judge chk2 候选训练实施计划

状态：执行中  
授权时间：2026-08-10（UTC）  
执行范围：仅 checkpoint-200 promotion、TOTALSL overlay 与独立 chk2 candidate

## 固定输入

- chk1 adapter：`output/training/retrain_v2/chk1_clean_v2_override_full3ep_lr1e6_maxlen7168_20260810/adapters/chk1/checkpoint-200`
- adapter 权重 SHA-256：`7a5155562d9482028e0acb777a7b5758e4283f8463c0c39a0cade4062959b9a9`
- chk0 base：`models/DeepSeek-R1-Distill-Llama-8B`
- 父 chk2 数据：`dataset/processed/retrain_v2/analysis_base_full_v7_automated_v3_20260804/analysis_grpo`
- 新 reward：`grounded_analysis_v3_deepseek_high`
- policy 设备：物理 GPU1，单进程；GPU0 不加载本地 Judge。

## 执行顺序与停止条件

1. 创建 chk2-only 下游授权收据；旧 chk1-only 授权保持不变。
2. 从父数据生成不可变 TOTALSL overlay，仅修正单位字符串，并执行逐行/逐字段审计。
3. 只在 DeepSeek High reward 中增加显式货币单位换算验证，并完成回归测试。
4. 使用 CPU BF16、adapter copy 和 atomic no-overwrite 路径把 checkpoint-200 合并到全新目录。
5. 执行 exact-LoRA lineage；要求 291 个模型 tensors、448 个 adapter tensors、零 mismatch，且源 checkpoint fingerprint 不变。
6. 生成独立 chk2 YAML 与专用 launcher；不接入 canonical DAG，不恢复旧 chk2。
7. 当前状态门禁通过后，在 GPU1 fresh 启动；监控前两个 optimizer steps。
8. 写入启动/状态收据和数据质量技术报告。

以下任一情况立即停止且不启动训练：overlay 计数或不变性不符、merge lineage mismatch、源 checkpoint 改变、GPU1 可用显存低于 17 GiB、磁盘低于 250 GiB、API key 不存在、同名输出/session 已存在或任何绑定哈希不符。

训练中 DeepSeek 两次总尝试失败、OOM、context error、NaN 或候选分组错误时 fail closed；不在同一 run 中静默修改模型、reward 或 token 配置。

## Promotion 边界

本轮产物只能成为 chk2 candidate。它不构成旧 Qwen/DeepSeek 审核已通过，也不允许自动进入 chk3、chk4 或自动 merge chk2。训练结束后必须另行根据 reward、截断率、重复率及固定验证集选择 checkpoint。

## 交付目录

所有面向人的计划、授权、数据审计、lineage、启动收据与状态统一保存在：

`docs/summary/20260810T123000Z/chk1_cp200_deepseek_chk2/`
