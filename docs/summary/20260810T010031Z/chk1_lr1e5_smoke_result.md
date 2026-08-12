# chk1 clean-v2：LR 1e-5 十步试验结果

生成时间：2026-08-10T01:19:10Z

## 结论

本轮通过退化门禁。保持数据和所有其他训练条件不变、仅把峰值学习率从 `5e-5` 降到 `1e-5` 后，checkpoint-10 在固定 8-case 测试中没有出现上轮的无限重复、缺失 `</think>` 或无法停止问题。

这个结果支持“`5e-5` 在前 10 steps 已把模型推过生成稳定性阈值”的判断。它证明 `1e-5` 修复了已观察到的早期退化，但不等价于证明完整约 170-step 训练始终安全；正式训练仍应在后续 checkpoint 重复执行相同门禁。

## 单变量与训练收据

- 新旧 YAML 的语义差异严格只有三项：`learning_rate`、隔离的 `output_dir`、隔离的 `peft_merged_model_path`。
- 数据、split、shuffle、`seed=42`、`data_seed=42`、global batch 16、LoRA r32/alpha64、target modules、优化器、`warmup_steps=9`、`max_steps=10` 均不变。
- 从 chk0 fresh start，双 A30 DDP；运行时记录 `world_size=2`、`CUDA_VISIBLE_DEVICES=0,1`、LR=`1e-5`。
- 新配置 SHA256：`4fcc470ce5545963fc361c3c6a26733b8d3dd7c6a552a6187ce7f44ef024b4dc`。
- 训练成功退出，`global_step=10`；checkpoint 8/9/10 保留，checkpoint-10 含 adapter、optimizer、scheduler 和两个 rank 的 RNG state。
- 根 adapter 与 checkpoint-10 权重相同，SHA256：`81c7775e427cdeffe23f8809dac42bb5f191e5508e81950018d2e4f670b36426`。

## Loss 对比

| 十步 smoke | step-10 train loss | 十步 train loss 均值 | eval loss | step-10 grad norm |
|---|---:|---:|---:|---:|
| 旧 LR `5e-5` | 1.6130 | 1.8263 | 1.5789 | 0.6445 |
| 新 LR `1e-5` | 1.8578 | 1.9279 | 1.8019 | 1.4453 |

较低 LR 的 loss 收敛较慢，这是预期代价；本轮目标是生成稳定性，而不是用十步 loss 选择最终最优点。所有 loss、grad norm 和 LR 均为有限值，无 OOM、NaN、Inf 或 NCCL fatal。

## 8-case 退化门禁

测试固定复用同一 sample manifest、prompt、6 个 sampling seeds 和 2 个 greedy case；推理参数为 4-bit、SDPA、`temperature=0.6`、`top_p=0.95`、`max_new_tokens=3072`。

| 指标 | chk0 baseline | 新 LR `1e-5` checkpoint-10 | 门槛 |
|---|---:|---:|---:|
| finite | 8/8 | 8/8 | 8/8 |
| 自然 EOS | 8/8 | 8/8 | 8/8 |
| `</think>` + 非空 answer | 8/8 | 8/8 | 8/8 |
| 触及 3072 上限 | 0/8 | 0/8 | 0/8 |
| catastrophic repetition | 0/8 | 0/8 | 0/8 |
| strict periodic tail | 0/8 | 0/8 | 0/8 |
| 平均 4-gram repetition | 0.0827 | 0.0948 | 相对增幅不超过 0.10 |
| 最大 4-gram repetition | 0.1425 | 0.2244 | 每 case 相对增幅不超过 0.20 |

新模型输出长度为 311–650 tokens，中位数 584.5；8 条均恰有一个 `</think>`。正式 compare 的 `status=passed`，平均重复率相对 chk0 仅增加 `0.0120`，最大单 case 增量 `0.1255`，都在预注册门槛内。

## 与旧 LR 失败的直接复现

旧 `5e-5` checkpoint-10 对首个相同 case 的重放结果与历史失败逐字哈希一致：

- 3072/3072 tokens，`hit_eos=false`；
- 没有有效 boundary/answer；
- 全文 4-gram repetition=`0.92310199`；
- 尾部 repetition=`0.96474045`；
- completion SHA256=`b454d4f37ad1b3f235c213144f388e6feacc8303a00a55221da0c7a57e4a2e4f`。

这条已足以重现旧故障，因此旧模型的其余冗余长循环 replay 被主动停止；该不完整 replay 不作为新模型的正式门禁输入。

## 工件

- 计划：`chk1_lr1e5_smoke_plan.md`
- 新训练配置：`configs/retrain_v2/chk1_analysis_sft_clean_v2_override_smoke10_lr1e5_20260810.yaml`
- adapter：`output/training/retrain_v2/chk1_clean_v2_override_smoke10_lr1e5_20260810/adapters/chk1`
- loss：上述 adapter 目录内的 `loss_history.jsonl`
- 新模型 probe：`chk1_lr1e5_smoke_cp10_probe/{launch.json,results.jsonl,summary.json}`
- chk0 比较：`chk1_lr1e5_vs_chk0_comparison.json`
- 旧 LR 首案重放：`chk1_lr5e5_smoke_cp10_probe_replay/results.jsonl`

## 使用边界

本轮仍使用此前明确授权的 chk1-only semantic override。该 candidate 不得直接用于 chk2/chk3/chk4；并且本轮没有启动正式 chk1 全量训练。

