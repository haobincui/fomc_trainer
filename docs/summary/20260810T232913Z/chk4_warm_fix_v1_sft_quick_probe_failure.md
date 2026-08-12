# chk4 warm_fix_v1 Decision-SFT quick probe 失败报告

生成时间：2026-08-10T23:29:13Z

## 结论

`warm_fix_v1` 的 12-step Decision-SFT 训练本身正常完成，loss、梯度和 eval 均为有限值，也没有 OOM；但其首个完整 quick-probe 分组 5/5 都没有产生可计分的纯 JSON decision，reward 全为 `0`。该 adapter 未通过 warm-start 的最低生成门禁，不应合并，也不应作为 Decision-GRPO 的 parent。

quick probe 在得到首个完整 short 分组后由主流程安全停止。当前 `results.jsonl` 恰好包含这 5 条，未生成 `summary.json`；这不是一次声称完成 15 条的 formal/quick 全量评测，而是一份足以拒绝本 adapter 的 fail-fast 证据。

## 训练收据

- 配置：[chk4_decision_sft_from_chk1_cp200_warm_fix_v1_20260810.yaml](/home/haobin_cui/research_files_space_2/fomc_trainer/configs/retrain_v2/chk4_decision_sft_from_chk1_cp200_warm_fix_v1_20260810.yaml)，SHA256 `82353d2e51de3e0ec89c2cddb4258da8b750c7b4910920ec2ca2ccf0671ed2f6`。
- parent：`core_v3/parent/merged/chk1`，probe-launch artifact SHA256 `6ad91dbb1571485df995c2443b97149badc3b8b6a6528aff82f6e2e8c824df1e`。
- adapter：[chk4_sft](/home/haobin_cui/research_files_space_2/fomc_trainer/output/training/retrain_v2/chk4_from_chk1_cp200_sft_grpo_warm_fix_v1_20260810/adapters/chk4_sft)，probe-launch artifact SHA256 `54358acb0fa3aa4443c95d4c4070dccb4970fa973293c87b1b8c19c630153393`；其中 `adapter_model.safetensors` SHA256 为 `9295f20480881c9b90a23456ed2404bb796dfd369762e832c608b115f4ee7235`。
- 数据 release SHA256：`8b05dce09bcb0a0b27ee240da1e5d730be93921ce19e650a3b3e8b2689adb893`。
- 运行合同：GPU1 only，`CUDA_VISIBLE_DEVICES=1`、`world_size=1`、`per_device_train_batch_size=1`、`gradient_accumulation_steps=8`。
- 训练参数：LR `5e-6`、cosine、warmup 2 steps、`max_steps=12`、4-bit NF4、LoRA r32/alpha64、completion-only loss。
- 结果：global step `12`、epoch `0.680851`、runtime `100.48s`、报告的 train loss `2.04276`。
- 12 个 logged LR 的总和为 `3.0e-5`，平均为 `2.5e-6`。

训练 loss 的前后半段确有改善：

| 指标 | step 1–6 | step 7–12 |
|---|---:|---:|
| 平均 train loss | 2.10347 | 1.98205 |
| eval loss | 2.02542（step 6） | 1.99252（step 12） |
| eval token accuracy | 0.55097 | 0.55618 |

因此失败不是训练进程崩溃或数值异常；它说明 token-level loss 的小幅改善没有转换为所需的终止、boundary 和纯 JSON delivery 行为。

## 首组 5 条 quick probe

probe 使用 validation 的固定 short 样本 `dec-4b867d2ad3a89e8af08268c3`，prompt 长度 433 tokens；包含 1 条 greedy 和 4 条固定 seed 的 `temperature=0.7, top_p=0.9` sampled generation，`max_new_tokens=1024`。

| 门禁 | 结果 |
|---|---:|
| cases | 5 |
| contract valid | 0/5 |
| delivery valid | 0/5 |
| plain JSON answer | 0/5 |
| positive reward | 0/5 |
| exactly one `</think>` | 3/5 |
| natural EOS | 3/5 |
| hit 1024-token cap | 2/5 |
| repetition valid | 5/5 |
| sampled group reward mean/std | 0.0 / 0.0 |

具体失败分为两类：

1. greedy 和 1 条 sampled 输出均达到 `1024/1024` tokens，没有 `</think>`，也没有 EOS。
2. 其余 3 条 sampled 输出在 856–933 tokens 后到达 `</think>` 和 EOS，但 answer 使用 `````json`` Markdown fence，因而不是合同要求的单一纯 JSON object。

5 条 completion 长度为 `1024, 886, 1024, 933, 856`，中位数 `933`。没有观察到灾难性重复：full 4-gram repetition 最大 `0.1332`，tail 最大 `0.2451`；失败核心是 reasoning 过长和 final-answer 格式污染，而不是循环重复。

## 证据哈希

- sample manifest：[sample_manifest.json](/home/haobin_cui/research_files_space_2/fomc_trainer/output/eval/retrain_v2/chk4_decision_sft_generation_probe_warm_fix_v1_20260810/sample_manifest.json)，SHA256 `1a6d39638ccd90d3eb2471d9b09e34380269409354736cc70c745e29ceb21872`。
- probe launch：[launch.json](/home/haobin_cui/research_files_space_2/fomc_trainer/output/eval/retrain_v2/chk4_decision_sft_generation_probe_warm_fix_v1_20260810/quick_adapter_v1/launch.json)，SHA256 `2c56b7707f4721177d7ad3e157d04eb8a98f4971f1e90f7656c4c7c2621b8d72`。
- 首组结果：[results.jsonl](/home/haobin_cui/research_files_space_2/fomc_trainer/output/eval/retrain_v2/chk4_decision_sft_generation_probe_warm_fix_v1_20260810/quick_adapter_v1/results.jsonl)，5 rows，SHA256 `de4a537061e437b201838b0cf20e45a5a00e9dca35fa486adaef67ea899d8f89`。
- loss history：[loss_history.jsonl](/home/haobin_cui/research_files_space_2/fomc_trainer/output/training/retrain_v2/chk4_from_chk1_cp200_sft_grpo_warm_fix_v1_20260810/adapters/chk4_sft/loss_history.jsonl)，SHA256 `3617468ab85508447395bf98242ca81f489a210b00681105434ccdc581e23fbf`。
- resolved runtime：[resolved_runtime_config.json](/home/haobin_cui/research_files_space_2/fomc_trainer/output/training/retrain_v2/chk4_from_chk1_cp200_sft_grpo_warm_fix_v1_20260810/adapters/chk4_sft/resolved_runtime_config.json)，SHA256 `b0702275af9bf16f592f5c70819ecd52338b801743367a8d4cb371ca85e45014`。
- 本报告的结构化伴随文件：[chk4_warm_fix_v1_sft_quick_probe_failure.json](/home/haobin_cui/research_files_space_2/fomc_trainer/docs/summary/20260810T232913Z/chk4_warm_fix_v1_sft_quick_probe_failure.json)。

## GPU 隔离

SFT 的 resolved runtime 和 probe launch 都明确记录 `CUDA_VISIBLE_DEVICES=1`。停止后的快照中：

- GPU0 仍由此前已存在、与 chk4 无关的 `llama_factory` PID `1224210` 占用约 23,850 MiB；本流程没有启动、停止或 signal 该进程。
- GPU1 没有 compute process，剩余约 24,036 MiB。

因此本次 chk4 SFT/probe 只使用 GPU1，GPU0 上的既有任务未被触碰。

## 后续约束

- 不合并 `warm_fix_v1` adapter。
- 不启动其 Decision-GRPO。
- 保留 adapter、loss 和首组 5-row probe 作为失败诊断证据。
- 后续若重训，必须使用新 profile 和新输出目录，并重新执行固定 quick gate；loss 下降不能替代生成门禁。
