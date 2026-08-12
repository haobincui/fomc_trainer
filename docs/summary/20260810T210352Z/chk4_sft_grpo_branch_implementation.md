# chk1 cp200 → chk4 Decision-SFT → Decision-GRPO 实施记录

生成时间：2026-08-10T21:03:52Z

## 目标

实现一条与既有 canonical retrain-v2 DAG 隔离的独立训练分支：

```text
DeepSeek-R1-Distill-Llama-8B + chk1 checkpoint-200 adapter
  -> chk4-scoped chk1 parent merge
  -> chk4 Decision-SFT
  -> merge Decision-SFT adapter
  -> chk4 Decision-GRPO
```

本轮只准备代码、配置、数据绑定、血缘收据和测试，不自动启动训练。

## 固定输入

- chk1 adapter：`output/training/retrain_v2/chk1_clean_v2_override_full3ep_lr1e6_maxlen7168_20260810/adapters/chk1/checkpoint-200`
- base：`models/DeepSeek-R1-Distill-Llama-8B`
- chk4 数据 release：`dataset/processed/retrain_v2/chk4_decision_warmstart_grpo_core_v3_20260810`
- release manifest SHA-256：`8b05dce09bcb0a0b27ee240da1e5d730be93921ce19e650a3b3e8b2689adb893`
- 数据角色：Decision-SFT 与 Decision-GRPO 分别只读取各自的 `train`、`validation`；`test` 永不进入训练 loader。

## 安全边界

旧 cp200 merged artifact 的 attestation 只允许 `chk2_parent_model`，并明确禁止 chk3/chk4。因此新分支必须从相同 base 与 checkpoint-200 adapter 重新进行一次不可覆盖、非破坏的 CPU merge，并生成只允许本 chk4 分支的新 attestation。不得把旧 chk2-scoped merged model 改名或重新声明为 chk4 parent。

数据 release 是 core-only：train physical direction counts 为 hold/cut/hike=`89/16/36`，且 validation/test 有 7 个 meeting-action magnitude 不在 train 支持集。本分支保留这些已封存限制，不把它们隐藏为“完全平衡”或“全标签覆盖”。独立 source-only semantic judge 为 `not_run`；训练收据不得声称已运行该审核。

## 训练合同

- 两阶段使用完全相同、manifest 绑定的 system prompt（SHA-256 `426b64532dfb955a3941db2cc2bcfc9a785c0540ffba9ee94afabc189fe4ce18`）。
- Decision-SFT：completion-only loss，`max_length=3072`；实测 release 最大总长 946 tokens。
- Decision-GRPO：`max_prompt_length=2560`、`max_completion_length=512`；reward 仅为本地 `decision_dense_v2`。
- 两阶段均采用 4-bit NF4 QLoRA、BF16、仅 GPU1、LoRA rank 32 / alpha 64、所有七类线性投影；梯度累积为 8，保持原双卡方案相同的有效 batch size=8。
- 每 step 写完整 checkpoint，并永久保留所有 10 的倍数及最新 3 个。
- 每个训练目录允许自动恢复数值最大的 checkpoint；不同阶段的输出目录严格隔离。

## 实施工件

- `configs/retrain_v2/chk1_cp200_merge_for_chk4_core_v3_20260810.yaml`
- `configs/retrain_v2/chk4_decision_sft_from_chk1_cp200_core_v3_20260810.yaml`
- `configs/retrain_v2/chk4_decision_grpo_from_sft_core_v3_20260810.yaml`
- `jobs/retrain_v2/chk4_sft_grpo_branch.py`
- `run/retrain_v2/chk4_sft_grpo_branch.sh`
- trainer 的专用 chk4 release binding 与对应测试。

## 验收

1. 两份训练 YAML 都能由当前 `fomc_trainer` 的 `TrlParser` 解析。
2. loader 只接受 externally pinned、immutable、quality-passed 的 v3 release，逐文件复核 hash/bytes/rows、handoff 与 system prompt。
3. SFT config parent 必须是新 chk4-scoped cp200 merge；GRPO config parent 必须是同一分支的 SFT merge。
4. GRPO 在 SFT 完成与 attested merge 前 fail closed。
5. parent merge、SFT merge 都禁止覆盖目标，并验证源 adapter merge 前后 hash 不变。
6. dry-run 可显示精确 GPU1 单卡命令但不得启动 GPU 任务。
7. 在 `fomc_trainer` 环境通过定向单元/集成测试、Ruff、编译和 shell 语法检查。

## 最终实现状态

实现完成时间：2026-08-10T21:30:17Z。

- 新增独立 branch orchestrator，所有写操作都必须显式传入 `--execute`。
- parent merge 从原始 chk0 与 checkpoint-200 adapter 重新构建，只授权为 chk4 Decision-SFT parent；不复用旧的 chk2-scoped merge。
- SFT 只有在 parent merge attestation 与逐 tensor exact-merge evidence 均通过后才能启动。
- SFT adapter 只有在三轮训练完成、loss 有限、runtime release binding 正确后才能 merge。
- GRPO 只有在 SFT merge attestation 与逐 tensor exact-merge evidence 均通过后才能启动。
- 训练、parent merge 与 SFT merge 共用一个非阻塞 branch lock；训练期间不能重复启动本分支，也不能提前 merge。
- SFT 与 GRPO 均通过 `CUDA_VISIBLE_DEVICES=1` 和 `accelerate --num_processes 1` 仅使用 GPU1。Trainer 仍按数值最大的 checkpoint 自动恢复。
- release runtime verifier 验证并封存 test split，但 loader 只返回 train/validation，防止 test 进入训练。

## 验证结果

- 联合测试：`251 passed in 36.65s`。
- branch/config/release 定向测试（含训练锁、授权配置漂移和外来恢复目录）：`24 passed`。
- Ruff：全部通过。
- Python compileall、shell `bash -n`、Git whitespace check：全部通过。
- 真实 `fomc_trainer` 依赖：accelerate 1.4.0、bitsandbytes 0.48.2、PEFT 0.15.2、PyTorch 2.10.0+cu128、Transformers 4.57.6、TRL 1.2.0，全部匹配。
- 真实 release preflight：passed；当前状态为 parent=`not_prepared`、Decision-SFT=`not_started`、Decision-GRPO=`not_started`。
- dry-run 未创建 parent、adapter、merged model 或训练进程。
- `reward_history.jsonl` 同时保存总体 reward、`decision_dense_reward` 与全部 `rewards/*/mean` 分项；逐样本结构化明细继续落在 `reward.jsonl`。

## 执行接口

以下命令按顺序运行；不带 `--execute` 的 mutation/training 命令只是 dry-run：

```bash
# 只读检查
run/retrain_v2/chk4_sft_grpo_branch.sh preflight
run/retrain_v2/chk4_sft_grpo_branch.sh status

# CPU：构建 chk4-scoped chk1 cp200 parent，并做 exact merge 验证
run/retrain_v2/chk4_sft_grpo_branch.sh prepare-parent
run/retrain_v2/chk4_sft_grpo_branch.sh prepare-parent --execute

# GPU1：Decision-SFT；中断后仍从最新数字 checkpoint 恢复
run/retrain_v2/chk4_sft_grpo_branch.sh train-sft
run/retrain_v2/chk4_sft_grpo_branch.sh train-sft --execute

# CPU：验证 SFT 完整结束，再 merge adapter 并做 exact merge 验证
run/retrain_v2/chk4_sft_grpo_branch.sh merge-sft
run/retrain_v2/chk4_sft_grpo_branch.sh merge-sft --execute

# GPU1：Decision-GRPO；中断后仍从最新数字 checkpoint 恢复
run/retrain_v2/chk4_sft_grpo_branch.sh train-grpo
run/retrain_v2/chk4_sft_grpo_branch.sh train-grpo --execute
```

## 已知限制

- 数据是 core-only；SFT/GRPO 的物理 train/validation/test 数量为 `141/13/13`，train unique 为 102。
- train physical direction 为 hold/cut/hike=`89/16/36`；validation/test 共 7 个 action magnitude 不在 train 支持集。
- 数据完成了结构与 point-in-time lineage replay，但独立 source-only semantic judge 为 `not_run`。该限制会写入 authorization/runtime receipt，不能被表述为完整语义审核通过。
- 最终检查时两张 GPU 均空闲；本轮仍按“实现代码”范围没有自动执行 parent merge 或启动训练。
