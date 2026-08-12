# retrain_v2 执行状态

> 本页记录的是 2026-08-03 的历史状态，已被 2026-08-04 的 full-v7 执行结果取代。
> 当前状态请以
> [`../20260804T100111Z/chk1_completion_limit_removal_and_readiness.md`](../20260804T100111Z/chk1_completion_limit_removal_and_readiness.md)
> 为准；不要继续执行本页中的旧 generation、人工审核、approval 或环境指令。

更新时间：2026-08-03T19:54:52Z（UTC）

目标血缘固定为：

```text
chk0  DeepSeek-R1-Distill-Llama-8B
  └─ chk1  analysis SFT
       └─ chk2  analysis GRPO
            ├─ chk3  Minutes SFT
            └─ chk4  decision GRPO
```

本文记录当前工作树和本机的实际状态。完整操作命令见
[`run/retrain_v2/README.md`](../../../run/retrain_v2/README.md)。

## 当前结论

现在还不能直接启动 `chk0 -> chk1 -> chk2` 训练。chk1 的 point-in-time 输入已经
准备好，但 teacher targets 尚未生成；在完成 full generation、exact-200 人工审核、
独立 operator approval、canonical/base release 和正式 run 初始化之前，训练入口会
fail closed。

| 边界 | 当前状态 | 证据/数量 |
|---|---|---|
| chk1 prepared 输入 | 已完成 | train 1,715、eval 205、test 197，共 2,117 条 |
| chk1 teacher targets | 未生成 | smoke/pilot/full 的 `generation_handoff.json` 均不存在 |
| exact-200 人工审核 | 未进行 | cohort、filled audit 和独立 approval 尚未形成 |
| canonical release | 未发布 | `output/data/retrain_v2/chk1/canonical_releases/` 下无 handoff |
| analysis base release | 未发布 | 无 `dataset/processed/retrain_v2/*/base_release_manifest.json` |
| 正式 run | 未初始化 | 无 `output/training/retrain_v2/*/run_manifest.json` |
| chk1/chk2 训练 | 未启动 | 没有 sealed chk1 或 chk2 |

这里的 2,117 条只能称为 **teacher 输入**，不能称为 2,117 条 SFT 样本或监督
target。prepared handoff 还记录了 canonical population 2,287 条、sample exclusion
170 条；最终可训练数量要以 full generation、审核和 base builder 的不可变产物为准。

## 已完成的实现与验收

- DAG 已固定为 `chk0 -> chk1 -> chk2 -> {chk3, chk4}`；chk4 不经过 decision
  SFT，chk3/chk4 从同一个 sealed chk2 分叉。
- chk1 prepared 数据具备 split、topic、point-in-time lineage、evidence、style 和
  tokenizer 绑定；当前 handoff payload SHA256 为
  `9f621e8971abc5aa939983b0eb0b1c47fa803daf2fb3a3434fc221a3c0511875`。
- 最终真实数据 dry-run 已对当前代码、模型和 tokenizer 重放通过：2,117/2,117
  全部选中，planned prepared exclusion 为 0；preparation binding SHA256 为
  `e32b2c269e6148ff8cb07d82046f2dd34243b8b3ee2f6ba1feec1d0ca93d6ba6`，
  generation provenance SHA256 为
  `a3e04756da3af89be5799386cfe867b34a8f1edc8eac170258afd205884f1028`。
- 已实现 smoke 20、pilot 200、full 2,117 的可恢复 teacher generation；teacher、
  deterministic verifier、chk0 critic、sample cache 和 terminal handoff 均受哈希与
  provenance 约束。
- 已实现 deterministic exact-200 cohort、严格四字段人工审核、独立 operator
  approval、全文件图 replay 和原子 no-clobber canonical publication。工具不会代填
  人工结论或 approval。
- 已实现独立 analysis base builder。它会重放 generation/preparation、人工审核、
  PIT/reference leakage/split/target/tokenizer 等审计，再生成 chk1 的 `analysis_sft`
  和 chk2 的 `analysis_grpo`。
- 正式 run manifest、stage lock/recovery、training/merge receipt、CPU merge、seal、
  parent/config/data/environment 哈希和后续篡改拒绝已经实现。
- chk1 固定使用 GPU0+GPU1 双 rank bf16 QLoRA DDP；chk2 固定使用 GPU0 上的
  Qwen3.5-9B vLLM judge 和 GPU1 上的 policy QLoRA。
- `fomc_train_v2`、`fomc_judge_v2` 两套隔离环境的当前只读检查、`pip check` 和
  CPU/import smoke 已通过；本次流程不使用 base Python，也不使用旧 README 中的
  legacy 环境。当前没有需要安装的版本缺失；若以后检查失败，只使用
  `run/setup_retrain_v2_envs.sh --train|--judge --skip-checks` 按 lock 修复，再重跑
  `--skip-gpu` 检查，不在环境中零散改包。

## 本地 target 复用结论

旧 `deepseek-reasoner` response 不能直接复用为 retrain-v2 监督 target。旧 prompt
曾暴露 same-meeting reference、current rate/rate change，且缺少固定 revision、
response ID 与 evidence ID，无法满足当前 provenance 和防泄漏合同。

可以复用的是已准备的 point-in-time 输入、split、topic 和风格资产。当前可执行路径
是固定本地 Qwen3.5-9B teacher，加 deterministic evidence verifier 和独立 chk0
critic。若以后改回具有固定 revision/凭证的远程 DeepSeek teacher，必须使用新的
release ID 从 smoke 重新生成，不能覆盖本地 release。详细审计见
[`chk1_local_reuse_audit.md`](chk1_local_reuse_audit.md)。

## 当前实际 blocker

1. **Teacher targets 缺失。** 必须按 smoke、pilot、full 顺序生成并验证，不能直接
   从 prepared 输入构建 base release。
2. **exact-200 外部审核缺失。** full generation 通过后，必须由人工填写恰好 200
   行的绑定 cohort；critical unsupported claim 必须为 0，平均 style score 至少 4.0。
3. **独立 approval 缺失。** approval 必须由 reviewer 之外的 operator 创建，并绑定
   release ID、cohort、filled audit、generation 和 preparation 的原始 SHA。
4. **canonical/base release 与 run 均不存在。** 这三层必须依序发布、验证和初始化，
   不能用手工文件或旧 release 绕过。
5. **GPU0 被外部任务占用。** 本次复核时 GPU0 利用率 98–99%，有三个
   `/home/haobin_cui/.conda/envs/py312/bin/python` compute process；GPU1 空闲。
   launcher 要求目标 GPU 没有 compute process、至少 22,000 MiB free、利用率不高于
   10%，因此 generation、双卡 chk1 和 GPU0 judge 当前都会被拒绝。没有也不应终止、
   暂停或迁移这些外部进程。

磁盘当前约有 455 GiB 可用，满足 stage 前 250 GiB 和 CPU merge 前 150 GiB 的
门禁；磁盘不是当前 blocker。

## 到 chk2 的唯一执行顺序

以下命令都在仓库根目录执行。先保持只读，直到 GPU0/GPU1 都满足 idle gate。

```bash
# 0. 固定环境与资源合同
run/check_retrain_v2_envs.sh --train --skip-gpu
run/check_retrain_v2_envs.sh --judge --skip-gpu
run/retrain_v2/resource_gate.sh

# 1. 只读复验 2,117 条 prepared 输入
run/retrain_v2/chk1_data.sh dry-run

# 2. 两卡空闲后依次生成/验证 teacher targets
run/retrain_v2/chk1_data.sh generate --mode smoke
run/retrain_v2/chk1_data.sh verify --mode smoke
run/retrain_v2/chk1_data.sh generate --mode pilot
run/retrain_v2/chk1_data.sh verify --mode pilot
run/retrain_v2/chk1_data.sh generate --mode full
run/retrain_v2/chk1_data.sh verify --mode full

# 3. 生成空结论的 exact-200 模板；由外部 reviewer 和独立 operator 完成文件
run/retrain_v2/chk1_data.sh audit-template
run/retrain_v2/chk1_data.sh verify-audit \
  --release-id <CANONICAL_RELEASE_ID> \
  --filled-audit output/data/retrain_v2/chk1/human_audit_full_v2/human_audit.reviewed.jsonl \
  --approval output/data/retrain_v2/chk1/human_audit_full_v2/operator_approval.json

# 4. 发布 immutable canonical release
run/retrain_v2/chk1_data.sh publish \
  --release-id <CANONICAL_RELEASE_ID> \
  --filled-audit output/data/retrain_v2/chk1/human_audit_full_v2/human_audit.reviewed.jsonl \
  --approval output/data/retrain_v2/chk1/human_audit_full_v2/operator_approval.json

# 5. 构建并验证 analysis base release
conda run --no-capture-output -n fomc_train_v2 \
  python -m jobs.retrain_v2.build_base_release \
  --repo-root "$PWD" \
  --canonical-handoff output/data/retrain_v2/chk1/canonical_releases/<CANONICAL_RELEASE_ID>/handoff.json \
  --base-release-id <BASE_RELEASE_ID>

# 6. 先只读验收，再用一个从未使用的 ID 初始化唯一正式 run
run/retrain_v2/pipeline.sh --run-id <RUN_ID> --base-release-id <BASE_RELEASE_ID>
run/retrain_v2/pipeline.sh --run-id <RUN_ID> --base-release-id <BASE_RELEASE_ID> --initialize

# 7. 两卡空闲后训练、merge、seal chk1
run/retrain_v2/stage.sh \
  --run-manifest output/training/retrain_v2/<RUN_ID>/run_manifest.json \
  --stage chk1 --execute

# 8. chk1 sealed 且 GPU0 释放后：终端 A 启动 judge
run/retrain_v2/judge.sh

# 9. 终端 B 在 judge healthy 后训练、merge、seal chk2（policy 固定 GPU1）
run/retrain_v2/stage.sh \
  --run-manifest output/training/retrain_v2/<RUN_ID>/run_manifest.json \
  --stage chk2 --execute
```

generation 因基础设施中断时只能在同一 mode 使用 `--resume`；stage 中断时重跑同一
`--execute` 命令，由 immutable recovery state 判断续做位置。不要删除 cache、receipt、
adapter 或 merged 目录来“重试”。

## 当前验证记录

- `run/check_retrain_v2_envs.sh --train --skip-gpu`：通过。
- `run/check_retrain_v2_envs.sh --judge --skip-gpu`：通过。
- `run/retrain_v2/resource_gate.sh`：通过；它不表示 GPU idle gate 已通过。
- `run/retrain_v2/chk1_data.sh dry-run`：通过；population/selected 均为
  2,117，train/eval/test 为 1,715/205/197，planned prepared exclusion 为 0。
- 当前稳定工作树的联合回归为 `486 passed, 14 warnings`：

  ```bash
  CUDA_VISIBLE_DEVICES=1 conda run -n fomc_train_v2 python -m pytest -q \
    tests/test_retrain_v2_*.py tests/test_chk1_*.py \
    tests/test_optional_dependency_detection.py
  ```

  14 条 warning 均来自 matplotlib/pyparsing 的上游 deprecation；回归只暴露空闲
  GPU1，没有在 GPU0 上启动计算。
- canonical/release bridge 的当前精确回归：`32 passed`：

  ```bash
  conda run -n fomc_train_v2 python -m pytest -q \
    tests/test_retrain_v2_chk1_canonical_workflow.py \
    tests/test_chk1_release.py \
    tests/test_chk1_generation_pipeline.py \
    tests/test_retrain_v2_chk1_workflow.py
  ```

- 相关 Python `py_compile`、Ruff、launcher `bash -n` 和 scoped `git diff --check`
  已通过。
- 旧状态页记录的 `234 passed` 是更早工作树的历史结果，不能作为当前并行修改后的
  完整回归结论；当前结论以上述 486 项联合回归为准。

## 不可绕过的限制

- 不把 prepared 输入误写为已生成 target，不承诺现在已可立即训练。
- 不复用旧泄漏/无 provenance 的 teacher response。
- 不由程序代填人工审核或 operator approval。
- 不在 canonical/base release 前初始化 run，不修改已经初始化的 run manifest。
- 不在 GPU0 外部进程存在时抢占运行，不干扰不属于本任务的进程。
- 不自动降 token、截断 prompt、把 judge 错误记为零分，或以不同 judge alias/端口
  冒充已封存服务。
