# chk2 Analysis GRPO：实现、运行与验收交接文档

生成时间：2026-08-05 12:40:42 UTC  
状态快照：2026-08-05 12:41:52 UTC  
适用仓库：`/home/haobin_cui/research_files_space_2/fomc_trainer`  
正式运行：`retrain_v2_full_v7_automated_v9_20260804`

## 技术摘要

chk2 的职责是从 sealed chk1 出发，在 point-in-time FOMC 证据上继续执行 analysis
GRPO。正式运行固定由 GPU1 训练 DeepSeek policy QLoRA，GPU0 使用本地
Qwen3.5-9B 作为 LLM-as-judge。Judge 只能看到完整候选 response（显式包含
`<think>` 和 `<answer>`）以及允许的 point-in-time evidence，不能看到会议日期、
Minutes、参考答案或实际决议。

当前唯一应继续运行的 chk2 是 v9。它于 2026-08-04 22:55:18 UTC 在持久 tmux
session 中启动，复用了 v5 已 seal 的 chk1 artifact，并使用新的 Judge 合同：
`max_model_len=8192`、`max_completion_tokens=2048`。截至本报告快照，训练完成
175/247 optimizer steps（70.9%），最新完整恢复点为 `checkpoint-175`，未发生 Judge
重试、HTTP 非 200、OOM、Traceback 或自动重启。

运行层面是稳定的，但训练信号存在明显压缩：截至 step 175，平均 reward 为
0.2560，中位数为 0.2454；98.22% 的 individual completion 触发了 0.25 hard cap，
50.86% 的 optimizer steps 中四个候选全部达到 1024-token policy 上限，进而在
`mask_truncated_completions=true` 下不产生梯度。因此，应让本次运行完成并作为可审计
baseline，但不能只根据训练 reward 宣布 chk2 优于 chk1；必须在 seal 后进行独立比较。

## 1. 边界与模型血缘

```text
chk0  DeepSeek-R1-Distill-Llama-8B
  └─ chk1  analysis SFT（v5 sealed artifact）
       └─ chk2  analysis GRPO（v9 正在训练）
            ├─ derived release → chk3 Minutes SFT
            └─ derived release → chk4 decision GRPO
```

chk2 输入输出合同：

- 输入：一个 atomic topic 对应的 point-in-time evidence prompt。
- rollout：每个 prompt 生成 4 个 policy completion。
- reward：`grounded_analysis_v2`，由本地确定性检查和 Qwen3.5-9B Judge 共同计算。
- 输出 artifact：训练完成后的 merged chk2 model。
- 下游边界：chk3/chk4 数据必须由同一个 sealed chk2 SHA、`temperature=0`、
  `do_sample=false` 生成；chk2 未 seal 前不得绑定 derived release。

本阶段不负责重新生成 chk1 teacher target，也不以 Minutes 或 vote accuracy 作为 reward。

## 2. 正式 v9 的不可变血缘

| 对象 | 固定值 |
|---|---|
| Run ID | `retrain_v2_full_v7_automated_v9_20260804` |
| Run 创建时间 | `2026-08-04T22:49:50Z` |
| Execution contract SHA256 | `2442e4c3870da7214b3c311aa45bd9cb2a8a6e20e6c2d0cd7b21a96a49ae7c40` |
| Environment SHA256 | `e02d78e0ee05fd971f42e7039cc3e9405c0d9eb6246d07fb7a22e5f5c65400fe` |
| Base release | `analysis_base_full_v7_automated_v3_20260804` |
| Base release SHA256 | `b8128b394420f90d8e7cb2e0e422a7dc097b5d6c68adb30f73ded29f764c040f` |
| analysis_grpo SHA256 | `21695c76be83658dce57d453d8d470504c1482765c1c6f3bc5e93fc3ac4ffec2` |
| chk1 merged SHA256 | `9b355f903b7722f274bca1bc226bf75c1da4ebe449de726174bd1d054ff8948c` |
| chk1 来源 | `retrain_v2_full_v7_automated_v5_20260804` |
| chk1 import attestation SHA256 | `3247091ff8bd8cc792f3793228da068cff22c1bcfc833a61c1233c8da1b5a820` |
| chk2 template SHA256 | `cb633ef15a3ab80ff334b7ad4091895428de97a01d984dad2a1e0cf6ec4258a2` |
| chk2 resolved config SHA256 | `361ef34b5beb045bbfa9608f39ace27021fb7f5726cb88767b8d226cd6c9e5f9` |
| Judge artifact SHA256 | `15e37eaf4ecb293adf484aa25ce3bf5d7ec91b1e75abde0a152921e37d200026` |

权威 manifest：
[`run_manifest.json`](../../../output/training/retrain_v2/retrain_v2_full_v7_automated_v9_20260804/run_manifest.json)。
manifest 中 chk2 的 `status=pending` 在训练、post-attestation、receipt、merge 和 seal
全部完成前是正确状态，不能手工改为 sealed。

## 3. 数据合同与 token 审计

`analysis_grpo` 数据由 immutable base release 绑定：

| Split | 行数 | 当前用途 |
|---|---:|---|
| train | 493 | 正式 GRPO |
| validation | 199 | 已绑定但 `do_eval=false`，不在训练中自动评估 |
| test | 190 | 保留给独立验收 |
| 合计 | 882 | release population |

数据路径：
[`analysis_grpo`](../../../dataset/processed/retrain_v2/analysis_base_full_v7_automated_v3_20260804/analysis_grpo)。

训练前 token gate 使用精确 chk1 tokenizer 重放数据：

- policy prompt 合同：最多 2560 token，禁止运行时截断；
- policy completion 合同：最多 1024 token；
- 审计观测最大 prompt：train 2180、validation 2304；
- Judge 完整请求合同：prompt token 加 2048 output reserve 必须不超过 8192；
- 另保留 candidate reserve 1536 和边界 margin 32，用于训练前保守预算；
- 任一 batch 越界必须在 HTTP 之前整体失败，不能截断 evidence 或 candidate 后继续。

泄漏防护同时存在于 release audit 和 reward runtime：Judge evidence 会拒绝 Minutes、
reference answer、gold label、actual decision、meeting date、sample ID、cutoff timestamp
等字段；精确会议日期只保留为本地审计 metadata，不进入 Judge prompt。

## 4. 硬件拓扑和隔离环境

| 角色 | 物理设备 | 模型/任务 | 正式配置 |
|---|---|---|---|
| Judge | GPU0，A30 24 GiB | Qwen3.5-9B | BnB 4-bit、bf16、vLLM、4 并发 |
| Policy | GPU1，A30 24 GiB | merged chk1 → LoRA chk2 | NF4 QLoRA、bf16、SDPA |

正式环境：

- `fomc_trainer`：Python 3.10.9、Torch 2.10.0+cu128、Transformers 5.5.4、
  TRL 1.2.0、PEFT 0.15.2、bitsandbytes 0.48.2、Accelerate 1.4.0；
- `fomc_judge_v2`：Torch 2.10.0+cu128、Transformers 5.5.4、vLLM 0.19.1、
  bitsandbytes 0.48.2；
- 精确依赖以 `requirements/retrain_v2_{train,judge}.{lock,freeze}.txt` 为准；
- 不允许使用 base Python 或旧 README 环境启动该 run。

快照时 GPU0 使用 17,663 MiB、空闲 6,387 MiB、35°C；GPU1 使用 17,221 MiB、
空闲 6,829 MiB、98% utilization、56°C。当前没有显存压力或热异常。

## 5. Policy 训练配置

权威模板：
[`chk2_analysis_grpo.yaml`](../../../configs/retrain_v2/chk2_analysis_grpo.yaml)。
正式运行必须读取 run 内的
[`resolved_configs/chk2.yaml`](../../../output/training/retrain_v2/retrain_v2_full_v7_automated_v9_20260804/resolved_configs/chk2.yaml)，
不能在运行中编辑模板或 resolved config。

| 参数 | 值 |
|---|---:|
| Epoch | 1 |
| Optimizer steps | 247 |
| Per-device batch | 1 |
| Gradient accumulation | 8 |
| Generations per prompt | 4 |
| Generation batch | 4 |
| Prompt/completion | 2560 / 1024 |
| Temperature / top-p | 0.7 / 0.9 |
| Learning rate | `5e-7` |
| Scheduler | cosine，warmup ratio 0.05 |
| GRPO loss | DAPO，group reward scaling |
| KL beta | 0.04 |
| Precision | bf16，TF32 enabled |
| Quantization | NF4、double quant、bf16 storage |
| LoRA | r=32、alpha=64、dropout=0.05 |
| LoRA targets | q/k/v/o、gate/up/down projection |
| Truncated completion | mask，不参与梯度 |
| Save interval | 每个 optimizer step |
| Retained checkpoints | 最新 3 个 |
| Seed / data seed | 42 / 42 |

`use_vllm=false` 针对 policy；vLLM 只用于 GPU0 Judge。generation batch=4 是 A30
实测后的上限：四路 2560-token prefill 约 18.715 GiB reserved，更大的 rollout batch
超过 22 GiB 资源门禁。

## 6. `grounded_analysis_v2` reward 合同

实现：
[`analysis_reward_v2.py`](../../../src/open_r1/trainer/rewards/reward_funcs/analysis_reward_v2.py)。

### 6.1 候选内容

Judge 接收完整 response，而不是 answer-only：

```text
<think>
{parsed reasoning}
</think>
<answer>
{parsed answer}
</answer>
```

若生成不满足结构合同，Judge 仍会看到完整原文，但用 `<malformed_response>` 标记；
不能静默丢弃 reasoning 或只转发 answer。

### 6.2 评分公式

Judge 五个 0–4 分 rubric 的内部权重为：

- data fidelity：0.30；
- trend reasoning：0.25；
- policy relevance：0.20；
- uncertainty calibration：0.15；
- FOMC style：0.10。

总 reward：

```text
R_raw = 0.60 * judge_weighted_score
      + 0.25 * numeric_grounding
      + 0.10 * structured_contract
      + 0.05 * concision

if unsupported_numbers or unsupported_claims:
    R = min(R_raw, 0.25)
else:
    R = clip(R_raw, 0, 1)
```

数字 grounding 只允许 candidate 中的数值在 evidence 中直接出现或在小容差内可推导；
provenance ID 中的数字会先被排除。结构合同要求 reasoning 和 answer 都非空；concision
同时惩罚超长和重复四元组。

### 6.3 Judge 请求合同

- model alias：`Qwen3.5-9B`；
- temperature=0、top-p=1、thinking disabled；
- strict JSON schema；
- `unsupported_claims` 最多 8 条，每条不超过 240 字符；
- 一个 reward batch 最多四路并发；
- 每个请求最多三次重试，1/2 秒指数 backoff；
- 三次都不能得到严格 JSON 时抛出 `JudgeInfrastructureError`，整步失败；
- 基础设施错误永远不能被记成 reward=0 或静默跳过。

## 7. Judge 服务与 attestation

启动器：[`judge.sh`](../../../run/retrain_v2/judge.sh)。固定合同如下：

- GPU0；
- 只允许 `models/Qwen3.5-9B`；
- endpoint `127.0.0.1:8000`；
- served alias `Qwen3.5-9B`；
- `max_model_len=8192`；
- `max_num_seqs=4`；
- `gpu_memory_utilization=0.72`；
- bitsandbytes dynamic load、bf16、language-model-only、eager mode。

GPU0 若已有其他 compute process、空闲显存不足 22,000 MiB 或 utilization 超过 10%，
服务会在模型加载前拒绝启动。

pre-attestation 已于 2026-08-04 23:02:47 UTC 完成：

- 模型目录 SHA 与 manifest 一致；
- `/v1/models` 只暴露一个正确 alias/root；
- `/tokenize` 与本地 tokenizer 的完整 token ID 序列一致；
- server `max_model_len=8192`；
- golden strict-JSON 请求首次成功；
- attestation canonical SHA：
  `8af31823c24c915f85ea6debf20603617c514104f4c322ce554aa3f5d0123352`。

证据：
[`judge.pre.json`](../../../output/training/retrain_v2/retrain_v2_full_v7_automated_v9_20260804/attestations/judge.pre.json)。
`judge.post.json` 尚不存在，因为训练未结束；launcher 会在训练完成后使用同一服务记录
post-attestation，并验证 pre/post identity pair。

## 8. 无人值守和恢复设计

持久化入口：
[`start_chk2_background.sh`](../../../run/retrain_v2/start_chk2_background.sh)。

```bash
run/retrain_v2/start_chk2_background.sh \
  --run-manifest \
    output/training/retrain_v2/retrain_v2_full_v7_automated_v9_20260804/run_manifest.json
```

它创建的 session 名称固定为：

```text
fomc_chk2_retrain_v2_full_v7_automated_v9_20260804
```

恢复机制：

1. tmux 脱离调用终端持续运行；
2. 每次 attempt 启动全新的 Judge process group；
3. 最多等待 30 分钟直到 `/v1/models` ready；
4. `stage.sh` 在独占 stage lock 中执行 parent/config/data/source/environment/token gate；
5. 每个 optimizer step 保存完整 adapter、optimizer、scheduler、RNG 和 trainer state；
6. 失败后终止整个 Judge process group，等待 30 秒；
7. 最多自动尝试 6 次，并从最新完整 checkpoint 恢复；
8. 训练结束后自动执行 post-attestation、training receipt、CPU merge、merge receipt 和 seal；
9. worker 退出时无论成功失败都会清理 Judge。

恢复状态只能由 `jobs.retrain_v2.stage_recovery` 判定：

- `fresh_train`；
- `checkpoint_resume_ready`；
- `training_complete_ready_to_record`；
- `training_receipt_ready_to_merge`；
- `attested_merge_ready_to_record`；
- `merge_receipt_ready_to_seal`；
- `sealed_complete`。

不要删除 checkpoint、receipt、attestation、adapter 或 merged 目录来“强制恢复”。

## 9. 故障历史与修复时间线

| Run | 结果 | 发现/处理 |
|---|---|---|
| v5 | chk1 sealed，chk2 未启动 | 作为所有后续 run 的 chk1 权威来源；Judge 仍是 6144/512 |
| v6 | 未形成正式 chk2 artifact | chk2 reward/candidate 传递合同迭代 |
| v7 | 未形成正式 chk2 artifact | 修复 Judge 应读取完整 response，而非错误的局部文本 |
| v8 | 到 step 10 后停止 | Judge 512-token 输出产生非完整 JSON；三次确定性重试均失败；保留 `checkpoint-10` |
| v9 | 正在运行 | 新不可变 run；Judge 8192/2048；逐步 checkpoint；最多 6 次自动恢复 |

v8 的终止错误为：

```text
JudgeInfrastructureError: judge failed after retries:
attempt=1/2/3: ValueError: judge response must be exactly one JSON object
```

修复不是放宽 JSON parser，而是：

- Judge context 从 6144 提升至 8192；
- Judge output reserve 从 512 提升至 2048；
- `unsupported_claims` 限制为最多 8 条、每条最多 240 字符；
- 保持严格 JSON 和 fail-closed；
- checkpoint 从每 10 步改为每步；
- 增加 persistent supervisor、Judge 重启和最多 6 次 stage attempt；
- 用 v9 重新初始化不可变合同，导入已 seal 的 chk1；
- v8/checkpoint-10 原样保留，未跨合同强行导入 v9。

## 10. v9 运行快照

快照时间：2026-08-05 12:41:52 UTC。

| 指标 | 状态 |
|---|---:|
| 完成 optimizer steps | 175 / 247（70.9%） |
| 剩余 steps | 72 |
| 最新完整 checkpoint | `checkpoint-175` |
| 平均 step time | 277.25 秒 |
| 粗略剩余训练时间 | 约 5 小时 33 分钟，不含最终 merge/seal |
| Judge chat HTTP 200 | 1,405（1 health + 1,404 reward） |
| Judge HTTP 非 200 | 0 |
| Judge records requiring retry | 0 |
| Fatal error / OOM / Traceback | 0 |
| Supervisor attempt | 1 / 6，未发生重启 |

快照时预计约在 2026-08-05 18:15 UTC 完成 optimizer steps，实际时间会随 completion
长度变化；之后还需 post-attestation、receipt、CPU merge 和 seal。

实时日志：

- [`chk2.log`](../../../output/training/retrain_v2/retrain_v2_full_v7_automated_v9_20260804/logs/chk2.log)
- [`judge.log`](../../../output/training/retrain_v2/retrain_v2_full_v7_automated_v9_20260804/logs/judge.log)
- [`orchestrator.log`](../../../output/training/retrain_v2/retrain_v2_full_v7_automated_v9_20260804/logs/orchestrator.log)

## 11. Reward 现状：运行稳定，但 hard cap 和截断压缩了学习信号

以下是 step-level 日志统计：

| 指标 | 全部 175 步 | 前 20 步 | 最近 20 步 |
|---|---:|---:|---:|
| 平均 reward | 0.2560 | 0.2619 | 0.2529 |
| reward 中位数 | 0.2454 | 0.2457 | 0.2419 |
| reward P10 / P90 | 0.2332 / 0.2500 | 0.2338 / 0.2664 | 0.2304 / 0.3347 |
| batch 内 reward std | 0.0304 | 0.0326 | 0.0433 |
| completion 平均长度 | 976.2 | 959.2 | 992.8 |
| completion 截断比例 | 83.93% | 80.00% | 90.00% |
| 四个候选全部截断的 step | 50.86% | 50.00% | 60.00% |
| 零梯度 step | 51.43% | 55.00% | 60.00% |

individual completion 共 1,404 条：

| 指标 | 结果 |
|---|---:|
| 正式平均 reward | 0.2559 |
| 诊断性 cap 前平均 reward | 0.4879 |
| 触发 0.25 hard cap | 98.22% |
| 存在 Judge unsupported claim | 95.01% |
| 存在 deterministic unsupported number | 86.47% |
| 完整 reasoning+answer contract | 17.95% |
| Judge retry | 0 |

截至 step 176 的交叉检查显示：89 个“四候选全部截断”的 step 全部为零梯度；没有任何
全截断 step 产生非零梯度。另有 1 个非全截断 step 因组内信号不足产生零梯度。这与
`mask_truncated_completions=true` 的设计一致，说明约一半 optimizer steps 没有执行有效
parameter update。

Judge 平均 rubric（0–4）也偏低：data fidelity 1.43、trend reasoning 1.46、policy
relevance 1.30、uncertainty calibration 1.45、style 1.40。训练中偶尔仍会出现 0.34–0.44
reward 和正常梯度，因此不能称为完全 reward collapse；更准确的表述是“reward 大规模
压缩、有效更新稀疏”。

### 11.1 已知 reward 风险

Judge system prompt 明确要求把格式、遗漏和风格问题只放入 rubric，不得写入
`unsupported_claims`。但实际记录中仍出现“未包含字段”“malformed JSON”“遗漏
evidence ID”等描述，一旦进入 `unsupported_claims` 就会触发 0.25 cap。对前 1,408 条
记录中 9,646 条 claims 做关键词粗筛，约 42.32% 疑似属于格式/遗漏类；该比例只是
heuristic，不是人工标注精度，但足以说明 hard cap 需要在下一版重新校准。

reward 明细来源：

- [`reward.jsonl`](../../../output/training/retrain_v2/retrain_v2_full_v7_automated_v9_20260804/adapters/chk2/reward.jsonl)
- [`reward_history.jsonl`](../../../output/training/retrain_v2/retrain_v2_full_v7_automated_v9_20260804/adapters/chk2/reward_history.jsonl)
- [`loss_history.jsonl`](../../../output/training/retrain_v2/retrain_v2_full_v7_automated_v9_20260804/adapters/chk2/loss_history.jsonl)
- [`completions/`](../../../output/training/retrain_v2/retrain_v2_full_v7_automated_v9_20260804/adapters/chk2/completions)

## 12. 当前操作手册

### 12.1 只读查看状态

```bash
RUN_ROOT=output/training/retrain_v2/retrain_v2_full_v7_automated_v9_20260804

tmux list-sessions
tail -f "$RUN_ROOT/logs/chk2.log"
tail -f "$RUN_ROOT/logs/judge.log"
tail -f "$RUN_ROOT/logs/orchestrator.log"
nvidia-smi
find "$RUN_ROOT/adapters/chk2" -maxdepth 1 \
  -type d -name 'checkpoint-*' -printf '%f\n' | sort -V | tail
```

### 12.2 错误扫描

```bash
rg -n \
  'Traceback|JudgeInfrastructureError|CUDA out of memory|OutOfMemory|ERROR:' \
  "$RUN_ROOT/logs/chk2.log" \
  "$RUN_ROOT/logs/judge.log" \
  "$RUN_ROOT/logs/orchestrator.log"
```

### 12.3 如果 tmux 消失但 chk2 未 seal

先检查 `orchestrator.log`、最新 checkpoint 和 manifest；确认没有另一个 launcher 持有
stage lock 后，重新执行同一个持久化入口：

```bash
run/retrain_v2/start_chk2_background.sh \
  --run-manifest "$RUN_ROOT/run_manifest.json"
```

不要手工添加 `--resume_from_checkpoint`；stage recovery 会选择最新完整 checkpoint。

### 12.4 当前禁止操作

- 不要结束当前 tmux、policy 或 Judge process；
- 不要在训练中修改 `jobs/train`、`src/open_r1`、`jobs/retrain_v2`、`run/retrain_v2`
  或 requirements；这些目录属于 sealed source bundle；
- 不要编辑 v9 resolved config 或 manifest；
- 不要删除旧 checkpoint 来腾空间；`save_total_limit=3` 会自动维护；
- 不要另起一个 8000 端口的 Judge 或用同 alias 冒充服务；
- 不要把 v8/checkpoint-10 复制进 v9；
- 不要在训练结束前手工运行 merge/seal；现有 launcher 会自动执行。

## 13. 完成、merge 与 seal 的验收清单

训练完成后，现有 `stage.sh` 应自动完成以下顺序：

1. policy trainer 正常返回；
2. 记录 `judge.post.json`；
3. 验证 pre/post Judge identity pair；
4. 记录 training execution receipt；
5. 执行至少 150 GiB 空闲磁盘的 merge resource gate；
6. CPU merge LoRA adapter；
7. 记录 merge attestation/receipt；
8. fingerprint merged chk2；
9. 将 manifest 中 chk2 更新为 sealed；
10. 清理 Judge 并结束 tmux session。

完成后检查：

```bash
conda run --no-capture-output -n fomc_trainer \
  python -m jobs.retrain_v2.stage_recovery \
  --repo-root "$PWD" \
  --run-manifest "$RUN_ROOT/run_manifest.json" \
  --stage chk2
```

只有返回 `sealed_complete`，并且 manifest 中存在 chk2 artifact SHA、training receipt SHA
和 merge receipt SHA，才能向 derived-data workflow 交付。

## 14. Seal 后必须做的独立模型验收

本次 GRPO 的训练 reward 被 cap 和 completion length 强烈影响，因此最终 checkpoint 不能
仅按训练 reward 选择。至少应在冻结 test/validation population 上比较 chk1 与 chk2：

- 完整 `<think>/<answer>` 或目标 JSON contract 率；
- completion 截断率和终止长度；
- deterministic numeric grounding；
- unsupported claim 数量及事实/格式分类；
- Judge 五维 rubric，但报告 cap 前分数和 cap 后 reward；
- 对相同 prompt 的 paired win/tie/loss；
- 手工抽查只作为质量分析，不重新引入已取消的 exact-200 approval gate。

在独立验收通过前，不应把 chk2 的训练 reward 均值 0.256 解释为绝对模型质量，也不应
直接宣称 chk2 比 chk1 更好。

## 15. 下一版建议，不得作用于当前 v9

以下项目需要新 run ID、新 immutable config 和重新 smoke；不能中途修改 v9：

1. 将 policy `max_completion_length` 从 1024 试升到 1536，先在 GPU1 做最坏 prompt、
   四路 rollout 和 backward smoke；当前剩余显存说明值得测试，但不等于已证明安全。
2. 把事实性 unsupported claim 与 format/omission/style issue 拆成不同 schema 字段；
   hard cap 只允许经过验证的事实冲突触发。
3. 对 numeric grounding 的常见 unsupported number 做误报审计，区分 evidence 数值、
   可推导数值、日期、序号和外部臆测。
4. 同时记录 `raw_reward_before_cap`、cap reason 和 cap 后 reward，使 reward compression
   可直接监控。
5. 增加按 step 的有效候选率、全截断组比例和 nonzero-gradient 比例 guardrail；
   不应只监控平均 reward。
6. 用独立 evaluator 或人工小样本校准 Qwen rubric；不要让同一个 hard-cap Judge 成为
   唯一模型选择依据。

## 16. 代码与测试责任地图

| 文件 | chk2 职责 |
|---|---|
| `configs/retrain_v2/chk2_analysis_grpo.yaml` | policy、GRPO、Judge、保存配置 |
| `configs/retrain_v2/dag.yaml` | DAG、硬件角色和不可变 Judge/token 合同 |
| `src/open_r1/trainer/rewards/reward_funcs/analysis_reward_v2.py` | 完整 candidate、local checks、Judge request、reward |
| `jobs/retrain_v2/dag.py` | init/verify/seal 时的合同验证 |
| `jobs/retrain_v2/token_budget_gate.py` | policy/Judge token preflight |
| `jobs/retrain_v2/judge_health.py` | model root、alias、token-ID parity、golden request |
| `jobs/retrain_v2/judge_attestation.py` | pre/post 服务身份封存 |
| `jobs/retrain_v2/stage_recovery.py` | checkpoint/receipt/merge/seal 恢复状态 |
| `jobs/retrain_v2/execution_receipt.py` | training/merge execution receipt |
| `jobs/retrain_v2/merge_adapter.py` | CPU LoRA merge |
| `run/retrain_v2/judge.sh` | GPU0 固定 Judge launcher |
| `run/retrain_v2/stage.sh` | GPU1 training → receipt → merge → seal |
| `run/retrain_v2/start_chk2_background.sh` | tmux、Judge 生命周期、6 次自动尝试 |
| `run/retrain_v2/README.md` | 正式操作手册 |
| `tests/test_retrain_v2_*.py` | DAG、reward、health、attestation、recovery、token、launcher 回归 |

2026-08-05 12:43 UTC 使用 `fomc_trainer` 重新执行：

```bash
bash -n \
  run/retrain_v2/start_chk2_background.sh \
  run/retrain_v2/judge.sh \
  run/retrain_v2/stage.sh

conda run --no-capture-output -n fomc_trainer \
  python -m pytest -q \
  tests/test_retrain_v2_*.py \
  tests/test_optional_dependency_detection.py
```

结果：`371 passed in 29.10s`，shell syntax 检查通过。

## 17. 限制与未完成事项

- 本文是训练中的冻结快照；step、checkpoint 和 Judge 请求数会继续增长。
- v9 当前尚无 post-attestation、training receipt、merged chk2 或 sealed artifact SHA。
- 当前 reward 诊断是训练 population 上的描述性统计，不是独立泛化评估。
- 对 style/omission-like unsupported claims 的 42.32% 是关键词 heuristic，不能替代人工
  claim-level 标注；它用于识别 reward 设计风险，不用于宣布 Judge 精度。
- 没有在本文生成趋势图：精确运行合同、故障时间线和验收步骤更适合表格；完整 per-step
  数据已经保存在 reward/loss history 中，后续独立验收报告应再生成曲线。

## 18. 后续需要回答的问题

1. final chk2 相对 sealed chk1 是否提高了独立 paired preference，而不是只提高训练 reward？
2. 1024-token 截断是否集中在特定 atomic topic 或 evidence 长度区间？
3. unsupported number 的高比例中有多少是真实幻觉、多少是 deterministic checker 误报？
4. Qwen 写入 `unsupported_claims` 的格式/遗漏问题占比经人工小样本复核后是多少？
5. chk3/chk4 derived release 应使用 final sealed chk2，还是独立验收选择的某个 checkpoint？
   任何选择都必须重新封存明确 artifact SHA，不能只按训练日志挑选。

