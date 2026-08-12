# chk4 Decision SFT → GRPO 实施记录

## 固定血缘

```text
chk0 → chk1 → chk2 ─→ chk3
                   └→ chk4_sft → chk4
```

`chk4_sft` 是 Decision-SFT warm start；最终 stage ID `chk4` 保留给 GRPO，且其
直接父模型必须是 sealed `chk4_sft`。经用户批准，Decision 数据生产不再等待
chk2：producer 改为 gold-blind DeepSeek V4 Pro meeting-brief stage。模型训练父链
仍可使用 chk2 → chk4_sft → chk4，但数据构建不读取 chk2 artifact。

当前 v8 chk2 正在运行。为不改变它已经固定的 execution contract，在 chk2 完成
training、merge、receipt 和 seal 前，不修改正式的 `jobs/retrain_v2`、
`run/retrain_v2`、`src/open_r1` 或现有训练配置。上述修改先在
`.codex_tmp/chk4_impl/` 隔离实现并测试；`jobs/generation`、测试与本记录不属于
当前运行的 execution-contract source roots，可以先正式交付。

## Decision 数据合同

- 一会议一唯一样本；核心 population 固定继承 102/13/13 chronological split。
- 1993–2008 supplement 只允许进入 train；有 supplement 时至少录取 98/109，
  且保留全部原动作类别。
- gold 只从
  `archive/code/process_decsion_output/summary_base_with_meeting_date.xlsx` 的
  `meeting_date`、`rate_change` 两列确定性生成。
- 必须得到 241 行、237 个唯一会议、零冲突，并与 `ffr_hist.xlsx` 同日或次日
  rate change 完全对账。
- student prompt 只包含 target-neutral `meeting_decision_brief`，不得包含会议日、
  Minutes、current rate、gold 或真实行动。

brief 由 `generate_chk4_meeting_briefs.py` 从 chk1 canonical atomic final answers
自动生成，输出目录：

```text
output/data/retrain_v2/chk4/meeting_decision_briefs_v1/
  train.jsonl
  validation.jsonl
  test.jsonl
```

核心行至少包含：

```json
{
  "meeting_date": "2021-12-15",
  "meeting_decision_brief": "Target-neutral pre-meeting analysis...",
  "source_ids": ["opaque-source-id"]
}
```

supplement 行还必须包含：

```json
{
  "valid_atomic_topic_count": 11,
  "category_coverage": [
    "prices",
    "employment_activity",
    "financial_conditions"
  ]
}
```

meeting date 只用于本地 join 和 manifest，绝不进入 student/teacher analysis prompt。

完整可恢复流水线入口：

```bash
jobs/generation/run_chk4_deepseek_pipeline.sh
```

它依次执行 gold-blind meeting briefs、target dry-run、hidden-gold reasoning 和
SFT/GRPO repeat materialization；使用独占锁，任一阶段失败即停止后续阶段。

## DeepSeek teacher acquisition

实现入口：`jobs/generation/generate_chk4_sft_targets.py`。

固定合同：`deepseek-v4-pro`、thinking enabled、reasoning effort high、JSON output、
`max_tokens=2048`、并发 8、只读取 `DEEPSEEK_API_KEY`、禁止 fallback、只允许一次
内容修复。teacher content 必须与本地 gold JSON 等价，但最终 SFT JSON 永远由本地
程序重新序列化；teacher 只提供 `reasoning_content`。

```bash
# 只准备和验证；绝不调用 API
conda run -n fomc_trainer python -m \
  jobs.generation.generate_chk4_sft_targets --dry-run

# 全量 teacher acquisition
conda run -n fomc_trainer python -m \
  jobs.generation.generate_chk4_sft_targets

# 中断后严格恢复
conda run -n fomc_trainer python -m \
  jobs.generation.generate_chk4_sft_targets --resume
```

在 brief 未齐时 `--dry-run` 必须 fail closed 并明确报告缺失文件，不发送 API 请求。
生成期间模型名、fingerprint、response ID 或模型合同漂移必须阻断；失败写入
`failures.jsonl`，不得静默删除。

teacher acquisition 完整通过后，使用离线、确定性的 materializer 生成 SFT/GRPO
物理训练行和独立 repeat manifest：

```bash
conda run -n fomc_trainer python -m \
  jobs.generation.materialize_chk4_training_data
```

该步骤不会调用 chk2 或任何 API。它仅重复 train；validation/test 始终一会议一行，
并为每个物理行生成独立 `training_row_id → source_sample_id` 审计映射。

## 训练合同

- chk4_sft：2×A30 DDP、NF4 QLoRA、bf16、double quant、paged AdamW 8-bit、
  SDPA + Liger、LoRA r32/alpha64、batch 1/GPU、grad accumulation 8、
  completion-only loss、3072 total tokens、1 epoch、LR 5e-6。
- 只对 train 物化可审计方向平衡 repeat manifest；validation/test 不重复。
- chk4 GRPO 直接父模型改为 sealed chk4_sft，保留 decision_dense_v2、
  2560/512、4 generations、LR 1e-6、最多 3 epochs、双卡 DDP 和 truncation mask。
- GRPO 前对 sealed chk4_sft 执行 validation rollout pilot：truncation `<10%`、
  JSON 合法率 `≥80%`；不得通过关闭 truncation mask 绕过。
- 正式 SFT 前执行双 rank、3072-token forward/backward smoke，单卡峰值 `<22 GiB`。

## 当前验证结果

- 归档 label audit：241 行、237 个唯一会议、4 个一致重复、0 个冲突。
- FFR 同日/次日对账：0 个 mismatch。
- 动作分布：hold 168；cut 25/50/75/100 = 14/9/2/1；
  hike 25/50/75 = 34/5/4。
- supplement 候选：109。
- teacher acquisition、全 128 核心 mock、resume 和 repeat materialization 测试：
  7 passed。
- 启动前 dry-run：128 个会议、2,072 条 atomic analyses，0 次 API 调用。
