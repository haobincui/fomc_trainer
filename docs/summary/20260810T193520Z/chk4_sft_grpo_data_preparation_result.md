# chk4 Decision-SFT warm start + GRPO 数据准备结果

完成时间：2026-08-10 20:38:28 UTC

## 结论

已发布并用外部固定 manifest SHA 完整回放验证新的不可变 core-only release：

```text
dataset/processed/retrain_v2/
  chk4_decision_warmstart_grpo_core_v3_20260810/
    decision_sft/
    decision_grpo/
```

训练顺序固定为：

```text
selected merged chk1 checkpoint-200
  -> decision_sft warm start
  -> merge decision_sft adapter
  -> decision_grpo
```

本轮没有启动 chk4 训练、没有调用外部 API，也没有覆盖 source、teacher cache、模型或
历史 release。`core_v1` 和 `core_v2` 被保留用于审计，但均已 superseded，不得用于训练。

## 两阶段数据规模

| 数据角色 | train | validation | test |
|---|---:|---:|---:|
| unique meetings | 102 | 13 | 13 |
| Decision-SFT physical rows | 141 | 13 | 13 |
| Decision-GRPO physical rows | 141 | 13 | 13 |

train unique direction 为 `cut/hold/hike=4/89/9`；固定 capped repeat 后为
`16/89/36`。SFT 与 GRPO 的每个 physical row prompt byte-for-byte 相同；GRPO label、
SFT final JSON 和 unique manifest 三方一致。

## 数据修复与完整 lineage

1. 对 23 个 unique SFT rationale 做了 27 处确定性改写，移除数字字符、显式月份/
   季度/年份及 `near zero`、`two percent` 等英文数量。prompt 和 final decision JSON
   完全不变；修复后 explicit quantity/date 违规为 0。
2. 128 条监督目标全部验证：raw target teacher response → teacher SFT → 确定性 repair →
   clean SFT；128 条 meeting brief 全部验证：raw brief teacher response → brief output。
3. 128 个 meeting prompt 进一步回放到 canonical chk1 的 2,072 条 atomic analysis 和
   16,212 条 evidence；future evidence、meeting/cutoff mismatch、atomic text mismatch 均为 0。
4. canonical chk1 的 selected-evidence 引用状态为 exact/prefix/empty/unresolved=
   `2003/11/50/8`。8 条无法解析的是上游 citation 元数据缺陷；完整 provided evidence
   lineage 仍 hash-bound 且 cutoff-safe，citation ID 不进入 chk4 prompt。
5. publisher 和 single-BOS renderer 的源码快照已嵌入 release；handoff unsigned payload
   由 manifest 绑定。未列文件、handoff 路由篡改、同步修改 GRPO label/内部哈希、替换
   SFT rationale 均在测试中 fail closed。

## Target-decision-blind 的准确口径

- 允许 cutoff 前已公开的 effective federal funds rate、pre-meeting target range、历史
  政策行动和资产负债表状态；实际 128 行中 target range/federal-funds/balance-sheet
  分别出现 `46/109/66` 行。
- 禁止直接 target direction、magnitude、vote、Minutes、会后信息、sample ID 和 exact
  target-meeting date 字段。
- prompt 中仍有月份、年份和政策状态，会议身份可能被间接推断；release 不声称
  identity inference-proof。
- deterministic `known-pattern target-outcome scan` 为 0 hits，但它不是穷举语义证明；
  独立 source-only semantic Judge 本轮未运行。target blindness 的主证据是字段排除、
  raw-teacher 绑定和 cutoff-safe canonical lineage，而不是把 regex 描述成语义验证器。

## Token 结果

使用 `fomc_trainer`、Transformers `4.57.6`、TRL `1.2.0` 和 selected chk1 的真实
`LlamaTokenizerFast`/single-BOS renderer：

| split | prompt max | SFT completion max | SFT total max | overflow |
|---|---:|---:|---:|---:|
| train | 811 | 167 | 946 | 0 |
| validation | 685 | 171 | 798 | 0 |
| test | 775 | 178 | 868 | 0 |

合同上限为 SFT `max_length=3072`、GRPO `max_prompt_length=2560`、
`max_completion_length=512`；全部零截断。tokenizer 仍会输出已知 `fix_mistral_regex`
warning；审计端故意保持与当前训练 runtime 一致，没有单独改变 tokenization 行为。

## 覆盖边界

- 本轮仅包含 2009 年后的 128 个 core meetings；1993–2008 supplement 仍未完成，
  `supplement_admitted=0`。
- train 只覆盖 `hold:0`、`cut:25`、`cut:100`、`hike:25`；validation/test 有 7 个
  meeting 使用 train 未见 magnitude：`hike:50` 两条、`hike:75` 四条、`cut:50` 一条。
- teacher SFT rationale 是已封存的监督目标，不是独立 source-only 语义真值。
- test 只允许最终评测，不得用于 warm start、GRPO 或 checkpoint 选择。
- 数据层已准备完成；正式训练仍需新增绑定此 v3 manifest SHA 的 chk4-only runtime verifier、
  chk1→chk4 分支授权/lineage 和两份新配置。现有旧 chk4 配置仍指向 chk2，不能复用。

## 固定哈希

- `release_manifest.json`：
  `8b05dce09bcb0a0b27ee240da1e5d730be93921ce19e650a3b3e8b2689adb893`
- `handoff.json`：
  `78958e3275ee7b43533e6fd824e885d011e69e8fc70ece9b8b78e11fe60d6773`
- `audits/data_quality.json`：
  `cd14a4c5f8f54525eb6f3d2f0aaa6cae43f7263f5f84110f8a6419f6f1694abc`
- `audits/point_in_time_lineage.json`：
  `486cbd418edcf7e778c88b011eee72a16130654390e89e14e8a719a354625895`
- `contracts/decision_input_contract.json` 文件 SHA：
  `4784da5e1cc0bffe9c79f9f02c3cb94359a588a8841f5fac353b50875886e5cf`
- input-contract payload SHA：
  `502a7c99694c97f83823dc39470c8f2fea90322b648c03e72acdb2c7876a6cae`
- `manifests/reasoning_repairs.jsonl`：
  `ed6022e65e6d81330a42346b9b72e9b27429a35f6c8bd2eca0637906aff9ff2c`
- embedded publisher snapshot：
  `02334bee9a84150398eab828199343ef52f770d3d8d8b54e5732e071c28084ac`

## 验证

固定哈希回放命令：

```bash
PYTHONPATH=src:. /home/haobin_cui/.conda/envs/fomc_trainer/bin/python \
  -m jobs.retrain_v2.publish_chk4_training_data verify \
  --release dataset/processed/retrain_v2/chk4_decision_warmstart_grpo_core_v3_20260810 \
  --expected-manifest-sha256 \
  8b05dce09bcb0a0b27ee240da1e5d730be93921ce19e650a3b3e8b2689adb893
```

回归命令：

```bash
/home/haobin_cui/.conda/envs/fomc_trainer/bin/python -m pytest -q \
  tests/test_generate_chk4_sft_targets.py \
  tests/test_chk4_supplement_pipeline.py \
  tests/test_publish_chk4_training_data.py
```

结果：`71 passed`。publisher/test 的 Ruff 与 `py_compile` 均通过；正式 v3 固定哈希回放
返回 `status=verified`。
