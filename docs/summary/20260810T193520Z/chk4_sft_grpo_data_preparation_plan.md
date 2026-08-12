# chk4 Decision-SFT warm start + GRPO 数据准备计划

生成时间：2026-08-10 19:35:20 UTC

## 目标

从已完成的本地、gold-blind chk4 数据流水线发布一个新的不可变训练 release：

```text
merged chk1 checkpoint-200
  -> Decision-SFT warm start (`decision_sft`)
  -> merged chk4_sft
  -> Decision GRPO (`decision_grpo`, reward=`decision_dense_v2`)
```

本轮只准备和验证两阶段数据，不启动训练、不调用 DeepSeek API、不修改现有
teacher cache 或 `output/data/retrain_v2/chk4/training_data_v2`。

目标 release：

```text
dataset/processed/retrain_v2/chk4_decision_warmstart_grpo_core_v3_20260810/
```

## 固定输入

- Teacher acquisition：`output/data/retrain_v2/chk4/deepseek_v4_pro_v2`
  - 状态必须为 `complete`；128/128 accepted；0 failure。
- Gold-blind meeting briefs：
  `output/data/retrain_v2/chk4/meeting_decision_briefs_v1`
  - 状态必须为 `complete`；128/128 accepted；0 failure。
- 已物化候选：`output/data/retrain_v2/chk4/training_data_v2`
  - unique split 固定为 train/validation/test=`102/13/13`；
  - train 只按方向做可审计 repeat，物理行固定为 `141/13/13`；
  - validation/test 不重复，test 仅用于最终封存评测。

当前 supplement 流水线尚未完成，`supplement_admitted=0`。因此本轮 release 明确标记为
`core-only`；不得把它描述为覆盖 1993–2008 supplement 的完整 chk4 数据。

## Target-decision-blind 合同修订

旧 meeting-brief prompt 把“目标会议的答案”和“会议前已知的政策状态”一起禁止，实际
数据却保留了 46 条含 `target range` 的 point-in-time 描述。直接删去这些句子既会丢失
决策所需的当前政策立场，也会使已生成的 SFT rationale 与输入失配。

新 release 使用版本化的 `target-decision-blind` 合同：

- 允许 cutoff 前已经公开的有效联邦基金利率、当时的 target range、此前政策行动和资产
  负债表状态；
- 继续禁止目标会议的 direction、magnitude、vote、Minutes、会后信息和 gold/teacher 字段；
  prompt 不含直接 sample ID 或 exact target-meeting date 字段，但月份、年份和政策状态可能
  间接暴露会议身份，因此不得宣称 identity 不可推断；
- 每条输入仍绑定原 point-in-time atomic analyses，且 brief 中全部数字和日期必须来自这些
  source analyses；
- 此修订只改变合同解释并生成新的合同哈希，不覆盖或伪称旧合同已经通过；
- outcome 文本检查是 `known-pattern scan`，不是穷举语义证明。独立 source-only semantic
  Judge 本轮未运行；安全主证据是 target 字段排除与 cutoff-safe canonical lineage 回放。

## 两阶段数据合同

### Decision-SFT

每行只能包含：

```json
{"prompt":"target-decision-blind pre-meeting analysis", "response":"reasoning\n</think>\n{strict decision JSON}"}
```

- response 恰有一个 `</think>`；两侧非空；
- 最终 JSON 键精确为 `direction,magnitude_bp`；
- `hold` 只允许 0 bp；`cut/hike` 只允许 25/50/75/100 bp；
- final JSON 必须与 unique manifest 的本地 gold 完全一致；
- prompt 禁止直接 sample ID、exact target-meeting date、`rate_change`/gold/target outcome
  字段；允许 point-in-time 历史政策状态，但不得含目标会议的真实行动。历史上下文可能
  使会议身份可推断，本合同不作 inference-proof 承诺。

### Decision-GRPO

每行只能包含：

```json
{"sample_id":"...", "prompt":"与 SFT 完全相同", "direction":"cut|hold|hike", "magnitude_bp":0}
```

- 同一 physical row 的 prompt 与 SFT byte-for-byte 相同；
- label 与 SFT final JSON、unique manifest 三方一致；
- GRPO 输入不包含 SFT reasoning、final JSON、meeting date 或真实决策描述。

## 发布与门禁

1. 新增版本化 publisher/validator；逐行关联 unique/repeat manifest，并进一步串联
   raw teacher response → teacher SFT → 确定性 repair → clean SFT，以及 raw brief teacher
   response → brief output → canonical chk1 atomic analyses。
2. 验证完整性、唯一性、split 隔离、repeat factor、label 域、prompt 泄漏、SFT/GRPO
   prompt parity、response boundary、JSON 和来源哈希。
   复验识别到 8 个 unique SFT reasoning 含数字或明确月份/季度日期，另有 15 个含
   `near zero` / `two percent` 等英文拼写数量；新 release 对合计 23 个 unique rationale
   作 27 处确定性定点改写。GRPO prompt/label 和 SFT final JSON 不变，并逐行记录旧/新
   response SHA 与 repair method。
3. 使用 `fomc_trainer` 和真实 merged chk1 tokenizer 复算：
   - SFT 单 BOS、最终 EOS、completion mask、`max_length=3072` 零截断；
   - GRPO rendered prompt `<=2560`，不把 label 或 response 放入 prompt。
4. 对 canonical chk1 的 2,072 行、16,212 条 evidence 重放 meeting/cutoff/availability；
   所有 evidence 必须不晚于 cutoff。上游 selected-evidence ID 的 exact/prefix/empty/
   unresolved 状态必须如实审计，不能以 citation 缺陷冒充时序泄漏。
5. 生成 `audits/data_quality.json`、`release_manifest.json`、`handoff.json` 和逐文件
   SHA-256/row count；manifest 绑定 handoff unsigned payload、publisher/renderer 源码快照。
   staging 内先验证，再用 no-replace 发布，拒绝覆盖、symlink 或未列文件。
6. 正式验证必须传入外部固定的 release-manifest SHA；同步修改数据与内部哈希、替换
   rationale、篡改 handoff 路由均必须 fail closed。
7. 运行定向单元测试、Ruff/compile，并对最终 release 重放同一 validator。

任何门禁失败均停止发布；不自动删除样本、不降低阈值、不重新调用 API。

## 已知训练覆盖边界

- unique train 的 `cut/hold/hike=4/89/9`；repeat 后为 `16/89/36`，并非完全平衡；
- train 只覆盖 `hold:0`、`cut:25`、`cut:100`、`hike:25`；validation/test 中共有 7 个
  meeting 的 magnitude 在 train 未出现；
- 这些限制必须写入 release audit/handoff，不能通过复制 eval/test 样本或伪造标签来补齐。
- canonical chk1 的 selected-evidence 引用中有 8 行无法解析、50 行为空；完整 provided
  evidence lineage 仍全部绑定且 cutoff 安全。这些引用不会进入 chk4 prompt，但必须披露。
- teacher rationale 是已封存的本地监督目标，不是独立 source-only 语义裁决；本轮不能把
  `known-pattern hits=0` 写成全面的语义无泄漏证明。

## 训练边界

- chk4 SFT 的父模型是独立封存的 chk1 checkpoint-200 merge；
- chk4 GRPO 的父模型必须是本轮 SFT warm start 后的新 merged `chk4_sft`，不能仍指向 chk1；
- 当前 chk1 promotion 收据原先只授权 chk2。正式训练前需另建明确的 chk4 分支授权/lineage
  收据，并保留 chk1 source semantic audit 的既有失败事实，不得宣称 canonical audit passed。

## 版本处理

开发过程中生成的 `core_v1` 只覆盖 8 个显式数字/日期 repair；`core_v2` 虽覆盖全部
23 个 rationale，但旧 verifier 未充分绑定 handoff、raw teacher target 和 canonical evidence
回放。两者均只读保留并标记 superseded；训练只允许绑定通过外部 SHA 验证的 `core_v3`。
