# Chapter 2 评审意见与修订建议

## 文档目的

本文档将博士论文第二章的主要评审意见转化为可执行的修订方案。它与同目录下的英文正式评审报告配套使用：

- 正式评审报告：`chapter2_phd_referee_report.md`
- 本文档：面向实际修改、补实验和后续维护的中文行动指南

当前总体评审结论为：

> **Major Revision。当前版本尚不足以支持提交或博士论文答辩。**

章节选题具有潜力，工程实现也有相当工作量，但现阶段最需要解决的不是语言润色，而是实验有效性、数据血缘、模型身份、样本口径、统计推断和结论边界。建议先冻结和重建证据链，再重写正文。

---

## 一、建议保留并强化的核心贡献

不建议放弃本章，而应收窄并重新定义其核心贡献。一个较稳健的定位是：

> **Historical, indicator-conditioned generation of FOMC-Minutes-style text and ex-post policy-action inference.**

中文可表述为：

> 本章研究如何基于历史会议时点可获得的宏观经济指标，生成具有 FOMC Minutes 文体特征的文本，并考察这些表示对历史政策行动推断的辅助价值。

在这一定位下，建议重点保留：

1. 宏观指标到央行沟通文本的数据到文本生成任务；
2. 分阶段后训练方法及清晰的 checkpoint 比较；
3. 对生成文本的语义、数值、方向和事实一致性评价；
4. 对模型输入敏感性的干预式分析；
5. 在严格限定为历史、事后分析的前提下，评价政策行动推断能力。

建议删除或显著弱化以下未经现有证据充分支持的表述：

- real-time forecasting；
- actionable policy support；
- market impact；
- causal interpretation；
- faithful interpretability；
- “first”或“novel”类优先权声明；
- 将事后 Minutes 输入称为严格意义上的会前预测信息。

---

## 二、必须优先解决的关键问题

### 1. 清除评估集中的目标泄漏

#### 发现的问题

当前流水线在构建 `eval` 和 `test` 的分析教师提示时仍包含官方 `reference_excerpt`，随后又将教师答案作为 `raw_analysis` 注入 Minutes 重写提示。因此，最终模型虽然没有直接看到目标文本字段，却可能通过教师答案间接接触目标信息。

这会使 held-out evaluation 不再是严格的 reference-free evaluation，并直接威胁文本生成结果的有效性。

#### 建议修改

1. 训练集可以保留 reference-conditioned teacher distillation，但必须在论文中明确说明。
2. `eval` 和 `test` 的教师提示只能包含：
   - meeting date；
   - section name；
   - 当时可获得的指标；
   - 不含目标内容的任务说明。
3. 官方 Minutes excerpt 只能作为评价目标，不能进入任何评估输入、中间分析或提示模板。
4. 每一行样本增加并保存如下审计字段：
   - `split`
   - `meeting_id`
   - `reference_used_in_prompt`
   - `teacher_model`
   - `teacher_prompt_hash`
   - `teacher_response_hash`
   - `target_hash`
   - `source_dataset_version`
5. 在构建完成后加入自动断言：
   - `eval/test.reference_used_in_prompt == false`
   - prompt 与 target 不存在长片段逐字重合；
   - 同一会议不能跨 split；
   - 中间教师回答能够追溯到唯一 prompt 和模型版本。

#### 验收标准

- 所有 `eval` 和 `test` 样本均能通过 reference-free 审计；
- 重新生成的评估结果与旧结果分开命名，旧结果不得继续作为主结果；
- 论文明确区分训练蒸馏与无参考评估；
- 附录或仓库中提供一份机器生成的数据审计报告。

---

### 2. 重建 checkpoint 血缘和实验身份

#### 发现的问题

论文、活动配置、归档配置和 metadata 对 `chk-0` 至 `chk-4` 的父模型关系并不一致。同一组文本相似度结果在不同文档中还被标记为不同的 checkpoint 对比。由此无法确定每个结果究竟来自哪个模型。

#### 建议修改

先以真实训练产物为准建立不可变的 checkpoint manifest，不要为了匹配论文叙述而人为改写历史。建议的概念结构可以是：

```text
chk-0: base model
  └── chk-1: analysis SFT
        └── chk-2: analysis GRPO
              ├── chk-3: Minutes SFT
              └── decision SFT
                    └── chk-4: decision GRPO
```

如果真实训练关系不是上述结构，应在论文中报告真实关系，并重新命名 checkpoint。

每个 checkpoint 至少记录：

- immutable model ID；
- parent checkpoint ID；
- base model name and revision；
- training dataset manifest hash；
- training config hash；
- code commit；
- start/end time；
- random seed；
- tokenizer revision；
- adapter/merged status；
- artifact checksum；
- 对应结果文件清单。

#### 验收标准

- 每个表格数字都能追溯到唯一 checkpoint、数据版本和评价脚本；
- 论文中只有一张权威 checkpoint lineage 表；
- 活动配置、归档记录、正文和图表中的命名完全一致；
- 无法证明身份的历史结果只能标记为 exploratory，不能放在主结果表。

---

### 3. 统一样本数量、分析单位和数据划分

#### 发现的问题

章节中同时出现多组不能相互核对的会议数、section 数、label 数和训练样本数。部分描述混合了 meeting、meeting-section、question-answer pair 和 generated trace 等不同分析单位。现有构建器使用 chronological meeting split，但部分 metadata 又写成 seed-42 random split。

#### 建议修改

建立唯一的 sample-flow ledger，并由代码自动生成论文中的数据流程表。建议至少包含：

| 阶段 | 分析单位 | Train | Eval | Test | 排除原因 | Manifest |
|---|---:|---:|---:|---:|---|---|
| Raw meetings | meeting |  |  |  |  |  |
| Meeting sections | meeting-section |  |  |  |  |  |
| QA construction | QA row |  |  |  |  |  |
| Analysis SFT | trace |  |  |  |  |  |
| Analysis GRPO | prompt |  |  |  |  |  |
| Minutes SFT | meeting-section |  |  |  |  |  |
| Decision task | meeting |  |  |  |  |  |

当前仓库快照中可观察到的若干口径包括：

- QA：4,887；
- raw split：3,890 / 535 / 462；
- analysis SFT：3,112 / 533 / 461；
- analysis GRPO：1,082 / 533 / 461；
- Minutes：3,883 / 533 / 461；
- unique meeting-section：577 / 72 / 70。

这些数字只能作为修订时的核对起点，最终应由新的 canonical manifest 自动计算，而不能直接复制进论文。

建议采用 chronological meeting split 作为主分析：

- 按 meeting 划分，而不是按行划分；
- 同一会议的所有 section、问题和 trace 必须继承同一 split；
- 后续所有任务继承同一 split manifest；
- random split 如需保留，只能作为稳健性分析。

#### 验收标准

- 正文、附录、配置和数据 manifest 的所有计数一致；
- 每个数字明确标注分析单位；
- 一个会议不会跨越 train/eval/test；
- 论文中的样本流程表可通过单个脚本重新生成。

---

### 4. 修正 leave-one-out 指标定义和统计检验

#### 发现的问题

论文将 leave-one-out 量定义为：

```text
Δ = cos(full output, target) - cos(masked output, target)
```

但当前实现将实际 Minutes 同时作为 `with-p` 输出和 target，因此保存的量实质上是：

```text
1 - cos(masked output, actual Minutes)
```

这属于 cosine distance，而不是论文所称的 cosine similarity difference。当前 0.20–0.26 的结果因而被错误命名和解释。此外，聚合后使用不配对 Welch 检验也不符合样本的配对结构。

#### 建议修改

对于每个 meeting-section、indicator 和 seed，显式保存：

```text
s_full   = cos(g(full indicators), target)
s_masked = cos(g(masked indicators), target)
delta    = s_full - s_masked
```

建议将该部分重新命名为：

> **Indicator Sensitivity Analysis**

不要使用 Shapley value 或 causal attribution 等更强术语，除非实施真正的联合子集采样和相应识别设计。

统计分析建议：

1. 对同一 meeting-section 的 full/masked 输出做配对比较；
2. 以 meeting 为 cluster 进行 bootstrap；
3. 报告 point estimate、95% CI 和有效样本量；
4. 对多个指标的检验使用 Holm 或 FDR 校正；
5. 至少加入以下对照：
   - neutral-value replacement；
   - within-date indicator shuffle；
   - directional counterfactual，例如通胀上升/下降；
   - repeated decoding seeds。

#### 验收标准

- 公式、代码变量、结果列、图标题和正文解释一致；
- 每个 LOO 结果包含 `s_full`、`s_masked` 和 `delta`；
- 推断过程保留配对关系并对 meeting 聚类；
- 论文不再把 cosine distance 写成 cosine similarity。

---

### 5. 重做文本生成评价

#### 发现的问题

当前文本评价主要依赖 embedding cosine 和 BERTScore 等语义相似度，而且现有活动评价器缺少用于复核显著性检验的逐行分数。论文中极大的 t-statistics 没有对应的样本量、标准误、零假设和可重现结果文件。语义相似并不能证明宏观数值、经济方向和政策表述正确。

#### 建议修改

主实验应在同一份干净、无泄漏、固定的 test set 上至少比较：

- `chk-0`
- `chk-1`
- `chk-2`
- `chk-3`

这样才能回答分阶段后训练是否产生增量价值，而不仅仅是比较首尾模型。

自动评价建议分为四组：

1. **语义相似性**
   - BERTScore；
   - embedding cosine；
   - 至少一个与训练奖励不同的独立 encoder。
2. **表层文本质量**
   - ROUGE-L；
   - 长度、重复率和格式合规率。
3. **事实和数值一致性**
   - 指标值是否正确；
   - 单位是否正确；
   - 时间范围是否正确；
   - 是否产生输入中不存在的数字；
   - unsupported claim rate。
4. **方向一致性**
   - inflation、growth、employment 等变化方向是否与输入一致；
   - policy stance 语言是否与输入证据相符。

所有统计量应从逐行结果计算，并至少报告：

- paired mean difference；
- meeting-clustered bootstrap 95% CI；
- sample size；
- effect size；
- multiple-comparison correction；
- invalid/missing output 数量。

#### 人工专家评价

建议抽取 50–100 个按时期和 section 分层的 meeting-section 样本，由至少两名不知道模型身份的评价者进行盲评。评价维度可包括：

- factual accuracy；
- numerical accuracy；
- directional consistency；
- unsupported claims；
- coherence；
- FOMC-style fidelity；
- overall usefulness。

同时报告：

- 评价说明和量表；
- 评价者背景；
- inter-rater agreement；
- disagreement adjudication；
- 模型顺序随机化方法。

#### 验收标准

- 结果来自共同、无泄漏的 test sample；
- 所有 checkpoint 使用相同输入、解码设置和评价脚本；
- 逐行评价文件可复查；
- 统计显著性与实际效应大小同时报告；
- 至少有一项人工事实性评价支持核心结论。

---

### 6. 重做政策行动分类比较

#### 发现的问题

`chk-4` 与 `chk-0` 当前使用不同的有效样本数，因此不能直接比较。部分 `chk-4` 训练奖励直接使用 vote accuracy，正文却称模型未使用 realized decisions 训练。此外，现有重建数据表明决策训练样本可能与 GRPO eval/test 会议发生重叠。

#### 建议修改

1. 重新构建 meeting-level decision split，并继承主 chronological split。
2. 检查 decision SFT、decision GRPO 和最终 test meeting 之间的交集。
3. 所有模型仅在共同有效样本上比较。
4. 将 invalid parsing 计为错误，并另行报告 invalid rate。
5. 至少报告：
   - accuracy；
   - balanced accuracy；
   - macro-F1；
   - per-class precision/recall/F1；
   - confusion matrix；
   - paired bootstrap CI；
   - McNemar test。
6. 加入简单基线：
   - majority class；
   - previous-meeting action；
   - 简单 multinomial/logistic model；
   - 如可行，加入仅使用数值指标的传统分类器。

#### 结论边界

如果模型输入包含会后 Minutes 或决定发生后的信息，任务只能称为：

> historical policy-action inference 或 ex-post classification

不能称为严格的 real-time policy forecasting。

#### 验收标准

- 训练、验证和测试会议完全不重叠；
- 所有模型在同一个 meeting set 上评价；
- 标签在训练阶段如何使用被准确披露；
- 简单基线与模型使用完全一致的信息集。

---

### 7. 重建“时点可获得信息”定义

#### 发现的问题

当前数据筛选主要依据 `observation_date <= meeting_date`，但没有使用真实发布日期或 vintage。某个观察月份的数据可能在会议后才发布。例如，2023 年 7 月 PCE 数据在 2023 年 8 月底才发布，不能作为 2023 年 7 月会议时的实时输入。

此外，当前 target-rate 读取方式可能使用 meeting 后一天的数据，也需要解释这代表决定后的目标值还是预测输入。

#### 建议修改

每条宏观数据至少记录：

- `observation_period`
- `release_timestamp`
- `vintage_timestamp`
- `meeting_timestamp`
- `available_at_meeting`
- `source`

主分析中只能使用：

```text
release_timestamp <= meeting_timestamp
```

如无法获得完整 vintage：

1. 明确称其为 revised historical data；
2. 不声称 real-time forecasting；
3. 对有实时 vintage 的指标做子样本稳健性分析；
4. 在限制部分说明修订值可能带来的 look-ahead bias。

#### 验收标准

- 每个输入变量都能回答“会议当天是否已经公开”；
- 论文清楚区分 observation date、release date 和 vintage；
- 所有 ex-ante 声明仅建立在真正可获得的信息上。

---

### 8. 处理计量经济学和“市场影响”部分

#### 发现的问题

当前计量部分没有充分说明模型、变量、标准误、样本构成、固定效应和识别假设。已有结果的统计证据较弱，而且未公开发布的合成文本不能直接产生可观测的市场影响。

#### 建议方案

在当前修订阶段，推荐将该部分降级为：

> exploratory appendix

如果希望保留在正文，则至少需要：

1. 明确写出完整回归方程；
2. 定义因变量、核心自变量、控制变量和样本窗口；
3. 说明标准误聚类方式；
4. 区分描述性相关与因果效应；
5. 提供完整回归表、样本数和缺失处理；
6. 说明生成文本没有公开发布，因此“market impact”只是文本表示与历史市场变量的关联，而不是市场对该文本的反应。

#### 验收标准

- 不再使用无法由设计识别的因果语言；
- 弱或不显著结果被如实解释；
- 若保留正文，数据和回归代码必须可完全复现；
- 若不能满足以上条件，则移入附录并从贡献声明中删除。

---

## 三、建议重写研究问题

建议将研究问题改写为三个可以由修订后实验直接回答的问题。

### RQ1：分阶段后训练是否改善生成质量？

> Does staged post-training improve held-out, reference-free generation of FOMC-Minutes-style text relative to the base model and intermediate checkpoints?

对应证据：

- `chk-0`、`chk-1`、`chk-2`、`chk-3` 的共同测试集对比；
- paired and meeting-clustered uncertainty；
- 自动指标和盲评。

### RQ2：生成内容是否忠实并合理响应输入变化？

> Are the generated texts numerically and directionally consistent with the supplied indicators, and do they respond coherently to controlled input perturbations?

对应证据：

- 数值和事实一致性；
- unsupported claim rate；
- indicator sensitivity；
- counterfactual tests；
- expert evaluation。

### RQ3：模型表示是否对历史政策行动推断有辅助价值？

> Do the learned representations provide incremental information for historical policy-action inference under a common, leakage-free evaluation design?

对应证据：

- 共同 meeting sample；
- 简单基线；
- balanced metrics；
- paired inference；
- 明确限定为 historical/ex-post。

---

## 四、建议的最小补实验集合

若计算预算有限，建议按以下最小集合补实验。它们比扩大模型规模或加入更多外围分析更重要。

### 实验 A：干净测试集上的 checkpoint ablation

在同一个 reference-free test set 上运行：

```text
chk-0 vs chk-1 vs chk-2 vs chk-3
```

控制：

- 相同 prompt；
- 相同 decoding parameters；
- 相同最大长度；
- 相同评估器；
- 相同有效样本。

### 实验 B：文本真实性评价

至少补充：

- 数值正确率；
- 方向正确率；
- unsupported claim rate；
- 独立语义 encoder；
- meeting-clustered paired CI。

### 实验 C：盲法专家评价

使用 50–100 个分层样本，至少两名评价者，报告 agreement 和 adjudication。

### 实验 D：输入敏感性

对选定核心指标进行：

- full input；
- masked input；
- neutral replacement；
- shuffled value；
- directional counterfactual。

每个条件使用多个 decoding seeds。

### 实验 E：政策行动共同样本比较

在完全不重叠的共同 test meetings 上比较：

```text
majority baseline
lag-1 baseline
traditional classifier
chk-0
chk-4
```

---

## 五、论文结构调整建议

建议按如下顺序重写本章。

### 1. Introduction

- 明确任务是历史、指标条件化的文本生成；
- 提出三个可检验 RQ；
- 将贡献限定为方法、数据构建和实证评价；
- 删除超出证据范围的政策和市场影响主张。

### 2. Related Literature

当前文献综述过短。建议扩展并分为：

- central-bank communication and FOMC Minutes；
- data-to-text generation；
- domain adaptation and post-training；
- factuality and numerical faithfulness；
- model sensitivity and counterfactual evaluation；
- monetary-policy action classification；
- real-time macroeconomic data and data vintages。

文献综述应形成研究缺口，而不只是列举相邻工作。

### 3. Data and Information Set

- 原始数据来源；
- observation/release/vintage 定义；
- 分析单位；
- chronological meeting split；
- sample-flow table；
- leakage prevention；
- data manifest。

### 4. Model and Training

- 唯一 checkpoint DAG；
- 每阶段目标和数据；
- 实际量化配置；
- 超参数；
- reward function；
- seed 和模型选择规则。

注意核对正文中的 QLoRA/FP4 描述与配置中的 `load_in_4bit` 是否一致。

### 5. Evaluation Design

- reference-free test protocol；
- checkpoint ablation；
- automatic metrics；
- factual/numerical evaluation；
- human evaluation；
- statistical inference；
- invalid-output handling。

### 6. Results

建议依次报告：

1. sample and audit results；
2. checkpoint generation comparison；
3. factuality and human evaluation；
4. indicator sensitivity；
5. historical policy-action inference；
6. robustness checks。

### 7. Limitations

至少讨论：

- revised versus real-time data；
- teacher and judge dependence；
- limited number of meetings；
- stochastic decoding；
- stylistic similarity不等于政策分析正确；
- ex-post task不能等同于实时预测；
- external validity。

### 8. Conclusion

仅总结被主实验直接支持的发现，不引入新的政策或市场含义。

---

## 六、建议的实际执行顺序

不要先逐句润色正文。建议按以下顺序推进：

1. **冻结当前版本**
   - 保存代码 commit、配置、已有结果和数据哈希；
   - 将旧结果标记为 pre-audit。
2. **修复数据泄漏**
   - 重建 reference-free eval/test；
   - 生成自动审计报告。
3. **统一 split 和 sample ledger**
   - 选定唯一 chronological meeting manifest；
   - 让所有下游任务继承。
4. **确认 checkpoint lineage**
   - 为每个模型建立不可变 manifest；
   - 删除或隔离身份不明的结果。
5. **运行最小核心实验**
   - checkpoint ablation；
   - factuality；
   - sensitivity；
   - decision comparison。
6. **完成统计分析**
   - paired comparisons；
   - meeting-clustered bootstrap；
   - multiple-testing correction。
7. **完成人工评价**
   - 盲法；
   - 多评价者；
   - agreement。
8. **重写研究问题和结论**
   - 依据新结果决定保留哪些主张。
9. **最后进行语言和版式编辑**
   - 统一术语、表格、图题、交叉引用和时态。

---

## 七、建议加入仓库的维护产物

为了让本章在后续修改中不再出现口径漂移，建议新增以下机器可读文件：

```text
manifests/
  canonical_meeting_split.json
  sample_flow.json
  checkpoint_lineage.json
  evaluation_runs.json
  data_sources_and_vintages.json
```

以及以下自动报告：

```text
reports/
  leakage_audit.md
  sample_flow.md
  checkpoint_audit.md
  evaluation_summary.md
```

每张论文主表最好通过脚本从 row-level results 生成，并在表格旁记录：

- run ID；
- checkpoint ID；
- data manifest hash；
- evaluator version；
- code commit。

同时应删除代码库中任何明文 API credential，并立即轮换已经暴露的密钥。论文复现说明中只保留环境变量名称，不保留真实值。

---

## 八、提交前检查清单

### 数据

- [ ] Eval/test 提示及所有中间输入不包含 reference-derived information。
- [ ] 所有任务使用同一 meeting-level split。
- [ ] 数据按 release time/vintage 审计。
- [ ] 所有计数由 canonical manifest 自动生成。
- [ ] 分析单位在每张表中明确标注。

### 模型

- [ ] 每个 checkpoint 的父模型唯一且可验证。
- [ ] 配置、metadata、正文和图表命名一致。
- [ ] 每项主结果能追溯到唯一 artifact。
- [ ] 超参数和量化描述与实际配置一致。

### 评价

- [ ] `chk-0` 至 `chk-3` 在共同干净测试集上比较。
- [ ] 文本评价包含事实、数值和方向一致性。
- [ ] LOO 公式与实现一致。
- [ ] 决策模型在共同会议样本上比较。
- [ ] invalid outputs 被计入失败并单独报告。
- [ ] 至少完成一项盲法人工评价。

### 统计

- [ ] 使用配对检验。
- [ ] 对 meeting 进行聚类或 cluster bootstrap。
- [ ] 报告效应大小、置信区间和样本量。
- [ ] 多重比较经过校正。
- [ ] 所有结果均保留 row-level artifacts。

### 写作

- [ ] 研究问题与实际实验直接对应。
- [ ] 不再使用未经识别的因果语言。
- [ ] 区分 historical/ex-post 与 real-time/ex-ante。
- [ ] 对弱结果和限制进行如实讨论。
- [ ] 计量部分已完整重做或移入探索性附录。
- [ ] 摘要、引言、结果和结论中的数字完全一致。

---

## 九、建议的达标条件

只有在以下条件同时满足后，才建议将本章重新提交评审：

1. reference leakage 已被排除并由自动审计证明；
2. checkpoint 与结果身份可唯一追溯；
3. 样本计数、分析单位和 split 完全统一；
4. 核心结果已在干净测试集上重新运行；
5. 文本质量不仅由相似度指标评价，还包含事实性证据；
6. sensitivity 指标的公式、实现和统计解释一致；
7. decision task 使用无重叠的共同样本和合理基线；
8. 所有 real-time 或 causal 主张均有相应设计支持，否则已删除；
9. 主表和统计量可从 row-level artifacts 复现；
10. 论文结论已根据新实验结果重写，而不是继续沿用旧结论。

完成上述工作后，本章有机会从一个工程内容丰富但证据链不闭合的项目，转变为一章边界清楚、可复现、论证可信的博士论文研究。
