# Chapter 2 与当前 retrain-v2 实现的一致性审阅

审阅时间：2026-08-09 15:51 UTC  
审阅对象：`docs/Chapter2` 现有正文、分节文件与提纲  
审阅口径：以不可变 release/lineage manifest、具体 run 的 resolved runtime config、当前源码及训练日志为优先事实源  
结论等级：**Major revision（必须重构后再作为当前实现说明）**

## 技术摘要

当前文章不能直接作为现行训练链路的准确说明。它混合了三类彼此不同的材料：

1. 2025 年旧归档实验及其结果；
2. 尚未完全实现的原始设计；
3. 2026 年实际运行的 `retrain_v2` 数据、训练与 reward 合同。

仓库目前可以严格证明的主链是：

```text
DeepSeek-R1-Distill-Llama-8B (chk0)
  -> compressed-reasoning SFT (chk1，已训练并合并)
  -> grounded_analysis_v3 GRPO (chk2，训练到 durable step 183/247)
  -> checkpoint-150（已选作 evaluation candidate 并精确合并）
```

当前**不能**严格证明的部分是：

- checkpoint-150 已经在同任务 held-out evaluation 上优于 chk1；
- 当前 chk3 已由所选 chk2 生成数据并训练完成；
- 当前 chk4 已由所选 chk2 训练完成；
- 旧分支 chk2/chk3/chk4 的结果代表当前顺序依赖链；
- synthetic text 具有可识别的历史市场影响；
- 生成的 `<think>` 内容是模型真实、忠实且可解释的内部推理过程。

因此，文章应改成两层叙事：

- **Current retrain-v2 implementation**：只报告当前可核验的 chk0、chk1、chk2 candidate、尚未完成的 chk3/chk4；
- **Legacy archived experiments**：保留旧曲线和旧结果，但明确它们来自不同父模型、数据合同、输出解析器或 reward 版本，不能用于证明当前五 checkpoint 流水线。

## 1. 审阅范围与证据规则

本次审阅覆盖：

- `docs/Chapter2/chapter2.tex`；
- `docs/Chapter2/sections/{intro,literature_review,dataset_construction,model_training}.tex`；
- `docs/Chapter2/chapter2_outline.md` 与既有 `chapter2_review.md`；
- 当前 chk0–chk4 数据 release、训练配置、解析器、reward 源码、训练日志和 lineage manifest；
- 与 retrain-v2 reward、输出解析、DAG 和执行收据有关的自动化测试。

本文中的行号对应上述审阅时间点的工作树；由于 Chapter 2 源文件正在修改，后续编辑可能使行号漂移，应同时按段落主题定位。

发生冲突时，本报告采用以下优先级：

```text
不可变 release/checkpoint/lineage manifest
  > 具体 run 的 resolved_runtime_config.json
  > 当前版本化源码与测试
  > 当前 YAML 模板
  > summary/handoff 文档
  > Chapter 2 现有叙述
```

原因是当前 YAML 可能在 run 结束后继续被修改，不能反推旧 run 的精确参数。例如，同一 chk2 系列出现过 Judge completion budget 2048、4000 和 4096；论文必须绑定 run ID 和 resolved config，不能只引用当前可变 YAML。

## 2. 当前实现事实表

| 节点 | 当前父节点与任务 | 当前状态 | 论文可安全声明的边界 |
|---|---|---|---|
| chk0 | `models/DeepSeek-R1-Distill-Llama-8B` | 本地基座已验证 | 所有当前后继均沿用该权重血缘；不能写成先用 LLaMA-3-8B-Instruct，再在 RL 阶段切换到 DeepSeek |
| chk1 | chk0 + completion-only QLoRA SFT；PIT evidence → analysis | 已完成 2 epochs/170 steps，已合并 | 可以报告 teacher-forced train/eval 指标；不能据此断言自由生成优于 chk0 |
| chk2 | compressed chk1 + `grounded_analysis_v3` GRPO；PIT evidence → analysis | fresh run durable 到 183/247；cp150 已选中并合并为 evaluation candidate | 可以报告血缘与训练诊断；在同任务 held-out 对比完成前，不能称 cp150 为“最佳”或“已晋升的最终 chk2” |
| chk3 | intended：selected chk2 analysis → Minutes-style prose | clean v3 数据已准备，但来自 chk1 analysis，`dag_bindable=false`；当前无绑定 cp150 的训练模型 | 只能称 standalone/provisional proxy data；不能称已完成 chk2→chk3，也不能称 target 为官方 Minutes 原文 |
| chk4 | intended：frozen chk2 analysis → decision JSON | core 数据与直接 GRPO 配置已准备；当前无训练完成的当前链模型 | 只能报告数据/接口准备状态；直接 GRPO 与 Decision-SFT→GRPO 两种方案仍需统一 |

截至本次审阅，cp150 的 common-test generation pipeline 正在运行，结果目录尚无正式 evaluation result。该 common test 本身又是 raw evidence → Minutes，属于 chk1/chk2 的 out-of-task stress test，因此即使完成也不能替代 evidence → analysis 的阶段匹配验证。

### 2.1 可复核的模型血缘

当前 cp150 manifest 记录的关键 SHA256 为：

| Artifact | SHA256 |
|---|---|
| chk0 model | `bfb086cb87e9616e60805fc6f6f95826294b1de70b158cfea2d3e213df38ca11` |
| merged chk1 | `70be17c6d879f70eb98ab54bcf2a29f2f4991b2d7ab73b5696cbb6a0139f83aa` |
| chk1 adapter | `b99fe46315f5ab495ed906906de928856a5f54b20e23797fae01a5818152c8e0` |
| chk2 cp150 adapter | `2130a852906cc201aee192e0927a7ea43f52e1dca0c0fac28900e42f85ef0e81` |
| merged chk2 cp150 candidate | `fe4a207a75d3883606e565e7001dbf66bda81563677dd061bc13fd5440b56c18` |

这证明 cp150 的父子权重关系，但**不证明其任务质量优于 chk1**。

## 3. 当前可信的数据血缘

文章的数据章节不能只替换几个样本数；teacher 输入、数据粒度、split 和目标来源都需要重写。当前可信主流程是：

```text
129 raw meetings
  -> 排除 2009-11-04 解析失败
128 eligible meetings
  -> chronological meeting split: 102 / 13 / 13
2,287 unique meeting_date × atomic_topic candidates
  -> 排除 170 个无法证明 exact D-1 vintage 的样本
2,117 teacher candidates
  -> 排除 2 个 teacher response contract invalid
  -> 排除 43 个 response component 为空的样本
2,072 canonical accepted rows
  -> train / validation / test = 1,683 / 199 / 190
```

训练角色采用确定性 sample-level 哈希分配：

| Train role | Unique rows |
|---|---:|
| SFT-only | 1,190 |
| GRPO-only | 328 |
| Shared replay | 165 |
| Physical SFT train | 1,355 |
| Physical GRPO train | 493 |

reasoning compression 语义审计后排除 1 条存在千倍单位错误和不支持因果推断的样本，所以当前 chk1 active release 为：

| Split | Rows |
|---|---:|
| train | 1,354 |
| validation | 199 |
| sealed test | 190 |

meeting-level 时间边界是：

| Split | Meetings | Date range |
|---|---:|---|
| train | 102 | 2009-01-28 至 2021-11-03 |
| validation | 13 | 2021-12-15 至 2023-06-14 |
| test | 13 | 2023-07-26 至 2025-01-29 |

### 3.1 Teacher 的真实合同

当前 chk1 teacher 不是文章所写的 DeepSeek-R1，也没有看到同会议 Minutes excerpt。实际 acquisition contract 为：

- teacher：provider 返回的 `deepseek-v4-pro`，revision 记录为 `provider-current`；
- 输入：`atomic_topic`、严格 point-in-time fact card、无事实的固定 style guide、输出合同；
- 不输入：目标会议 Minutes、reference excerpt、`current_rate`、`rate_change` 或 decision label；
- response：teacher reasoning 加 plain-text answer；
- 后续 reasoning compressor/auditor：`deepseek-v4-flash`，reasoning effort max；
- 1,744 条被压缩，最终 1,743 条通过审计，answer 在压缩过程中保持不变。

manifest 中的 teacher model binding/hash 是调用与 provenance 绑定，不是可下载远程模型权重的内容哈希；论文不能把它表述为固定 teacher weights SHA。

文章 `dataset_construction.tex:383-412` 当前描述的“meeting date/section/reference excerpt → DeepSeek-R1”不仅过时，而且会把实现中主动移除的目标泄漏重新写成正式方法，必须删除。

需要保留的限定是：payload/provenance 审计只能证明调用合同没有暴露目标材料，不能证明远程 teacher 不会从预训练记忆中回忆历史会议。并且 compressor 与 semantic auditor 同属 DeepSeek V4 Flash 路线，不应称为“外部独立审核”或“人工验证”。

## 4. 当前输出合同

文章多处要求：

```text
<think>...</think><answer>...</answer>
```

这与当前解析器冲突。实际合同为：

```text
[chat template 已提供 opening <think>]
reasoning text
</think>
plain-text answer
```

解析器以**第一个 `</think>`** 为唯一权威边界；其后的全部内容都是 answer。无需 `<answer>` 标签，chk2 的 final answer 还明确禁止 JSON、Markdown、evidence IDs、schema commentary 和格式规划。

| Stage | 当前 final answer 合同 |
|---|---|
| chk1/chk2 | 首个 `</think>` 后的一段简洁纯文本 FOMC analysis |
| chk3 | 首个 `</think>` 后的一段 Minutes-style prose |
| chk4 | 首个 `</think>` 后的严格 JSON：`{"direction":"cut|hold|hike","magnitude_bp":0|25|50|75|100}` |

必须修订 `model_training.tex:89-117`、`:290`、`:442-451` 及 `chapter2.tex:826-829`。否则论文给出的 parser 会把有效输出误判为 fatal output，也会错误诱导模型生成并不存在的 `<answer>` wrapper。

## 5. 当前 chk1 训练事实

文章 `chapter2.tex:594-612` 和 `model_training.tex:738-780` 报告的是旧 run。当前 compressed chk1 的权威口径为：

| Item | Current chk1 |
|---|---|
| Base | DeepSeek-R1-Distill-Llama-8B |
| Data | 1,354 train / 199 validation / 190 sealed test |
| Objective | completion-only teacher-forced SFT |
| Quantization | 4-bit NF4 QLoRA，double quant，BF16 |
| Attention | FlexAttention |
| LoRA | r=32, alpha=64, dropout=0.05 |
| Target modules | q/k/v/o + gate/up/down projections |
| Trainable parameters | 83,886,080 |
| Max sequence length | 4,608 |
| Optimizer / LR | paged AdamW 8-bit / 5e-5 |
| Epochs / steps | 2 / 170 |
| Hardware | 2 × NVIDIA A30 24GB，DDP |
| Effective train batch | 16 |
| Train loss | 1.2323049265 |
| Validation loss | 1.1228511333 |
| Validation token accuracy | 0.6890714103 |

旧文中的 3,128 samples、3 epochs、2,346 steps、约 3.4M trainable parameters、r=8/alpha=16、仅 q/v 以及旧 loss 曲线均不得作为 current chk1 结果。若保留，应整体移入 legacy subsection 并绑定旧 artifact。

这些指标是 teacher-forced token metrics，不能单独证明 chk1 在自由生成时优于 chk0，也不能证明 factual grounding。

## 6. 当前 chk2、reward v3 与训练诊断

### 6.1 训练合同

fresh run 的 resolved runtime config 显示：

| Item | Current fresh chk2 run |
|---|---|
| Parent | merged compressed chk1 |
| Data | analysis_grpo：493 train / 199 validation / 190 sealed test；训练时 `do_eval=false` |
| Policy GPU | GPU1，一张 A30；4-bit QLoRA policy |
| Judge GPU | GPU0，Qwen3.5-9B HTTP Judge |
| Prompt / completion limit | 2,560 / 4,096 |
| Judge max model len / completion / reserve | 12,288 / 4,000 / 4,352 |
| Generations / generation batch | 4 / 4 |
| Optimizer / LR | paged AdamW 8-bit / 5e-7 |
| Loss | DAPO-style，group reward scaling，token-level importance sampling |
| Epochs / intended optimizer steps | 1 / 247 |
| Durable endpoint | 183/247 steps（74.09%）；trainer epoch 字段为 0.74239，`best_model_checkpoint=null` |

论文不能笼统写“两张 GPU 同时训练 policy”。当前 chk2 的拓扑是 GPU1 运行 policy，GPU0 独占 Judge；不是双卡 policy DDP。

### 6.2 `grounded_analysis_v3` 的真实 reward

Judge 接收完整 reconstructed think + answer 以及允许的 PIT evidence。五项 rubric 均为 0..4：

- data fidelity；
- trend reasoning；
- policy relevance；
- uncertainty calibration；
- FOMC style。

Judge 还可返回最多 8 条结构化 violation。只有经过大小写/空白归一化后能在对应 think/answer 原文中匹配的 quote 才能进入事实 penalty；format、omission、style violation 不触发事实 penalty。

当前公式是：

```text
base =
    0.50 * judge_score
  + 0.25 * answer_numeric_score
  + 0.15 * structure_score
  + 0.05 * answer_concision
  + 0.05 * reasoning_efficiency

penalty = min(
    0.50,
    0.20 * major_answer_error
  + 0.08 * minor_answer_error
  + 0.04 * major_think_error
  + 0.02 * minor_think_error
)

reward = clip(base - penalty, 0, 1)
```

空 answer 或经 quote 验证的目标会议 decision/Minutes 泄漏直接为 0。不存在文章所写的普遍 0.25 hard cap，也不存在“七项 1–5 分后除以 35”的当前实现。

`reasoning_efficiency` 在 768 tokenizer tokens 内为满分，至 1,536 线性降为 0，并叠加 token 4-gram 重复惩罚。

一个需要在发表前核对的实现漂移是：数字辅助评分当前仍会记录部分 list/observation count，而 reward-v3 的设计意图是排除字段数量等非事实数字。建议为这一点增加明确单测并决定以设计还是当前代码为准。

### 6.3 fresh run 到 step 183 的训练诊断

对重启日志做逆向 reconciliation、移除重复历史后，当前可报告：

| Metric | Step 1–183 |
|---|---:|
| mean step reward | 0.1720 |
| median step reward | 0.1605 |
| completion median reward | 0.0807 |
| completion reward = 0 | 38.46% |
| Judge 五项全零 | 91.74% |
| Judge 五项全四 | 6.76% |
| Judge intermediate | 1.37% |
| invalid violation share | 47.35% |
| mean answer numeric score | 0.9260 |
| structure pass | 96.93% |
| mean answer concision | 0.9018 |
| mean reasoning efficiency | 0.3574 |
| mean base reward | 0.4757 |
| mean penalty | 0.3270 |
| penalty cap reached | 36.89% |
| target leakage | 0 |
| Judge retry rate | 0.41% |
| clipped completion ratio | 3.01% |
| reward slope per 100 steps | +0.00143 |

这一轨迹基本平坦且 Judge 极度二元化。它不能证明 chk2 已改善；还提示以下有效性风险：

- rubric collapse：约 91.7% 全零，约 6.8% 全四，中间分辨率极低；
- reward 可能同时通过 rubric 和 factual penalty 重复惩罚同一事实问题；
- invalid violation 接近一半，说明 Judge 的 quote 定位可靠性有限；
- think violations 为 0 不能自动解释为 reasoning 无错误，也可能是 Judge 没有有效审查 think；
- 训练 reward 不是独立 held-out quality metric。

cp150 的平均 step reward 约 0.1757，最后 10 步约 0.2075；但 131–140 步约 0.2424，高于 141–150。cp150 是合理的**操作性候选**，不是由 validation model selection 证明的全局最佳 checkpoint。

## 7. 阻断级问题与所需修改

### P0-1：当前链与 legacy 分支混写

主要位置：`chapter2.tex:70-80`、`:594-693`、`:705-915`、`:1171-1290`、`:1297-1352`。

现文一方面说没有真实 chk1→chk2 链，另一方面又把旧 chk1–chk4 结果写成整体流水线结果。前者对 legacy artifacts 可成立，但对当前 chk0→compressed chk1→cp150 血缘已不成立。

所需修改：给所有表、图和结果增加 `lineage_scope`、run ID、dataset release、parent SHA、reward version；将旧实验整体隔离为 archival evidence。

### P0-2：基座叙事错误

主要位置：`chapter2.tex:92-146`，`chapter2_outline.md:31-34`。

当前文章暗示 SFT 先从 LLaMA-3-8B-Instruct 开始，RL 时再换成 DeepSeek。实际 chk0 从一开始就是 DeepSeek-R1-Distill-Llama-8B，后续 chk1/chk2 沿同一权重链。

所需修改：将 Llama 3.1 只作为架构背景，随后明确 chk0 和全部后继血缘。128k 是模型架构容量，不是本项目实际训练 context。

### P0-3：teacher 和目标泄漏合同写反

主要位置：`dataset_construction.tex:383-420`。

现文写 teacher 看同会议 Minutes reference excerpt；当前实现明确禁止该输入。现文 teacher 名称也与 acquisition manifest 不符。

所需修改：按第 3.1 节重写，并把 fact-free style guide 与 target Minutes 清楚区分。

### P0-4：输出标签与 parser 错误

主要位置：`model_training.tex:89-117`、`:290`、`:442-451`，`chapter2.tex:826-829`。

所需修改：统一为 first-`</think>` boundary；删除 chk1/chk2/chk3 的 `<answer>` wrapper 要求。

### P0-5：数据计数、粒度和 split 已过时

主要位置：`dataset_construction.tex:20-122`、`:569-601`。

现文同时出现 123/128 meetings、486/718 sections、4,887/4,889 rows，以及互不闭合的 section/trace 数。`.xlsx` 与 `.csv` 双格式副本也不能作为独立样本相加。

所需修改：所有计数从不可变 manifest 生成，并为每个表增加 `grain`：meeting、meeting-section、meeting-topic、unique sample 或 physical oversampled row。

### P0-6：chk1/chk2 超参数、结果和 reward 是旧版

主要位置：`chapter2.tex:594-645`，`model_training.tex:425-638`、`:738-780`。

所需修改：用第 5、6 节当前事实替换；旧 0.70–0.78 reward、约 1094-step GRPO 只能标成 legacy。GRPO loss 可以为负，不能与 SFT cross-entropy loss 横向比较。

### P0-7：chk3 的父节点、输入和 target 均写错

主要位置：`chapter2.tex:38`、`:61`、`:649-653`，`model_training.tex:124-173`，`dataset_construction.tex:425` 以后。

当前 prepared release：

- 来源是 chk1 canonical analysis，不是 sealed selected chk2；
- student prompt 只有 analysis，不含 section title 或 meeting date；
- target 是 DeepSeek 生成并复核的 Minutes-style paragraph，不是官方 Minutes 原文；
- release 明确 `dag_bindable=false`；
- 当前无绑定 cp150 的 chk3 模型。

所需修改：把它写成 standalone provisional dataset，并在方法层明确二选一：继续 synthetic Minutes-style distillation，或重新构建 official-Minutes target。两者的研究问题和 fidelity 验证不同。

### P0-8：chk4 的 prompt、输出、reward 与训练状态写错

主要位置：`chapter2.tex:40`、`:476-578`、`:666-682`、`:1173-1207`，`model_training.tex:179-218`、`:642-668`。

当前 prepared `decision_grpo` rows 的 prompt 只包含 target-neutral pre-meeting analysis；模板 system prompt 则写成“pre-meeting evidence and frozen chk2 analyses”。这两者仍需统一，但都不包含 meeting date、current rate 或 gold action。最终 answer 是 direction/magnitude JSON。`decision_dense_v2` 是确定性 reward：

```text
0.05 format
+ 0.45 direction correctness
+ 0.30 magnitude proximity
+ 0.20 exact match
```

它不是 LLM-as-Judge。gold decision 没有进入 prompt，但会直接进入 GRPO reward，所以“模型没有在 realized decisions 上训练”是错误的。

当前计数为：128 unique meetings，102/13/13；physical train 141 是 cut/hike oversampling 的结果，不是 141 次独立会议。pre-2009 supplement 仍在 preflight/in-progress，不能写成已进入主结果。

所需修改：先确定唯一正式路线——直接 Decision-GRPO，或 Decision-SFT warm-start→GRPO——再写论文。当前没有 current-chain chk4 model result。

### P0-9：核心评估任务与各节点产品不匹配

主要位置：`chapter2.tex:186-247`、`:705-915`、`:1303-1318`。

旧 common test 是 raw D-1 evidence → Minutes section；chk1/chk2 的任务是 evidence → analysis，chk3 是 analysis → Minutes。因此旧 test 只能作为 robustness stress test，不能作为 chk1/chk2 的主要能力比较。

所需修改：采用第 10 节的 stage-aligned evaluation，并将旧 benchmark 移到 legacy/robustness 小节。

## 8. 高优先级有效性与复现问题

### P1-1：当前数据源应写成 ALFRED exact D-1

`dataset_construction.tex:142-147` 的 FRED+WRDS、102 indicators 属于 legacy raw corpus。当前 active retrain evidence 使用 ALFRED exact D-1 vintage；same-day observation 被排除，current-vintage fallback 被禁止。人工示例不得包含会议当天尚未发布的数据，例如当前文稿中的 2023 年 7 月 PCE 示例。

### P1-2：shared replay 与会议覆盖需披露

165 个 train rows 同时进入 chk1 SFT 和 chk2 GRPO，这是显式 replay，不是 split 泄漏，但意味着 chk1/chk2 的 train metrics 不能被当成独立样本比较。GRPO train 的 493 rows 实际覆盖 101 个而不是 102 个 train meetings，也应按 manifest 报告。

### P1-3：训练配置必须绑定不可变 run

每个实验表至少报告：

- run ID；
- parent model/release SHA；
- dataset release SHA；
- resolved runtime config；
- reward name/version；
- Judge model、server budget 和 tokenizer；
- checkpoint adapter/merged SHA；
- log reconciliation 规则。

### P1-4：硬件和环境应精确写出

- 硬件：2 × A30 24GB；
- chk1：双卡 DDP；
- chk2：GPU1 policy、GPU0 Judge；
- 验证本报告所用环境：`fomc_trainer`；PyTorch 2.10.0+cu128、Transformers 4.57.6、TRL 1.2.0、PEFT 0.15.2、Accelerate 1.4.0、bitsandbytes 0.48.2。

环境版本应最终从 run receipt/lockfile 固化，而不是依赖审阅时的可变环境。

### P1-5：决策类别不能用 daily EFFR zero days 构造

`chapter2.tex:487-530` 用 2000–2025 每日 effective FFR 变化频数，其中大量 zero days，解释 meeting-level FOMC decision 类别。这在构造效度上不成立。应使用 meeting-level target range changes；EFFR 仅用于 same/next-day reconciliation 或审计。

### P1-6：Research questions 与贡献过度声明

`intro.tex:26-36` 的“market impact”不可识别，因为 synthetic text 没有在历史时点发布。应改成探索性的 historical association，并明确非因果。

`intro.tex:60-71` 中以下表达缺乏当前证据：continuous distribution、meaningful FedWatch comparison、one of the first、interpretable reasoning paths、actionable policy insights、broader realism。建议把贡献限定为：

- PIT/leakage-controlled 数据和 teacher provenance；
- 可审计的五 checkpoint 计算图；
- 版本化 grounded-analysis 和 decision reward；
- fatal-output accounting、meeting-level split 和 lineage-aware evaluation；
- 明确区分 similarity、sensitivity、grounding、association 和 decision accuracy。

### P1-7：文献综述不足以支撑当前实现

`literature_review.tex` 仅 29 行，至少应补入：

- instruction tuning、SFT、LoRA/QLoRA；
- RLHF/RLAIF 与 GRPO/DAPO-style optimization；
- LLM-as-Judge bias、reward hacking、rubric collapse；
- numerical grounding、temporal leakage、PIT/vintage data；
- CoT faithfulness，以及 rationale 不等于 interpretability；
- central-bank communication NLP；
- synthetic text external validity 与非因果边界。

## 9. 技术、公式和语言问题

以下问题不一定改变主结论，但正式版本必须修复：

| Location | Problem | Required correction |
|---|---|---|
| `chapter2.tex:101-129` | SwiGLU 说明与符号不够准确 | 按实际 Llama block 定义重写，区分 gate/up/down projections |
| `chapter2.tex:160` | 声称 LLM 总选最高概率 token | 仅 greedy decoding 如此；采样会按温度/top-p 随机选择 |
| `chapter2.tex:165` | 把 F1 称为 classification accuracy | 改为 precision/recall 的调和平均，另报 accuracy |
| `chapter2.tex:253` | `ROGUE`、`BOG` 拼写错误 | 改为 ROUGE、Bag-of-Words/BoW |
| `chapter2.tex:255` | cosine 一般范围写成 [0,1] | 数学范围通常为 [-1,1]；若映射/经验非负需另说明 |
| `chapter2.tex:259` | 说使用 LLaMA hidden states | formal test 实际使用 frozen MPNet，应统一 |
| `chapter2.tex:286` | `Gemini 3-27B` 很可能写错 | 核对是否为 Gemma 3 27B，并给版本化来源 |
| `model_training.tex:326` | 按 SGD 描述当前优化 | 当前为 paged AdamW 8-bit；按 teacher forcing 和真实 optimizer 重写 |
| `model_training.tex:329-339` | cross-entropy/response 公式含混 | 明确只对 completion tokens 计算负对数似然及 mask |
| `model_training.tex:734` | LoRA rank 与 FP4 说明错误 | 当前 r=32，4-bit NF4 QLoRA；准确区分存储与计算 dtype |
| `chapter2.tex:1173` | 把 decision accuracy 称为 factual grounding | 改为 held-out label-prediction performance |
| 多处 | grammar 与术语不稳定 | 统一为 FOMC Minutes、point-in-time、teacher、checkpoint、reward |

章标题建议改为：`Synthetic Text Generation with a Fine-Tuned Large Language Model`。

TeX 静态检查方面，当前 begin/end 环境数量平衡；旧 review 中提到的 label mismatch、表格列数问题部分已被修复。当前环境没有检测到可用的 `latexmk`/`pdflatex`，因此本次没有完成实际 PDF 编译验证。

## 10. 建议的阶段匹配评估

### 10.1 chk0 / chk1 / chk2：PIT evidence → analysis

在同一 frozen validation/test prompts、同一 decoding seed 下比较：

- fatal/empty/`</think>` completion rate；
- answer length 与 truncation；
- answer factual/numerical support；
- trend direction 与 uncertainty calibration；
- target leakage；
- v3 reward 的每个分量，而不仅是总分；
- Judge disagreement 或至少小规模 blinded manual audit。

只有这一层可以回答“chk2 是否比 chk1 改善 analysis”。checkpoint 选择不得使用 sealed test。

### 10.2 chk3：frozen selected-chk2 analysis → Minutes

所有模型使用同一 frozen analysis 输入，评估：

- claim、quantity、date、direction、uncertainty 保真；
- 新增 unsupported claims；
- Minutes style/similarity；
- fatal/length completeness。

style similarity 不等于 fidelity，应分开报告。如果继续使用 synthetic Minutes-style target，必须披露 teacher provenance；若改用官方 Minutes target，则重新定义 leakage 与时态边界。

### 10.3 chk4：pre-meeting analysis → decision JSON

在相同 held-out meetings 上报告：

- valid JSON rate；
- direction accuracy 与 macro-F1；
- magnitude MAE / within-25bp；
- exact match；
- confusion matrix；
- 按 cut/hold/hike 的类别指标；
- unique-meeting 与 physical-oversampled row 的严格区分。

不同 information set 的 FedWatch 或其他 benchmark 不得组合成统一 leaderboard。

### 10.4 Legacy common test

保留 frozen manifest、D-1 evidence、target scorer-only、fatal row accounting、meeting-cluster bootstrap 与 Holm correction，这些设计是现文较扎实的部分；但将其明确标为 out-of-task robustness stress test。

## 11. 推荐的文章结构

1. **Research scope and claim boundary**  
   明确本章是当前实现报告、legacy reconstruction，还是二者并列。

2. **Data and temporal controls**  
   以 128 meetings、2,072 canonical rows、PIT contract、teacher provenance 和 release audit 为主线。

3. **Versioned checkpoint graph**  
   `chk0 → chk1 → chk2 → {chk3, chk4}`，每条已实现和 intended edge 使用不同线型/状态。

4. **Current chk1 implementation and diagnostics**  
   报告 compressed teacher SFT 和 teacher-forced metrics。

5. **Current chk2 implementation and diagnostics**  
   报告 reward v3、cp150 candidate、reward validity caveats；不预设 improvement。

6. **Prepared downstream tasks**  
   chk3 proxy release、chk4 core data 和仍未解决的接口选择；不得写成已训练结果。

7. **Stage-aligned evaluation protocol**  
   分别评价 analysis、Minutes rewrite、decision。

8. **Legacy archived experiments**  
   旧曲线、旧 LOO、旧 decision/econometric结果全部在此，并保留血缘限制。

9. **Limitations and next validation**  
   Judge collapse、teacher memory、CoT faithfulness、out-of-task stress test、不同 information sets。

10. **Conclusion**  
    只总结实际已验证的节点，不将 prepared data 或 legacy branches 写成完成的当前 pipeline。

## 12. 建议的研究问题

将当前较宽泛且部分不可识别的问题改为：

1. 在 frozen PIT evidence → analysis 任务上，chk1 和 chk2 相对父模型是否提高事实、数字、趋势、完成率和不确定性表达？
2. 在相同 frozen selected-chk2 analysis 输入下，chk3 是否在提高 Minutes-style alignment 的同时保留 claims、quantities 和 uncertainty？
3. 配对 LOO 删除干预是否表现出稳定的局部输入敏感性？
4. 在相同 held-out meetings 上，chk4 是否改善 direction、magnitude 和 exact decision prediction？

其中 LOO 是 sensitivity diagnostic，不是 factual grounding 的证明；generated rationale 也不是模型内部机制的忠实解释。

## 13. 可保留的现有内容

以下内容方向合理，可在更新 scope 后保留：

- `chapter2.tex:44-68` 的五 checkpoint 概念骨架；
- 旧 artifacts 的 lineage audit，但必须标为 legacy；
- `chapter2.tex:188-247` 的 frozen manifest、D-1 evidence、fatal-row accounting、meeting-cluster bootstrap 和 Holm correction；
- similarity 不等于 factual/numeric/reasoning fidelity 的限定；
- signed LOO delta、internal/external target 区分、保留负值及 meeting-level resampling；
- synthetic-text market regression 的非因果声明；
- unequal denominator 和 different information set 的警告；
- literature review 中 internal coherence 与 input fidelity 的概念区分。

## 14. 推荐修改顺序

1. 冻结论文事实截止日，并决定 current 与 legacy 的章节边界；
2. 以 manifests 重建数据 flow table，删除 teacher reference excerpt；
3. 统一 `</think>` parser 与三个下游输出合同；
4. 用 resolved runtime config 重写 chk1/chk2 训练与 reward；
5. 明确 chk3 target 选择、chk4 唯一正式路线；
6. 完成同任务 chk0/chk1/chk2 held-out evaluation 后再选择/晋升 chk2；
7. 用阶段匹配结果替换核心 out-of-task 结论；
8. 最后补文献、公式、语言、引用和 LaTeX 编译 QA。

## 15. 未决问题

在文章可以形成最终版本前，需要作出以下设计决定：

1. chk3 的 target 是 official Minutes 原文，还是 synthetic Minutes-style paragraph？
2. chk3 是否需要 section title 和 meeting date？当前 prompt 没有这两个字段。
3. chk4 采用直接 GRPO，还是 Decision-SFT warm-start 后再 GRPO？
4. cp150 的 model selection 指标、validation prompts 和晋升门槛是什么？
5. reward v3 的 Judge collapse 如何处理，是否增加独立或人工小样本校准？
6. `<think>` 是否作为可发布研究对象？若是，需要 CoT faithfulness 的独立验证；若不是，应只把它视为训练时生成轨迹。

## 16. 验证记录与限制

在 `fomc_trainer` conda 环境执行：

```text
pytest -q \
  tests/test_retrain_v2_rewards.py \
  tests/test_structured_response.py \
  tests/test_retrain_v2_dag.py \
  tests/test_retrain_v2_execution_receipt.py
```

结果：**241 passed in 20.98s**。这验证了当前 reward/parser/DAG/receipt 的实现一致性，但不验证模型生成质量或论文实证结论。

本次没有修改 `docs/Chapter2` 源文件，没有启动、停止或改动任何训练/评估任务，也没有把仍在运行的 cp150 common-test generation 当作完成结果。
