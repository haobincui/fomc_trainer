# Chapter 2 重新审阅意见

**审阅标准：金融学博士论文 / 匿名审稿口径**

**审阅日期：2026-08-26**

**审阅对象：当前工作区中的 Chapter 2 现行版本，而非 archive 版本**

**总体结论：Major Revision（尚不建议进入答辩稿或投稿稿）**

## 一、执行摘要

本章相较此前版本已有实质进步。最重要的改进是：作者已明确采用共享父模型下的两条独立任务分支，并在结论中主动收窄了 synthetic reference、LOO、市场回归和政策决策实验的解释边界；GRPO 未通过预设 gate、决策准确率较低等负结果也得到如实披露。这些处理符合博士论文应有的透明度。

但当前版本仍有数项会改变主要结论的关键问题，因而尚不能视为“只需语言润色”的完成稿：

1. 引言和研究问题仍把证据延伸为 market impact、policy decision support 和 actionable insights，而现有设计最多识别对冻结 synthetic reference 的相对对齐、输入敏感性和历史统计关联。
2. chapter2.tex:893 把核心文本评价误写成 post-2009；冻结 roster 和构造代码已确认实际为 1993–2008、且不与被评价 rewriting-model lineage 的训练会议重叠。该事实错误会误导读者判断外部效度。
3. 本地工件已证明主要数据路径使用严格 ALFRED vintages，这是本章的真实优点；但旧 core ledger 以会议结束日为锚，128 场中有 119 场两日会议的 D-1 实际落在会议第一日，pre-2009 supplement 另有 25 场同类问题。
4. post-2008 core/native decision pathway 的 topic 集合由目标会议的会后 Minutes 提取，topic presence/missingness 本身可能构成结构性前视信息；pre-2009 supplement 和 external Core8 路径不受同一问题影响。
5. chk-1 训练参数数量与本地运行工件存在约 24.6 倍的硬性冲突；结果工件已绑定 cp200 hashes，但正文的 numerator、denominator 和比例均已陈旧。
6. 决策模型没有超过最简单的 Always-Hold 基准。按正文给出的 31 场会议计算，Always-Hold 的 direction/modal/exact-action accuracy 均为 61.29%，而最佳学习模型的 direction accuracy 只有 24.84%，balanced accuracy 也低于 33.33% 的朴素基准。
7. 市场回归不能识别“合成文本的市场影响”，而且合成系数在 FF3 和 FF6 上与官方 Minutes 系数异号；当前也没有正式等价检验。
8. 当前 checkout 缺失正文引用的 checkpoint appendix，并存在未定义表格引用、错误图示和图题裁切等可编译性/呈现问题。
9. 文献综述仅约 528 词、30 行，不足以支撑金融博士论文对货币政策沟通、资产价格识别和实时预测的贡献定位。
10. 核心 external semantic 分数是把 hard-gate failures 置零后的复合 estimand，而非纯语义相似度；保留模型只有 585/1,280 次生成获得 semantic score，数值忠实度通过率仅 46.09%。LOO 中 95.43% outputs 也未保持 numeric multiset。当前主结果必须拆分 coverage、validity 与 conditional semantic alignment。

因此，本章最可信、也最有机会形成可辩护贡献的定位，应是：

> 本章研究任务特定后训练能否提高模型在 meeting-indexed、balanced single-topic panel 上对冻结、确定性、source-grounded synthetic monetary-policy reference 的对齐，并以八主题输入敏感性、历史系数关联和严格的负向政策分类结果作为边界化诊断；它不识别完整 Minutes 的生成质量、合成文本的因果市场影响，也不证明模型具备实时政策预测或决策支持能力。

若作者接受这一收窄定位，本章可以发展为一篇方法透明、负结果有价值的金融文本生成研究；若仍要保留“市场影响”或“政策预测”作为主要贡献，则需要重建实时信息集、加入可交易市场基准，并重新设计识别与推断。

## 二、审阅范围与证据状态

本次审阅逐行检查了以下现行文件：

- docs/Chapter2/chapter2.tex
- docs/Chapter2/sections/intro.tex
- docs/Chapter2/sections/literature_review.tex
- docs/Chapter2/sections/dataset_construction.tex
- docs/Chapter2/sections/model_training.tex

同时只读核验了当前图表、结果说明、训练配置和诊断工件。archive/v20260826 仅用于判断版本变化，不作为当前章节内容。工作区中没有完整论文主文件和当前主 bibliography，且现行 sections 目录缺少被引用的 checkpoint_selection_appendix.tex，因此本次只能进行静态引用检查，不能完成整篇论文级 LaTeX 编译和参考文献解析。文中以下判断分为：

- **已验证事实**：可由当前正文、表格或本地运行工件直接复核；
- **推断性风险**：现有文件不足以排除泄漏、重叠或选择偏误；
- **待补证据**：需要作者提供冻结 manifest、原始来源或重新估计。

## 三、本轮版本值得保留的改进

### 3.1 两分支研究设计已经基本讲清

intro.tex:25 及 chapter2.tex:1589–1601 已明确：chk-2 是 Minutes-style rewriting 分支，chk-3 是独立的 historical action-classification 分支；后者不消费 chk-2 的输出。这一澄清消除了“先生成 Minutes，再据此作政策决定”的错误因果链，应在全文保持一致。

### 3.2 统计单位意识明显改善

dataset_construction.tex:1108–1121 区分了 atomic rows、unique meetings、physical resamples 和 stochastic generations，并采用 meeting-level chronological split。chapter2.tex:951–963 也先在会议内聚合随机生成，再进行配对比较。该做法比把每次生成当作独立样本更符合金融事件研究的统计单位要求。

### 3.3 结果披露较为诚实

chapter2.tex:818 明确报告没有 GRPO checkpoint 通过全部 gate；chapter2.tex:1568–1576 承认绝对决策性能较弱且 GRPO 后表现下降。负结果没有被隐藏，这是本章可信度的重要资产。

### 3.4 结论中的识别边界基本正确

chapter2.tex:1612–1618 已把语义指标限定为 synthetic-reference alignment，虽然其中仍把外部确定性模板错误称为 teacher-generated；1633–1636 将 LOO 限定为 reference-relative sensitivity；1649–1652 否认市场因果解释；1671–1682 也否认实时预测、交易或政策支持能力。当前最有效的总体修订策略，是先纠正 reference provenance，再用结论中的谨慎边界反向统一摘要、引言、研究问题、方法标题和贡献段。

### 3.5 对失败 gate 和小样本选点的局限已有披露

chk-2 选点段承认 12 个确定性样本不构成总体证据；GRPO 段也披露 cp450 的 blind delivery-valid rate 只有 0.30。建议保留这种透明度，并进一步把 formal selected checkpoint 与 exploratory failed-gate artifact 明确区分。

## 四、核心命题与当前证据的匹配

| 当前或隐含命题 | 当前设计实际识别的对象 | 审阅判断 | 建议表述 |
|---|---|---|---|
| 微调提高 authentic FOMC Minutes quality | meeting-indexed、每会一条 balanced Core8 atomic topic 对确定性 synthetic reference 的 gate-adjusted MPNet/BERTScore 对齐 | 证据范围明显较窄 | single-topic synthetic-reference alignment |
| LOO 证明信息充分性或经济重要性 | 删除输入后，相对于 full-information reference 的分数变化 | 不支持充分性或因果重要性 | target-relative prompt sensitivity |
| 合成文本具有市场影响 | 未被市场观察的合成情绪与历史发布日收益的系数关联 | 不识别市场影响 | historical coefficient-association diagnostic |
| 合成文本保留官方 Minutes 的经济内容 | 合成与官方情绪回归系数的距离比较 | 尚无等价检验，且部分期限异号 | limited coefficient proximity |
| 模型可预测 FOMC 行动 | 31 场 retrospective、task-specific-training-held-out meetings 上的随机分类 | 未超过 Always-Hold | no demonstrated predictive value |
| GRPO 改善决策能力 | 单次训练、无 checkpoint 通过 gate，cp450 探索性评估 | 不支持改善 | exploratory failed-gate result |

建议将这张“命题—estimand—证据”对照表转化为正文的方法路线图。金融学读者首先需要知道每个检验回答什么问题，以及它明确不回答什么问题。

## 五、必须优先解决的问题（P0 / submission blockers）

### P0-1. 引言、研究问题与贡献仍显著超出可识别范围

**证据。** intro.tex:43–57 将核心问题写为生成文本能否保留 market impact 和 policy-rate outcomes；intro.tex:82–86 使用 economics-based validation、actionable insights 和 augment human decision-making。chapter2.tex:287 又称显著关系可以说明 realistic content mirroring actual market impact。上述表述与结论中的谨慎界定直接冲突。

**金融学含义。** 合成文本从未在历史时点被市场参与者观察，因此资产价格不可能对它产生历史因果反应。它的情绪分数与收益相关，可能只是二者共同反映同一宏观状态。类似地，事后历史行动分类不等于决策支持。

**必须修改。**

1. 把主问题改写为 task-specific post-training、synthetic-reference alignment、prompt sensitivity 和 historical association；
2. 删除 market impact、realism validation、decision support、actionable insights 等无识别支撑的词；
3. 将 RQ2 的 structured deliberation、RQ3 的 authentic Minutes style、RQ4 的 policy forecasting 改成与实际报告指标一一对应的可检验问题；
4. 删除 intro.tex:14 关于增强 volatility models 的未实施主张，或将其另立为需要新增波动率实验的研究命题；
5. 修复 intro.tex:56 不完整的 RQ4 句子，并给 intro.tex:37 补句号。

### P0-2. 128 场核心评价会议的年份在正文中写错

**已验证事实。**

- chapter2.tex:893：128 FOMC Minutes 被写为 post-2009；
- chapter2.tex:976：Core8 外部评价被写为 1993–2008；
- dataset_construction.tex:1271–1314：正式 external rewriting panel 定义为 1993–2008；
- 冻结 inputs/samples_n128_k10.json 的 128 个 meeting IDs 全部位于 1993–2008；
- external builder 对被评价 rewriting-model lineage 的 meeting overlap 采用 fail-closed 检查，release manifest 也标记 evaluation_only=true、trainable=false。

因此，chapter2.tex:893 的 post-2009 已可确认为正文笔误，而不是仍待判定的样本身份。该错误会让读者误以为核心评价与 2009–2021 rewriting 训练期重叠，必须在提交前修正。这里的 held-out 仅指被评价的 chk-0→chk-1→chk-2 rewriting-model lineage 及其 checkpoint selection；其中 109 个日期进入独立 decision branch 并不形成 rewriting 权重泄漏。该定义也不排除基础模型预训练语料中的潜在记忆。

**必须修改。**

1. 将 chapter2.tex:893 改为 128 regular FOMC meetings from 1993–2008；
2. 在结果表注绑定现有 frozen roster、release ID 和 hash；
3. 明确 evaluated rewriting-lineage overlap=0，并披露独立 decision branch 的日期重叠，但不声称排除了 base-model pretraining contamination；
4. 增加 contamination/fingerprinting 检查，或把该限制写入正文；
5. 只有在结果表并非来自上述冻结工件时才需要重跑；若绑定无误，当前问题主要是正文事实修正。

### P0-3. ALFRED vintage 已实现，但旧 core 与 supplement 的 cutoff 锚点错位

**已验证优点。** 本地 indicator_inputs 和 snapshot manifests 显示：post-2008 core、pre-2009 supplement 及 external paths 均绑定 ALFRED vintage snapshots；snapshot policy 禁止 current-value fallback，并记录 requested vintage、availability-as-of、information-as-of 和 meeting timestamp。因而不能笼统批评本章“使用了今天的修订终值”。

**关键剩余问题。** 实际用于 chk-1、chk-2 和 native decision core 的旧 ledger 把 Minutes/decision date，即会议结束日，作为 meeting_date。修正版 official roster 显示，post-2008 core 的 128 场会议中有 119 场为两日会议；这些旧样本的 end-date minus one day 实际是会议第一日，而不是 meeting-start minus one day。对 3,034 个旧 meeting-topic rows 的工件比较显示，改用 start-date D-1 后有 689 行的 series/observation payload 实际变化（train 501、validation 92、test 96，另有 1 个映射缺口），因此这不是仅影响 metadata 的理论风险。pre-2009 supplement 另有 25 场同类锚点错位。由此可能纳入会议第一日发布的信息，违反严格会前信息集。

应区分未受同一问题影响的路径：1993–2008 external rewriting panel 和 31 场 external decision panels 已使用 official meeting-start D-1。正文目前没有把这些路径差异和已有 ALFRED 证据讲清。

**必须修改。**

1. 以官方 meeting-start timestamp 为唯一锚点，重建受影响的 119 场 core 和 25 场 supplement 输入；
2. 重新训练或至少提供 start-date D-1 的严格敏感性结果，不能只把它列为未来工作；
3. 把现有 ALFRED snapshot policy、vintage fields、fallback prohibition 和 manifest hash 写入正文或数据附录；
4. 在逐序列数据字典中继续披露 source、series ID、vintage date、release timestamp、timezone、变换和 staleness；
5. 金融变量应明确取会议开始前最后一个可交易收盘。

ALFRED 官方说明明确区分历史时点可见数据与后续修订值。本章已经采取了正确方向，当前 blocker 是大范围 meeting-anchor 错位和论文披露，而不是要求从零改用 ALFRED。

### P0-4. post-2008 core pathway 的会后 topic mask 存在结构性 look-ahead 风险

**证据。** dataset_construction.tex:74–129 描述的 post-2008 core/native pathway 先用目标会议的官方 Minutes 段落生成 indicator labels，再形成 meeting-topic observation；735–747 将 available atomic analyses 聚合为决策 brief。虽然 teacher prompt 排除了政策决定、投票和同会议原文，但 topic 是否出现、哪些 topic 被强调，仍由会后文档决定。

该风险不应错误外推到全部路径：pre-2009 supplement 的构造代码冻结 26-topic roster，纳入由 ALFRED 可得性和 coverage 规则决定；external evaluation 固定 exactly Core8。问题只针对 post-2008 native decision path。

**金融学含义。** topic presence/missingness 本身可能编码委员会在该次会议上重点讨论什么，从而泄露事后会议状态。排除原文内容不能自动排除这种 selection-on-topics。与此同时，chapter2.tex:396 又称评价输入固定为 Core8，训练和评价 schema 是否一致也不清楚。

**必须修改。**

1. 重建 post-2008 native decision path，使其预先固定 26-topic 或 Core8 universe；
2. topic inclusion 只能由会前数据 availability 决定，缺失应显式编码；
3. 增加 topic-mask-only、固定 topic、Minutes-derived mask 三组消融；
4. 仅改名不能消除 outcome-informed feature selection；若保留旧路径，只能把结果标为带已知 topic-selection bias 的探索性结果，并提供固定-topic sensitivity。

### P0-5. raw-to-canonical 谱系仍缺失，但 1,683→1,354 的差额主要是可审计的 role allocation

**证据。**

- dataset_construction.tex:102–129 承认缺少从 18,957 到 6,266、5,290/5,397、4,887/5,327、最终 2,072 行的 row-level manifest；
- 513–514 称 2,072 行用于 chk-1；
- canonical train 的 1,683 行由稳定 hash 规则分为 sft_only=1,190、shared=165、grpo_only=328，因此 SFT source 为 1,355；semantic audit 再排除 1 行，最终 cp200 train=1,354；
- split_integrity 和相关工件已记录该 role allocation、topic/year 分布与 sample hashes；
- merge attestation 绑定的 semantic audit 针对 237 个 changed rows：235 个完成判定，其中 1 个通过、234 个失败，记录 611 个 blocking violations，另有 2 个 judge errors；随后由 documented override 放行，其余 1,506 个 unchanged rows 沿用前一 release；
- chk-2 又使用完整 1,683 个训练行。

因此，不能把 329 行全部称为“审计删除”：其中 328 行是旧 SFT/GRPO 角色分配，真正 semantic exclusion 只有 1 行。真实问题有两个：早期 raw-to-canonical 流程仍缺逐行链；当前 chk-1 只做 SFT，却沿用旧的 SFT/GRPO role split，使 328 个合格 train rows 未进入 cp200，而 chk-2 使用全部 1,683 行。该设计选择可能影响阶段间比较。

**必须修改。**

1. 为每条样本提供稳定 row ID、source hash、meeting ID、topic、split、每一步过滤和 rejection reason；
2. 披露 semantic audit 的失败维度、阈值、override 决策人和日期；
3. 在正文准确区分 canonical sample、role allocation、semantic exclusion 和实际 SFT exposure；
4. 解释为何纯 SFT cp200 仍排除 grpo_only 的 328 个合格行，以及为何 chk-1 与 chk-2 使用不同训练总体；
5. 提供 all-1,683 SFT 或等价敏感性分析，并对 override 做 pass-only sensitivity。

### P0-6. chk-1 参数统计存在 manuscript–artifact 硬性冲突

**已验证事实。**

- chapter2.tex:519 写明 QLoRA rank 32、alpha 64，并作用于 q/k/v/o/gate/up/down 七类 projection；
- chapter2.tex:574 报告 3,407,872 个可训练参数，占 8,051,232,768 的 0.0423%；
- 最终 lr=1e-6 checkpoint-200 的 adapter_config.json 给出 r=32，并确认上述七类 target modules；基础模型 config 给出 32 层、hidden size 4,096、intermediate size 14,336 和 8 个 KV heads。按各 projection 的 LoRA A/B 矩阵维度逐层求和，得到 83,886,080 个 adapter 参数；
- base weight index 对应 8,030,261,248 个基础参数，实际 PEFT total 为 8,114,147,328；
- 因而 trainable/PEFT-total = 83,886,080 / 8,114,147,328 = 1.033825%；若以 frozen base 为分母则为 1.044625%。

因此，正文的 numerator、denominator 和 percentage 三项都与当前工件不闭合。smoke diagnosis 只能作为旁证，不应被误写成最终 cp200 的运行日志。

**必须修改。**

1. 以结果 manifest 和 merge attestation 已绑定的 cp200 checkpoint 为唯一根工件；
2. 在正文列出模型 revision、adapter hash、target modules、rank、alpha、83,886,080 trainable 和 8,114,147,328 PEFT-total；
3. 修正参数数量和比例；
4. 将已有逐文件 SHA、resolved runtime config 和代码 commit 汇总到可复现清单。

这项硬错主要影响论文中的模型配置和参数效率陈述；现有结果 hashes 已提供较好的工件身份依据，不应无证据地推断全部结果需要重跑。

### P0-7. 决策模型未超过 Always-Hold，必须将其作为主结果而非附加稳健性

正文给出 31 场会议，其中 7 次 cut、19 次 hold、5 次 hike。由此可直接得到：

| 方法 | Direction accuracy | Balanced accuracy | Meeting-modal accuracy | Exact action |
|---|---:|---:|---:|---:|
| Always-Hold / 0bp | 61.29% | 33.33% | 61.29% | 61.29% |
| 最佳学习模型 chk-3 SFT cp38 | 24.84% | 32.73% | 12.90% | 约 19.35% |

最佳模型不仅在 direction/modal/exact-action 上大幅低于多数类基准，balanced accuracy 也没有超过 33.33%。因此 chapter2.tex:1568 所称 strongest overall decision-quality 只能表示“四个较弱 checkpoint 中的相对最高者”，不能表示存在预测价值。

冻结 evaluation manifest 已对 237 场 task-specific exposure roster 作 fail-closed 排查，并确认这 31 场 direct-decision overlap=0；这是可信的 task-specific holdout 控制。但两组评价均为 retrospective，历史 N=19 仍可能出现在基础模型预训练语料中，不能称 prospective forecast。

还应限定输入合同：外部 N19/N12 使用 deterministic fixed-Core8 contract，而训练 release 使用 teacher-compressed、variable-topic briefs；manifest 明确标记两者 byte_comparable=false。因而当前负结果是“在该外部 Core8 合同下未超过基准”，同时混合了决策能力与 schema distribution shift，不能无条件外推为模型在所有输入合同下的总体能力。

**必须修改。**

1. 至少加入 Always-Hold，并把两个时代不同的 external panels 分别报告；
2. 以 meeting 为独立单位进行配对推断，不能把 310 次生成当作 310 场政策事件；
3. 报告 confusion matrix、macro-F1、strict/recoverable delivery 和 invalid rate；
4. 将结论改为“未显示超过朴素基准的政策方向或幅度预测能力”；
5. 若保留“预测贡献”，则必须进一步加入均匀随机、经验类别概率、lagged action/no-change，以及 Taylor-rule/ordered logit 和会前 Fed Funds futures 隐含概率等基准；若由重复抽样定义概率预测，还应报告 Brier score、log score 与 calibration；
6. 增加 same-contract 训练/评价或 input-normalization sensitivity，把 schema shift 与分类能力分开；
7. 资产价格响应的是相对于市场预期的 target/path surprise，而不是 cut/hold/hike 标签本身。若要连接资产定价，应检验模型概率相对会前期货隐含概率是否含有对政策意外成分的增量信息。

### P0-8. GRPO 没有正式通过选点规则

chapter2.tex:818 已承认没有 checkpoint 通过所有 gate，cp450 的 blind delivery-valid rate 仅为 0.30，低于 0.75 门槛。正文中的 all 77 gates 显然应为 all seven gates。

**必须修改。**

1. formal selected GRPO checkpoint 应记录为 null；
2. cp450 只能称 exploratory failed-gate candidate；
3. 将其结果放入 failure analysis，而不是与正式通过 gate 的模型并列作确认性比较；
4. 使用多个预先设定的独立训练 seeds 重复 SFT/GRPO，并根据目标精度、统计功效和计算预算解释数量；十个 decoding seeds 不能替代训练不确定性；
5. 若要论证 GRPO 的一般效果，需要跨训练运行的均值、方差和 reward-component ablation。

### P0-9. 市场回归不识别 market impact，也尚未证明经济等价

**事件时间问题。** 正文没有清楚披露事件映射，但冻结工件实际已绑定 official Minutes release-date ledger；analysis panel 记录 trading date、release-event flag、meeting ID 和 lag meeting ID，t-1 被定义为前一次 regular meeting。这个实现应写进论文。即使映射正确，合成文本仍由会前信息生成且从未被市场观察，因此它的分数最多代理官方 Minutes 发布时的信息，不能被赋予自身的历史 communication shock。同期 VIX 变化还可能是共同结果或后处理变量。

**数值问题。** 正文表中官方 Minutes 系数为 FF1 -0.0833、FF3 -0.3490、FF6 -0.7240、FF12 -1.5981；chk-2 分别为 -0.0346、+0.0738、+0.0022、-0.3065。FF3 和 FF6 异号，不能用“某些系数显著”替代与官方系数相近的检验。chk-2 与 chk-0 的平均绝对距离仅相差 0.0055，且均值容易被 FF12 主导。

结果表中的部分星号检验实际针对“模型系数减官方系数”的差异；在这种列定义下，显著星号意味着拒绝相等，而不是模型表现更好。表题、列名和注释必须避免把“不等价”视觉呈现为“有效性”。

**单位问题。** 代码先计算 100 倍 log futures price return，再将回归系数乘 100 报告；因此表中数值可称 futures-price log-return basis points，但不能不加限定地称 interest-rate basis points。若希望给出经济上更直观的利率含义，应以 100-P 构造隐含利率 bp 变化并重估。FF1/3/6/12 的合约选择、换月规则和到期处理仍需在正文说明。

**推断问题。** NS_t 的非零识别变异集中于 82 个 Minutes 事件；非事件日仍会影响 nuisance parameters 和方差估计，因此不能把表中的 2,338–2,613 个日度观测等同于 82 个独立事件，也不能说全部样本信息只来自事件日。单个系数表仍报告 HC1；model-minus-official contrasts 则已使用 10,000 次 paired、regime-stratified calendar-year block bootstrap 和 Holm 校正，这是应保留的优点。但该 bootstrap 不自动构成事件窗因果识别或经济等价检验，正文还需解释为何 calendar-year blocks 足以处理事件间依赖和四期限共同冲击。2004–2015 回归也与 2009–2021 的任务训练期部分重叠。

**复制口径问题。** 冻结 manifest 明确标记 exact_tadle_2022_replication=false：本文使用的 LM dictionary 不同于原研究的 custom monetary-policy dictionary，Core8 生成文档与 full official Minutes 的信息范围不同，WRDS continuation convention 也由本项目定义。因此 chapter2.tex:1152 的 same dataset as the original test 应改成 project-specific Tadle-form adaptation。

**必须修改。**

1. 将小节改名为 historical sentiment–Fed Funds futures coefficient-association diagnostic；
2. 画出完整时间线：vintage → meeting → Statement → Minutes release → return window，并把现有 release ledger 与 lag definition 写入正文；
3. 将现有单位准确标为 futures-price log-return bp；若要解释为隐含利率 bp，则重构因变量，并明确合约拼接；
4. 以会议为关键识别单位报告 82 个事件，并解释现有 year-block bootstrap 的依赖结构假设；对单个系数增加适当的时序/事件级稳健推断，并报告 leverage 与 leave-one-event-out sensitivity；
5. 对四期限作联合检验和多重校正；
6. 预设经济含义明确的 equivalence margin，直接检验 synthetic 与 official 的系数差，或使用 TOST；
7. 分别报告 pre-2009 untouched subset 与 2009–2015 overlap subset；
8. 将 chapter2.tex:1154 的 K=10 修正为冻结工件确认的 K=5；现有 bootstrap 和结果均已绑定五个 replicates，若不改变 K 无需因此重算；
9. 将 same dataset/replication 改为 project-specific adaptation，并逐项披露与原研究的不同；
10. 即便改用 Minutes 发布时间附近 30/60 分钟窗口，也只能检验 synthetic score 是否代理实际 Minutes 的发布信息，仍不能识别未被市场观察的 synthetic text 的因果 market impact。

### P0-10. 训练 target 与外部 reference 的 provenance 在正文中混淆

dataset_construction.tex:523–528 称 teacher reasoning 与 official Minutes paragraph 配对；535–547、623–628 和 694–702 又明确说明 teacher 看不到 official Minutes，训练 answer 是 DeepSeek-V4 Pro 生成的 synthetic target。

外部 1993–2008 评价又是第三种对象：构造代码使用确定性 Python 模板从 source evidence 分别生成 analysis 和 reference_text，release 类型为 deterministic_source_grounded_minutes_style_v1；它不是同一 teacher 生成，也不是 official Minutes。chapter2.tex:1615–1616 把外部参考统称 teacher-generated 同样不准确。

**必须修改。**

1. 建立三类对象的 provenance 表：official Minutes、training teacher target、external deterministic synthetic reference；
2. 以冻结 manifest 为准修正 dataset_construction.tex:527 及结论；
3. 外部结果统一称 deterministic source-grounded synthetic-reference alignment；
4. 不应称 official target、official Minutes supervision、teacher-reference evaluation 或 institutional authenticity；
5. 三类文本在字段名、表注和图例中必须始终分开。

### P0-11. 基座模型身份与真实 chat template 存在硬性矛盾

chapter2.tex:70 将 DeepSeek-R1-Distill-Llama-8B 追溯至 Llama-3.1-8B Base，但邻近位置又写成 Llama-3.1-Instruct-8B 或 LLaMA 3-8B-Instruct。model_training.tex:241–263 展示通用 Llama role tokens，并包含不存在的 end_id 写法；本地 checkpoint 的真实模板使用 DeepSeek 风格的 User/Assistant/think/end-of-sentence tokens。模板会改变监督边界、EOS 和生成终止，不能以近似示例替代。

**必须修改。** 建立唯一模型谱系表，绑定 exact repository/revision、父 checkpoint、tokenizer hash、真实序列化模板、EOS/PAD、loss mask 和 adapter hash。chapter2.tex:92–136 的架构教程应删除或按真实配置重写：本地配置为 mlp_bias=false，而正文加入 bias；W1/W2 的三维写法与输入不相容；SwiGLU 也未准确对应 gate/up/down projection。

### P0-12. 核心 semantic estimand 混合了 gate coverage 与条件语义相似度

**评价粒度。** external N=128 semantic panel 不是每场会议的完整 Core8 或完整 Minutes，而是每场会议调度一个 atomic topic，并在 128 场之间实现 topic balance。因此它是 meeting-indexed balanced atomic-topic rewriting panel。LOO 则为每场会议拼接八个 Core8 blocks 后逐块干预，两项实验的输入合同不同，不能合并解释为“完整会议纪要质量”。

**外部 N=128 主结果。** reliability-adjusted MPNet/BERTScore 不是纯 semantic similarity，而是 hard-gate failure 置零后的复合分数。冻结 bootstrap 工件显示，论文 chk-2/cp318（artifact label chk3）的 1,280 次生成中：

- numeric fidelity 仅 590/1,280 = 46.09% 通过，690 次失败；
- 119/128 场会议至少出现一次 numeric-gate failure；
- 只有 585/1,280 = 45.70% 获得 semantic score，695 次被置零；
- chk-1 只有 512 次获得 semantic score，768 次被置零。

因此，表中的 0.061446/0.057130 等增量可能同时来自 gate pass-rate 和条件语义质量变化。它们不能被直接解释为“在可比有效输出上，语言质量提高了同样幅度”。更重要的是，source-preserving 的绝对有效率本身较弱。

**LOO 主结果。** chk-2 的 SFT 合同是 atomic analysis → one paragraph，external semantic 也只调度一个 atomic topic；LOO Full 却把八个 topic blocks 拼成单一输入，仍要求一个段落输出，因而首先是明显的 multi-block out-of-distribution stress test。21,760 个输出全部进入 raw-score estimand，但工件记录 21,196 行 generation diagnostic failure（97.41%）、20,766 行 numeric multiset not preserved（95.43%）和 5,621 行 delivery invalid（25.83%）。这不自动使 target-relative sensitivity 无效，却说明它主要是在大量未通过生成质量诊断的 OOD 输出上估计，必须作为结果的一部分披露。

**必须修改。**

1. 明确定义 reliability-adjusted 公式、hard gates、置零顺序和 estimand；
2. 将 external semantic panel 明确称为每会一个 balanced atomic topic，并与 full-Core8 LOO 合同分开；
3. 分别报告 gate pass rates、unconditional composite score、common-valid-support conditional semantic score；
4. 报告按 meeting 聚合的 any-failure rate 和各 gate 原因；
5. 对 LOO 按 validity status 分层，并在 Full 与 intervention 共同有效的支持集上做 paired sensitivity；
6. 把 LOO 明确定位为 multi-block OOD stress test，不外推为常规 atomic rewriting task 的信息充分性；
7. 将 source-preserving 的低绝对有效率写入主要发现，不能只报告模型间均值差。

## 六、重要方法与呈现问题（P1 / Major）

### P1-1. 确定性 synthetic reference 仍不能验证 institutional authenticity

外部参考由独立的确定性 Python 模板生成，并非训练 teacher 生成；这避免了“同一 teacher 同时训练和打分”的错误描述，也是应在正文讲清的设计优点。但模板参考仍不是官方 Minutes 或人工专家标准，因此当前语义结果只支持“更接近预先冻结的 source-grounded synthetic target”，不能独立验证机构文体、政策含义或专家可接受性。建议增加：

- 官方 Minutes 和 teacher target 的分开对照；
- 货币经济学专家盲评 factual grounding、unsupported claims、tone、omission 和 policy interpretation；
- 数字、日期、方向和单位的 claim-level audit；
- deterministic-template reference、training teacher target 与 human/official benchmark 的三方比较；
- 人际一致性及不同 reference-construction mechanism 下的稳健性。

### P1-2. LOO 应统一称 target-relative prompt sensitivity

以 full-information deterministic synthetic reference 为固定目标时，删除一个 topic 后相似度下降在设计上部分是机械的，因为 reference 仍包含该 topic。它不证明 information sufficiency、经济重要性或因果贡献。建议：

- 明确披露 atomic-training contract 到 eight-block LOO contract 的 input shift；
- 主表报告 10,000 次 hierarchical bootstrap 的 primary confidence intervals，而不是只展示 supplemental paired t-tests；
- 统一方向性/双侧假设与 Holm family；
- 加入等长度无关数字块、随机 topic、顺序置换和 placebo 删除；
- 构建 intervention-consistent reference；
- 报告 mask-only 和 topic coalition effects；
- 在 replicate adequacy 仍标记 increase_k_recommended 时，不作细粒度 meeting-specific 解释。

### P1-3. 文本评价指标的实现和命名不一致

chapter2.tex:185–198 用 Llama hidden layers 解释 sentence embedding，但结果使用 MPNet；cosine similarity 的一般范围是 [-1,1]，正文却写成 0–1；reliability-adjusted score 没有给出公式。还需冻结并披露 exact MPNet/BERTScore revision、pooling、layer、IDF、baseline rescaling、invalid output 和 truncation 处理。Levenshtein、ROUGE、BOW 等基础概念段存在多处术语或拼写错误，建议精简为实际使用指标，而不是保留泛化教科书式介绍。

### P1-4. 情绪测量的经济有效性不足

Loughran–McDonald 词典服务于公司披露，不天然等于 hawkish/dovish monetary-policy stance。固定使用 0.368 的过滤参数，也未说明是否适用于不同来源、不同方差和自相关结构的生成文本。建议加入中央银行专用 stance 指标、人工标签或 embedding measure，并在 bootstrap draw 中重新估计过滤参数，或给出统一训练期预估和跨来源可比性依据。

K=5/K=10 的硬冲突已列为 P0-9。小节标题 Sentiment–Treasury Association 也与实际 30-Day Federal Funds futures 因变量不符。

### P1-5. 外部 Core8 信息集与训练合同不同，且缺少反应函数的基本状态变量

外部 deterministic Core8 合同排除了当前目标利率/区间、上一行动、PCE、通胀预期、工资、产出缺口、金融条件和市场预期；这不等于训练中的 variable-topic teacher-compressed briefs 也具有完全相同字段。政策方向和幅度无法脱离当前 stance 与 forward guidance 判断。teacher 先看到真实行动再生成 rationale，本质上也更接近 hindsight rationalization。

若保留预测语言，应先统一训练与评价合同，再纳入会前已知的当前目标区间、上一行动、实时预测、通胀预期和期货隐含概率，并做有/无 current stance 的消融。否则应把任务限定为在特定 Core8 schema 下的 historical action classification。

### P1-6. 跨货币政策制度的行动标签缺少统一定义

1993–2008 期间包括显性目标制度形成、点目标和 2008 年开始的目标区间。正文未说明 action 使用 midpoint、upper bound 还是其他序列，也未说明临时会议、会议间行动、不规则幅度和技术调整。2008-12-16 之前不应统一称 target-range change。应发布逐会议标签、官方来源、冲突裁决和 operating-regime 分层结果。

### P1-7. 样本粒度、长度与截断风险需统一

dataset_construction.tex:25 称 486 section-level samples，48–61 又报告 3,890 extracted section rows，实际更像 paragraph rows；需要区分 source sections、paragraph rows、meeting-topic rows 和 generation units。

同一数据表报告最大长度约 7,128 tokens，却以“sections shorter than 4,096”为由设置 4,096 cap；chk-1 使用 7,168 context，chk-2 使用 4,096。应报告每阶段 prompt/target token 分布、截断比例、截断位置，以及 reasoning/answer boundary 是否丢失。

### P1-8. 弱监督标签需要独立质量验证

ChatGPT-4o 产生 topic labels 后虽经过 ontology 清洗，但没有 exact model snapshot、prompt、sampling settings、双人标注 codebook 或 precision/recall。建议对按年代和 topic 分层的样本进行人工双盲复核，并报告 multi-label precision、recall、F1、inter-annotator agreement 和主要混淆对。

### P1-9. checkpoint 选择面板过小且单次训练不能支撑一般化训练效果

chk-1 主要依赖 8 个 probe completions，但实际只有 6 个 unique atomic rows/meetings，其中 3 个来自 validation、3 个来自内部 test，两个 rows 又以 sampled/greedy 重复；这些 probes 被用于排除 cp80/150/255 并选择 cp200。chk-2 的 sealed N12 是 12 个 atomic rows、仅 9 个 unique meetings，且全部来自 minutes_alignment test manifest，评分卡据此选择 cp318。因此内部 test 只保持 gradient-held-out，不再保持 selection-held-out，不能称 blind test 或用于无偏内部泛化结论。1993–2008 external N128 仍保持 selection-held-out，核心外部结果不因该问题自动失效。

此外，validation loss 的 199 行实际上只有 13 场会议，topic 较多的会议可能权重更高。建议：

- 扩大 meeting-stratified selection panel；
- 把已打开的内部 test 明确重命名为 selection panel，并另留真正 blind panel；
- 补 meeting-equal-weighted validation loss；
- 对 checkpoint probes 同时报告 completion、atomic-row 和 unique-meeting 分母，并采用 meeting-equal weighting；
- 报告全部候选 checkpoint 和 gate uncertainty；
- 每个训练阶段至少重复多个训练 seed；
- 明确选点后的外部评价没有再反向用于模型选择。

### P1-10. 当前文献综述不足以支撑博士论文定位

现行 literature_review.tex 仅约 528 词、30 行，且把 ARMA/GARCH 作为 synthetic data 例子并不能建立本章的金融学问题。至少应形成以下五条文献线：

1. **货币政策 surprise 与收益率曲线**：target/path factors、期货隐含预期及高频识别；
2. **中央银行信息效应**：政策行动与央行对经济状态的信息披露如何共同影响资产价格；
3. **FOMC communication/text-as-data**：Statement、Minutes、新闻发布会、hawkish/dovish tone 与市场反应；
4. **实时数据与政策反应函数**：real-time vintages、Taylor-rule/ordered response、forecast benchmarks；
5. **LLM 金融文本生成与验证**：domain adaptation、teacher dependence、contamination、hallucination、human evaluation 和 external validity。

综述末尾应明确区分三类潜在贡献：文本生成方法、金融测量工具和政策预测。当前证据最接近前两者中的“受限测量工具”，不支持第三类。

## 七、计算、表格和复现核查

### 7.1 已复核的关键数量

| 核查项 | 正文或表格 | 独立复核 | 结论 |
|---|---:|---:|---|
| chk-1 trainable parameters | 3,407,872 | 83,886,080 | 硬性冲突 |
| chk-1 trainable share | 0.0423% | 1.033825% of actual PEFT total | 硬性冲突 |
| Always-Hold direction accuracy | 未报告 | 19/31 = 61.29% | 应成为主基准 |
| 最佳模型 direction accuracy | 24.84% | 与表格一致 | 远低于 Always-Hold |
| 最佳模型 exact-action | 未汇总 | 60/310 = 19.35% | 远低于 Always-Hold |
| chk-2 相对 chk-0 的平均距离优势 | 文字称很小 | 0.6278 - 0.6223 = 0.0055 | 需直接推断 |
| LOO 生成数 | 128×17×10 | 21,760 | 算术一致 |

### 7.2 表格与正文问题

- chapter2.tex:522–570 的 chk-1 gate 表声明四列，但数据行字段不足，Model result 列疑似为空，需重新排版。
- chapter2.tex:659 将 Figure 写成 Table。
- 决策方法承诺 macro-F1、exact-action 和 delivery，主结果却未完整报告。
- meeting-modal 在 K=10 时可能平票，未定义 tie-breaking 和 invalid output 处理。
- all 77 gates 应改为 all seven gates。
- chapter2.tex:1149 的 TThis、a empirical，597 的 chk-1checkpoint、600 的 summaried，以及 Chain of Though、ROGUE、BOG 等语言/术语错误应系统清理。

## 八、LaTeX、图表与可读性审阅

### 8.1 当前缺失或未定义的交叉引用

1. chapter2.tex:590、662 引用 app:checkpoint_selection，但现行 sections 目录中没有 checkpoint_selection_appendix.tex，也没有有效 input；仅 archive 中存在旧文件。
2. dataset_construction.tex:1265–1266 引用 tab:ch2:chk3_training_composition 和 tab:ch2:chk3_bounded_resampling，现行源文件中没有对应 label。
3. GRPO 图在 chapter2.tex:811–815 没有 caption，label 不能形成完整、稳定的读者引用。
4. 当前工作区缺少完整 thesis master 和主 bibliography，无法验证 ch:ch0、sec:ch0:ml-finance 及全部 citation 是否在整篇论文中解析。

以上问题应在提交前通过一次 clean full build 验证，要求无 undefined references、multiply defined labels、missing citations 或 overfull critical boxes。

### 8.2 图表实查

- **stage1.png**：两分支主线整体已更新且可读，但 Domain-Specific Knowledge 容易把 synthetic-target fitting 写成更强的知识获取，可改为 task-specific synthetic-target adaptation。
- **stage2.png**：决策分支输入仍画成 Minutes，与 target-neutral pre-meeting brief 的正文设计直接冲突。必须重绘，不能只靠 caption 或脚注纠错。
- **model_training.tex:39–46**：称图中仍使用旧 chk-3/chk-4 标识，但当前图已经显示 paper-level chk-2/chk-3；该 caveat 本身已经过期。
- **chk-2 与 chk-3 SFT loss 图**：主体曲线清楚，但两份 PDF 的标题 bounding box 均越过页面上边界，底部 note 也过于贴近页面边缘，实际渲染可出现裁切；应统一增加上下 margins 后重新导出。chk-2 图已区分 batch-level train loss 与 199 个验证样本的 evaluation loss，也标记 cp250 与 cp318；正文仍需解释为何放弃最低 loss 而保留 hard-gate checkpoint。
- **GRPO reward 图**：图下注明阴影只是 step-level SE、不是 confidence interval，这一限定是正确的。仍应强调 20-step overlapping window、序列相关和单次训练运行使其只能作描述性训练诊断。

### 8.3 章节结构失衡

基础模型架构、激活函数和 chain-of-thought 的解释篇幅较长，而文献综述和经济识别明显不足。金融博士论文的读者更关心：

- 为什么该信息集在决策时点可得；
- 何种经济机制使文本指标与不同期限期货相关；
- benchmark 和 counterfactual 是什么；
- 样本、估计量与推断单位是否一致；
- 结论能否跨制度、跨时期外推。

建议把通用 Transformer 数学背景、完整 prompts、gate 明细和 checkpoint 表移至附录，把主文篇幅让给识别、实时数据、经济量级和稳健性。

此外，当前大量主要层级使用 starred section/subsection，约百页的章节内容因而不能稳定进入目录，相关 label 也可能落到父层级。主要研究设计、数据、评价和结果应采用有编号标题。chapter2_outline.md 仍保留旧的端到端流程、旧模型身份、旧 reward 和 Treasury association 表述，也应更新或明确标为 archive，避免其与现行研究设计并存。

## 九、建议的章节重构

### 9.1 建议标题

当前 Synthetic Texts Generation... 不够自然，也把“生成”写得比识别更突出。可考虑：

> Source-Grounded Synthetic Monetary-Policy Text Generation and Evaluation

若保留决策负结果：

> Task-Specific Adaptation for Monetary-Policy Text Generation: Alignment, Sensitivity, and Limits of Historical Action Classification

### 9.2 建议主文顺序

1. **Introduction**：收窄研究问题，给出可识别 estimands 和明确非目标；
2. **Literature and Economic Motivation**：建立 monetary surprise、information effect、FOMC text、real-time data、LLM validation 五条线；
3. **Institutional Setting and Information Timeline**：先讲会议、Statement、Minutes、数据 vintage 和市场窗口；
4. **Data Construction and Leakage Controls**：meeting ledger、topic universe、point-in-time vintage、split；
5. **Shared Parent and Two Independent Branches**：只保留必要的模型结构与训练配置；
6. **Primary Rewriting Evidence**：deterministic synthetic-reference alignment，清楚标明外部样本；
7. **Sensitivity and Economic Diagnostics**：LOO 与市场系数关联，全部按边界化命名；
8. **Independent Decision Branch: A Negative Benchmarking Result**：先放 Always-Hold 和市场基准，再放模型结果；
9. **Limitations and Conclusion**：保留当前结论中较谨慎的限定；
10. **Appendix**：prompts、architecture、checkpoint gates、全配置、数据字典、会议名单、额外表格。

### 9.3 建议的 RQ 重写

**RQ1.** On a meeting-indexed balanced atomic-topic panel held out from all task-specific post-training and checkpoint selection in the evaluated rewriting-model lineage, does post-training improve alignment with a frozen deterministic source-grounded synthetic reference?

**RQ2.** How sensitive is the retained rewriting model's reference alignment to exact deletion and neutral replacement of prespecified, point-in-time economic input blocks?

**RQ3.** Do sentiment measures extracted from generated documents reproduce selected historical coefficient patterns associated with Federal Funds futures, without implying causal market impact or equivalence?

**RQ4.** Do the historical-action classifiers outperform prespecified naive, econometric, and market-implied benchmarks on meeting-level panels held out from task-specific training?

这样写后，即使 RQ4 得到明确负结果，也仍然是可以解释、可以复核的博士论文结果。

## 十、建议的修订路线与验收标准

### 第一阶段：冻结事实与复现身份

**任务。**

- 冻结 meeting ledger、split、overlap matrix、generation manifests；
- 绑定准确 checkpoint、tokenizer、chat template、adapter hash 和 trainable parameter count；
- 补齐 row-level attrition、数据字典和 override 记录；
- 恢复/更新 checkpoint appendix，修复所有引用和图表。

**验收标准。**

- 任一结果表均可追溯到唯一的会议名单、生成工件和模型 hash；
- 128 场样本年份在所有位置完全一致；
- clean full build 无未定义引用；
- 参数数量与运行诊断一致。

### 第二阶段：修复信息集和基准

**任务。**

- 统一 meeting-start date，并保留已验证的 ALFRED real-time vintages；
- 固定 topic universe，完成 mask leakage 消融；
- 加入 Always-Hold、规则模型、econometric model 和 futures-implied benchmark；
- 按历史 panel 与 post-cutoff panel 分开报告。

**验收标准。**

- 每个输入值均可证明在会议开始前可得；
- topic mask 不由同会议会后 Minutes 决定；
- 决策结论以相对 prespecified benchmark 的 meeting-level 性能为依据。

### 第三阶段：重估推断与构念

**任务。**

- 展示 LOO primary hierarchical-bootstrap intervals；
- 对市场系数作事件级、联合和等价推断；
- 将正文 K=10 修正为工件已确认的 K=5；
- 加入独立人工/官方文本验证；
- 多训练 seed 评估训练不确定性。

**验收标准。**

- 随机生成不再被当作独立会议；
- 任何“相近、等价、改善”均有直接差值及不确定性；
- teacher alignment 与 institutional authenticity 被明确分开；
- GRPO 若无 gate-passing checkpoint，则不作确认性改善结论。

### 第四阶段：统一叙事

**任务。**

- 用结论中的谨慎边界重写引言、贡献、RQ、标题和摘要；
- 重建文献综述；
- 压缩通用模型教材内容；
- 完成英文专业润色。

**验收标准。**

- 每个贡献句都能指向一张表或一个可识别 estimand；
- 不再出现 market impact、decision support、authenticity 等超出证据的未限定主张；
- 负向决策结果被准确呈现为主要发现之一。

## 十一、建议保留的最窄主结论

下面这段可作为英文摘要或结论的事实边界参考：

> Using a shared parent model and two separate task branches, this chapter finds that post-training improves gate-adjusted alignment with a frozen deterministic source-grounded synthetic reference on a meeting-indexed balanced atomic-topic panel held out from all task-specific post-training and checkpoint selection in the evaluated rewriting-model lineage. Date overlap with the independent decision branch does not enter the rewriting weights. This is not a full-Minutes evaluation and does not exclude possible exposure through the base model's pretraining corpus. Input-deletion results under a different full-Core8 prompt contract indicate reference-relative sensitivity to selected economic blocks, but do not identify causal information importance or sufficiency. Sentiment-based futures regressions provide only a limited historical coefficient-association diagnostic and do not establish market impact or equivalence with official Minutes. The independent historical-action classifiers fail to outperform a simple no-change benchmark, and no GRPO checkpoint satisfies the prespecified selection gates. The evidence therefore supports a bounded research-instrument interpretation, rather than real-time forecasting, trading, or policy-decision use.

这一定义牺牲了一些宣传性，但显著提高了金融学上的可辩护性，也与当前结果和结论基本一致。

## 十二、最终审阅意见

**建议：Major Revision。**

当前版本的主要价值在于：透明地展示了一个 source-grounded、teacher-supervised 的货币政策文本适配流程，并且没有掩盖决策分支和 GRPO 的负结果。当前最主要的障碍并非“模型效果不够高”，而是数据时点、topic 选择、样本身份、模型工件和 benchmark 尚未形成闭合的金融实证链条；与此同时，引言仍把有限的诊断性证据写成更强的市场和政策含义。

如能优先解决 P0-2 至 P0-8 的事实与复现问题，再按 P0-1 和 P0-9 收窄经济解释，本章可以形成可信的博士论文方法章节，并把“模型未超过简单政策基准”转化为有信息含量的负结果。若不解决这些问题，则即使增加更多语言润色或图表，也不足以支持答辩级别的核心实证结论。

## 参考核验来源

- DeepSeek-R1-Distill-Llama-8B 官方模型卡：<https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B>
- ALFRED 实时数据与历史 vintage 说明：<https://alfred.stlouisfed.org/help>
- Federal Reserve 对 FOMC Minutes 发布制度历史的说明：<https://www.federalreserve.gov/pubs/bulletin/2005/spring05_fomc.pdf>
- 2008-12-16 FOMC 关于建立 0–1/4 percent target range 的声明：<https://www.federalreserve.gov/newsevents/pressreleases/monetary20081216b.htm>
